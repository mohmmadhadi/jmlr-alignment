import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import TextProcessor
from src.evaluation.evaluator import AlignmentEvaluator


class TestTextProcessor:
    """Unit tests for the TextProcessor class."""

    def setup_method(self) -> None:
        self.processor = TextProcessor()

    def test_clean_lowercases(self) -> None:
        assert self.processor.clean("Hello World") == "hello world"

    def test_clean_removes_special_chars(self) -> None:
        result = self.processor.clean("machine learning! @#$%")
        assert "!" not in result
        assert "@" not in result

    def test_clean_collapses_whitespace(self) -> None:
        result = self.processor.clean("too   many    spaces")
        assert "  " not in result

    def test_clean_handles_non_string(self) -> None:
        assert self.processor.clean(None) == ""
        assert self.processor.clean(123) == ""

    def test_clean_batch_returns_correct_length(self) -> None:
        texts = ["Hello", "World", "Machine Learning"]
        result = self.processor.clean_batch(texts)
        assert len(result) == 3

    def test_process_records_drops_empty_abstracts(self) -> None:
        records = [
            {"title": "Paper A", "abstract": "valid abstract", "authors": "X", "year": 2022, "doi": "", "arxiv_id": "1"},
            {"title": "Paper B", "abstract": "", "authors": "Y", "year": 2022, "doi": "", "arxiv_id": "2"},
            {"title": "Paper C", "abstract": None, "authors": "Z", "year": 2022, "doi": "", "arxiv_id": "3"},
        ]
        df = self.processor.process_records(records)
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Paper A"

    def test_process_records_adds_clean_abstract_column(self) -> None:
        records = [{"title": "A", "abstract": "Some Text!", "authors": "X", "year": 2021, "doi": "", "arxiv_id": "1"}]
        df = self.processor.process_records(records)
        assert "clean_abstract" in df.columns


class TestAlignmentEvaluator:
    """Unit tests for the AlignmentEvaluator class."""

    def setup_method(self) -> None:
        self.scope = np.array([1.0, 0.0, 0.0])
        self.abstracts = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        self.evaluator = AlignmentEvaluator(self.scope, self.abstracts)

    def test_compute_scores_returns_correct_shape(self) -> None:
        scores = self.evaluator.compute_scores()
        assert scores.shape == (3,)

    def test_compute_scores_identical_vector_is_one(self) -> None:
        scores = self.evaluator.compute_scores()
        assert pytest.approx(scores[0], abs=1e-6) == 1.0

    def test_compute_scores_orthogonal_vector_is_zero(self) -> None:
        scores = self.evaluator.compute_scores()
        assert pytest.approx(scores[1], abs=1e-6) == 0.0

    def test_compute_scores_opposite_vector_is_negative(self) -> None:
        scores = self.evaluator.compute_scores()
        assert scores[2] < 0

    def test_attach_scores_adds_column(self) -> None:
        df = pd.DataFrame({"title": ["A", "B", "C"]})
        scores = np.array([0.1, 0.5, 0.9])
        result = self.evaluator.attach_scores(df, scores, prefix="sbert_")
        assert "sbert_alignment_score" in result.columns

    def test_attach_scores_does_not_mutate_original(self) -> None:
        df = pd.DataFrame({"title": ["A", "B", "C"]})
        scores = np.array([0.1, 0.5, 0.9])
        self.evaluator.attach_scores(df, scores, prefix="sbert_")
        assert "sbert_alignment_score" not in df.columns

    def test_empty_abstract_matrix_raises(self) -> None:
        with pytest.raises(Exception):
            ev = AlignmentEvaluator(self.scope, np.array([]))
            ev.compute_scores()


class TestOutlierDetection:
    """Unit tests for the detect_outliers method."""

    def setup_method(self) -> None:
        scope = np.array([1.0, 0.0, 0.0])
        abstracts = np.array([[1.0, 0.0, 0.0]] * 10)
        self.evaluator = AlignmentEvaluator(scope, abstracts)

    def test_outlier_columns_added(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.9])
        df = pd.DataFrame({"title": [f"Paper {i}" for i in range(10)]})
        df["sbert_alignment_score"] = scores
        result = self.evaluator.detect_outliers(df)
        assert "outlier_type" in result.columns
        assert "score_mean" in result.columns
        assert "score_std" in result.columns

    def test_high_outlier_flagged(self) -> None:
        scores = np.array([0.3] * 9 + [0.99])
        df = pd.DataFrame({"title": [f"Paper {i}" for i in range(10)]})
        df["sbert_alignment_score"] = scores
        result = self.evaluator.detect_outliers(df, threshold=2.0)
        assert result.iloc[-1]["outlier_type"] == "high"

    def test_low_outlier_flagged(self) -> None:
        scores = np.array([-0.5] + [0.3] * 9)
        df = pd.DataFrame({"title": [f"Paper {i}" for i in range(10)]})
        df["sbert_alignment_score"] = scores
        result = self.evaluator.detect_outliers(df, threshold=2.0)
        assert result.iloc[0]["outlier_type"] == "low"

    def test_no_mutation_of_original(self) -> None:
        df = pd.DataFrame({"title": ["A"], "sbert_alignment_score": [0.3]})
        self.evaluator.detect_outliers(df)
        assert "outlier_type" not in df.columns


from src.models.umap_projector import UMAPProjector

class TestUMAPProjector:
    """Unit tests for the UMAPProjector class."""

    def test_output_shape(self) -> None:
        embeddings = np.random.rand(50, 384)
        projector = UMAPProjector(n_neighbors=5, random_state=42)
        result = projector.fit_transform(embeddings)
        assert result.shape == (50, 2)

    def test_attach_coordinates_adds_columns(self) -> None:
        embeddings = np.random.rand(50, 384)
        projector = UMAPProjector(n_neighbors=5, random_state=42)
        projector.fit_transform(embeddings)
        df = pd.DataFrame({"title": [f"Paper {i}" for i in range(50)]})
        result = projector.attach_coordinates(df)
        assert "umap_x" in result.columns
        assert "umap_y" in result.columns

    def test_attach_before_fit_raises(self) -> None:
        projector = UMAPProjector()
        df = pd.DataFrame({"title": ["A"]})
        with pytest.raises(RuntimeError):
            projector.attach_coordinates(df)

    def test_does_not_mutate_original(self) -> None:
        embeddings = np.random.rand(50, 384)
        projector = UMAPProjector(n_neighbors=5, random_state=42)
        projector.fit_transform(embeddings)
        df = pd.DataFrame({"title": [f"Paper {i}" for i in range(50)]})
        projector.attach_coordinates(df)
        assert "umap_x" not in df.columns