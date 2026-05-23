import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


class AlignmentEvaluator:
    """Computes thematic alignment scores between a reference and a corpus."""

    def __init__(self, scope_vector: np.ndarray, abstract_matrix: np.ndarray) -> None:
        """Initialize with precomputed embeddings.

        Args:
            scope_vector: 1D array representing the Aims & Scope embedding.
            abstract_matrix: 2D array of shape (n_papers, embedding_dim).
        """
        self.scope_vector = scope_vector.reshape(1, -1)
        self.abstract_matrix = abstract_matrix

    def compute_scores(self) -> np.ndarray:
        """Compute cosine similarity between scope and every abstract.

        Returns:
            1D array of alignment scores in range [0, 1], one per paper.
        """
        scores = cosine_similarity(self.scope_vector, self.abstract_matrix)
        return scores.flatten()

    def get_outliers(self, scores: np.ndarray, top_pct: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Identify the highest and lowest scoring papers.

        Args:
            scores: 1D array of alignment scores.
            top_pct: Fraction of papers to treat as outliers on each end.

        Returns:
            Tuple of (top_indices, bottom_indices) sorted by score.
        """
        n = max(1, int(len(scores) * top_pct))
        top_indices = np.argsort(scores)[::-1][:n]
        bottom_indices = np.argsort(scores)[:n]
        return top_indices, bottom_indices

    def attach_scores(self, df: pd.DataFrame, scores: np.ndarray, prefix: str = "") -> pd.DataFrame:
        """Merge alignment scores into the paper dataframe.

        Args:
            df: DataFrame with paper metadata.
            scores: 1D array of scores aligned to df rows.
            prefix: Optional column name prefix (e.g. 'tfidf_' or 'sbert_').

        Returns:
            DataFrame with a new alignment score column added.
        """
        col = f"{prefix}alignment_score"
        df = df.copy()
        df[col] = scores
        return df

    def detect_outliers(
            self,
            df: pd.DataFrame,
            score_col: str = "sbert_alignment_score",
            threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Flag papers as outliers if their score deviates beyond a std dev threshold.

        Args:
            df: DataFrame with alignment scores attached.
            score_col: Name of the score column to evaluate.
            threshold: Number of standard deviations from the mean to use as cutoff.

        Returns:
            DataFrame with three new columns:
                - score_mean: corpus mean alignment score
                - score_std: corpus standard deviation
                - outlier_type: 'high', 'low', or None
        """
        mean = df[score_col].mean()
        std = df[score_col].std()

        df = df.copy()
        df["score_mean"] = mean
        df["score_std"] = std

        df["outlier_type"] = None
        df.loc[df[score_col] > mean + threshold * std, "outlier_type"] = "high"
        df.loc[df[score_col] < mean - threshold * std, "outlier_type"] = "low"

        return df