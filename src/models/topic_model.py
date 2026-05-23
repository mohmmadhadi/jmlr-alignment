import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple


class JMLRTopicModel:
    """Fits a BERTopic model on paper abstracts and extracts topic distributions."""

    def __init__(
        self,
        n_topics: int = 15,
        min_topic_size: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        """Initialize BERTopic with sensible defaults for a ML journal corpus.

        Args:
            n_topics: Target number of topics. 'auto' lets HDBSCAN decide.
            min_topic_size: Minimum papers per topic cluster.
            ngram_range: N-gram range for topic keyword extraction.
        """
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=2,
        )
        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

        self.model = BERTopic(
            nr_topics=n_topics,
            min_topic_size=min_topic_size,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            calculate_probabilities=True,
            verbose=True,
        )
        self.topics = None
        self.probs = None
        self.topic_info = None

    def fit(
        self,
        abstracts: list[str],
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        """Fit the topic model using precomputed SBERT embeddings.

        Args:
            abstracts: List of cleaned abstract strings.
            embeddings: Precomputed embedding matrix of shape (n_papers, dim).

        Returns:
            DataFrame with topic assignments and top keywords per topic.
        """
        self.topics, self.probs = self.model.fit_transform(
            abstracts,
            embeddings=embeddings,
        )
        self.topic_info = self.model.get_topic_info()
        return self.topic_info

    def get_topic_labels(self) -> dict[int, str]:
        """Return a dict mapping topic ID to its top 3 keywords.

        Returns:
            Dict of {topic_id: 'keyword1, keyword2, keyword3'}.
        """
        labels = {}
        for tid in self.topic_info["Topic"].tolist():
            if tid == -1:
                labels[tid] = "outlier / noise"
                continue
            words = self.model.get_topic(tid)
            if words:
                labels[tid] = ", ".join([w for w, _ in words[:3]])
        return labels

    def attach_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add topic assignment and label columns to the paper DataFrame.

        Args:
            df: Paper DataFrame aligned to the fitted abstracts.

        Returns:
            DataFrame with new columns: topic_id, topic_label.
        """
        labels = self.get_topic_labels()
        df = df.copy()
        df["topic_id"] = self.topics
        df["topic_label"] = df["topic_id"].map(labels)
        return df

    def get_topic_alignment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize mean alignment score per topic.

        Args:
            df: DataFrame with topic_id and sbert_alignment_score columns.

        Returns:
            Summary DataFrame sorted by mean alignment score descending.
        """
        summary = (
            df[df["topic_id"] != -1]
            .groupby(["topic_id", "topic_label"])
            .agg(
                n_papers=("title", "count"),
                mean_alignment=("sbert_alignment_score", "mean"),
                std_alignment=("sbert_alignment_score", "std"),
            )
            .reset_index()
            .sort_values("mean_alignment", ascending=False)
        )
        return summary