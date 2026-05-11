import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from src.base import BaseEmbedder


class TFIDFEmbedder(BaseEmbedder):
    """Baseline text embedder using TF-IDF bag-of-words representation."""

    def __init__(self, max_features: int = 5000) -> None:
        """Initialize the TF-IDF vectorizer.

        Args:
            max_features: Maximum number of vocabulary terms to keep.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.fitted = False

    def fit(self, texts: List[str]) -> None:
        """Fit the vectorizer on a corpus.

        Args:
            texts: List of strings to fit on.
        """
        self.vectorizer.fit(texts)
        self.fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into TF-IDF vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            2D numpy array of shape (n_texts, max_features).
        """
        if not self.fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray()