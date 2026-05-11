from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseFetcher(ABC):
    """Abstract base class for all API fetchers."""

    @abstractmethod
    def fetch(self, query: str, max_results: int) -> list:
        """Fetch records from a data source.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw records (dicts).
        """
        pass


class BaseEmbedder(ABC):
    """Abstract base class for all text embedding models."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into a matrix of vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            2D numpy array of shape (n_texts, embedding_dim).
        """
        pass