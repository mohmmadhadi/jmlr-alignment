import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.base import BaseEmbedder


class SBERTEmbedder(BaseEmbedder):
    """Semantic text embedder using a pre-trained Sentence-BERT model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize and load the Sentence-BERT model.

        Args:
            model_name: HuggingFace model identifier. 
                        'all-MiniLM-L6-v2' is fast and lightweight.
                        'all-mpnet-base-v2' is slower but more accurate.
        """
        print(f"Loading SBERT model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into dense semantic vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            2D numpy array of shape (n_texts, embedding_dim).
            For all-MiniLM-L6-v2, embedding_dim = 384.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        return embeddings