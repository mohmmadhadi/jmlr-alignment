import numpy as np
import pandas as pd
from umap import UMAP


class UMAPProjector:
    """Reduces high-dimensional embeddings to 2D for visualization."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize UMAP with reproducible settings.

        Args:
            n_neighbors: Controls how UMAP balances local vs global structure.
                         Lower = more local clusters, higher = broader structure.
            min_dist: Minimum distance between points in 2D space.
                      Lower = tighter clusters, higher = more spread.
            random_state: Seed for reproducibility.
        """
        self.reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
        )
        self.embedding_2d = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to 2D.

        Args:
            embeddings: Array of shape (n_papers, embedding_dim).

        Returns:
            Array of shape (n_papers, 2) with x, y coordinates.
        """
        print(f"Running UMAP on {embeddings.shape[0]} vectors of dim {embeddings.shape[1]}...")
        self.embedding_2d = self.reducer.fit_transform(embeddings)
        print("Done.")
        return self.embedding_2d

    def attach_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add umap_x and umap_y columns to the paper DataFrame.

        Args:
            df: Paper DataFrame aligned to the fitted embeddings.

        Returns:
            DataFrame with umap_x and umap_y columns added.
        """
        if self.embedding_2d is None:
            raise RuntimeError("Call fit_transform() before attach_coordinates().")
        df = df.copy()
        df["umap_x"] = self.embedding_2d[:, 0]
        df["umap_y"] = self.embedding_2d[:, 1]
        return df