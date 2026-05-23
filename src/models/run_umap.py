import numpy as np
import pandas as pd
from src.models.umap_projector import UMAPProjector


def run() -> None:
    """Project SBERT embeddings to 2D and save coordinates to the dataset."""

    df = pd.read_csv("data/processed/jmlr_papers_scored.csv")
    embeddings = np.load("data/embeddings/sbert_abstracts.npy")

    projector = UMAPProjector(n_neighbors=15, min_dist=0.1, random_state=42)
    projector.fit_transform(embeddings)
    df = projector.attach_coordinates(df)

    df.to_csv("data/processed/jmlr_papers_umap.csv", index=False)
    print(f"Saved UMAP coordinates to data/processed/jmlr_papers_umap.csv")
    print(f"x range: [{df['umap_x'].min():.2f}, {df['umap_x'].max():.2f}]")
    print(f"y range: [{df['umap_y'].min():.2f}, {df['umap_y'].max():.2f}]")


if __name__ == "__main__":
    run()