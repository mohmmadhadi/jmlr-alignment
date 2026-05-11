import os
import numpy as np
import pandas as pd
from src.models.tfidf_embedder import TFIDFEmbedder
from src.models.sbert_embedder import SBERTEmbedder


def run() -> None:
    """Generate and save embeddings for the Aims & Scope and all abstracts."""

    os.makedirs("data/embeddings", exist_ok=True)

    df = pd.read_csv("data/processed/jmlr_papers_clean.csv")
    print(f"Loaded {len(df)} papers.")

    with open("src/data/aims_and_scope.txt", "r") as f:
        scope_text = f.read()

    abstracts = df["clean_abstract"].fillna("").tolist()
    all_texts = [scope_text] + abstracts

    # --- TF-IDF ---
    print("\nGenerating TF-IDF embeddings...")
    tfidf = TFIDFEmbedder(max_features=5000)
    tfidf_matrix = tfidf.encode(all_texts)
    np.save("data/embeddings/tfidf_scope.npy", tfidf_matrix[0])
    np.save("data/embeddings/tfidf_abstracts.npy", tfidf_matrix[1:])
    print(f"TF-IDF shape: {tfidf_matrix[1:].shape}")

    # --- SBERT ---
    print("\nGenerating SBERT embeddings...")
    sbert = SBERTEmbedder(model_name="all-MiniLM-L6-v2")
    sbert_scope = sbert.encode([scope_text])
    sbert_abstracts = sbert.encode(abstracts)
    np.save("data/embeddings/sbert_scope.npy", sbert_scope[0])
    np.save("data/embeddings/sbert_abstracts.npy", sbert_abstracts)
    print(f"SBERT shape: {sbert_abstracts.shape}")

    print("\nAll embeddings saved to data/embeddings/")


if __name__ == "__main__":
    run()