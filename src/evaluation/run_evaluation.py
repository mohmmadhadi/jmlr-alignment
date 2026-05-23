import numpy as np
import pandas as pd
from src.evaluation.evaluator import AlignmentEvaluator


def run() -> None:
    """Load embeddings, compute alignment scores, and save results."""

    df = pd.read_csv("data/processed/jmlr_papers_clean.csv")
    print(f"Loaded {len(df)} papers.")

    # --- TF-IDF scores ---
    print("\nComputing TF-IDF alignment scores...")
    tfidf_scope = np.load("data/embeddings/tfidf_scope.npy")
    tfidf_abstracts = np.load("data/embeddings/tfidf_abstracts.npy")

    tfidf_eval = AlignmentEvaluator(tfidf_scope, tfidf_abstracts)
    tfidf_scores = tfidf_eval.compute_scores()
    df = tfidf_eval.attach_scores(df, tfidf_scores, prefix="tfidf_")

    # --- SBERT scores ---
    print("Computing SBERT alignment scores...")
    sbert_scope = np.load("data/embeddings/sbert_scope.npy")
    sbert_abstracts = np.load("data/embeddings/sbert_abstracts.npy")

    sbert_eval = AlignmentEvaluator(sbert_scope, sbert_abstracts)
    sbert_scores = sbert_eval.compute_scores()
    df = sbert_eval.attach_scores(df, sbert_scores, prefix="sbert_")

    # --- Save results ---
    df.to_csv("data/processed/jmlr_papers_scored.csv", index=False)
    print(f"\nSaved scored dataset to data/processed/jmlr_papers_scored.csv")

    # --- Outlier detection ---


    print("\nDetecting outliers (±2 std dev threshold)...")
    df = sbert_eval.detect_outliers(df, score_col="sbert_alignment_score", threshold=2.0)

    n_high = (df["outlier_type"] == "high").sum()
    n_low = (df["outlier_type"] == "low").sum()
    mean = df["score_mean"].iloc[0]
    std = df["score_std"].iloc[0]

    print(f"  Corpus mean:  {mean:.4f}")
    print(f"  Corpus std:   {std:.4f}")
    print(f"  Upper bound:  {mean + 2 * std:.4f}  →  {n_high} high outliers")
    print(f"  Lower bound:  {mean - 2 * std:.4f}  →  {n_low} low outliers")

    df.to_csv("data/processed/jmlr_papers_scored.csv", index=False)
    print("\nUpdated scored CSV with outlier flags.")

    print("\nHigh outliers (above mean + 2σ):")
    high = df[df["outlier_type"] == "high"].sort_values("sbert_alignment_score", ascending=False)
    for _, row in high.iterrows():
        print(f"  [{row['sbert_alignment_score']:.4f}] {row['title'][:75]}")

    print("\nLow outliers (below mean − 2σ):")
    low = df[df["outlier_type"] == "low"].sort_values("sbert_alignment_score")
    for _, row in low.iterrows():
        print(f"  [{row['sbert_alignment_score']:.4f}] {row['title'][:75]}")




if __name__ == "__main__":
    run()