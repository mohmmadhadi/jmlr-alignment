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

    # --- Print summary statistics ---
    for prefix in ["tfidf_", "sbert_"]:
        col = f"{prefix}alignment_score"
        print(f"\n{col} summary:")
        print(f"  mean:   {df[col].mean():.4f}")
        print(f"  std:    {df[col].std():.4f}")
        print(f"  min:    {df[col].min():.4f}")
        print(f"  max:    {df[col].max():.4f}")

    # --- Print top and bottom 5 papers by SBERT score ---
    for label, ascending in [("Bottom 5 (lowest alignment)", True), ("Top 5 (highest alignment)", False)]:
        print(f"\n{label}:")
        subset = df.sort_values("sbert_alignment_score", ascending=ascending).head(5)
        for _, row in subset.iterrows():
            print(f"  [{row['sbert_alignment_score']:.4f}] {row['title'][:80]}")


if __name__ == "__main__":
    run()