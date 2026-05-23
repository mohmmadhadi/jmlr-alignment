import numpy as np
import pandas as pd
from src.models.topic_model import JMLRTopicModel


def run() -> None:
    """Fit BERTopic on SBERT embeddings and save topic assignments."""

    df = pd.read_csv("data/processed/jmlr_papers_scored.csv")
    embeddings = np.load("data/embeddings/sbert_abstracts.npy")
    abstracts = df["clean_abstract"].fillna("").tolist()

    print(f"Fitting BERTopic on {len(abstracts)} abstracts...")
    model = JMLRTopicModel(n_topics=20, min_topic_size=8)
    topic_info = model.fit(abstracts, embeddings)

    print("\nTopic overview:")
    print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

    df = model.attach_topics(df)
    df.to_csv("data/processed/jmlr_papers_topics.csv", index=False)
    print("\nSaved: data/processed/jmlr_papers_topics.csv")

    summary = model.get_topic_alignment_summary(df)
    summary.to_csv("data/processed/topic_alignment_summary.csv", index=False)
    print("Saved: data/processed/topic_alignment_summary.csv")

    print("\nTopic alignment summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()