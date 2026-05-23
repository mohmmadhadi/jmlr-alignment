import os
import pandas as pd
from src.data.fetcher import ArxivFetcher
from src.data.preprocessor import TextProcessor


def run(max_results: int = 500) -> None:
    """Run the full data collection and preprocessing pipeline.

    Args:
        max_results: Number of papers to fetch from arXiv.
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    fetcher = ArxivFetcher(delay=1.0)
    records = fetcher.fetch(
        query='jo:"Journal of Machine Learning Research"',
        max_results=max_results,
    )

    raw_df = pd.DataFrame(records)
    raw_df.to_csv("data/raw/jmlr_papers.csv", index=False)
    print(f"Saved {len(raw_df)} raw records to data/raw/jmlr_papers.csv")

    processor = TextProcessor()
    processed_df = processor.process_records(records)
    processed_df = processed_df[
        ~processed_df["clean_abstract"].str.contains(r'\bjo\b', regex=True)
    ]
    processed_df.to_csv("data/processed/jmlr_papers_clean.csv", index=False)
    print(f"Saved {len(processed_df)} processed records to data/processed/jmlr_papers_clean.csv")


if __name__ == "__main__":
    run(max_results=500)