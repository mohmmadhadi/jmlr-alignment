import re
import pandas as pd
from typing import List, Dict, Any


class TextProcessor:
    """Cleans and normalizes raw text for embedding and comparison."""

    def clean(self, text: str) -> str:
        """Clean a single string.

        Args:
            text: Raw input string.

        Returns:
            Cleaned, normalized string.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9\s\.\,\-]', '', text)
        text = text.strip()
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a list of strings.

        Args:
            texts: List of raw strings.

        Returns:
            List of cleaned strings.
        """
        return [self.clean(t) for t in texts]

    def process_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert raw records to a cleaned DataFrame.

        Args:
            records: List of dicts from the fetcher.

        Returns:
            DataFrame with original fields plus a clean_abstract column.
        """
        df = pd.DataFrame(records)
        df = df.dropna(subset=["abstract"])
        df = df[df["abstract"].str.strip() != ""]
        df["clean_abstract"] = self.clean_batch(df["abstract"].tolist())
        df = df.reset_index(drop=True)
        return df