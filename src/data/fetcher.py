import arxiv
import time
import logging
from typing import List, Dict, Any
from src.base import BaseFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivFetcher(BaseFetcher):
    """Fetches paper metadata from arXiv for a given journal/query."""

    def __init__(self, delay: float = 1.0) -> None:
        """Initialize the fetcher.

        Args:
            delay: Seconds to wait between paginated requests (rate limiting).
        """
        self.delay = delay
        self.client = arxiv.Client()

    def fetch(self, query: str, max_results: int = 500) -> List[Dict[str, Any]]:
        """Fetch paper metadata from arXiv.

        Args:
            query: arXiv search query (e.g. journal name or topic).
            max_results: Maximum number of papers to retrieve.

        Returns:
            List of dicts with keys: title, abstract, authors, year, doi, arxiv_id.
        """
        logger.info(f"Fetching up to {max_results} results for query: '{query}'")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        records = []
        for i, result in enumerate(self.client.results(search)):
            records.append({
                "title": result.title,
                "abstract": result.summary,
                "authors": ", ".join(a.name for a in result.authors),
                "year": result.published.year,
                "doi": result.doi or "",
                "arxiv_id": result.entry_id,
            })

            if (i + 1) % 50 == 0:
                logger.info(f"  Fetched {i + 1} records...")
                time.sleep(self.delay)

        logger.info(f"Done. Total records fetched: {len(records)}")
        return records