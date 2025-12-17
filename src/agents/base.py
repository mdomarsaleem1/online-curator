"""Base scraper class with common functionality."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import httpx

from src.models.article import Article, ArticleCategory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Base class for all content scrapers.

    Provides common functionality for fetching, parsing, and storing articles.
    """

    def __init__(self, timeout: int = 30):
        """Initialize the scraper."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        self.source_name: str = "Unknown"
        self.default_category: ArticleCategory = ArticleCategory.OTHER

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()

    @abstractmethod
    def scrape(self, query: Optional[str] = None, limit: int = 10) -> List[Article]:
        """
        Scrape articles from the source.

        Args:
            query: Optional search query or filter
            limit: Maximum number of articles to return

        Returns:
            List of Article objects
        """
        pass

    @abstractmethod
    def parse_item(self, item: Any) -> Optional[Article]:
        """
        Parse a single item into an Article.

        Args:
            item: Raw item data from the source

        Returns:
            Article object or None if parsing fails
        """
        pass

    def fetch_url(self, url: str, headers: Optional[Dict] = None) -> Optional[str]:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch
            headers: Optional headers to include

        Returns:
            Response text or None if fetch fails
        """
        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def fetch_json(self, url: str, headers: Optional[Dict] = None) -> Optional[Dict]:
        """
        Fetch JSON from a URL.

        Args:
            url: URL to fetch
            headers: Optional headers to include

        Returns:
            JSON data or None if fetch fails
        """
        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch JSON from {url}: {e}")
            return None

    def clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Limit length
        if len(text) > 10000:
            text = text[:10000] + "..."
        return text

    def extract_tags(self, text: str) -> List[str]:
        """Extract potential tags from text."""
        # Simple keyword extraction - can be enhanced
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Deduplicate and limit
        seen = set()
        tags = []
        for word in words:
            lower = word.lower()
            if lower not in seen and len(word) > 2:
                seen.add(lower)
                tags.append(word)
        return tags[:10]
