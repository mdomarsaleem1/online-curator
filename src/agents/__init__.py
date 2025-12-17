"""
Scraper Agents for various content sources.

Supports:
- ArXiv: Research papers
- HuggingFace: Models, datasets, spaces, papers
- YouTube: Video transcripts and metadata
- RSS: Generic RSS/Atom feeds
"""

from .base import BaseScraper
from .arxiv_scraper import ArxivScraper
from .huggingface_scraper import HuggingFaceScraper
from .youtube_scraper import YouTubeScraper
from .rss_scraper import RSSScraper
from .summarizer import SummarizerAgent

__all__ = [
    "BaseScraper",
    "ArxivScraper",
    "HuggingFaceScraper",
    "YouTubeScraper",
    "RSSScraper",
    "SummarizerAgent",
]
