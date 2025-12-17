"""ArXiv scraper for research papers."""

import re
from typing import List, Optional, Any
from datetime import datetime
import xml.etree.ElementTree as ET

from .base import BaseScraper, logger
from src.models.article import Article, ArticleCategory


class ArxivScraper(BaseScraper):
    """
    Scraper for ArXiv research papers.

    Uses the ArXiv API to fetch papers by category or search query.
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    # ArXiv categories relevant to ML/AI
    ML_CATEGORIES = [
        "cs.LG",   # Machine Learning
        "cs.AI",   # Artificial Intelligence
        "cs.CL",   # Computation and Language (NLP)
        "cs.CV",   # Computer Vision
        "cs.NE",   # Neural and Evolutionary Computing
        "stat.ML", # Statistics - Machine Learning
    ]

    def __init__(self, timeout: int = 30):
        """Initialize the ArXiv scraper."""
        super().__init__(timeout)
        self.source_name = "ArXiv"
        self.default_category = ArticleCategory.IDEA

    def scrape(
        self,
        query: Optional[str] = None,
        limit: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[Article]:
        """
        Scrape papers from ArXiv.

        Args:
            query: Search query (searches title, abstract, authors)
            limit: Maximum number of papers to return
            categories: List of ArXiv categories to filter by

        Returns:
            List of Article objects
        """
        # Build query
        search_parts = []

        if query:
            # Search in title and abstract
            search_parts.append(f'all:"{query}"')

        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_parts.append(f"({cat_query})")
        elif not query:
            # Default to ML categories
            cat_query = " OR ".join([f"cat:{cat}" for cat in self.ML_CATEGORIES])
            search_parts.append(f"({cat_query})")

        search_query = " AND ".join(search_parts) if search_parts else "cat:cs.LG"

        # Build URL
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        url = f"{self.BASE_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        logger.info(f"Fetching ArXiv: {url}")

        # Fetch and parse
        xml_content = self.fetch_url(url)
        if not xml_content:
            return []

        return self._parse_feed(xml_content)

    def _parse_feed(self, xml_content: str) -> List[Article]:
        """Parse ArXiv Atom feed."""
        articles = []

        try:
            root = ET.fromstring(xml_content)
            # ArXiv uses Atom namespace
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            for entry in root.findall('atom:entry', ns):
                article = self._parse_entry(entry, ns)
                if article:
                    articles.append(article)

        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {e}")

        return articles

    def _parse_entry(self, entry: ET.Element, ns: dict) -> Optional[Article]:
        """Parse a single ArXiv entry."""
        try:
            # Extract fields
            title = entry.find('atom:title', ns)
            title_text = self.clean_text(title.text) if title is not None else ""

            summary = entry.find('atom:summary', ns)
            summary_text = self.clean_text(summary.text) if summary is not None else ""

            # Get the ArXiv ID and construct URLs
            id_elem = entry.find('atom:id', ns)
            arxiv_url = id_elem.text if id_elem is not None else ""

            # Extract arxiv ID from URL
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""

            # Get published date
            published = entry.find('atom:published', ns)
            pub_date = None
            if published is not None and published.text:
                try:
                    pub_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                except ValueError:
                    pass

            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None and name.text:
                    authors.append(name.text)

            # Get categories
            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term')
                if term:
                    categories.append(term)

            # Extract key insights from abstract (first few sentences)
            key_insights = self._extract_insights(summary_text)

            # Extract concepts from title and abstract
            concepts = self._extract_concepts(title_text, summary_text, categories)

            return Article(
                url=arxiv_url,
                title=title_text,
                source=self.source_name,
                summary=summary_text[:1000] + "..." if len(summary_text) > 1000 else summary_text,
                key_insights=key_insights,
                category=self.default_category,
                tags=categories[:5] + (["arxiv"] if "arxiv" not in categories else []),
                related_concepts=concepts,
                prerequisite_concepts=self._infer_prerequisites(concepts),
                published_at=pub_date,
            )

        except Exception as e:
            logger.error(f"Failed to parse ArXiv entry: {e}")
            return None

    def parse_item(self, item: Any) -> Optional[Article]:
        """Parse a single item (for interface compatibility)."""
        if isinstance(item, ET.Element):
            return self._parse_entry(item, {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            })
        return None

    def _extract_insights(self, abstract: str) -> List[str]:
        """Extract key insights from abstract."""
        if not abstract:
            return []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', abstract)

        # Take first 3-4 sentences as key insights
        insights = []
        for sent in sentences[:4]:
            sent = sent.strip()
            if len(sent) > 20:  # Skip very short sentences
                insights.append(sent)

        return insights

    def _extract_concepts(
        self,
        title: str,
        abstract: str,
        categories: List[str]
    ) -> List[str]:
        """Extract key concepts from the paper."""
        concepts = []

        # Add category names as concepts
        cat_names = {
            "cs.LG": "Machine Learning",
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Natural Language Processing",
            "cs.CV": "Computer Vision",
            "cs.NE": "Neural Networks",
            "stat.ML": "Statistical Learning",
        }
        for cat in categories:
            if cat in cat_names:
                concepts.append(cat_names[cat])

        # Extract capitalized terms from title
        title_concepts = re.findall(r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b', title)
        concepts.extend([c for c in title_concepts if len(c) > 2])

        # Look for common ML/AI terms in abstract
        ml_terms = [
            "Transformer", "Attention", "BERT", "GPT", "LLM", "CNN", "RNN", "LSTM",
            "GAN", "VAE", "Diffusion", "Reinforcement Learning", "Fine-tuning",
            "Pre-training", "Transfer Learning", "Few-shot", "Zero-shot",
            "Embedding", "Encoder", "Decoder", "Multimodal"
        ]
        text = title + " " + abstract
        for term in ml_terms:
            if term.lower() in text.lower():
                concepts.append(term)

        # Deduplicate
        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

        return unique[:10]

    def _infer_prerequisites(self, concepts: List[str]) -> List[str]:
        """Infer prerequisite knowledge from concepts."""
        prereq_map = {
            "Transformer": ["Attention Mechanism", "Neural Networks"],
            "BERT": ["Transformers", "NLP Basics"],
            "GPT": ["Transformers", "Language Modeling"],
            "LLM": ["Transformers", "Pre-training"],
            "Diffusion": ["Probability Theory", "Neural Networks"],
            "GAN": ["Neural Networks", "Generative Models"],
            "Reinforcement Learning": ["Markov Decision Processes", "Neural Networks"],
            "Fine-tuning": ["Pre-training", "Transfer Learning"],
        }

        prereqs = set()
        for concept in concepts:
            if concept in prereq_map:
                prereqs.update(prereq_map[concept])

        return list(prereqs)[:5]
