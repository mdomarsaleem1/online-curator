"""RSS/Atom feed scraper for blogs and news sources."""

from typing import List, Optional, Any
from datetime import datetime
import time

from .base import BaseScraper, logger
from src.models.article import Article, ArticleCategory


class RSSScraper(BaseScraper):
    """
    Scraper for RSS and Atom feeds.

    Supports any standard RSS 2.0 or Atom feed.
    Includes predefined feeds for popular ML/AI blogs.
    """

    # Popular ML/AI RSS feeds
    ML_FEEDS = {
        "openai": "https://openai.com/blog/rss/",
        "anthropic": "https://www.anthropic.com/rss.xml",
        "deepmind": "https://deepmind.google/blog/rss.xml",
        "google_ai": "https://blog.google/technology/ai/rss/",
        "meta_ai": "https://ai.meta.com/blog/rss/",
        "huggingface": "https://huggingface.co/blog/feed.xml",
        "pytorch": "https://pytorch.org/blog/feed.xml",
        "tensorflow": "https://blog.tensorflow.org/feeds/posts/default",
        "distill": "https://distill.pub/rss.xml",
        "ml_mastery": "https://machinelearningmastery.com/feed/",
        "towards_ds": "https://towardsdatascience.com/feed",
        "mit_news_ai": "https://news.mit.edu/topic/artificial-intelligence2/feed",
    }

    def __init__(self, timeout: int = 30):
        """Initialize the RSS scraper."""
        super().__init__(timeout)
        self.source_name = "RSS"
        self.default_category = ArticleCategory.NEWS

    def scrape(
        self,
        query: Optional[str] = None,
        limit: int = 10,
        feed_url: Optional[str] = None,
        feed_name: Optional[str] = None
    ) -> List[Article]:
        """
        Scrape articles from RSS feeds.

        Args:
            query: Filter articles containing this text (optional)
            limit: Maximum number of articles to return
            feed_url: Direct URL to RSS feed
            feed_name: Name of predefined feed (e.g., 'openai', 'deepmind')

        Returns:
            List of Article objects
        """
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed. Install with: pip install feedparser")
            return []

        # Determine feed URL
        if feed_url:
            url = feed_url
            source = self._extract_source_name(feed_url)
        elif feed_name and feed_name in self.ML_FEEDS:
            url = self.ML_FEEDS[feed_name]
            source = feed_name.replace("_", " ").title()
        else:
            # Scrape all predefined feeds
            return self._scrape_all_feeds(query, limit)

        # Fetch and parse feed
        logger.info(f"Fetching RSS feed: {url}")
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            logger.error(f"Failed to parse feed: {url}")
            return []

        articles = []
        for entry in feed.entries[:limit]:
            article = self._parse_entry(entry, source)
            if article:
                # Apply query filter if specified
                if query:
                    if query.lower() not in (article.title + article.summary).lower():
                        continue
                articles.append(article)

        return articles[:limit]

    def _scrape_all_feeds(self, query: Optional[str], limit: int) -> List[Article]:
        """Scrape from all predefined feeds."""
        all_articles = []
        per_feed_limit = max(3, limit // len(self.ML_FEEDS))

        for feed_name, feed_url in self.ML_FEEDS.items():
            try:
                articles = self.scrape(
                    query=query,
                    limit=per_feed_limit,
                    feed_url=feed_url
                )
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"Failed to scrape {feed_name}: {e}")
                continue

            # Small delay to be respectful
            time.sleep(0.5)

        # Sort by date and limit
        all_articles.sort(
            key=lambda a: a.published_at or datetime.min,
            reverse=True
        )

        return all_articles[:limit]

    def _parse_entry(self, entry: Any, source: str) -> Optional[Article]:
        """Parse a single feed entry into an Article."""
        try:
            # Get basic fields
            title = entry.get("title", "Untitled")
            link = entry.get("link", "")

            if not link:
                return None

            # Get content/summary
            content = ""
            if "content" in entry and entry.content:
                content = entry.content[0].get("value", "")
            elif "summary" in entry:
                content = entry.get("summary", "")
            elif "description" in entry:
                content = entry.get("description", "")

            # Clean HTML from content
            summary = self._clean_html(content)

            # Get published date
            pub_date = None
            for date_field in ["published_parsed", "updated_parsed", "created_parsed"]:
                if hasattr(entry, date_field) and getattr(entry, date_field):
                    try:
                        pub_date = datetime(*getattr(entry, date_field)[:6])
                        break
                    except (TypeError, ValueError):
                        continue

            # Get tags/categories
            tags = []
            if "tags" in entry:
                tags = [t.get("term", "") for t in entry.tags if t.get("term")]
            elif "category" in entry:
                tags = [entry.category] if isinstance(entry.category, str) else []

            # Get author
            author = entry.get("author", "")

            # Determine category
            category = self._determine_category(title, summary, source)

            # Extract concepts
            concepts = self._extract_concepts(title, summary)

            # Extract key insights
            key_insights = self._extract_insights(summary)

            return Article(
                url=link,
                title=self.clean_text(title),
                source=source,
                summary=summary[:1500] if summary else f"Article from {source}",
                key_insights=key_insights,
                category=category,
                tags=tags[:8] + [source.lower().replace(" ", "-")],
                related_concepts=concepts,
                prerequisite_concepts=[],
                published_at=pub_date,
            )

        except Exception as e:
            logger.error(f"Failed to parse RSS entry: {e}")
            return None

    def parse_item(self, item: Any) -> Optional[Article]:
        """Parse a single item (interface compatibility)."""
        return self._parse_entry(item, "RSS")

    def _clean_html(self, html: str) -> str:
        """Remove HTML tags and clean text."""
        import re

        if not html:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)

        # Decode HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")

        # Clean whitespace
        text = " ".join(text.split())

        return text

    def _extract_source_name(self, url: str) -> str:
        """Extract a readable source name from URL."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc

        # Remove common prefixes
        for prefix in ["www.", "blog.", "blogs.", "news."]:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]

        # Get main domain name
        parts = domain.split(".")
        if len(parts) >= 2:
            name = parts[-2]
        else:
            name = domain

        return name.title()

    def _determine_category(self, title: str, summary: str, source: str) -> ArticleCategory:
        """Determine article category."""
        text = (title + " " + summary).lower()
        source_lower = source.lower()

        # Source-based hints
        if any(s in source_lower for s in ["arxiv", "distill", "research"]):
            return ArticleCategory.IDEA

        # Content-based detection
        if any(w in text for w in ["tutorial", "how to", "guide", "getting started"]):
            return ArticleCategory.TUTORIAL
        elif any(w in text for w in ["paper", "research", "study", "propose", "novel"]):
            return ArticleCategory.IDEA
        elif any(w in text for w in ["release", "announcing", "introducing", "launch"]):
            return ArticleCategory.NEWS
        elif any(w in text for w in ["demo", "implementation", "github", "code"]):
            return ArticleCategory.DEMO
        elif any(w in text for w in ["tool", "library", "framework", "api"]):
            return ArticleCategory.TOOL

        return ArticleCategory.NEWS  # Default for RSS/blogs

    def _extract_concepts(self, title: str, summary: str) -> List[str]:
        """Extract concepts from article."""
        import re

        text = title + " " + summary
        concepts = []

        # ML/AI terms
        ml_terms = [
            "LLM", "GPT", "BERT", "Transformer", "Neural Network",
            "Deep Learning", "Machine Learning", "AI", "NLP",
            "Computer Vision", "Reinforcement Learning", "Diffusion"
        ]

        text_lower = text.lower()
        for term in ml_terms:
            if term.lower() in text_lower:
                concepts.append(term)

        # Capitalized phrases
        caps = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', title)
        concepts.extend([c for c in caps if len(c) > 2 and c not in ["The", "This", "That"]])

        # Deduplicate
        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

        return unique[:10]

    def _extract_insights(self, summary: str) -> List[str]:
        """Extract key insights from summary."""
        import re

        if not summary:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', summary)
        insights = []

        for sent in sentences[:4]:
            sent = sent.strip()
            if 30 < len(sent) < 300:
                insights.append(sent)

        return insights[:3]

    def add_custom_feed(self, name: str, url: str):
        """Add a custom feed to the scraper."""
        self.ML_FEEDS[name] = url
        logger.info(f"Added custom feed: {name} -> {url}")

    def list_feeds(self) -> dict:
        """List all available feeds."""
        return self.ML_FEEDS.copy()
