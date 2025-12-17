"""YouTube scraper for video content and transcripts."""

import re
from typing import List, Optional, Any
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from .base import BaseScraper, logger
from src.models.article import Article, ArticleCategory


class YouTubeScraper(BaseScraper):
    """
    Scraper for YouTube videos with transcripts.

    Fetches video metadata and transcripts for ML/AI channels.
    Uses youtube-transcript-api for transcripts (when available).
    """

    # Popular ML/AI YouTube channels
    ML_CHANNELS = {
        "UCbfYPyITQ-7l4upoX8nvctg": "Two Minute Papers",
        "UCWN3xxRkmTPmbKwht9FuE5A": "Siraj Raval",
        "UCZHmQk67mN2biG-Q1KrOBeg": "Yannic Kilcher",
        "UCr8O8l5cCX85Oem1d18EezQ": "DeepMind",
        "UCddiUEpeqJcYeBxX1IVBKvQ": "The AI Epiphany",
        "UCXZCJLdBC09xxGZ6gcdrc6A": "Andrej Karpathy",
    }

    def __init__(self, timeout: int = 30, api_key: Optional[str] = None):
        """Initialize the YouTube scraper."""
        super().__init__(timeout)
        self.source_name = "YouTube"
        self.default_category = ArticleCategory.TUTORIAL
        self.api_key = api_key

    def scrape(
        self,
        query: Optional[str] = None,
        limit: int = 10,
        video_urls: Optional[List[str]] = None
    ) -> List[Article]:
        """
        Scrape videos from YouTube.

        Args:
            query: Search query for YouTube
            limit: Maximum number of videos to return
            video_urls: Specific video URLs to scrape

        Returns:
            List of Article objects
        """
        articles = []

        if video_urls:
            # Scrape specific videos
            for url in video_urls[:limit]:
                article = self.scrape_video(url)
                if article:
                    articles.append(article)
        elif query and self.api_key:
            # Search YouTube (requires API key)
            articles = self._search_videos(query, limit)
        else:
            # Return placeholder for manual URL input
            logger.info("YouTube scraper: Provide video_urls or API key for search")

        return articles

    def scrape_video(self, url: str) -> Optional[Article]:
        """
        Scrape a single YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            Article object or None
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from: {url}")
            return None

        # Get video metadata via oEmbed (no API key needed)
        metadata = self._get_video_metadata(video_id)
        if not metadata:
            return None

        # Try to get transcript
        transcript = self._get_transcript(video_id)

        # Build article
        title = metadata.get("title", "YouTube Video")
        author = metadata.get("author_name", "Unknown")

        # Create summary from transcript or title
        if transcript:
            summary = self._summarize_transcript(transcript)
            key_insights = self._extract_insights_from_transcript(transcript)
        else:
            summary = f"Video by {author}: {title}"
            key_insights = [f"Video content by {author}"]

        # Determine category
        category = self._determine_category(title, transcript or "")

        # Extract concepts
        concepts = self._extract_concepts(title, transcript or "")

        return Article(
            url=f"https://www.youtube.com/watch?v={video_id}",
            title=title,
            source=f"YouTube - {author}",
            summary=summary,
            key_insights=key_insights,
            category=category,
            tags=["youtube", "video"] + self._extract_tags(title),
            related_concepts=concepts,
            prerequisite_concepts=[],
        )

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        # Handle different URL formats
        parsed = urlparse(url)

        if "youtube.com" in parsed.netloc:
            if parsed.path == "/watch":
                query_params = parse_qs(parsed.query)
                return query_params.get("v", [None])[0]
            elif parsed.path.startswith("/embed/"):
                return parsed.path.split("/embed/")[1].split("?")[0]
            elif parsed.path.startswith("/v/"):
                return parsed.path.split("/v/")[1].split("?")[0]
        elif "youtu.be" in parsed.netloc:
            return parsed.path[1:].split("?")[0]

        # Try regex as fallback
        patterns = [
            r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
            r'^([0-9A-Za-z_-]{11})$'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _get_video_metadata(self, video_id: str) -> Optional[dict]:
        """Get video metadata using oEmbed."""
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

        try:
            return self.fetch_json(oembed_url)
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            return None

    def _get_transcript(self, video_id: str) -> Optional[str]:
        """
        Get video transcript.

        Uses youtube-transcript-api if available.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Combine transcript segments
            full_transcript = " ".join([
                segment.get("text", "")
                for segment in transcript_list
            ])

            return full_transcript

        except ImportError:
            logger.warning("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")
            return None
        except Exception as e:
            logger.warning(f"Could not get transcript for {video_id}: {e}")
            return None

    def _summarize_transcript(self, transcript: str) -> str:
        """Create a summary from transcript."""
        # Take first ~500 words as summary
        words = transcript.split()
        if len(words) > 500:
            summary = " ".join(words[:500]) + "..."
        else:
            summary = transcript

        return self.clean_text(summary)

    def _extract_insights_from_transcript(self, transcript: str) -> List[str]:
        """Extract key insights from transcript."""
        # Simple extraction: get sentences with key phrases
        key_phrases = [
            "important", "key", "main", "significant", "breakthrough",
            "novel", "new approach", "state of the art", "sota",
            "in summary", "to summarize", "the key takeaway"
        ]

        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        insights = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(phrase in sentence_lower for phrase in key_phrases):
                clean_sent = self.clean_text(sentence)
                if 20 < len(clean_sent) < 300:
                    insights.append(clean_sent)

        # If no key phrases found, take first few sentences
        if not insights and sentences:
            for sent in sentences[:3]:
                clean_sent = self.clean_text(sent)
                if len(clean_sent) > 20:
                    insights.append(clean_sent)

        return insights[:5]

    def _determine_category(self, title: str, transcript: str) -> ArticleCategory:
        """Determine video category based on content."""
        text = (title + " " + transcript).lower()

        if any(w in text for w in ["tutorial", "how to", "step by step", "learn", "guide"]):
            return ArticleCategory.TUTORIAL
        elif any(w in text for w in ["demo", "showcase", "implementation", "build", "code"]):
            return ArticleCategory.DEMO
        elif any(w in text for w in ["paper", "research", "study", "arxiv", "theory"]):
            return ArticleCategory.IDEA
        elif any(w in text for w in ["news", "announcement", "release", "update"]):
            return ArticleCategory.NEWS
        elif any(w in text for w in ["tool", "library", "framework"]):
            return ArticleCategory.TOOL

        return ArticleCategory.TUTORIAL  # Default for YouTube

    def _extract_concepts(self, title: str, transcript: str) -> List[str]:
        """Extract ML/AI concepts from video."""
        text = title + " " + transcript

        # Common ML/AI concepts to look for
        concepts = []
        ml_terms = [
            "transformer", "attention", "bert", "gpt", "llm",
            "neural network", "deep learning", "machine learning",
            "cnn", "rnn", "lstm", "gan", "vae", "diffusion",
            "reinforcement learning", "nlp", "computer vision",
            "fine-tuning", "pre-training", "embedding"
        ]

        text_lower = text.lower()
        for term in ml_terms:
            if term in text_lower:
                concepts.append(term.title())

        # Also extract capitalized terms from title
        title_concepts = re.findall(r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b', title)
        concepts.extend([c for c in title_concepts if len(c) > 2])

        # Deduplicate
        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

        return unique[:10]

    def _extract_tags(self, title: str) -> List[str]:
        """Extract tags from title."""
        # Remove special characters and split
        clean_title = re.sub(r'[^\w\s]', ' ', title)
        words = clean_title.lower().split()

        # Filter common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"}
        tags = [w for w in words if w not in stop_words and len(w) > 2]

        return tags[:5]

    def _search_videos(self, query: str, limit: int) -> List[Article]:
        """Search YouTube videos using API."""
        if not self.api_key:
            logger.warning("YouTube API key required for search")
            return []

        search_url = (
            f"https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&q={query}&type=video&maxResults={limit}"
            f"&key={self.api_key}"
        )

        data = self.fetch_json(search_url)
        if not data or "items" not in data:
            return []

        articles = []
        for item in data["items"]:
            video_id = item.get("id", {}).get("videoId")
            if video_id:
                article = self.scrape_video(f"https://www.youtube.com/watch?v={video_id}")
                if article:
                    articles.append(article)

        return articles

    def parse_item(self, item: Any) -> Optional[Article]:
        """Parse a single item (interface compatibility)."""
        if isinstance(item, str):
            return self.scrape_video(item)
        return None
