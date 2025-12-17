"""YouTube scraper for video content and transcripts."""

import re
import os
from typing import List, Optional, Any, Dict
from datetime import datetime
from urllib.parse import urlparse, parse_qs, quote

from .base import BaseScraper, logger
from src.models.article import Article, ArticleCategory


class YouTubeScraper(BaseScraper):
    """
    Scraper for YouTube videos with transcripts.

    Supports:
    - Individual video URLs
    - Channel scraping (latest videos from a channel)
    - Search queries
    - Playlist scraping

    Uses youtube-transcript-api for transcripts (when available).
    """

    # Popular ML/AI YouTube channels with their handles/IDs
    ML_CHANNELS = {
        # Channel ID: (Name, Handle)
        "UCbfYPyITQ-7l4upoX8nvctg": ("Two Minute Papers", "@TwoMinutePapers"),
        "UCZHmQk67mN2biG-Q1KrOBeg": ("Yannic Kilcher", "@YannicKilcher"),
        "UCr8O8l5cCX85Oem1d18EezQ": ("DeepMind", "@GoogleDeepMind"),
        "UCddiUEpeqJcYeBxX1IVBKvQ": ("The AI Epiphany", "@TheAIEpiphany"),
        "UCXZCJLdBC09xxGZ6gcdrc6A": ("Andrej Karpathy", "@AndrejKarpathy"),
        "UCYO_jab_esuFRV4b17AJtAw": ("3Blue1Brown", "@3blue1brown"),
        "UCJ0-OtVpF0wOKEqT2Z1HEtA": ("EleutherAI", "@EleutherAI"),
        "UC4UJ26WkceqONNF5S26OiVw": ("Lex Fridman", "@lexfridman"),
        "UCVHFbqXqoYvEWM1Ddxl0QKg": ("AI Explained", "@aiaboratory"),
        "UCMLtBahI5DMrt0NPvDSoIRQ": ("Machine Learning Street Talk", "@MachineLearningStreetTalk"),
        "UCtxCXg-UvSnTKPOzLH4wJaQ": ("Cognitive Revolution", "@CognitiveRevolutionAI"),
        "UCgBncpylJ1kiVaPyP-PZauQ": ("Weights & Biases", "@WeightsBiases"),
    }

    # Search terms for ML/AI content
    ML_SEARCH_TERMS = [
        "machine learning tutorial",
        "deep learning explained",
        "transformer architecture",
        "LLM tutorial",
        "GPT explained",
        "neural network tutorial",
        "AI research paper",
        "PyTorch tutorial",
        "TensorFlow tutorial",
        "computer vision tutorial",
        "NLP tutorial",
        "reinforcement learning",
    ]

    def __init__(self, timeout: int = 30, api_key: Optional[str] = None):
        """Initialize the YouTube scraper."""
        super().__init__(timeout)
        self.source_name = "YouTube"
        self.default_category = ArticleCategory.TUTORIAL
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")

    def scrape(
        self,
        query: Optional[str] = None,
        limit: int = 10,
        video_urls: Optional[List[str]] = None,
        channel_id: Optional[str] = None,
        channel_handle: Optional[str] = None,
        search_terms: Optional[List[str]] = None,
        playlist_id: Optional[str] = None
    ) -> List[Article]:
        """
        Scrape videos from YouTube.

        Args:
            query: Search query for YouTube
            limit: Maximum number of videos to return
            video_urls: Specific video URLs to scrape
            channel_id: YouTube channel ID to scrape
            channel_handle: YouTube channel handle (e.g., @AndrejKarpathy)
            search_terms: List of search terms to use
            playlist_id: YouTube playlist ID to scrape

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

        elif channel_id or channel_handle:
            # Scrape from channel
            articles = self.scrape_channel(
                channel_id=channel_id,
                channel_handle=channel_handle,
                limit=limit
            )

        elif playlist_id:
            # Scrape playlist
            articles = self.scrape_playlist(playlist_id, limit)

        elif search_terms:
            # Search using multiple terms
            per_term_limit = max(1, limit // len(search_terms))
            for term in search_terms:
                results = self._search_videos(term, per_term_limit)
                articles.extend(results)
            articles = articles[:limit]

        elif query:
            # Single search query
            articles = self._search_videos(query, limit)

        else:
            logger.info("YouTube scraper: Provide video_urls, channel, search terms, or query")

        return articles

    def scrape_channel(
        self,
        channel_id: Optional[str] = None,
        channel_handle: Optional[str] = None,
        limit: int = 10
    ) -> List[Article]:
        """
        Scrape latest videos from a YouTube channel.

        Args:
            channel_id: YouTube channel ID
            channel_handle: YouTube channel handle (e.g., @AndrejKarpathy)
            limit: Maximum number of videos

        Returns:
            List of Article objects
        """
        if not self.api_key:
            logger.warning("YouTube API key required for channel scraping")
            # Try RSS feed fallback
            return self._scrape_channel_rss(channel_id, channel_handle, limit)

        # Resolve channel handle to ID if needed
        if channel_handle and not channel_id:
            channel_id = self._resolve_channel_handle(channel_handle)

        if not channel_id:
            logger.error("Could not resolve channel ID")
            return []

        # Get channel uploads playlist
        channel_url = (
            f"https://www.googleapis.com/youtube/v3/channels"
            f"?part=contentDetails&id={channel_id}&key={self.api_key}"
        )

        channel_data = self.fetch_json(channel_url)
        if not channel_data or "items" not in channel_data:
            return []

        uploads_playlist_id = (
            channel_data["items"][0]
            .get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )

        if not uploads_playlist_id:
            return []

        return self.scrape_playlist(uploads_playlist_id, limit)

    def _scrape_channel_rss(
        self,
        channel_id: Optional[str],
        channel_handle: Optional[str],
        limit: int
    ) -> List[Article]:
        """Fallback: scrape channel using RSS feed (no API key needed)."""
        if not channel_id:
            # Try to find channel ID from handle in our known list
            if channel_handle:
                for cid, (name, handle) in self.ML_CHANNELS.items():
                    if handle.lower() == channel_handle.lower():
                        channel_id = cid
                        break

        if not channel_id:
            logger.warning("Could not resolve channel ID for RSS fallback")
            return []

        # YouTube channel RSS feed
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

        try:
            import feedparser
            feed = feedparser.parse(rss_url)

            if not feed.entries:
                return []

            articles = []
            channel_name = feed.feed.get("title", "YouTube Channel")

            for entry in feed.entries[:limit]:
                video_id = entry.get("yt_videoid")
                if not video_id:
                    # Try to extract from link
                    link = entry.get("link", "")
                    video_id = self._extract_video_id(link)

                if video_id:
                    article = self._create_article_from_rss_entry(entry, channel_name)
                    if article:
                        articles.append(article)

            return articles

        except ImportError:
            logger.warning("feedparser not installed for RSS fallback")
            return []
        except Exception as e:
            logger.error(f"RSS fallback failed: {e}")
            return []

    def _create_article_from_rss_entry(self, entry: dict, channel_name: str) -> Optional[Article]:
        """Create article from RSS feed entry."""
        try:
            video_id = entry.get("yt_videoid")
            title = entry.get("title", "YouTube Video")
            summary = entry.get("summary", entry.get("media_description", ""))
            published = entry.get("published_parsed")

            pub_date = None
            if published:
                pub_date = datetime(*published[:6])

            # Try to get transcript
            transcript = self._get_transcript(video_id) if video_id else None

            if transcript:
                summary = self._summarize_transcript(transcript)
                key_insights = self._extract_insights_from_transcript(transcript)
            else:
                key_insights = [f"Video content from {channel_name}"]

            category = self._determine_category(title, transcript or summary)
            concepts = self._extract_concepts(title, transcript or summary)

            return Article(
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=title,
                source=f"YouTube - {channel_name}",
                summary=summary[:1000] if summary else f"Video by {channel_name}",
                key_insights=key_insights,
                category=category,
                tags=["youtube", "video"] + self._extract_tags(title),
                related_concepts=concepts,
                prerequisite_concepts=[],
                published_at=pub_date
            )
        except Exception as e:
            logger.error(f"Failed to create article from RSS: {e}")
            return None

    def _resolve_channel_handle(self, handle: str) -> Optional[str]:
        """Resolve channel handle to channel ID."""
        # Remove @ if present
        handle = handle.lstrip("@")

        # Check our known channels
        for channel_id, (name, known_handle) in self.ML_CHANNELS.items():
            if known_handle.lstrip("@").lower() == handle.lower():
                return channel_id

        # Try API if available
        if self.api_key:
            search_url = (
                f"https://www.googleapis.com/youtube/v3/search"
                f"?part=snippet&q={quote(handle)}&type=channel&maxResults=1"
                f"&key={self.api_key}"
            )

            data = self.fetch_json(search_url)
            if data and "items" in data and data["items"]:
                return data["items"][0].get("snippet", {}).get("channelId")

        return None

    def scrape_playlist(self, playlist_id: str, limit: int = 10) -> List[Article]:
        """
        Scrape videos from a YouTube playlist.

        Args:
            playlist_id: YouTube playlist ID
            limit: Maximum number of videos

        Returns:
            List of Article objects
        """
        if not self.api_key:
            logger.warning("YouTube API key required for playlist scraping")
            return []

        playlist_url = (
            f"https://www.googleapis.com/youtube/v3/playlistItems"
            f"?part=snippet&playlistId={playlist_id}&maxResults={limit}"
            f"&key={self.api_key}"
        )

        data = self.fetch_json(playlist_url)
        if not data or "items" not in data:
            return []

        articles = []
        for item in data["items"]:
            video_id = item.get("snippet", {}).get("resourceId", {}).get("videoId")
            if video_id:
                article = self.scrape_video(f"https://www.youtube.com/watch?v={video_id}")
                if article:
                    articles.append(article)

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

    def search(self, query: str, limit: int = 10) -> List[Article]:
        """
        Search YouTube for videos.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of Article objects
        """
        return self._search_videos(query, limit)

    def scrape_from_search_terms(
        self,
        search_terms: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Article]:
        """
        Scrape videos using multiple search terms.

        Args:
            search_terms: List of search terms (uses ML defaults if None)
            limit: Total maximum articles

        Returns:
            List of Article objects
        """
        terms = search_terms or self.ML_SEARCH_TERMS
        per_term_limit = max(1, limit // len(terms))

        articles = []
        seen_ids = set()

        for term in terms:
            results = self._search_videos(term, per_term_limit)
            for article in results:
                video_id = self._extract_video_id(article.url)
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)
                    articles.append(article)

            if len(articles) >= limit:
                break

        return articles[:limit]

    def scrape_ml_channels(self, limit_per_channel: int = 5) -> List[Article]:
        """
        Scrape latest videos from popular ML/AI channels.

        Args:
            limit_per_channel: Max videos per channel

        Returns:
            List of Article objects
        """
        articles = []

        for channel_id, (name, handle) in self.ML_CHANNELS.items():
            logger.info(f"Scraping channel: {name}")
            channel_articles = self.scrape_channel(
                channel_id=channel_id,
                limit=limit_per_channel
            )
            articles.extend(channel_articles)

        return articles

    def get_available_channels(self) -> Dict[str, str]:
        """Get list of available ML/AI channels."""
        return {handle: name for cid, (name, handle) in self.ML_CHANNELS.items()}

    def add_custom_channel(self, channel_id: str, name: str, handle: str):
        """Add a custom channel to the scraper."""
        self.ML_CHANNELS[channel_id] = (name, handle)

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        parsed = urlparse(url)

        if "youtube.com" in parsed.netloc:
            if parsed.path == "/watch":
                query_params = parse_qs(parsed.query)
                return query_params.get("v", [None])[0]
            elif parsed.path.startswith("/embed/"):
                return parsed.path.split("/embed/")[1].split("?")[0]
            elif parsed.path.startswith("/v/"):
                return parsed.path.split("/v/")[1].split("?")[0]
            elif parsed.path.startswith("/shorts/"):
                return parsed.path.split("/shorts/")[1].split("?")[0]
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
        """Get video transcript using youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            full_transcript = " ".join([
                segment.get("text", "")
                for segment in transcript_list
            ])

            return full_transcript

        except ImportError:
            logger.warning("youtube-transcript-api not installed")
            return None
        except Exception as e:
            logger.warning(f"Could not get transcript for {video_id}: {e}")
            return None

    def _summarize_transcript(self, transcript: str) -> str:
        """Create a summary from transcript."""
        words = transcript.split()
        if len(words) > 500:
            summary = " ".join(words[:500]) + "..."
        else:
            summary = transcript
        return self.clean_text(summary)

    def _extract_insights_from_transcript(self, transcript: str) -> List[str]:
        """Extract key insights from transcript."""
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

        return ArticleCategory.TUTORIAL

    def _extract_concepts(self, title: str, transcript: str) -> List[str]:
        """Extract ML/AI concepts from video."""
        text = title + " " + transcript
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

        title_concepts = re.findall(r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b', title)
        concepts.extend([c for c in title_concepts if len(c) > 2])

        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

        return unique[:10]

    def _extract_tags(self, title: str) -> List[str]:
        """Extract tags from title."""
        clean_title = re.sub(r'[^\w\s]', ' ', title)
        words = clean_title.lower().split()
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
            f"?part=snippet&q={quote(query)}&type=video&maxResults={limit}"
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
