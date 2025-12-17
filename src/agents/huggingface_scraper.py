"""HuggingFace scraper for models, datasets, papers, and spaces."""

from typing import List, Optional, Any
from datetime import datetime

from .base import BaseScraper, logger
from src.models.article import Article, ArticleCategory


class HuggingFaceScraper(BaseScraper):
    """
    Scraper for HuggingFace Hub content.

    Fetches trending models, datasets, spaces, and papers.
    """

    BASE_URL = "https://huggingface.co"
    API_URL = "https://huggingface.co/api"

    def __init__(self, timeout: int = 30, token: Optional[str] = None):
        """Initialize the HuggingFace scraper."""
        super().__init__(timeout)
        self.source_name = "HuggingFace"
        self.default_category = ArticleCategory.TOOL
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def scrape(
        self,
        query: Optional[str] = None,
        limit: int = 10,
        content_type: str = "models"  # models, datasets, spaces, papers
    ) -> List[Article]:
        """
        Scrape content from HuggingFace Hub.

        Args:
            query: Optional search query
            limit: Maximum number of items to return
            content_type: Type of content to scrape (models, datasets, spaces, papers)

        Returns:
            List of Article objects
        """
        if content_type == "papers":
            return self._scrape_papers(query, limit)
        elif content_type == "models":
            return self._scrape_models(query, limit)
        elif content_type == "datasets":
            return self._scrape_datasets(query, limit)
        elif content_type == "spaces":
            return self._scrape_spaces(query, limit)
        else:
            logger.warning(f"Unknown content type: {content_type}")
            return []

    def _scrape_models(self, query: Optional[str], limit: int) -> List[Article]:
        """Scrape trending/searched models."""
        url = f"{self.API_URL}/models"
        params = {
            "limit": limit,
            "sort": "downloads",
            "direction": -1
        }
        if query:
            params["search"] = query

        url_with_params = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        data = self.fetch_json(url_with_params, self.headers)

        if not data:
            return []

        articles = []
        for item in data[:limit]:
            article = self._parse_model(item)
            if article:
                articles.append(article)

        return articles

    def _parse_model(self, item: dict) -> Optional[Article]:
        """Parse a model into an Article."""
        try:
            model_id = item.get("modelId", item.get("id", ""))
            if not model_id:
                return None

            # Build URL
            url = f"{self.BASE_URL}/{model_id}"

            # Get model info
            pipeline_tag = item.get("pipeline_tag", "")
            tags = item.get("tags", [])
            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)

            # Build summary
            summary_parts = []
            if pipeline_tag:
                summary_parts.append(f"Pipeline: {pipeline_tag}")
            summary_parts.append(f"Downloads: {downloads:,}")
            summary_parts.append(f"Likes: {likes:,}")

            # Get description from model card if available
            description = item.get("description", "")
            if description:
                summary_parts.insert(0, description[:500])

            summary = " | ".join(summary_parts)

            # Determine category
            category = ArticleCategory.TOOL
            if "demo" in tags or "space" in str(item):
                category = ArticleCategory.DEMO

            # Extract concepts
            concepts = [pipeline_tag] if pipeline_tag else []
            concepts.extend([t for t in tags if not t.startswith("license:")][:5])

            return Article(
                url=url,
                title=f"{model_id}",
                source=self.source_name,
                summary=summary,
                key_insights=[
                    f"Model type: {pipeline_tag}" if pipeline_tag else "General purpose model",
                    f"Community engagement: {likes:,} likes, {downloads:,} downloads"
                ],
                category=category,
                tags=tags[:8],
                related_concepts=concepts,
                prerequisite_concepts=self._get_model_prerequisites(pipeline_tag),
            )

        except Exception as e:
            logger.error(f"Failed to parse HuggingFace model: {e}")
            return None

    def _scrape_datasets(self, query: Optional[str], limit: int) -> List[Article]:
        """Scrape trending/searched datasets."""
        url = f"{self.API_URL}/datasets"
        params = {
            "limit": limit,
            "sort": "downloads",
            "direction": -1
        }
        if query:
            params["search"] = query

        url_with_params = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        data = self.fetch_json(url_with_params, self.headers)

        if not data:
            return []

        articles = []
        for item in data[:limit]:
            article = self._parse_dataset(item)
            if article:
                articles.append(article)

        return articles

    def _parse_dataset(self, item: dict) -> Optional[Article]:
        """Parse a dataset into an Article."""
        try:
            dataset_id = item.get("id", "")
            if not dataset_id:
                return None

            url = f"{self.BASE_URL}/datasets/{dataset_id}"
            tags = item.get("tags", [])
            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)

            description = item.get("description", f"Dataset: {dataset_id}")

            return Article(
                url=url,
                title=f"Dataset: {dataset_id}",
                source=self.source_name,
                summary=f"{description[:500]} | Downloads: {downloads:,} | Likes: {likes:,}",
                key_insights=[
                    f"Dataset for ML/AI training and evaluation",
                    f"Community usage: {downloads:,} downloads"
                ],
                category=ArticleCategory.TOOL,
                tags=tags[:8] + ["dataset"],
                related_concepts=["Datasets", "Training Data"] + tags[:3],
                prerequisite_concepts=["Machine Learning Basics"],
            )

        except Exception as e:
            logger.error(f"Failed to parse HuggingFace dataset: {e}")
            return None

    def _scrape_spaces(self, query: Optional[str], limit: int) -> List[Article]:
        """Scrape trending/searched spaces (demos)."""
        url = f"{self.API_URL}/spaces"
        params = {
            "limit": limit,
            "sort": "likes",
            "direction": -1
        }
        if query:
            params["search"] = query

        url_with_params = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        data = self.fetch_json(url_with_params, self.headers)

        if not data:
            return []

        articles = []
        for item in data[:limit]:
            article = self._parse_space(item)
            if article:
                articles.append(article)

        return articles

    def _parse_space(self, item: dict) -> Optional[Article]:
        """Parse a space into an Article."""
        try:
            space_id = item.get("id", "")
            if not space_id:
                return None

            url = f"{self.BASE_URL}/spaces/{space_id}"
            tags = item.get("tags", [])
            likes = item.get("likes", 0)
            sdk = item.get("sdk", "unknown")

            return Article(
                url=url,
                title=f"Space: {space_id}",
                source=self.source_name,
                summary=f"Interactive demo built with {sdk}. Likes: {likes:,}",
                key_insights=[
                    f"Interactive ML/AI demo",
                    f"Built with {sdk}",
                    f"Community appreciation: {likes:,} likes"
                ],
                category=ArticleCategory.DEMO,
                tags=tags[:8] + ["space", "demo", sdk],
                related_concepts=["Interactive Demo", sdk] + tags[:3],
                prerequisite_concepts=[],
            )

        except Exception as e:
            logger.error(f"Failed to parse HuggingFace space: {e}")
            return None

    def _scrape_papers(self, query: Optional[str], limit: int) -> List[Article]:
        """Scrape daily papers from HuggingFace."""
        # HuggingFace daily papers endpoint
        url = f"{self.API_URL}/daily_papers"
        data = self.fetch_json(url, self.headers)

        if not data:
            return []

        articles = []
        for item in data[:limit]:
            article = self._parse_paper(item)
            if article:
                articles.append(article)

        return articles

    def _parse_paper(self, item: dict) -> Optional[Article]:
        """Parse a paper into an Article."""
        try:
            paper = item.get("paper", {})
            paper_id = paper.get("id", "")
            if not paper_id:
                return None

            title = paper.get("title", "Untitled Paper")
            summary = paper.get("summary", "")
            authors = paper.get("authors", [])
            author_names = [a.get("name", "") for a in authors[:5]]

            # Use arxiv URL if available, otherwise HF papers
            arxiv_id = paper.get("arxiv_id")
            if arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
            else:
                url = f"{self.BASE_URL}/papers/{paper_id}"

            pub_date = None
            if paper.get("publishedAt"):
                try:
                    pub_date = datetime.fromisoformat(paper["publishedAt"].replace('Z', '+00:00'))
                except ValueError:
                    pass

            return Article(
                url=url,
                title=title,
                source="HuggingFace Papers",
                summary=summary[:1000] if summary else f"Paper by {', '.join(author_names)}",
                key_insights=self._extract_paper_insights(summary),
                category=ArticleCategory.IDEA,
                tags=["paper", "research", "huggingface-daily"],
                related_concepts=self._extract_concepts_from_text(title + " " + summary),
                prerequisite_concepts=[],
                published_at=pub_date,
            )

        except Exception as e:
            logger.error(f"Failed to parse HuggingFace paper: {e}")
            return None

    def parse_item(self, item: Any) -> Optional[Article]:
        """Parse a single item (interface compatibility)."""
        if isinstance(item, dict):
            if "modelId" in item or "pipeline_tag" in item:
                return self._parse_model(item)
            elif "paper" in item:
                return self._parse_paper(item)
        return None

    def _get_model_prerequisites(self, pipeline_tag: str) -> List[str]:
        """Get prerequisites based on pipeline tag."""
        prereqs = {
            "text-generation": ["Transformers", "NLP Basics"],
            "text-classification": ["NLP Basics", "Classification"],
            "image-classification": ["Computer Vision", "CNN"],
            "object-detection": ["Computer Vision", "Neural Networks"],
            "text-to-image": ["Diffusion Models", "Image Generation"],
            "automatic-speech-recognition": ["Audio Processing", "Transformers"],
        }
        return prereqs.get(pipeline_tag, ["Machine Learning Basics"])

    def _extract_paper_insights(self, summary: str) -> List[str]:
        """Extract insights from paper summary."""
        if not summary:
            return []

        import re
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        return [s.strip() for s in sentences[:3] if len(s) > 20]

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract concepts from text."""
        import re
        # Find capitalized phrases
        concepts = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)

        # Common ML terms to look for
        ml_terms = ["LLM", "GPT", "BERT", "Transformer", "Diffusion", "CNN", "GAN", "LoRA"]
        for term in ml_terms:
            if term.lower() in text.lower() and term not in concepts:
                concepts.append(term)

        # Deduplicate
        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen and len(c) > 2:
                seen.add(c.lower())
                unique.append(c)

        return unique[:10]
