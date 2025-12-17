"""
Summarizer Agent with structured output.

Uses LLMs to generate structured summaries including:
- High-level summary
- Key insights (bullet points)
- Related concepts
- Prerequisite knowledge
- Category classification
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import logging

from src.models.article import Article, ArticleCategory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StructuredSummary:
    """Structured output from the summarizer."""
    title: str
    summary: str                    # 2-3 sentence high-level summary
    key_insights: List[str]         # 3-5 bullet points
    category: str                   # demo, idea, tool, tutorial, news, other
    tags: List[str]                 # 5-10 relevant tags
    related_concepts: List[str]     # Key concepts covered
    prerequisite_concepts: List[str] # What you need to know first
    difficulty_level: str           # beginner, intermediate, advanced
    estimated_read_time: int        # minutes
    why_read: str                   # One sentence on why this is worth reading

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class SummarizerAgent:
    """
    Agent for generating structured summaries of articles.

    Can use either OpenAI API or local models via Ollama.
    Falls back to rule-based extraction if no LLM available.
    """

    SYSTEM_PROMPT = """You are an expert AI research assistant that creates structured summaries of technical articles.

Your task is to analyze the given content and produce a JSON response with the following structure:
{
    "title": "Cleaned/improved title",
    "summary": "2-3 sentence high-level summary of the main contribution",
    "key_insights": ["insight 1", "insight 2", "insight 3"],
    "category": "one of: demo, idea, tool, tutorial, news, other",
    "tags": ["tag1", "tag2", "tag3"],
    "related_concepts": ["concept1", "concept2"],
    "prerequisite_concepts": ["prereq1", "prereq2"],
    "difficulty_level": "one of: beginner, intermediate, advanced",
    "estimated_read_time": 5,
    "why_read": "One compelling sentence on why this is worth reading"
}

Guidelines:
- Summary should focus on the "what" and "why", not implementation details
- Key insights should be actionable or memorable takeaways
- Category should reflect the primary nature of the content
- Tags should be specific and useful for searching
- Related concepts are the main topics/technologies covered
- Prerequisite concepts are what readers should know beforehand
- Be concise but informative"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_ollama: bool = False,
        ollama_model: str = "llama3.2"
    ):
        """
        Initialize the summarizer.

        Args:
            api_key: OpenAI API key (uses env OPENAI_API_KEY if not provided)
            model: OpenAI model to use
            use_ollama: Whether to use Ollama for local inference
            ollama_model: Ollama model name
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model

        self._client = None
        if self.api_key and not use_ollama:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI package not installed")

    def summarize(self, article: Article) -> StructuredSummary:
        """
        Generate a structured summary for an article.

        Args:
            article: Article to summarize

        Returns:
            StructuredSummary with all fields populated
        """
        # Prepare content for summarization
        content = self._prepare_content(article)

        # Try LLM-based summarization
        if self._client:
            summary = self._summarize_with_openai(content, article)
            if summary:
                return summary

        if self.use_ollama:
            summary = self._summarize_with_ollama(content, article)
            if summary:
                return summary

        # Fallback to rule-based extraction
        return self._summarize_with_rules(article)

    def _prepare_content(self, article: Article) -> str:
        """Prepare article content for summarization."""
        parts = [f"Title: {article.title}"]

        if article.source:
            parts.append(f"Source: {article.source}")

        if article.summary:
            parts.append(f"Content: {article.summary}")

        if article.key_insights:
            parts.append(f"Existing insights: {'; '.join(article.key_insights)}")

        if article.tags:
            parts.append(f"Tags: {', '.join(article.tags)}")

        return "\n".join(parts)

    def _summarize_with_openai(
        self,
        content: str,
        article: Article
    ) -> Optional[StructuredSummary]:
        """Generate summary using OpenAI API."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)
            return self._parse_llm_response(result, article)

        except Exception as e:
            logger.error(f"OpenAI summarization failed: {e}")
            return None

    def _summarize_with_ollama(
        self,
        content: str,
        article: Article
    ) -> Optional[StructuredSummary]:
        """Generate summary using Ollama (local LLM)."""
        try:
            import httpx

            response = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": f"{self.SYSTEM_PROMPT}\n\n{content}\n\nRespond with JSON only:",
                    "format": "json",
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()

            result = json.loads(response.json()["response"])
            return self._parse_llm_response(result, article)

        except Exception as e:
            logger.error(f"Ollama summarization failed: {e}")
            return None

    def _parse_llm_response(
        self,
        result: dict,
        article: Article
    ) -> StructuredSummary:
        """Parse LLM response into StructuredSummary."""
        # Map category string to enum
        category_map = {
            "demo": ArticleCategory.DEMO,
            "idea": ArticleCategory.IDEA,
            "tool": ArticleCategory.TOOL,
            "tutorial": ArticleCategory.TUTORIAL,
            "news": ArticleCategory.NEWS,
            "other": ArticleCategory.OTHER,
        }

        return StructuredSummary(
            title=result.get("title", article.title),
            summary=result.get("summary", article.summary or ""),
            key_insights=result.get("key_insights", article.key_insights or [])[:5],
            category=result.get("category", "other"),
            tags=result.get("tags", article.tags or [])[:10],
            related_concepts=result.get("related_concepts", article.related_concepts or [])[:10],
            prerequisite_concepts=result.get("prerequisite_concepts", [])[:5],
            difficulty_level=result.get("difficulty_level", "intermediate"),
            estimated_read_time=result.get("estimated_read_time", 5),
            why_read=result.get("why_read", "Valuable content for learning.")
        )

    def _summarize_with_rules(self, article: Article) -> StructuredSummary:
        """Fallback rule-based summarization."""
        import re

        # Category detection
        text = (article.title + " " + (article.summary or "")).lower()
        category = "other"

        if any(w in text for w in ["tutorial", "how to", "guide", "learn"]):
            category = "tutorial"
        elif any(w in text for w in ["paper", "research", "arxiv", "study"]):
            category = "idea"
        elif any(w in text for w in ["demo", "implementation", "github", "code"]):
            category = "demo"
        elif any(w in text for w in ["tool", "library", "framework", "release"]):
            category = "tool"
        elif any(w in text for w in ["news", "announcement", "update"]):
            category = "news"

        # Difficulty detection
        difficulty = "intermediate"
        if any(w in text for w in ["beginner", "introduction", "basics", "101"]):
            difficulty = "beginner"
        elif any(w in text for w in ["advanced", "deep dive", "expert"]):
            difficulty = "advanced"

        # Extract key insights from summary
        key_insights = article.key_insights or []
        if not key_insights and article.summary:
            sentences = re.split(r'(?<=[.!?])\s+', article.summary)
            key_insights = [s.strip() for s in sentences[:3] if len(s) > 20]

        # Estimate read time (rough: 200 words per minute)
        word_count = len((article.summary or "").split())
        read_time = max(1, word_count // 200)

        # Generate "why read" based on category
        why_read_templates = {
            "idea": "Explores novel concepts that could shape your understanding of the field.",
            "demo": "Provides hands-on examples you can learn from and adapt.",
            "tool": "Introduces a tool that could enhance your workflow.",
            "tutorial": "Offers step-by-step guidance to build practical skills.",
            "news": "Keeps you updated on important developments in the field.",
            "other": "Contains valuable information worth exploring.",
        }
        why_read = why_read_templates.get(category, why_read_templates["other"])

        return StructuredSummary(
            title=article.title,
            summary=article.summary or f"Content from {article.source}",
            key_insights=key_insights[:5],
            category=category,
            tags=article.tags or [],
            related_concepts=article.related_concepts or [],
            prerequisite_concepts=article.prerequisite_concepts or [],
            difficulty_level=difficulty,
            estimated_read_time=read_time,
            why_read=why_read
        )

    def summarize_batch(
        self,
        articles: List[Article],
        update_articles: bool = True
    ) -> List[StructuredSummary]:
        """
        Summarize multiple articles.

        Args:
            articles: List of articles to summarize
            update_articles: Whether to update article objects with summary data

        Returns:
            List of StructuredSummary objects
        """
        summaries = []

        for article in articles:
            summary = self.summarize(article)
            summaries.append(summary)

            if update_articles:
                article.summary = summary.summary
                article.key_insights = summary.key_insights
                article.tags = summary.tags
                article.related_concepts = summary.related_concepts
                article.prerequisite_concepts = summary.prerequisite_concepts

                # Map category string back to enum
                category_map = {
                    "demo": ArticleCategory.DEMO,
                    "idea": ArticleCategory.IDEA,
                    "tool": ArticleCategory.TOOL,
                    "tutorial": ArticleCategory.TUTORIAL,
                    "news": ArticleCategory.NEWS,
                    "other": ArticleCategory.OTHER,
                }
                article.category = category_map.get(summary.category, ArticleCategory.OTHER)

        return summaries

    def enhance_article(self, article: Article) -> Article:
        """
        Enhance an article with structured summary data.

        Returns the same article object with updated fields.
        """
        summary = self.summarize(article)

        article.summary = summary.summary
        article.key_insights = summary.key_insights
        article.tags = summary.tags
        article.related_concepts = summary.related_concepts
        article.prerequisite_concepts = summary.prerequisite_concepts
        article.recommendation_reason = summary.why_read

        category_map = {
            "demo": ArticleCategory.DEMO,
            "idea": ArticleCategory.IDEA,
            "tool": ArticleCategory.TOOL,
            "tutorial": ArticleCategory.TUTORIAL,
            "news": ArticleCategory.NEWS,
            "other": ArticleCategory.OTHER,
        }
        article.category = category_map.get(summary.category, ArticleCategory.OTHER)

        return article
