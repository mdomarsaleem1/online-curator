"""
Article Categorizer - Automatically categorize articles into sections.

Categories:
- DEMO: Hands-on tutorials, code demos, implementations, projects
- IDEA: Conceptual papers, research ideas, theories, proposals
- TOOL: Tools, libraries, frameworks, platforms
- TUTORIAL: Educational content, how-tos, guides
- NEWS: Industry news, announcements, updates
- OTHER: Uncategorized
"""

import re
from typing import List, Tuple
from src.models.article import Article, ArticleCategory


class ArticleCategorizer:
    """
    Categorizes articles based on content analysis.

    Uses keyword matching and heuristics to determine article category.
    Can be enhanced with ML-based classification in the future.
    """

    def __init__(self):
        """Initialize the categorizer with keyword patterns."""
        # Keywords strongly associated with each category
        self.category_keywords = {
            ArticleCategory.DEMO: [
                "demo", "implementation", "build", "code", "project",
                "hands-on", "walkthrough", "step-by-step", "tutorial",
                "github", "repository", "example", "colab", "notebook",
                "reproduce", "run", "execute", "deploy", "showcase"
            ],
            ArticleCategory.IDEA: [
                "paper", "research", "theory", "concept", "approach",
                "method", "novel", "propose", "hypothesis", "framework",
                "architecture", "design", "rethinking", "towards",
                "understanding", "analysis", "study", "survey", "review",
                "arxiv", "preprint", "abstract", "contribution"
            ],
            ArticleCategory.TOOL: [
                "tool", "library", "framework", "platform", "sdk",
                "api", "package", "release", "version", "install",
                "pip", "npm", "download", "cli", "interface",
                "integration", "plugin", "extension", "utility"
            ],
            ArticleCategory.TUTORIAL: [
                "tutorial", "guide", "learn", "beginner", "introduction",
                "getting started", "basics", "fundamentals", "course",
                "lesson", "chapter", "explained", "understand", "how to",
                "what is", "complete guide", "masterclass", "bootcamp"
            ],
            ArticleCategory.NEWS: [
                "announcement", "release", "update", "news", "launch",
                "introducing", "new", "latest", "breaking", "today",
                "just released", "coming soon", "roadmap", "changelog"
            ]
        }

        # Source-based hints
        self.source_hints = {
            "arxiv": ArticleCategory.IDEA,
            "papers with code": ArticleCategory.IDEA,
            "huggingface": ArticleCategory.TOOL,
            "github": ArticleCategory.DEMO,
            "youtube": ArticleCategory.TUTORIAL,
            "medium": ArticleCategory.TUTORIAL,
            "dev.to": ArticleCategory.TUTORIAL,
            "techcrunch": ArticleCategory.NEWS,
            "venturebeat": ArticleCategory.NEWS,
        }

    def categorize(self, article: Article) -> Tuple[ArticleCategory, float]:
        """
        Categorize an article and return category with confidence score.

        Returns:
            Tuple of (category, confidence) where confidence is 0-1
        """
        scores = {cat: 0.0 for cat in ArticleCategory}

        # Combine text for analysis
        text = self._prepare_text(article)
        text_lower = text.lower()

        # Score based on keywords
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Longer keywords are more specific, weight them higher
                    weight = 1.0 + (len(keyword) / 20.0)
                    scores[category] += weight

        # Apply source hints
        if article.source:
            source_lower = article.source.lower()
            for source_pattern, category in self.source_hints.items():
                if source_pattern in source_lower:
                    scores[category] += 3.0  # Strong hint

        # URL-based hints
        if article.url:
            url_lower = article.url.lower()
            if "arxiv.org" in url_lower:
                scores[ArticleCategory.IDEA] += 5.0
            elif "github.com" in url_lower:
                scores[ArticleCategory.DEMO] += 3.0
            elif "youtube.com" in url_lower or "youtu.be" in url_lower:
                scores[ArticleCategory.TUTORIAL] += 2.0

        # Find the best category
        max_score = max(scores.values())
        if max_score == 0:
            return ArticleCategory.OTHER, 0.0

        best_category = max(scores, key=scores.get)

        # Calculate confidence (normalize by max possible score)
        # Approximate max score based on average keyword matches
        confidence = min(1.0, max_score / 15.0)

        return best_category, confidence

    def _prepare_text(self, article: Article) -> str:
        """Prepare article text for analysis."""
        parts = [article.title]
        if article.summary:
            parts.append(article.summary)
        if article.key_insights:
            parts.extend(article.key_insights)
        if article.tags:
            parts.extend(article.tags)
        return " ".join(parts)

    def categorize_batch(self, articles: List[Article]) -> List[Tuple[Article, ArticleCategory, float]]:
        """Categorize multiple articles at once."""
        results = []
        for article in articles:
            category, confidence = self.categorize(article)
            article.category = category
            results.append((article, category, confidence))
        return results

    def get_category_description(self, category: ArticleCategory) -> str:
        """Get a human-readable description of a category."""
        descriptions = {
            ArticleCategory.DEMO: "🔧 Demos & Implementations - Hands-on code, projects, and practical examples",
            ArticleCategory.IDEA: "💡 Ideas & Research - Papers, theories, and conceptual frameworks",
            ArticleCategory.TOOL: "🛠️ Tools & Libraries - Frameworks, SDKs, and developer tools",
            ArticleCategory.TUTORIAL: "📚 Tutorials & Guides - Educational content and how-tos",
            ArticleCategory.NEWS: "📰 News & Updates - Industry news and announcements",
            ArticleCategory.OTHER: "📎 Other - Uncategorized content",
        }
        return descriptions.get(category, "Unknown category")

    def extract_concepts(self, article: Article) -> List[str]:
        """
        Extract key concepts from an article.

        This is a simplified extraction using common NLP patterns.
        Could be enhanced with NER or LLM-based extraction.
        """
        text = self._prepare_text(article)

        # Simple extraction: look for capitalized phrases (likely proper nouns/concepts)
        concepts = []

        # Extract capitalized words/phrases (potential concepts)
        cap_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        matches = re.findall(cap_pattern, text)

        # Filter common non-concepts
        stop_words = {
            "The", "This", "That", "These", "Those", "What", "How", "Why",
            "When", "Where", "Who", "Which", "If", "But", "And", "Or",
            "For", "From", "With", "About", "Into", "Through"
        }

        for match in matches:
            if match not in stop_words and len(match) > 2:
                concepts.append(match)

        # Deduplicate while preserving order
        seen = set()
        unique_concepts = []
        for c in concepts:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_concepts.append(c)

        return unique_concepts[:15]  # Limit to top 15 concepts
