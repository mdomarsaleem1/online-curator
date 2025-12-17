"""
Knowledge Growth Engine - The core recommendation algorithm.

This engine recommends articles based on "semantic growth" rather than
simple similarity. The goal is to maximize learning efficiency by suggesting
articles that:
1. Build on what you already know (not too far from current knowledge)
2. Introduce new concepts (not too similar/redundant)
3. Fill knowledge gaps in your learning path
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.models.article import Article, ArticleRecommendation, ArticleCategory
from src.models.database import DatabaseManager
from src.services.vector_store import VectorStore


@dataclass
class GrowthScore:
    """Breakdown of growth potential score."""
    novelty: float          # How new is this content (0-1)
    relevance: float        # How related to interests (0-1)
    foundation: float       # How well does prior knowledge support this (0-1)
    gap_filling: float      # Does this fill a knowledge gap (0-1)
    total: float            # Weighted combination


class KnowledgeEngine:
    """
    Engine for calculating knowledge growth potential and recommendations.

    The core idea: Your time is precious. We want to recommend articles that:
    - Are in your zone of proximal development (not too easy, not too hard)
    - Introduce genuinely new concepts
    - Build coherently on your existing knowledge graph
    - Don't waste time on redundant content

    Growth Potential Formula:
    growth = (novelty * 0.4) + (relevance * 0.25) + (foundation * 0.2) + (gap_filling * 0.15)

    Where:
    - novelty: 0 = exact duplicate, 1 = completely new
    - relevance: 0 = unrelated to interests, 1 = perfect match
    - foundation: 0 = missing prerequisites, 1 = well prepared
    - gap_filling: 0 = doesn't fill gaps, 1 = fills important gap
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize the knowledge engine."""
        self.vector_store = vector_store or VectorStore()
        self.db = DatabaseManager()

        # Weights for the growth score components
        self.weights = {
            "novelty": 0.40,       # Most important: avoid redundancy
            "relevance": 0.25,    # Stay in areas of interest
            "foundation": 0.20,   # Build on solid ground
            "gap_filling": 0.15   # Fill knowledge gaps
        }

        # Optimal novelty range (sweet spot for learning)
        self.optimal_novelty_min = 0.3  # Not too familiar
        self.optimal_novelty_max = 0.7  # Not too foreign

    def calculate_growth_score(self, article: Article) -> GrowthScore:
        """
        Calculate the growth potential score for an article.

        Returns a GrowthScore with breakdown of all components.
        """
        # Get novelty from vector store
        novelty = self.vector_store.calculate_novelty(article)

        # Adjust novelty to prefer the optimal learning zone
        adjusted_novelty = self._adjust_novelty_for_learning(novelty)

        # Calculate relevance based on user preferences and category
        relevance = self._calculate_relevance(article)

        # Calculate foundation score (do we have prerequisites?)
        foundation = self._calculate_foundation(article)

        # Calculate gap-filling score
        gap_filling = self._calculate_gap_filling(article)

        # Weighted combination
        total = (
            adjusted_novelty * self.weights["novelty"] +
            relevance * self.weights["relevance"] +
            foundation * self.weights["foundation"] +
            gap_filling * self.weights["gap_filling"]
        )

        return GrowthScore(
            novelty=novelty,
            relevance=relevance,
            foundation=foundation,
            gap_filling=gap_filling,
            total=total
        )

    def _adjust_novelty_for_learning(self, novelty: float) -> float:
        """
        Adjust novelty score to prefer the optimal learning zone.

        Articles that are moderately novel (30-70% new) are best for learning.
        Too familiar = boring/redundant
        Too novel = overwhelming/incomprehensible
        """
        if novelty < self.optimal_novelty_min:
            # Too familiar - reduce score
            return novelty * 0.5
        elif novelty > self.optimal_novelty_max:
            # Too novel - reduce score but not as much
            excess = novelty - self.optimal_novelty_max
            return self.optimal_novelty_max - (excess * 0.3)
        else:
            # In the sweet spot - boost slightly
            return min(1.0, novelty * 1.2)

    def _calculate_relevance(self, article: Article) -> float:
        """
        Calculate relevance to user's interests.

        Based on:
        - Preferred categories
        - Topics in liked articles
        - Source preferences
        """
        liked_articles = self.db.get_liked_articles()
        if not liked_articles:
            return 0.5  # Neutral if no history

        relevance_score = 0.5  # Base score

        # Category preference
        category_counts = {}
        for liked in liked_articles:
            cat = liked.category.value if liked.category else "other"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if category_counts:
            total = sum(category_counts.values())
            article_cat = article.category.value if article.category else "other"
            cat_weight = category_counts.get(article_cat, 0) / total
            relevance_score += cat_weight * 0.3

        # Source preference
        source_counts = {}
        for liked in liked_articles:
            if liked.source:
                source_counts[liked.source] = source_counts.get(liked.source, 0) + 1

        if source_counts and article.source:
            total = sum(source_counts.values())
            source_weight = source_counts.get(article.source, 0) / total
            relevance_score += source_weight * 0.2

        return min(1.0, relevance_score)

    def _calculate_foundation(self, article: Article) -> float:
        """
        Calculate how well the user's knowledge supports understanding this article.

        Checks if prerequisite concepts are covered in liked articles.
        """
        if not article.prerequisite_concepts:
            return 0.8  # No prerequisites = easy to understand

        liked_articles = self.db.get_liked_articles()
        if not liked_articles:
            return 0.5  # Unknown if we can understand

        # Collect all concepts from liked articles
        known_concepts = set()
        for liked in liked_articles:
            if liked.related_concepts:
                try:
                    concepts = json.loads(liked.related_concepts)
                    known_concepts.update(c.lower() for c in concepts)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Check how many prerequisites we know
        prereqs = [p.lower() for p in article.prerequisite_concepts]
        if not prereqs:
            return 0.8

        known_prereqs = sum(1 for p in prereqs if any(p in kc or kc in p for kc in known_concepts))
        foundation_score = known_prereqs / len(prereqs)

        return foundation_score

    def _calculate_gap_filling(self, article: Article) -> float:
        """
        Calculate if this article fills a gap in the knowledge graph.

        A gap exists when:
        - We know concepts A and C, but not B (which connects them)
        - We have many articles in a category but miss key subtopics
        """
        # Simplified implementation: check if article introduces new concepts
        # that are related to but not duplicate of known concepts
        if not article.related_concepts:
            return 0.3

        liked_articles = self.db.get_liked_articles()
        if not liked_articles:
            return 0.5

        # Collect known concepts
        known_concepts = set()
        for liked in liked_articles:
            if liked.related_concepts:
                try:
                    concepts = json.loads(liked.related_concepts)
                    known_concepts.update(c.lower() for c in concepts)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Check article concepts
        article_concepts = [c.lower() for c in article.related_concepts]
        new_concepts = [c for c in article_concepts if c not in known_concepts]

        if not new_concepts:
            return 0.1  # All concepts already known

        # Higher score if it introduces 1-3 new concepts (not overwhelming)
        new_count = len(new_concepts)
        if new_count <= 3:
            return 0.8
        elif new_count <= 5:
            return 0.6
        else:
            return 0.4  # Too many new concepts

    def get_recommendations(
        self,
        n: int = 10,
        category: Optional[ArticleCategory] = None
    ) -> List[ArticleRecommendation]:
        """
        Get top N article recommendations for knowledge growth.

        Returns articles sorted by growth potential with explanations.
        """
        # Get unrated articles
        unrated = self.db.get_unrated_articles(limit=100)

        if category:
            unrated = [a for a in unrated if a.category == category]

        if not unrated:
            return []

        recommendations = []
        for db_article in unrated:
            article = self.db.db_article_to_pydantic(db_article)
            score = self.calculate_growth_score(article)

            # Generate recommendation reason
            reason = self._generate_recommendation_reason(article, score)

            # Identify knowledge gaps this would fill
            gaps = self._identify_knowledge_gaps(article)

            # Calculate overlap with existing knowledge
            overlap = 1.0 - score.novelty

            recommendations.append(ArticleRecommendation(
                article=article,
                score=score.total,
                reason=reason,
                knowledge_gap=gaps,
                overlap_percentage=overlap * 100
            ))

        # Sort by growth potential (highest first)
        recommendations.sort(key=lambda r: r.score, reverse=True)

        return recommendations[:n]

    def _generate_recommendation_reason(self, article: Article, score: GrowthScore) -> str:
        """Generate a human-readable recommendation reason."""
        reasons = []

        if score.novelty > 0.6:
            reasons.append("introduces fresh perspectives")
        elif score.novelty > 0.3:
            reasons.append("builds naturally on your knowledge")
        else:
            reasons.append("reinforces existing concepts")

        if score.foundation > 0.7:
            reasons.append("you have strong prerequisites for this")
        elif score.foundation < 0.3:
            reasons.append("may require some background reading")

        if score.gap_filling > 0.6:
            reasons.append("fills gaps in your knowledge")

        if score.relevance > 0.7:
            reasons.append("matches your interests well")

        if not reasons:
            reasons.append("could expand your knowledge base")

        return "This article " + ", ".join(reasons) + "."

    def _identify_knowledge_gaps(self, article: Article) -> List[str]:
        """Identify which concepts this article would add to your knowledge."""
        if not article.related_concepts:
            return []

        liked_articles = self.db.get_liked_articles()
        known_concepts = set()

        for liked in liked_articles:
            if liked.related_concepts:
                try:
                    concepts = json.loads(liked.related_concepts)
                    known_concepts.update(c.lower() for c in concepts)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Find new concepts this article would introduce
        new_concepts = []
        for concept in article.related_concepts:
            if concept.lower() not in known_concepts:
                new_concepts.append(concept)

        return new_concepts[:5]  # Limit to top 5

    def process_rating(self, article_id: int, is_liked: bool):
        """
        Process a user rating and update the knowledge model.

        When an article is liked:
        - Add to knowledge collection
        - Update user statistics
        - Recalculate growth scores for pending articles

        When disliked:
        - Mark as not relevant
        - Slightly adjust category preferences
        """
        # Get the article
        db_article = self.db.get_article_by_id(article_id)
        if not db_article:
            return

        article = self.db.db_article_to_pydantic(db_article)

        # Update the database
        self.db.update_article_rating(article_id, is_liked)
        self.db.update_user_stats(is_liked)

        # Update the vector store
        if is_liked:
            self.vector_store.mark_as_known(article)
        else:
            self.vector_store.remove_from_knowledge(article_id)

    def recalculate_all_scores(self):
        """
        Recalculate growth scores for all unrated articles.

        Should be called periodically or after significant knowledge updates.
        """
        unrated = self.db.get_unrated_articles(limit=500)

        for db_article in unrated:
            article = self.db.db_article_to_pydantic(db_article)
            score = self.calculate_growth_score(article)
            reason = self._generate_recommendation_reason(article, score)

            self.db.update_article_scores(
                article_id=db_article.id,
                novelty_score=score.novelty,
                growth_potential=score.total,
                recommendation_reason=reason
            )

    def get_knowledge_summary(self) -> Dict:
        """Get a summary of the user's current knowledge state."""
        liked = self.db.get_liked_articles()

        # Category distribution
        categories = {}
        sources = {}
        concepts = set()

        for article in liked:
            cat = article.category.value if article.category else "other"
            categories[cat] = categories.get(cat, 0) + 1

            if article.source:
                sources[article.source] = sources.get(article.source, 0) + 1

            if article.related_concepts:
                try:
                    article_concepts = json.loads(article.related_concepts)
                    concepts.update(article_concepts)
                except (json.JSONDecodeError, TypeError):
                    pass

        return {
            "total_learned": len(liked),
            "categories": categories,
            "sources": sources,
            "concepts_count": len(concepts),
            "top_concepts": list(concepts)[:20],
            "vector_stats": self.vector_store.get_stats()
        }
