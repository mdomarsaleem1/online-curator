"""
Research-Backed Knowledge Growth Engine.

Implements learning science principles:
1. Zone of Proximal Development (Vygotsky) - Learn just beyond current ability
2. Bloom's Taxonomy - Progress through cognitive levels
3. Spaced Repetition (Ebbinghaus) - Optimal review timing
4. Interleaving - Mix related topics for deeper learning
5. Desirable Difficulties (Bjork) - Some challenge improves retention
6. Knowledge Transfer - Connect new to existing knowledge

References:
- Vygotsky, L.S. (1978). Mind in Society
- Bloom, B.S. (1956). Taxonomy of Educational Objectives
- Ebbinghaus, H. (1885). Memory: A Contribution to Experimental Psychology
- Bjork, R.A. (1994). Memory and Metamemory Considerations
- Rohrer, D. & Taylor, K. (2007). The Shuffling of Mathematics Problems
"""

import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.models.article import Article, ArticleRecommendation, ArticleCategory
from src.models.database import DatabaseManager
from src.services.vector_store import VectorStore
from src.services.user_profile import ProfileManager, SkillLevel, LearningDimension


class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels."""
    REMEMBER = "remember"        # Recall facts, basic concepts
    UNDERSTAND = "understand"    # Explain ideas, concepts
    APPLY = "apply"             # Use information in new situations
    ANALYZE = "analyze"         # Draw connections, organize
    EVALUATE = "evaluate"       # Justify decisions, critique
    CREATE = "create"           # Produce new or original work


@dataclass
class GrowthMetrics:
    """Detailed breakdown of growth potential scores."""
    # Core scores (0-1)
    zpd_score: float           # Zone of Proximal Development alignment
    bloom_progression: float   # Progress through cognitive levels
    spaced_repetition: float   # Optimal timing for review
    interleaving: float        # Topic mixing benefit
    desirable_difficulty: float # Appropriate challenge level
    knowledge_transfer: float  # Connection to existing knowledge
    topic_relevance: float     # Matches user's learning goals

    # Weighted total
    total_score: float

    # Explanations
    reasoning: List[str]


class ResearchBackedGrowthEngine:
    """
    Knowledge growth engine based on educational research.

    Weight Distribution (based on meta-analyses):
    - ZPD alignment: 25% (Vygotsky - most impactful for learning)
    - Knowledge transfer: 20% (builds on existing knowledge)
    - Topic relevance: 15% (motivation and engagement)
    - Desirable difficulty: 15% (Bjork - challenge improves retention)
    - Bloom progression: 10% (cognitive level advancement)
    - Spaced repetition: 10% (Ebbinghaus - memory consolidation)
    - Interleaving: 5% (topic mixing for deeper learning)

    Research sources:
    - Hattie (2008) effect sizes for learning strategies
    - Dunlosky et al. (2013) improving student learning
    """

    # Research-backed weights
    WEIGHTS = {
        "zpd": 0.25,
        "transfer": 0.20,
        "relevance": 0.15,
        "difficulty": 0.15,
        "bloom": 0.10,
        "spacing": 0.10,
        "interleaving": 0.05
    }

    # ZPD parameters
    ZPD_OPTIMAL_NOVELTY_MIN = 0.25  # 25% minimum novelty
    ZPD_OPTIMAL_NOVELTY_MAX = 0.65  # 65% maximum novelty
    ZPD_PEAK = 0.45                  # Peak learning at 45% novelty

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        profile_manager: Optional[ProfileManager] = None
    ):
        """Initialize the research-backed growth engine."""
        self.vector_store = vector_store or VectorStore()
        self.profile_manager = profile_manager or ProfileManager()
        self.db = DatabaseManager()

    def calculate_growth_metrics(self, article: Article) -> GrowthMetrics:
        """
        Calculate comprehensive growth metrics for an article.

        Returns detailed breakdown of all factors.
        """
        reasoning = []

        # 1. Zone of Proximal Development Score
        zpd_score, zpd_reason = self._calculate_zpd_score(article)
        reasoning.append(zpd_reason)

        # 2. Knowledge Transfer Score
        transfer_score, transfer_reason = self._calculate_transfer_score(article)
        reasoning.append(transfer_reason)

        # 3. Topic Relevance Score
        relevance_score, relevance_reason = self._calculate_relevance_score(article)
        reasoning.append(relevance_reason)

        # 4. Desirable Difficulty Score
        difficulty_score, difficulty_reason = self._calculate_difficulty_score(article)
        reasoning.append(difficulty_reason)

        # 5. Bloom's Taxonomy Progression
        bloom_score, bloom_reason = self._calculate_bloom_score(article)
        reasoning.append(bloom_reason)

        # 6. Spaced Repetition Score
        spacing_score, spacing_reason = self._calculate_spacing_score(article)
        reasoning.append(spacing_reason)

        # 7. Interleaving Score
        interleaving_score, interleaving_reason = self._calculate_interleaving_score(article)
        reasoning.append(interleaving_reason)

        # Calculate weighted total
        total = (
            zpd_score * self.WEIGHTS["zpd"] +
            transfer_score * self.WEIGHTS["transfer"] +
            relevance_score * self.WEIGHTS["relevance"] +
            difficulty_score * self.WEIGHTS["difficulty"] +
            bloom_score * self.WEIGHTS["bloom"] +
            spacing_score * self.WEIGHTS["spacing"] +
            interleaving_score * self.WEIGHTS["interleaving"]
        )

        return GrowthMetrics(
            zpd_score=zpd_score,
            bloom_progression=bloom_score,
            spaced_repetition=spacing_score,
            interleaving=interleaving_score,
            desirable_difficulty=difficulty_score,
            knowledge_transfer=transfer_score,
            topic_relevance=relevance_score,
            total_score=total,
            reasoning=[r for r in reasoning if r]
        )

    def _calculate_zpd_score(self, article: Article) -> Tuple[float, str]:
        """
        Calculate Zone of Proximal Development alignment.

        The ZPD is where learning is most effective - content that's
        challenging enough to promote growth but not so difficult
        as to be incomprehensible.

        Uses a Gaussian-like curve centered at optimal novelty.
        """
        # Get novelty from vector store
        novelty = self.vector_store.calculate_novelty(article)

        # Apply Gaussian curve centered at ZPD_PEAK
        # This creates a bell curve that peaks at optimal novelty
        sigma = 0.25  # Standard deviation
        zpd_score = math.exp(-((novelty - self.ZPD_PEAK) ** 2) / (2 * sigma ** 2))

        # Generate reasoning
        if novelty < self.ZPD_OPTIMAL_NOVELTY_MIN:
            reason = f"Content may be too familiar ({novelty:.0%} novel) - limited growth potential"
        elif novelty > self.ZPD_OPTIMAL_NOVELTY_MAX:
            reason = f"Content may be too advanced ({novelty:.0%} novel) - consider prerequisites first"
        else:
            reason = f"Optimal learning zone ({novelty:.0%} novel) - builds on your knowledge effectively"

        return zpd_score, reason

    def _calculate_transfer_score(self, article: Article) -> Tuple[float, str]:
        """
        Calculate knowledge transfer potential.

        Higher when article connects to existing knowledge,
        enabling transfer from known to new concepts.
        """
        liked_articles = self.db.get_liked_articles()
        if not liked_articles:
            return 0.5, "Building your knowledge base - this is a good starting point"

        # Get article concepts
        article_concepts = set()
        if article.related_concepts:
            article_concepts = set(c.lower() for c in article.related_concepts)
        if article.prerequisite_concepts:
            article_concepts.update(c.lower() for c in article.prerequisite_concepts)

        if not article_concepts:
            return 0.5, "Topic connections unclear"

        # Get known concepts from liked articles
        known_concepts = set()
        for liked in liked_articles:
            if liked.related_concepts:
                try:
                    concepts = json.loads(liked.related_concepts)
                    known_concepts.update(c.lower() for c in concepts)
                except:
                    pass

        # Calculate overlap
        overlap = article_concepts.intersection(known_concepts)
        new_concepts = article_concepts - known_concepts

        if not known_concepts:
            return 0.5, "Starting fresh - all concepts are new"

        # Optimal: some overlap (connection) + some new (growth)
        overlap_ratio = len(overlap) / len(article_concepts)
        new_ratio = len(new_concepts) / len(article_concepts)

        # Best transfer when ~40% familiar, ~60% new
        transfer_score = 1.0 - abs(overlap_ratio - 0.4) - abs(new_ratio - 0.6) * 0.5
        transfer_score = max(0, min(1, transfer_score))

        if overlap_ratio > 0.7:
            reason = f"Strong foundation - {len(overlap)} connected concepts"
        elif overlap_ratio > 0.3:
            reason = f"Good transfer potential - bridges {len(overlap)} known concepts to {len(new_concepts)} new ones"
        else:
            reason = f"Many new concepts ({len(new_concepts)}) - may require extra focus"

        return transfer_score, reason

    def _calculate_relevance_score(self, article: Article) -> Tuple[float, str]:
        """
        Calculate relevance to user's learning goals.

        Based on configured topics and keywords.
        """
        if not self.profile_manager.profile.topics:
            return 0.5, "Configure learning topics to get personalized recommendations"

        # Check match with user interests
        article_text = f"{article.title} {article.summary or ''} {' '.join(article.tags or [])}"
        matches, relevance = self.profile_manager.matches_user_interests(article_text)

        if relevance > 0.7:
            reason = "Highly relevant to your learning goals"
        elif relevance > 0.4:
            reason = "Moderately relevant to your interests"
        elif matches:
            reason = "Somewhat related to your topics"
        else:
            reason = "Outside your specified topics - could broaden perspective"
            # Don't penalize too much for exploration
            relevance = max(0.3, relevance)

        return relevance, reason

    def _calculate_difficulty_score(self, article: Article) -> Tuple[float, str]:
        """
        Calculate desirable difficulty alignment.

        Based on Bjork's research: some difficulty enhances learning,
        but too much difficulty hinders it.
        """
        user_level = self.profile_manager.profile.overall_level

        # Estimate article difficulty from various signals
        difficulty_signals = {
            "has_prerequisites": bool(article.prerequisite_concepts),
            "concept_count": len(article.related_concepts or []),
            "is_research": article.category == ArticleCategory.IDEA,
            "source_type": article.source or ""
        }

        # Estimate difficulty level (0-1)
        estimated_difficulty = 0.3  # Base

        if difficulty_signals["has_prerequisites"]:
            estimated_difficulty += 0.2
        if difficulty_signals["concept_count"] > 5:
            estimated_difficulty += 0.1
        if difficulty_signals["is_research"]:
            estimated_difficulty += 0.2
        if "arxiv" in difficulty_signals["source_type"].lower():
            estimated_difficulty += 0.1

        estimated_difficulty = min(1.0, estimated_difficulty)

        # Map user level to optimal difficulty
        level_to_optimal = {
            SkillLevel.NOVICE: 0.3,
            SkillLevel.ADVANCED_BEGINNER: 0.4,
            SkillLevel.COMPETENT: 0.5,
            SkillLevel.PROFICIENT: 0.65,
            SkillLevel.EXPERT: 0.8
        }

        optimal_difficulty = level_to_optimal.get(user_level, 0.5)

        # Score based on distance from optimal
        # Allow slight stretch (desirable difficulty principle)
        stretch_bonus = 0.1 if estimated_difficulty > optimal_difficulty else 0
        difficulty_score = 1.0 - abs(estimated_difficulty - optimal_difficulty) + stretch_bonus
        difficulty_score = max(0, min(1, difficulty_score))

        if estimated_difficulty < optimal_difficulty - 0.2:
            reason = "May be too easy for your level - limited challenge"
        elif estimated_difficulty > optimal_difficulty + 0.3:
            reason = "Challenging content - desirable difficulty for deeper learning"
        else:
            reason = "Appropriate difficulty level for optimal learning"

        return difficulty_score, reason

    def _calculate_bloom_score(self, article: Article) -> Tuple[float, str]:
        """
        Calculate Bloom's Taxonomy progression score.

        Encourages progression through cognitive levels:
        Remember → Understand → Apply → Analyze → Evaluate → Create
        """
        # Estimate article's Bloom level
        article_text = f"{article.title} {article.summary or ''}".lower()

        bloom_indicators = {
            BloomLevel.CREATE: ["implement", "build", "create", "design", "develop", "produce"],
            BloomLevel.EVALUATE: ["compare", "critique", "evaluate", "assess", "judge", "benchmark"],
            BloomLevel.ANALYZE: ["analyze", "examine", "investigate", "explore", "break down"],
            BloomLevel.APPLY: ["apply", "use", "demonstrate", "tutorial", "hands-on", "example"],
            BloomLevel.UNDERSTAND: ["explain", "describe", "discuss", "understand", "overview"],
            BloomLevel.REMEMBER: ["define", "list", "identify", "basics", "introduction", "what is"]
        }

        detected_level = BloomLevel.REMEMBER
        for level, indicators in bloom_indicators.items():
            if any(ind in article_text for ind in indicators):
                detected_level = level
                break

        # Get user's dominant Bloom level from history
        liked = self.db.get_liked_articles()
        user_bloom_counts = {level: 0 for level in BloomLevel}

        for liked_article in liked:
            liked_text = f"{liked_article.title} {liked_article.summary or ''}".lower()
            for level, indicators in bloom_indicators.items():
                if any(ind in liked_text for ind in indicators):
                    user_bloom_counts[level] += 1
                    break

        # Find user's current dominant level
        if sum(user_bloom_counts.values()) > 0:
            user_level = max(user_bloom_counts, key=user_bloom_counts.get)
        else:
            user_level = BloomLevel.REMEMBER

        # Score based on progression
        bloom_order = list(BloomLevel)
        user_idx = bloom_order.index(user_level)
        article_idx = bloom_order.index(detected_level)

        # Prefer articles at same level or one level up
        if article_idx == user_idx:
            bloom_score = 0.8
            reason = f"Reinforces your current level ({detected_level.value})"
        elif article_idx == user_idx + 1:
            bloom_score = 1.0
            reason = f"Advances to next cognitive level ({detected_level.value})"
        elif article_idx > user_idx:
            bloom_score = 0.6
            reason = f"Higher-order content ({detected_level.value}) - ambitious but valuable"
        else:
            bloom_score = 0.5
            reason = f"Foundational content ({detected_level.value}) - good for reinforcement"

        return bloom_score, reason

    def _calculate_spacing_score(self, article: Article) -> Tuple[float, str]:
        """
        Calculate spaced repetition score.

        Based on Ebbinghaus forgetting curve - optimal review timing
        follows an exponential pattern.
        """
        # Get articles on related topics
        article_concepts = set()
        if article.related_concepts:
            article_concepts = set(c.lower() for c in article.related_concepts)

        if not article_concepts:
            return 0.5, "New topic area"

        liked = self.db.get_liked_articles()
        if not liked:
            return 0.5, "Building initial knowledge base"

        # Find most recent related article
        last_related = None
        last_related_date = None

        for liked_article in liked:
            if not liked_article.read_at:
                continue

            liked_concepts = set()
            if liked_article.related_concepts:
                try:
                    liked_concepts = set(c.lower() for c in json.loads(liked_article.related_concepts))
                except:
                    pass

            if liked_concepts.intersection(article_concepts):
                if last_related_date is None or liked_article.read_at > last_related_date:
                    last_related = liked_article
                    last_related_date = liked_article.read_at

        if not last_related_date:
            return 0.7, "New topic - good for expanding knowledge"

        # Calculate days since last related content
        days_since = (datetime.utcnow() - last_related_date).days

        # Optimal spacing follows roughly: 1, 3, 7, 14, 30 days
        # Score based on whether timing is appropriate
        if days_since < 1:
            spacing_score = 0.4
            reason = "Recently covered - consider spacing for better retention"
        elif days_since < 3:
            spacing_score = 0.7
            reason = "Good timing for reinforcement"
        elif days_since < 7:
            spacing_score = 0.9
            reason = "Optimal spacing for memory consolidation"
        elif days_since < 14:
            spacing_score = 1.0
            reason = "Ideal review timing based on spaced repetition"
        elif days_since < 30:
            spacing_score = 0.8
            reason = "Good time to revisit and strengthen connections"
        else:
            spacing_score = 0.6
            reason = "Time for a refresher on related concepts"

        return spacing_score, reason

    def _calculate_interleaving_score(self, article: Article) -> Tuple[float, str]:
        """
        Calculate interleaving benefit score.

        Interleaving (mixing related but different topics) improves
        discrimination learning and transfer.
        """
        # Get recent reading history
        liked = self.db.get_liked_articles()
        if len(liked) < 3:
            return 0.5, "Building reading history"

        # Get last 5 articles' categories and topics
        recent = sorted(liked, key=lambda a: a.read_at or datetime.min, reverse=True)[:5]
        recent_categories = [a.category for a in recent]
        recent_concepts = []
        for a in recent:
            if a.related_concepts:
                try:
                    recent_concepts.extend(json.loads(a.related_concepts))
                except:
                    pass

        # Check if this article provides variety
        same_category_count = sum(1 for c in recent_categories if c == article.category)

        article_concepts = set(c.lower() for c in (article.related_concepts or []))
        recent_concepts_set = set(c.lower() for c in recent_concepts)
        concept_overlap = len(article_concepts.intersection(recent_concepts_set))

        # Optimal: same general area but different specific topic
        if same_category_count >= 3:
            if concept_overlap < 2:
                interleaving_score = 0.9
                reason = "Good variety within your focus area - enhances discrimination learning"
            else:
                interleaving_score = 0.6
                reason = "Similar to recent content - consider exploring adjacent topics"
        else:
            if concept_overlap > 0:
                interleaving_score = 0.8
                reason = "Connects different areas - promotes transfer learning"
            else:
                interleaving_score = 0.7
                reason = "Different domain - adds breadth to your knowledge"

        return interleaving_score, reason

    def get_recommendations(
        self,
        n: int = 10,
        category: Optional[ArticleCategory] = None
    ) -> List[ArticleRecommendation]:
        """
        Get top N recommendations using research-backed scoring.
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

            # Calculate comprehensive metrics
            metrics = self.calculate_growth_metrics(article)

            # Generate recommendation reason from top factors
            top_reasons = sorted(
                [r for r in metrics.reasoning if r],
                key=lambda x: len(x)
            )[:2]
            reason = " ".join(top_reasons) if top_reasons else "Good learning opportunity"

            # Knowledge gaps
            gaps = []
            if article.related_concepts:
                known = self._get_known_concepts()
                gaps = [c for c in article.related_concepts if c.lower() not in known][:5]

            recommendations.append(ArticleRecommendation(
                article=article,
                score=metrics.total_score,
                reason=reason,
                knowledge_gap=gaps,
                overlap_percentage=(1 - metrics.zpd_score) * 100
            ))

        # Sort by total score
        recommendations.sort(key=lambda r: r.score, reverse=True)

        return recommendations[:n]

    def _get_known_concepts(self) -> set:
        """Get all concepts from liked articles."""
        known = set()
        for article in self.db.get_liked_articles():
            if article.related_concepts:
                try:
                    concepts = json.loads(article.related_concepts)
                    known.update(c.lower() for c in concepts)
                except:
                    pass
        return known

    def process_rating(self, article_id: int, is_liked: bool):
        """Process a rating and update learning profile."""
        db_article = self.db.get_article_by_id(article_id)
        if not db_article:
            return

        article = self.db.db_article_to_pydantic(db_article)

        # Update database
        self.db.update_article_rating(article_id, is_liked)
        self.db.update_user_stats(is_liked)

        # Update vector store
        if is_liked:
            self.vector_store.mark_as_known(article)

            # Update topic progress
            for topic in self.profile_manager.profile.topics:
                topic_text = f"{topic.name} {' '.join(topic.keywords)}"
                article_text = f"{article.title} {article.summary or ''}"
                if any(kw.lower() in article_text.lower() for kw in [topic.name] + topic.keywords):
                    self.profile_manager.update_topic_progress(topic.name)

            # Update activity
            self.profile_manager.update_activity()
            self.profile_manager.profile.total_articles_read += 1
            self.profile_manager.save_profile()
        else:
            self.vector_store.remove_from_knowledge(article_id)

    def get_learning_insights(self) -> Dict:
        """Get comprehensive learning insights."""
        # Assess maturity
        overall_level, dimension_scores = self.profile_manager.assess_overall_maturity()

        # Get recommendations for improvement
        recommendations = self.profile_manager.get_learning_recommendations()

        # Calculate reading pace
        liked = self.db.get_liked_articles()
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_count = sum(1 for a in liked if a.read_at and a.read_at > week_ago)

        return {
            "overall_level": overall_level.value,
            "dimension_scores": {k: v.score for k, v in self.profile_manager.profile.dimension_scores.items()},
            "streak_days": self.profile_manager.profile.streak_days,
            "articles_this_week": recent_count,
            "target_per_week": self.profile_manager.profile.articles_per_week_target,
            "recommendations": recommendations,
            "topics": [
                {
                    "name": t.name,
                    "current_level": t.current_level.value,
                    "target_level": t.target_level.value,
                    "articles_read": t.articles_read
                }
                for t in self.profile_manager.profile.topics
            ]
        }
