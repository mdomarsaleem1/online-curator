"""
Dynamic Topic Dimensions Visualization.

Creates radar/spider charts and other visualizations showing
user's knowledge across 3-6 logical dimensions.

Dimensions are dynamically adjusted based on:
- User's learning topics
- Article categories read
- Concept coverage
- Skill progression
"""

import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.models.database import DatabaseManager
from src.services.user_profile import ProfileManager, LearningDimension, SkillLevel


@dataclass
class DimensionData:
    """Data for a single dimension in the visualization."""
    name: str
    score: float           # 0-100
    articles_count: int
    trend: str            # "up", "down", "stable"
    recent_activity: int  # articles in last 7 days
    top_concepts: List[str]
    suggested_focus: bool  # Should user focus here?


@dataclass
class RadarChartData:
    """Data structure for radar chart visualization."""
    dimensions: List[str]
    scores: List[float]
    max_score: float
    labels: List[str]


class DimensionVisualizer:
    """
    Visualizer for knowledge dimensions.

    Creates various visualizations:
    - Radar/spider charts for dimension overview
    - Progress bars for each dimension
    - Trend indicators
    - Focus recommendations
    """

    # Default ML/AI dimensions
    DEFAULT_DIMENSIONS = {
        "foundations": {
            "name": "Foundations",
            "keywords": ["linear algebra", "calculus", "probability", "statistics",
                        "optimization", "mathematics", "fundamentals", "basics"],
            "description": "Core mathematical and theoretical foundations"
        },
        "models": {
            "name": "Models & Architectures",
            "keywords": ["neural network", "transformer", "cnn", "rnn", "lstm",
                        "gan", "vae", "diffusion", "architecture", "model"],
            "description": "Neural network architectures and model types"
        },
        "nlp": {
            "name": "NLP & Language",
            "keywords": ["nlp", "language model", "llm", "bert", "gpt", "text",
                        "tokenization", "embedding", "sentiment", "translation"],
            "description": "Natural language processing and language models"
        },
        "vision": {
            "name": "Computer Vision",
            "keywords": ["computer vision", "image", "object detection", "segmentation",
                        "classification", "cnn", "visual", "video"],
            "description": "Image and video understanding"
        },
        "engineering": {
            "name": "ML Engineering",
            "keywords": ["training", "deployment", "mlops", "pipeline", "scaling",
                        "optimization", "distributed", "inference", "production"],
            "description": "Practical ML systems and deployment"
        },
        "applications": {
            "name": "Applications",
            "keywords": ["application", "use case", "industry", "product",
                        "real-world", "implementation", "demo", "project"],
            "description": "Real-world applications and implementations"
        }
    }

    def __init__(self, profile_manager: Optional[ProfileManager] = None):
        """Initialize the dimension visualizer."""
        self.profile_manager = profile_manager or ProfileManager()
        self.db = DatabaseManager()
        self._dimensions_cache: Optional[Dict[str, DimensionData]] = None
        self._cache_time: Optional[datetime] = None

    def get_dimensions(self, force_refresh: bool = False) -> Dict[str, DimensionData]:
        """
        Get computed dimension data.

        Args:
            force_refresh: Force recalculation even if cached

        Returns:
            Dictionary of dimension key -> DimensionData
        """
        # Check cache (valid for 5 minutes)
        if (not force_refresh and
            self._dimensions_cache and
            self._cache_time and
            (datetime.utcnow() - self._cache_time).seconds < 300):
            return self._dimensions_cache

        # Calculate dimensions
        dimensions = self._calculate_dimensions()
        self._dimensions_cache = dimensions
        self._cache_time = datetime.utcnow()

        return dimensions

    def _calculate_dimensions(self) -> Dict[str, DimensionData]:
        """Calculate scores for all dimensions."""
        liked_articles = self.db.get_liked_articles()
        week_ago = datetime.utcnow() - timedelta(days=7)

        dimensions = {}

        for dim_key, dim_config in self.DEFAULT_DIMENSIONS.items():
            # Count matching articles
            matching_articles = []
            recent_count = 0

            for article in liked_articles:
                if self._article_matches_dimension(article, dim_config["keywords"]):
                    matching_articles.append(article)
                    if article.read_at and article.read_at > week_ago:
                        recent_count += 1

            # Calculate score (0-100)
            # Score based on: article count, recency, diversity
            base_score = min(100, len(matching_articles) * 5)

            # Boost for recent activity
            recency_boost = min(20, recent_count * 5)

            # Extract top concepts
            concepts = self._extract_dimension_concepts(matching_articles, dim_config["keywords"])

            # Concept diversity boost
            diversity_boost = min(15, len(concepts) * 3)

            total_score = min(100, base_score + recency_boost + diversity_boost)

            # Calculate trend
            trend = self._calculate_trend(matching_articles, week_ago)

            # Determine if this needs focus
            suggested_focus = total_score < 40 or recent_count == 0

            dimensions[dim_key] = DimensionData(
                name=dim_config["name"],
                score=total_score,
                articles_count=len(matching_articles),
                trend=trend,
                recent_activity=recent_count,
                top_concepts=concepts[:5],
                suggested_focus=suggested_focus
            )

        return dimensions

    def _article_matches_dimension(self, article, keywords: List[str]) -> bool:
        """Check if article matches dimension keywords."""
        # Build searchable text
        text_parts = [article.title or ""]
        if article.summary:
            text_parts.append(article.summary)
        if article.related_concepts:
            try:
                concepts = json.loads(article.related_concepts)
                text_parts.extend(concepts)
            except:
                pass
        if article.tags:
            try:
                tags = json.loads(article.tags)
                text_parts.extend(tags)
            except:
                pass

        text = " ".join(text_parts).lower()

        # Check for keyword matches
        return any(kw.lower() in text for kw in keywords)

    def _extract_dimension_concepts(self, articles, keywords: List[str]) -> List[str]:
        """Extract relevant concepts from dimension articles."""
        concepts = {}

        for article in articles:
            if article.related_concepts:
                try:
                    article_concepts = json.loads(article.related_concepts)
                    for concept in article_concepts:
                        # Weight by relevance to dimension
                        weight = 1
                        if any(kw.lower() in concept.lower() for kw in keywords):
                            weight = 2
                        concepts[concept] = concepts.get(concept, 0) + weight
                except:
                    pass

        # Sort by weight and return top concepts
        sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1])
        return [c[0] for c in sorted_concepts[:10]]

    def _calculate_trend(self, articles, week_ago: datetime) -> str:
        """Calculate trend direction for a dimension."""
        if not articles:
            return "stable"

        # Count articles in last week vs previous week
        two_weeks_ago = week_ago - timedelta(days=7)

        recent = sum(1 for a in articles if a.read_at and a.read_at > week_ago)
        previous = sum(1 for a in articles
                      if a.read_at and two_weeks_ago < a.read_at <= week_ago)

        if recent > previous + 1:
            return "up"
        elif recent < previous - 1:
            return "down"
        return "stable"

    def get_radar_chart_data(self) -> RadarChartData:
        """
        Get data formatted for radar chart visualization.

        Returns:
            RadarChartData with dimension names and scores
        """
        dimensions = self.get_dimensions()

        names = []
        scores = []
        labels = []

        for dim_key, dim_data in dimensions.items():
            names.append(dim_data.name)
            scores.append(dim_data.score)
            labels.append(f"{dim_data.name}\n({dim_data.score:.0f})")

        return RadarChartData(
            dimensions=names,
            scores=scores,
            max_score=100,
            labels=labels
        )

    def get_progress_data(self) -> List[Dict]:
        """
        Get data for progress bar visualization.

        Returns:
            List of dimension progress data
        """
        dimensions = self.get_dimensions()

        progress_data = []
        for dim_key, dim_data in dimensions.items():
            trend_icon = {"up": "↑", "down": "↓", "stable": "→"}.get(dim_data.trend, "→")

            progress_data.append({
                "key": dim_key,
                "name": dim_data.name,
                "score": dim_data.score,
                "articles": dim_data.articles_count,
                "recent": dim_data.recent_activity,
                "trend": dim_data.trend,
                "trend_icon": trend_icon,
                "concepts": dim_data.top_concepts,
                "needs_focus": dim_data.suggested_focus
            })

        # Sort by score (lowest first to highlight areas needing attention)
        progress_data.sort(key=lambda x: x["score"])

        return progress_data

    def get_strengths_and_gaps(self) -> Tuple[List[str], List[str]]:
        """
        Identify user's strengths and knowledge gaps.

        Returns:
            Tuple of (strengths, gaps)
        """
        dimensions = self.get_dimensions()

        strengths = []
        gaps = []

        for dim_key, dim_data in dimensions.items():
            if dim_data.score >= 60:
                strengths.append(f"{dim_data.name} ({dim_data.score:.0f}/100)")
            elif dim_data.score < 30:
                gaps.append(f"{dim_data.name} ({dim_data.score:.0f}/100)")

        return strengths, gaps

    def get_learning_balance_score(self) -> float:
        """
        Calculate how balanced the user's learning is across dimensions.

        Returns:
            Score from 0-100 (100 = perfectly balanced)
        """
        dimensions = self.get_dimensions()
        scores = [d.score for d in dimensions.values()]

        if not scores:
            return 50

        # Calculate coefficient of variation (lower = more balanced)
        mean_score = sum(scores) / len(scores)
        if mean_score == 0:
            return 50

        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_score

        # Convert to 0-100 scale (lower CV = higher balance score)
        balance_score = max(0, 100 - (cv * 100))

        return balance_score

    def get_focus_recommendations(self) -> List[Dict]:
        """
        Get specific recommendations for what to focus on.

        Returns:
            List of focus recommendations
        """
        dimensions = self.get_dimensions()
        recommendations = []

        # Sort by need for focus
        sorted_dims = sorted(
            dimensions.items(),
            key=lambda x: (x[1].score, x[1].recent_activity)
        )

        for dim_key, dim_data in sorted_dims[:3]:  # Top 3 areas needing focus
            if dim_data.suggested_focus:
                rec = {
                    "dimension": dim_data.name,
                    "score": dim_data.score,
                    "reason": self._generate_focus_reason(dim_data),
                    "suggested_keywords": self.DEFAULT_DIMENSIONS[dim_key]["keywords"][:5],
                    "action": self._generate_action(dim_data)
                }
                recommendations.append(rec)

        return recommendations

    def _generate_focus_reason(self, dim_data: DimensionData) -> str:
        """Generate reason for focusing on a dimension."""
        if dim_data.score < 20:
            return "This area is largely unexplored in your learning journey"
        elif dim_data.score < 40:
            return "Building foundational knowledge here would strengthen your overall understanding"
        elif dim_data.recent_activity == 0:
            return "No recent activity - consider refreshing your knowledge"
        else:
            return "Continued learning here would help achieve balanced expertise"

    def _generate_action(self, dim_data: DimensionData) -> str:
        """Generate action suggestion for a dimension."""
        if dim_data.score < 20:
            return f"Search for beginner tutorials on {dim_data.name.lower()}"
        elif dim_data.score < 40:
            return f"Look for intermediate content about {', '.join(dim_data.top_concepts[:2]) if dim_data.top_concepts else dim_data.name.lower()}"
        else:
            return f"Explore advanced topics in {dim_data.name.lower()}"

    def create_matplotlib_radar(self, save_path: Optional[str] = None):
        """
        Create a matplotlib radar chart.

        Args:
            save_path: Optional path to save the figure

        Returns:
            matplotlib figure or None
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        data = self.get_radar_chart_data()
        n_dims = len(data.dimensions)

        if n_dims < 3:
            return None

        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        scores = data.scores + data.scores[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot data
        ax.plot(angles, scores, 'o-', linewidth=2, color='#667eea')
        ax.fill(angles, scores, alpha=0.25, color='#667eea')

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(data.dimensions, size=10)

        # Set y-axis
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)

        # Title
        ax.set_title('Knowledge Dimensions', size=14, y=1.08)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def get_streamlit_chart_data(self) -> Dict:
        """
        Get data formatted for Streamlit charts.

        Returns:
            Dictionary with chart data for various Streamlit visualizations
        """
        dimensions = self.get_dimensions()

        # For bar chart
        bar_data = {
            "Dimension": [],
            "Score": [],
            "Articles": []
        }

        for dim_data in dimensions.values():
            bar_data["Dimension"].append(dim_data.name)
            bar_data["Score"].append(dim_data.score)
            bar_data["Articles"].append(dim_data.articles_count)

        # For radar (using plotly format)
        radar_data = {
            "r": [d.score for d in dimensions.values()],
            "theta": [d.name for d in dimensions.values()]
        }

        return {
            "bar": bar_data,
            "radar": radar_data,
            "balance_score": self.get_learning_balance_score(),
            "strengths_gaps": self.get_strengths_and_gaps(),
            "recommendations": self.get_focus_recommendations()
        }
