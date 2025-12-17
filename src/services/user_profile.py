"""
User Learning Profile and Configuration System.

Manages:
- Learning topic preferences and goals
- Skill/maturity level assessment
- Learning pace and style preferences
- Progress tracking across dimensions
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import math

from src.models.database import DatabaseManager


class SkillLevel(str, Enum):
    """Skill level based on Dreyfus model of skill acquisition."""
    NOVICE = "novice"              # Following rules, limited context
    ADVANCED_BEGINNER = "advanced_beginner"  # Recognizes aspects, situational perception
    COMPETENT = "competent"        # Conscious planning, prioritization
    PROFICIENT = "proficient"      # Holistic view, intuitive decisions
    EXPERT = "expert"              # Intuitive, transcends rules


class LearningStyle(str, Enum):
    """Learning style preferences (VARK model simplified)."""
    VISUAL = "visual"              # Prefers diagrams, videos
    READING = "reading"            # Prefers text, documentation
    PRACTICAL = "practical"        # Prefers hands-on, demos
    BALANCED = "balanced"          # Mixed approach


@dataclass
class LearningTopic:
    """A topic the user wants to learn."""
    name: str
    priority: int = 1             # 1-5, higher = more important
    target_level: SkillLevel = SkillLevel.COMPETENT
    current_level: SkillLevel = SkillLevel.NOVICE
    articles_read: int = 0
    last_activity: Optional[datetime] = None
    subtopics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)  # For matching articles


@dataclass
class LearningDimension:
    """A dimension for visualizing knowledge (3-6 axes)."""
    name: str
    description: str
    score: float = 0.0            # 0-100
    articles_count: int = 0
    key_concepts: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """Complete user learning profile."""
    # Identity
    user_id: str = "default"
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Learning preferences
    topics: List[LearningTopic] = field(default_factory=list)
    learning_style: LearningStyle = LearningStyle.BALANCED
    daily_time_budget: int = 30   # minutes per day
    preferred_article_length: str = "medium"  # short, medium, long

    # Skill assessment
    overall_level: SkillLevel = SkillLevel.NOVICE
    dimension_scores: Dict[str, LearningDimension] = field(default_factory=dict)

    # Learning pace
    articles_per_week_target: int = 10
    difficulty_preference: str = "adaptive"  # easy, medium, hard, adaptive

    # Progress tracking
    total_articles_read: int = 0
    streak_days: int = 0
    last_active: Optional[datetime] = None

    # YouTube preferences
    youtube_channels: List[str] = field(default_factory=list)
    youtube_search_terms: List[str] = field(default_factory=list)

    # RSS preferences
    rss_feeds: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert enums to strings
        data["learning_style"] = self.learning_style.value
        data["overall_level"] = self.overall_level.value
        data["topics"] = [
            {**t, "target_level": t["target_level"].value if isinstance(t["target_level"], SkillLevel) else t["target_level"],
             "current_level": t["current_level"].value if isinstance(t["current_level"], SkillLevel) else t["current_level"]}
            for t in data["topics"]
        ]
        # Convert datetimes
        data["created_at"] = self.created_at.isoformat() if self.created_at else None
        data["last_active"] = self.last_active.isoformat() if self.last_active else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        """Create from dictionary."""
        if data.get("learning_style"):
            data["learning_style"] = LearningStyle(data["learning_style"])
        if data.get("overall_level"):
            data["overall_level"] = SkillLevel(data["overall_level"])
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_active") and isinstance(data["last_active"], str):
            data["last_active"] = datetime.fromisoformat(data["last_active"])

        # Convert topics
        if data.get("topics"):
            topics = []
            for t in data["topics"]:
                if isinstance(t.get("target_level"), str):
                    t["target_level"] = SkillLevel(t["target_level"])
                if isinstance(t.get("current_level"), str):
                    t["current_level"] = SkillLevel(t["current_level"])
                if t.get("last_activity") and isinstance(t["last_activity"], str):
                    t["last_activity"] = datetime.fromisoformat(t["last_activity"])
                topics.append(LearningTopic(**t))
            data["topics"] = topics

        # Convert dimensions
        if data.get("dimension_scores"):
            data["dimension_scores"] = {
                k: LearningDimension(**v) if isinstance(v, dict) else v
                for k, v in data["dimension_scores"].items()
            }

        return cls(**data)


# Default learning dimensions based on ML/AI taxonomy
DEFAULT_DIMENSIONS = {
    "foundations": LearningDimension(
        name="Foundations",
        description="Core ML concepts: linear algebra, calculus, probability, optimization",
        key_concepts=["Linear Algebra", "Calculus", "Probability", "Statistics", "Optimization"]
    ),
    "models": LearningDimension(
        name="Models & Architectures",
        description="Neural networks, transformers, diffusion models, etc.",
        key_concepts=["Neural Networks", "Transformers", "CNN", "RNN", "Diffusion", "GAN"]
    ),
    "applications": LearningDimension(
        name="Applications",
        description="NLP, computer vision, speech, multimodal AI",
        key_concepts=["NLP", "Computer Vision", "Speech", "Multimodal", "Robotics"]
    ),
    "engineering": LearningDimension(
        name="ML Engineering",
        description="Training, deployment, MLOps, scaling",
        key_concepts=["Training", "Deployment", "MLOps", "Scaling", "Infrastructure"]
    ),
    "research": LearningDimension(
        name="Research & Theory",
        description="Papers, novel techniques, theoretical advances",
        key_concepts=["Research", "Papers", "Theory", "Experiments", "Benchmarks"]
    ),
    "tools": LearningDimension(
        name="Tools & Frameworks",
        description="PyTorch, TensorFlow, HuggingFace, LangChain, etc.",
        key_concepts=["PyTorch", "TensorFlow", "HuggingFace", "LangChain", "Libraries"]
    )
}


class ProfileManager:
    """Manages user profiles and configurations."""

    PROFILE_FILE = "user_profile.json"

    def __init__(self, profile_path: Optional[str] = None):
        """Initialize the profile manager."""
        self.profile_path = profile_path or self.PROFILE_FILE
        self.profile: UserProfile = self._load_or_create_profile()
        self.db = DatabaseManager()

    def _load_or_create_profile(self) -> UserProfile:
        """Load existing profile or create new one."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    data = json.load(f)
                return UserProfile.from_dict(data)
            except Exception as e:
                print(f"Error loading profile: {e}")

        # Create default profile with dimensions
        profile = UserProfile()
        profile.dimension_scores = DEFAULT_DIMENSIONS.copy()
        return profile

    def save_profile(self):
        """Save profile to disk."""
        with open(self.profile_path, 'w') as f:
            json.dump(self.profile.to_dict(), f, indent=2, default=str)

    def add_topic(self, topic: LearningTopic):
        """Add a learning topic."""
        # Check if topic exists
        existing = next((t for t in self.profile.topics if t.name.lower() == topic.name.lower()), None)
        if existing:
            # Update existing
            existing.priority = topic.priority
            existing.target_level = topic.target_level
            existing.keywords = topic.keywords
        else:
            self.profile.topics.append(topic)
        self.save_profile()

    def remove_topic(self, topic_name: str):
        """Remove a learning topic."""
        self.profile.topics = [t for t in self.profile.topics if t.name.lower() != topic_name.lower()]
        self.save_profile()

    def update_topic_progress(self, topic_name: str, articles_read: int = 1):
        """Update progress for a topic."""
        for topic in self.profile.topics:
            if topic.name.lower() == topic_name.lower():
                topic.articles_read += articles_read
                topic.last_activity = datetime.utcnow()
                self._assess_topic_level(topic)
                break
        self.save_profile()

    def _assess_topic_level(self, topic: LearningTopic):
        """Assess skill level for a topic based on articles read."""
        # Simple heuristic based on article count
        # Can be enhanced with comprehension tests, time spent, etc.
        articles = topic.articles_read

        if articles < 5:
            topic.current_level = SkillLevel.NOVICE
        elif articles < 15:
            topic.current_level = SkillLevel.ADVANCED_BEGINNER
        elif articles < 30:
            topic.current_level = SkillLevel.COMPETENT
        elif articles < 50:
            topic.current_level = SkillLevel.PROFICIENT
        else:
            topic.current_level = SkillLevel.EXPERT

    def assess_overall_maturity(self) -> Tuple[SkillLevel, Dict[str, float]]:
        """
        Assess user's overall maturity level based on reading history.

        Returns:
            Tuple of (overall_level, dimension_breakdown)
        """
        liked_articles = self.db.get_liked_articles()

        if not liked_articles:
            return SkillLevel.NOVICE, {}

        # Calculate dimension scores
        dimension_scores = {}
        for dim_key, dimension in self.profile.dimension_scores.items():
            score = 0
            count = 0

            for article in liked_articles:
                # Check if article relates to this dimension
                article_concepts = []
                if article.related_concepts:
                    try:
                        article_concepts = json.loads(article.related_concepts)
                    except:
                        pass

                # Match concepts
                matches = sum(1 for c in article_concepts
                            if any(kc.lower() in c.lower() or c.lower() in kc.lower()
                                  for kc in dimension.key_concepts))

                if matches > 0:
                    count += 1
                    # Weight by article complexity (approximated by tag count)
                    complexity = 1 + (len(article_concepts) / 10)
                    score += matches * complexity

            # Normalize to 0-100
            normalized_score = min(100, (score / max(count, 1)) * 20) if count > 0 else 0
            dimension.score = normalized_score
            dimension.articles_count = count
            dimension_scores[dim_key] = normalized_score

        # Calculate overall level
        avg_score = sum(dimension_scores.values()) / max(len(dimension_scores), 1)

        if avg_score < 20:
            overall = SkillLevel.NOVICE
        elif avg_score < 40:
            overall = SkillLevel.ADVANCED_BEGINNER
        elif avg_score < 60:
            overall = SkillLevel.COMPETENT
        elif avg_score < 80:
            overall = SkillLevel.PROFICIENT
        else:
            overall = SkillLevel.EXPERT

        self.profile.overall_level = overall
        self.save_profile()

        return overall, dimension_scores

    def get_learning_recommendations(self) -> Dict[str, List[str]]:
        """
        Get personalized recommendations for improvement.

        Returns recommendations based on:
        - Weakest dimensions (need more coverage)
        - Stagnant topics (not recently active)
        - Next level suggestions
        """
        recommendations = {
            "weak_areas": [],
            "suggested_topics": [],
            "next_steps": [],
            "stretch_goals": []
        }

        # Find weakest dimensions
        sorted_dims = sorted(
            self.profile.dimension_scores.items(),
            key=lambda x: x[1].score if isinstance(x[1], LearningDimension) else 0
        )

        for dim_key, dim in sorted_dims[:2]:
            if isinstance(dim, LearningDimension) and dim.score < 50:
                recommendations["weak_areas"].append(
                    f"{dim.name}: Score {dim.score:.0f}/100. "
                    f"Focus on: {', '.join(dim.key_concepts[:3])}"
                )

        # Find stagnant topics
        now = datetime.utcnow()
        for topic in self.profile.topics:
            if topic.last_activity:
                days_inactive = (now - topic.last_activity).days
                if days_inactive > 7:
                    recommendations["suggested_topics"].append(
                        f"Resume '{topic.name}' - inactive for {days_inactive} days"
                    )

        # Next level suggestions based on current level
        level_suggestions = {
            SkillLevel.NOVICE: "Start with tutorials and beginner guides",
            SkillLevel.ADVANCED_BEGINNER: "Try implementing simple projects",
            SkillLevel.COMPETENT: "Explore research papers and advanced tutorials",
            SkillLevel.PROFICIENT: "Contribute to open source, write technical content",
            SkillLevel.EXPERT: "Focus on cutting-edge research and novel applications"
        }
        recommendations["next_steps"].append(level_suggestions.get(
            self.profile.overall_level,
            "Continue exploring diverse content"
        ))

        # Stretch goals
        for topic in self.profile.topics:
            if topic.current_level != topic.target_level:
                levels = list(SkillLevel)
                current_idx = levels.index(topic.current_level)
                target_idx = levels.index(topic.target_level)
                if target_idx > current_idx:
                    recommendations["stretch_goals"].append(
                        f"Advance '{topic.name}' from {topic.current_level.value} to {topic.target_level.value}"
                    )

        return recommendations

    def get_topic_keywords(self) -> List[str]:
        """Get all keywords from configured topics for article matching."""
        keywords = []
        for topic in self.profile.topics:
            keywords.append(topic.name.lower())
            keywords.extend([k.lower() for k in topic.keywords])
            keywords.extend([s.lower() for s in topic.subtopics])
        return list(set(keywords))

    def matches_user_interests(self, article_text: str) -> Tuple[bool, float]:
        """
        Check if article matches user's learning interests.

        Returns:
            Tuple of (matches, relevance_score)
        """
        if not self.profile.topics:
            return True, 0.5  # No preferences set, neutral

        text_lower = article_text.lower()
        keywords = self.get_topic_keywords()

        if not keywords:
            return True, 0.5

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in text_lower)
        relevance = min(1.0, matches / max(len(keywords), 1) * 5)

        return matches > 0, relevance

    def update_activity(self):
        """Update user activity tracking."""
        now = datetime.utcnow()

        if self.profile.last_active:
            # Check if streak continues (within 48 hours)
            hours_since = (now - self.profile.last_active).total_seconds() / 3600
            if hours_since < 48:
                # Check if it's a new day
                if self.profile.last_active.date() < now.date():
                    self.profile.streak_days += 1
            else:
                self.profile.streak_days = 1
        else:
            self.profile.streak_days = 1

        self.profile.last_active = now
        self.save_profile()
