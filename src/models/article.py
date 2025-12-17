"""Article and user preference models."""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ArticleCategory(str, Enum):
    """Categories for articles based on their nature."""
    DEMO = "demo"           # Hands-on tutorials, code demos, implementations
    IDEA = "idea"           # Conceptual papers, research ideas, theories
    TOOL = "tool"           # Tools, libraries, frameworks
    NEWS = "news"           # Industry news, announcements
    TUTORIAL = "tutorial"   # Educational content, how-tos
    OTHER = "other"         # Uncategorized


class ArticleDB(Base):
    """SQLAlchemy model for articles in the database."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), unique=True, nullable=False)
    title = Column(String(512), nullable=False)
    source = Column(String(256))  # ArXiv, HuggingFace, YouTube, RSS, etc.

    # Content
    raw_content = Column(Text)
    summary = Column(Text)  # High-level summary for display
    key_insights = Column(Text)  # JSON list of key insights

    # Categorization
    category = Column(SQLEnum(ArticleCategory), default=ArticleCategory.OTHER)
    tags = Column(Text)  # JSON list of tags

    # Knowledge graph connections
    related_concepts = Column(Text)  # JSON list of concepts this article covers
    prerequisite_concepts = Column(Text)  # JSON list of concepts needed to understand this

    # User interaction
    is_liked = Column(Boolean, nullable=True)  # None = not rated, True = liked, False = disliked
    read_at = Column(DateTime, nullable=True)

    # Recommendation metadata
    novelty_score = Column(Float, default=0.0)  # How novel is this compared to known knowledge
    growth_potential = Column(Float, default=0.0)  # How much will this expand knowledge
    recommendation_reason = Column(Text)  # Why we think this is worth reading

    # Timestamps
    scraped_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserPreferenceDB(Base):
    """SQLAlchemy model for tracking user's knowledge state."""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Knowledge state embedding (stored as JSON)
    knowledge_embedding = Column(Text)  # Current knowledge state vector

    # Statistics
    total_articles_read = Column(Integer, default=0)
    total_liked = Column(Integer, default=0)
    total_disliked = Column(Integer, default=0)

    # Preferences
    preferred_categories = Column(Text)  # JSON: category -> weight
    preferred_sources = Column(Text)  # JSON: source -> weight

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic models for API
class Article(BaseModel):
    """Pydantic model for article data."""
    id: Optional[int] = None
    url: str
    title: str
    source: Optional[str] = None
    summary: Optional[str] = None
    key_insights: Optional[List[str]] = Field(default_factory=list)
    category: ArticleCategory = ArticleCategory.OTHER
    tags: Optional[List[str]] = Field(default_factory=list)
    related_concepts: Optional[List[str]] = Field(default_factory=list)
    prerequisite_concepts: Optional[List[str]] = Field(default_factory=list)
    is_liked: Optional[bool] = None
    novelty_score: float = 0.0
    growth_potential: float = 0.0
    recommendation_reason: Optional[str] = None
    scraped_at: Optional[datetime] = None
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserPreference(BaseModel):
    """Pydantic model for user preferences."""
    id: Optional[int] = None
    total_articles_read: int = 0
    total_liked: int = 0
    total_disliked: int = 0
    preferred_categories: Optional[dict] = Field(default_factory=dict)
    preferred_sources: Optional[dict] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class ArticleRecommendation(BaseModel):
    """A recommended article with explanation."""
    article: Article
    score: float
    reason: str
    knowledge_gap: List[str]  # What new concepts will this teach
    overlap_percentage: float  # How much overlaps with current knowledge
