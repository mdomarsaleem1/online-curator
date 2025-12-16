"""Database connection and management."""

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Generator
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .article import Base, ArticleDB, UserPreferenceDB, Article, ArticleCategory


def get_database_url() -> str:
    """Get database URL from environment or default."""
    return os.getenv("DATABASE_URL", "sqlite:///./knowledge.db")


# Create engine and session factory
engine = create_engine(
    get_database_url(),
    connect_args={"check_same_thread": False} if "sqlite" in get_database_url() else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize the database tables."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get database session context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Manager for database operations."""

    @staticmethod
    def add_article(article: Article) -> ArticleDB:
        """Add a new article to the database."""
        with get_db() as db:
            db_article = ArticleDB(
                url=article.url,
                title=article.title,
                source=article.source,
                summary=article.summary,
                key_insights=json.dumps(article.key_insights) if article.key_insights else None,
                category=article.category,
                tags=json.dumps(article.tags) if article.tags else None,
                related_concepts=json.dumps(article.related_concepts) if article.related_concepts else None,
                prerequisite_concepts=json.dumps(article.prerequisite_concepts) if article.prerequisite_concepts else None,
                novelty_score=article.novelty_score,
                growth_potential=article.growth_potential,
                recommendation_reason=article.recommendation_reason,
                published_at=article.published_at,
            )
            db.add(db_article)
            db.commit()
            db.refresh(db_article)
            return db_article

    @staticmethod
    def get_article_by_url(url: str) -> Optional[ArticleDB]:
        """Get article by URL."""
        with get_db() as db:
            return db.query(ArticleDB).filter(ArticleDB.url == url).first()

    @staticmethod
    def get_article_by_id(article_id: int) -> Optional[ArticleDB]:
        """Get article by ID."""
        with get_db() as db:
            return db.query(ArticleDB).filter(ArticleDB.id == article_id).first()

    @staticmethod
    def get_all_articles(
        category: Optional[ArticleCategory] = None,
        is_liked: Optional[bool] = None,
        limit: int = 100
    ) -> List[ArticleDB]:
        """Get all articles with optional filters."""
        with get_db() as db:
            query = db.query(ArticleDB)
            if category:
                query = query.filter(ArticleDB.category == category)
            if is_liked is not None:
                query = query.filter(ArticleDB.is_liked == is_liked)
            return query.order_by(ArticleDB.scraped_at.desc()).limit(limit).all()

    @staticmethod
    def get_unrated_articles(limit: int = 50) -> List[ArticleDB]:
        """Get articles that haven't been rated yet."""
        with get_db() as db:
            return db.query(ArticleDB).filter(
                ArticleDB.is_liked.is_(None)
            ).order_by(ArticleDB.growth_potential.desc()).limit(limit).all()

    @staticmethod
    def update_article_rating(article_id: int, is_liked: bool) -> Optional[ArticleDB]:
        """Update the like/dislike status of an article."""
        with get_db() as db:
            article = db.query(ArticleDB).filter(ArticleDB.id == article_id).first()
            if article:
                article.is_liked = is_liked
                article.read_at = datetime.utcnow()
                db.commit()
                db.refresh(article)
            return article

    @staticmethod
    def get_liked_articles() -> List[ArticleDB]:
        """Get all liked articles (represents user's knowledge base)."""
        with get_db() as db:
            return db.query(ArticleDB).filter(ArticleDB.is_liked == True).all()

    @staticmethod
    def update_article_scores(
        article_id: int,
        novelty_score: float,
        growth_potential: float,
        recommendation_reason: str
    ):
        """Update recommendation scores for an article."""
        with get_db() as db:
            article = db.query(ArticleDB).filter(ArticleDB.id == article_id).first()
            if article:
                article.novelty_score = novelty_score
                article.growth_potential = growth_potential
                article.recommendation_reason = recommendation_reason
                db.commit()

    @staticmethod
    def get_user_preference() -> Optional[UserPreferenceDB]:
        """Get the user preference record (singleton)."""
        with get_db() as db:
            pref = db.query(UserPreferenceDB).first()
            if not pref:
                pref = UserPreferenceDB()
                db.add(pref)
                db.commit()
                db.refresh(pref)
            return pref

    @staticmethod
    def update_user_stats(liked: bool):
        """Update user statistics after rating an article."""
        with get_db() as db:
            pref = db.query(UserPreferenceDB).first()
            if not pref:
                pref = UserPreferenceDB()
                db.add(pref)
            pref.total_articles_read += 1
            if liked:
                pref.total_liked += 1
            else:
                pref.total_disliked += 1
            pref.updated_at = datetime.utcnow()
            db.commit()

    @staticmethod
    def db_article_to_pydantic(db_article: ArticleDB) -> Article:
        """Convert database article to Pydantic model."""
        return Article(
            id=db_article.id,
            url=db_article.url,
            title=db_article.title,
            source=db_article.source,
            summary=db_article.summary,
            key_insights=json.loads(db_article.key_insights) if db_article.key_insights else [],
            category=db_article.category,
            tags=json.loads(db_article.tags) if db_article.tags else [],
            related_concepts=json.loads(db_article.related_concepts) if db_article.related_concepts else [],
            prerequisite_concepts=json.loads(db_article.prerequisite_concepts) if db_article.prerequisite_concepts else [],
            is_liked=db_article.is_liked,
            novelty_score=db_article.novelty_score or 0.0,
            growth_potential=db_article.growth_potential or 0.0,
            recommendation_reason=db_article.recommendation_reason,
            scraped_at=db_article.scraped_at,
            published_at=db_article.published_at,
        )
