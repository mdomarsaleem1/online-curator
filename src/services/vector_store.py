"""Vector store service using ChromaDB for semantic embeddings."""

import os
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.models.article import Article


class VectorStore:
    """
    Vector database for storing and querying article embeddings.

    Uses ChromaDB with sentence-transformers for embeddings.
    Stores articles by their high-level summaries for semantic search.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize the vector store."""
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "./chroma_db"
        )

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Use sentence-transformers for embeddings (all-MiniLM-L6-v2 is fast and good)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Create collections for different purposes
        self._init_collections()

    def _init_collections(self):
        """Initialize the ChromaDB collections."""
        # Main collection for all articles (indexed by summary)
        self.articles_collection = self.client.get_or_create_collection(
            name="articles",
            embedding_function=self.embedding_function,
            metadata={"description": "Article summaries for semantic search"}
        )

        # Collection for user's known knowledge (liked articles)
        self.knowledge_collection = self.client.get_or_create_collection(
            name="user_knowledge",
            embedding_function=self.embedding_function,
            metadata={"description": "User's learned knowledge from liked articles"}
        )

        # Collection for concepts/topics
        self.concepts_collection = self.client.get_or_create_collection(
            name="concepts",
            embedding_function=self.embedding_function,
            metadata={"description": "Concepts and topics extracted from articles"}
        )

    def add_article(self, article: Article) -> bool:
        """
        Add an article to the vector store.

        The article is indexed by its summary for semantic search.
        Returns True if successful, False if article already exists.
        """
        # Check if article already exists
        existing = self.articles_collection.get(ids=[str(article.id)])
        if existing and existing["ids"]:
            return False

        # Prepare the document text (summary + key insights)
        doc_text = self._prepare_document_text(article)

        # Prepare metadata
        metadata = {
            "id": article.id,
            "url": article.url,
            "title": article.title,
            "source": article.source or "",
            "category": article.category.value,
            "is_liked": str(article.is_liked) if article.is_liked is not None else "none",
        }

        # Add to collection
        self.articles_collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[str(article.id)]
        )

        # If article has related concepts, add them
        if article.related_concepts:
            self._add_concepts(article.related_concepts, article.id)

        return True

    def _prepare_document_text(self, article: Article) -> str:
        """Prepare the document text for embedding."""
        parts = [article.title]
        if article.summary:
            parts.append(article.summary)
        if article.key_insights:
            parts.append(" ".join(article.key_insights))
        if article.tags:
            parts.append(" ".join(article.tags))
        return " ".join(parts)

    def _add_concepts(self, concepts: List[str], article_id: int):
        """Add concepts from an article to the concepts collection."""
        for concept in concepts:
            concept_id = f"concept_{concept.lower().replace(' ', '_')}"
            existing = self.concepts_collection.get(ids=[concept_id])
            if not existing or not existing["ids"]:
                self.concepts_collection.add(
                    documents=[concept],
                    metadatas=[{"concept": concept, "article_ids": str(article_id)}],
                    ids=[concept_id]
                )

    def mark_as_known(self, article: Article):
        """
        Mark an article as part of user's knowledge (liked).

        This adds the article to the knowledge collection for tracking
        what the user has already learned.
        """
        doc_text = self._prepare_document_text(article)

        # Add to knowledge collection
        existing = self.knowledge_collection.get(ids=[str(article.id)])
        if not existing or not existing["ids"]:
            self.knowledge_collection.add(
                documents=[doc_text],
                metadatas={
                    "id": article.id,
                    "title": article.title,
                    "category": article.category.value,
                },
                ids=[str(article.id)]
            )

        # Update the article's status in the main collection
        self.articles_collection.update(
            ids=[str(article.id)],
            metadatas=[{"is_liked": "true"}]
        )

    def remove_from_knowledge(self, article_id: int):
        """Remove an article from the knowledge collection (disliked)."""
        try:
            self.knowledge_collection.delete(ids=[str(article_id)])
        except Exception:
            pass  # Article might not be in knowledge collection

        # Update the article's status
        self.articles_collection.update(
            ids=[str(article_id)],
            metadatas=[{"is_liked": "false"}]
        )

    def find_similar_articles(
        self,
        query: str,
        n_results: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find articles similar to a query.

        Returns articles sorted by similarity score.
        """
        where_filter = None
        if category:
            where_filter = {"category": category}

        results = self.articles_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        return self._format_results(results)

    def find_articles_for_growth(
        self,
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find articles that would provide maximum knowledge growth.

        This uses the knowledge collection to find articles that are
        somewhat related to known knowledge but not too similar.

        Returns articles sorted by growth potential.
        """
        # Get the user's current knowledge
        knowledge_docs = self.knowledge_collection.get(include=["documents"])

        if not knowledge_docs or not knowledge_docs["documents"]:
            # No knowledge yet, return newest unrated articles
            return self._get_unrated_articles(n_results)

        # Combine all knowledge into a query
        knowledge_text = " ".join(knowledge_docs["documents"])

        # Find all articles and calculate growth scores
        all_articles = self.articles_collection.get(
            include=["documents", "metadatas"]
        )

        if not all_articles or not all_articles["ids"]:
            return []

        # Query to find similarity to current knowledge
        results = self.articles_collection.query(
            query_texts=[knowledge_text],
            n_results=min(100, len(all_articles["ids"])),
            include=["documents", "metadatas", "distances"]
        )

        return self._format_results(results)

    def _get_unrated_articles(self, n_results: int) -> List[Dict[str, Any]]:
        """Get unrated articles when there's no knowledge yet."""
        results = self.articles_collection.get(
            where={"is_liked": "none"},
            include=["documents", "metadatas"],
            limit=n_results
        )

        articles = []
        if results and results["ids"]:
            for i, id_ in enumerate(results["ids"]):
                articles.append({
                    "id": id_,
                    "document": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    "distance": 0.5  # Neutral distance
                })
        return articles

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results into a cleaner structure."""
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                formatted.append({
                    "id": id_,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0
                })
        return formatted

    def get_knowledge_embedding(self) -> Optional[List[float]]:
        """
        Get a single embedding representing the user's current knowledge.

        This is the centroid of all liked article embeddings.
        """
        knowledge = self.knowledge_collection.get(include=["embeddings"])
        if not knowledge or not knowledge["embeddings"]:
            return None

        # Calculate centroid of all knowledge embeddings
        import numpy as np
        embeddings = np.array(knowledge["embeddings"])
        centroid = np.mean(embeddings, axis=0)
        return centroid.tolist()

    def calculate_novelty(self, article: Article) -> float:
        """
        Calculate how novel an article is compared to known knowledge.

        Returns a score from 0 (completely known) to 1 (completely novel).
        """
        knowledge = self.knowledge_collection.get(include=["documents"])
        if not knowledge or not knowledge["documents"]:
            return 0.5  # Neutral if no knowledge yet

        doc_text = self._prepare_document_text(article)

        # Query against knowledge base
        results = self.knowledge_collection.query(
            query_texts=[doc_text],
            n_results=5,
            include=["distances"]
        )

        if not results or not results["distances"] or not results["distances"][0]:
            return 0.5

        # Average distance to closest known articles
        avg_distance = sum(results["distances"][0]) / len(results["distances"][0])

        # Normalize to 0-1 range (distances are typically 0-2 for cosine)
        novelty = min(1.0, avg_distance / 1.5)
        return novelty

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the vector store."""
        return {
            "total_articles": self.articles_collection.count(),
            "known_articles": self.knowledge_collection.count(),
            "concepts": self.concepts_collection.count(),
        }
