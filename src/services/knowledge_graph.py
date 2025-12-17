"""
Knowledge Graph visualization service.

Builds and visualizes the knowledge graph showing:
- Articles as nodes (colored by category)
- Concepts as nodes (shared between articles)
- Edges connecting articles to their concepts
- User's learned knowledge highlighted
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging

from src.models.database import DatabaseManager
from src.models.article import ArticleCategory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    label: str
    node_type: str  # 'article' or 'concept'
    category: Optional[str] = None
    is_learned: bool = False
    size: int = 10
    color: str = "#666666"
    url: Optional[str] = None


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "related_to"


class KnowledgeGraph:
    """
    Builds and manages the knowledge graph visualization.

    The graph shows:
    - Articles connected to their concepts
    - Concepts shared between articles (creating clusters)
    - User's learned articles highlighted
    - Category-based coloring
    """

    # Colors for different categories
    CATEGORY_COLORS = {
        ArticleCategory.DEMO: "#28a745",      # Green
        ArticleCategory.IDEA: "#6f42c1",      # Purple
        ArticleCategory.TOOL: "#fd7e14",      # Orange
        ArticleCategory.TUTORIAL: "#17a2b8",  # Teal
        ArticleCategory.NEWS: "#dc3545",      # Red
        ArticleCategory.OTHER: "#6c757d",     # Gray
    }

    CONCEPT_COLOR = "#ffc107"  # Yellow for concepts
    LEARNED_COLOR = "#00ff00"  # Bright green for learned

    def __init__(self):
        """Initialize the knowledge graph."""
        self.db = DatabaseManager()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def build_graph(
        self,
        include_unread: bool = True,
        max_articles: int = 100,
        min_concept_connections: int = 2
    ) -> Tuple[List[dict], List[dict]]:
        """
        Build the knowledge graph from articles.

        Args:
            include_unread: Whether to include unread/unrated articles
            max_articles: Maximum number of articles to include
            min_concept_connections: Minimum connections for a concept to be shown

        Returns:
            Tuple of (nodes_list, edges_list) for visualization
        """
        self.nodes = {}
        self.edges = []

        # Get articles
        if include_unread:
            articles = self.db.get_all_articles(limit=max_articles)
        else:
            articles = self.db.get_liked_articles()

        if not articles:
            return [], []

        # Track concept frequencies
        concept_counts: Dict[str, int] = {}
        concept_articles: Dict[str, List[str]] = {}

        # First pass: count concepts
        for article in articles:
            concepts = self._get_article_concepts(article)
            for concept in concepts:
                concept_lower = concept.lower()
                concept_counts[concept_lower] = concept_counts.get(concept_lower, 0) + 1
                if concept_lower not in concept_articles:
                    concept_articles[concept_lower] = []
                concept_articles[concept_lower].append(f"article_{article.id}")

        # Filter concepts by minimum connections
        valid_concepts = {
            c for c, count in concept_counts.items()
            if count >= min_concept_connections
        }

        # Second pass: build nodes and edges
        for article in articles:
            article_id = f"article_{article.id}"

            # Create article node
            category = article.category or ArticleCategory.OTHER
            color = self.CATEGORY_COLORS.get(category, "#6c757d")

            # Highlight learned articles
            if article.is_liked:
                color = self.LEARNED_COLOR

            self.nodes[article_id] = GraphNode(
                id=article_id,
                label=self._truncate(article.title, 40),
                node_type="article",
                category=category.value if category else "other",
                is_learned=article.is_liked or False,
                size=15 if article.is_liked else 10,
                color=color,
                url=article.url
            )

            # Create edges to concepts
            concepts = self._get_article_concepts(article)
            for concept in concepts:
                concept_lower = concept.lower()
                if concept_lower not in valid_concepts:
                    continue

                concept_id = f"concept_{concept_lower.replace(' ', '_')}"

                # Create concept node if not exists
                if concept_id not in self.nodes:
                    # Size based on how many articles connect to it
                    size = 8 + (concept_counts.get(concept_lower, 0) * 2)
                    self.nodes[concept_id] = GraphNode(
                        id=concept_id,
                        label=concept.title(),
                        node_type="concept",
                        size=min(size, 30),
                        color=self.CONCEPT_COLOR
                    )

                # Create edge
                self.edges.append(GraphEdge(
                    source=article_id,
                    target=concept_id,
                    edge_type="covers"
                ))

        return self._to_visualization_format()

    def _get_article_concepts(self, article) -> List[str]:
        """Extract concepts from an article."""
        concepts = []

        if article.related_concepts:
            try:
                if isinstance(article.related_concepts, str):
                    concepts = json.loads(article.related_concepts)
                else:
                    concepts = article.related_concepts
            except (json.JSONDecodeError, TypeError):
                pass

        if article.tags:
            try:
                if isinstance(article.tags, str):
                    tags = json.loads(article.tags)
                else:
                    tags = article.tags
                concepts.extend(tags[:5])
            except (json.JSONDecodeError, TypeError):
                pass

        return concepts

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def _to_visualization_format(self) -> Tuple[List[dict], List[dict]]:
        """Convert graph to visualization format."""
        nodes_list = []
        for node in self.nodes.values():
            nodes_list.append({
                "id": node.id,
                "label": node.label,
                "type": node.node_type,
                "category": node.category,
                "is_learned": node.is_learned,
                "size": node.size,
                "color": node.color,
                "url": node.url
            })

        edges_list = []
        for edge in self.edges:
            edges_list.append({
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "type": edge.edge_type
            })

        return nodes_list, edges_list

    def get_pyvis_graph(
        self,
        include_unread: bool = True,
        height: str = "600px",
        width: str = "100%"
    ):
        """
        Build and return a PyVis network graph for visualization.

        Args:
            include_unread: Whether to include unread articles
            height: Height of the visualization
            width: Width of the visualization

        Returns:
            PyVis Network object
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error("pyvis not installed. Install with: pip install pyvis")
            return None

        # Build graph data
        nodes, edges = self.build_graph(include_unread=include_unread)

        # Create PyVis network
        net = Network(
            height=height,
            width=width,
            bgcolor="#ffffff",
            font_color="#000000",
            directed=False
        )

        # Configure physics
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 100
                }
            },
            "nodes": {
                "font": {
                    "size": 12
                }
            },
            "edges": {
                "color": {
                    "opacity": 0.5
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100
            }
        }
        """)

        # Add nodes
        for node in nodes:
            shape = "dot" if node["type"] == "article" else "diamond"
            title = f"{node['label']}"
            if node.get("url"):
                title += f"\n{node['url']}"
            if node.get("is_learned"):
                title += "\n✓ Learned"

            net.add_node(
                node["id"],
                label=node["label"],
                title=title,
                size=node["size"],
                color=node["color"],
                shape=shape
            )

        # Add edges
        for edge in edges:
            net.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"]
            )

        return net

    def save_html(
        self,
        filepath: str = "knowledge_graph.html",
        include_unread: bool = True
    ) -> str:
        """
        Save the knowledge graph as an interactive HTML file.

        Args:
            filepath: Path to save the HTML file
            include_unread: Whether to include unread articles

        Returns:
            Path to the saved file
        """
        net = self.get_pyvis_graph(include_unread=include_unread)
        if net:
            net.save_graph(filepath)
            logger.info(f"Knowledge graph saved to {filepath}")
            return filepath
        return ""

    def get_graph_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        if not self.nodes:
            self.build_graph()

        article_nodes = [n for n in self.nodes.values() if n.node_type == "article"]
        concept_nodes = [n for n in self.nodes.values() if n.node_type == "concept"]
        learned = [n for n in article_nodes if n.is_learned]

        # Category distribution
        categories = {}
        for node in article_nodes:
            cat = node.category or "other"
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_articles": len(article_nodes),
            "total_concepts": len(concept_nodes),
            "learned_articles": len(learned),
            "total_edges": len(self.edges),
            "category_distribution": categories,
            "avg_concepts_per_article": len(self.edges) / max(len(article_nodes), 1)
        }

    def get_concept_clusters(self) -> Dict[str, List[str]]:
        """
        Get clusters of articles grouped by shared concepts.

        Returns:
            Dict mapping concept to list of article titles
        """
        if not self.edges:
            self.build_graph()

        clusters = {}
        for edge in self.edges:
            if edge.target.startswith("concept_"):
                concept = self.nodes[edge.target].label
                article = self.nodes[edge.source].label
                if concept not in clusters:
                    clusters[concept] = []
                clusters[concept].append(article)

        # Sort by cluster size
        return dict(sorted(clusters.items(), key=lambda x: -len(x[1])))

    def get_learning_path(self, target_concept: str) -> List[dict]:
        """
        Suggest a learning path to understand a concept.

        Finds articles that cover prerequisites before the target concept.

        Args:
            target_concept: The concept to learn

        Returns:
            Ordered list of articles forming a learning path
        """
        if not self.nodes:
            self.build_graph()

        # Find articles covering the target concept
        target_id = f"concept_{target_concept.lower().replace(' ', '_')}"
        if target_id not in self.nodes:
            return []

        # Find all articles connected to target
        target_articles = []
        for edge in self.edges:
            if edge.target == target_id:
                article_node = self.nodes.get(edge.source)
                if article_node:
                    target_articles.append(article_node)

        # Sort: learned first, then by size (importance)
        target_articles.sort(key=lambda a: (-int(a.is_learned), -a.size))

        return [
            {
                "title": a.label,
                "url": a.url,
                "is_learned": a.is_learned,
                "category": a.category
            }
            for a in target_articles
        ]
