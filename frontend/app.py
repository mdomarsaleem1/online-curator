"""
Knowledge Curator - Streamlit Frontend

A reader-friendly interface for exploring and learning from curated articles.
Features:
- Article recommendations based on knowledge growth potential
- Like/dislike buttons to train the model
- Category-based sections (Demos, Ideas, Tools, etc.)
- Knowledge summary and statistics
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import DatabaseManager, init_db
from src.models.article import Article, ArticleCategory
from src.services.vector_store import VectorStore
from src.services.knowledge_engine import KnowledgeEngine
from src.services.categorizer import ArticleCategorizer


# Page configuration
st.set_page_config(
    page_title="Knowledge Curator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
    .article-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .growth-score {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .category-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    .demo-badge { background-color: #28a745; color: white; }
    .idea-badge { background-color: #6f42c1; color: white; }
    .tool-badge { background-color: #fd7e14; color: white; }
    .tutorial-badge { background-color: #17a2b8; color: white; }
    .news-badge { background-color: #dc3545; color: white; }
    .recommendation-reason {
        font-style: italic;
        color: #6c757d;
        margin-top: 10px;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Initialize and cache services."""
    init_db()
    vector_store = VectorStore()
    knowledge_engine = KnowledgeEngine(vector_store)
    categorizer = ArticleCategorizer()
    return vector_store, knowledge_engine, categorizer


def get_category_color(category: ArticleCategory) -> str:
    """Get the color class for a category badge."""
    colors = {
        ArticleCategory.DEMO: "demo-badge",
        ArticleCategory.IDEA: "idea-badge",
        ArticleCategory.TOOL: "tool-badge",
        ArticleCategory.TUTORIAL: "tutorial-badge",
        ArticleCategory.NEWS: "news-badge",
        ArticleCategory.OTHER: "",
    }
    return colors.get(category, "")


def get_category_emoji(category: ArticleCategory) -> str:
    """Get emoji for category."""
    emojis = {
        ArticleCategory.DEMO: "🔧",
        ArticleCategory.IDEA: "💡",
        ArticleCategory.TOOL: "🛠️",
        ArticleCategory.TUTORIAL: "📚",
        ArticleCategory.NEWS: "📰",
        ArticleCategory.OTHER: "📎",
    }
    return emojis.get(category, "📄")


def render_article_card(article: Article, knowledge_engine: KnowledgeEngine, show_actions: bool = True):
    """Render a single article card."""
    category_emoji = get_category_emoji(article.category)

    with st.container():
        # Header row
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

        with col1:
            st.markdown(f"### {category_emoji} [{article.title}]({article.url})")

        with col2:
            if article.growth_potential > 0:
                st.metric("Growth Potential", f"{article.growth_potential:.0%}")

        with col3:
            if article.novelty_score > 0:
                st.metric("Novelty", f"{article.novelty_score:.0%}")

        # Category and source
        source_text = f" • Source: {article.source}" if article.source else ""
        st.caption(f"{article.category.value.upper()}{source_text}")

        # Summary
        if article.summary:
            st.markdown(article.summary)

        # Key insights
        if article.key_insights:
            with st.expander("Key Insights"):
                for insight in article.key_insights:
                    st.markdown(f"• {insight}")

        # Recommendation reason
        if article.recommendation_reason:
            st.markdown(f"*💭 {article.recommendation_reason}*")

        # Tags
        if article.tags:
            st.markdown(" ".join([f"`{tag}`" for tag in article.tags[:5]]))

        # Action buttons
        if show_actions and article.is_liked is None:
            col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
            with col1:
                if st.button("👍 Like", key=f"like_{article.id}", help="I read this and found it valuable"):
                    knowledge_engine.process_rating(article.id, True)
                    st.success("Added to your knowledge base!")
                    st.rerun()
            with col2:
                if st.button("👎 Skip", key=f"dislike_{article.id}", help="Not relevant or not interested"):
                    knowledge_engine.process_rating(article.id, False)
                    st.info("Marked as skipped")
                    st.rerun()
        elif article.is_liked is not None:
            status = "✅ Read & Liked" if article.is_liked else "⏭️ Skipped"
            st.caption(status)

        st.divider()


def render_sidebar(knowledge_engine: KnowledgeEngine):
    """Render the sidebar with stats and filters."""
    st.sidebar.title("🧠 Knowledge Curator")
    st.sidebar.markdown("*Your personal learning assistant*")

    st.sidebar.divider()

    # Knowledge Summary
    summary = knowledge_engine.get_knowledge_summary()

    st.sidebar.subheader("📊 Your Knowledge")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Articles Read", summary["total_learned"])
    with col2:
        st.metric("Concepts Known", summary["concepts_count"])

    # Category breakdown
    if summary["categories"]:
        st.sidebar.subheader("Categories Read")
        for cat, count in sorted(summary["categories"].items(), key=lambda x: -x[1]):
            st.sidebar.progress(count / max(summary["categories"].values()), text=f"{cat}: {count}")

    # Top concepts
    if summary["top_concepts"]:
        st.sidebar.subheader("Top Concepts")
        st.sidebar.markdown(" ".join([f"`{c}`" for c in summary["top_concepts"][:10]]))

    st.sidebar.divider()

    # Actions
    st.sidebar.subheader("🔧 Actions")
    if st.sidebar.button("🔄 Recalculate Scores"):
        with st.spinner("Recalculating..."):
            knowledge_engine.recalculate_all_scores()
        st.sidebar.success("Scores updated!")

    return summary


def main():
    """Main application entry point."""
    # Initialize services
    vector_store, knowledge_engine, categorizer = get_services()
    db = DatabaseManager()

    # Render sidebar
    summary = render_sidebar(knowledge_engine)

    # Main content area
    st.title("📚 Article Recommendations")
    st.markdown("*Curated for your optimal learning growth*")

    # View selection tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 For You",
        "🔧 Demos",
        "💡 Ideas",
        "🛠️ Tools",
        "📚 Tutorials",
        "✅ My Library"
    ])

    with tab1:
        st.subheader("Recommended for Maximum Growth")
        st.markdown("These articles are selected to expand your knowledge while building on what you already know.")

        recommendations = knowledge_engine.get_recommendations(n=10)

        if not recommendations:
            st.info("No new recommendations yet. Add some articles to get started!")
            st.markdown("""
            ### Getting Started
            1. Use the scraper to add articles from ArXiv, HuggingFace, or RSS feeds
            2. Or manually add articles using the API
            3. Rate articles with 👍 (like) or 👎 (skip) to train the system
            """)
        else:
            for rec in recommendations:
                render_article_card(rec.article, knowledge_engine)

    with tab2:
        st.subheader("🔧 Demos & Implementations")
        st.markdown("Hands-on code, projects, and practical examples")
        articles = db.get_all_articles(category=ArticleCategory.DEMO)
        pydantic_articles = [db.db_article_to_pydantic(a) for a in articles]
        for article in pydantic_articles:
            render_article_card(article, knowledge_engine)
        if not articles:
            st.info("No demo articles yet.")

    with tab3:
        st.subheader("💡 Ideas & Research")
        st.markdown("Papers, theories, and conceptual frameworks")
        articles = db.get_all_articles(category=ArticleCategory.IDEA)
        pydantic_articles = [db.db_article_to_pydantic(a) for a in articles]
        for article in pydantic_articles:
            render_article_card(article, knowledge_engine)
        if not articles:
            st.info("No research papers yet.")

    with tab4:
        st.subheader("🛠️ Tools & Libraries")
        st.markdown("Frameworks, SDKs, and developer tools")
        articles = db.get_all_articles(category=ArticleCategory.TOOL)
        pydantic_articles = [db.db_article_to_pydantic(a) for a in articles]
        for article in pydantic_articles:
            render_article_card(article, knowledge_engine)
        if not articles:
            st.info("No tool articles yet.")

    with tab5:
        st.subheader("📚 Tutorials & Guides")
        st.markdown("Educational content and how-tos")
        articles = db.get_all_articles(category=ArticleCategory.TUTORIAL)
        pydantic_articles = [db.db_article_to_pydantic(a) for a in articles]
        for article in pydantic_articles:
            render_article_card(article, knowledge_engine)
        if not articles:
            st.info("No tutorials yet.")

    with tab6:
        st.subheader("✅ Your Knowledge Library")
        st.markdown("Articles you've read and found valuable")
        liked_articles = db.get_liked_articles()
        pydantic_articles = [db.db_article_to_pydantic(a) for a in liked_articles]
        for article in pydantic_articles:
            render_article_card(article, knowledge_engine, show_actions=False)
        if not liked_articles:
            st.info("No articles in your library yet. Like articles to add them here!")


if __name__ == "__main__":
    main()
