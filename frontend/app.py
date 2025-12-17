"""
Knowledge Curator - Streamlit Frontend

A reader-friendly interface for exploring and learning from curated articles.
Features:
- Article recommendations based on knowledge growth potential
- Like/dislike buttons to train the model
- Category-based sections (Demos, Ideas, Tools, etc.)
- Knowledge summary and statistics
- Knowledge graph visualization
- Scraper interface for adding new content
- Email digest configuration
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import DatabaseManager, init_db
from src.models.article import Article, ArticleCategory
from src.services.vector_store import VectorStore
from src.services.knowledge_engine import KnowledgeEngine
from src.services.categorizer import ArticleCategorizer
from src.services.knowledge_graph import KnowledgeGraph
from src.services.email_digest import EmailDigest, DigestConfig


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
    .scraper-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .scraper-success { background-color: #d4edda; color: #155724; }
    .scraper-error { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Initialize and cache services."""
    init_db()
    vector_store = VectorStore()
    knowledge_engine = KnowledgeEngine(vector_store)
    categorizer = ArticleCategorizer()
    knowledge_graph = KnowledgeGraph()
    return vector_store, knowledge_engine, categorizer, knowledge_graph


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


def render_scraper_tab():
    """Render the scraper interface tab."""
    st.subheader("🔍 Content Scraper")
    st.markdown("Add new articles from various sources")

    # Source selection
    source = st.selectbox(
        "Select Source",
        ["ArXiv", "HuggingFace", "YouTube", "RSS Feed"]
    )

    if source == "ArXiv":
        st.markdown("#### ArXiv Research Papers")
        query = st.text_input("Search Query (e.g., 'transformer attention')", key="arxiv_query")
        categories = st.multiselect(
            "Categories",
            ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "stat.ML"],
            default=["cs.LG", "cs.AI"]
        )
        limit = st.slider("Number of papers", 5, 50, 10)

        if st.button("🔎 Fetch ArXiv Papers"):
            with st.spinner("Fetching from ArXiv..."):
                try:
                    from src.agents.arxiv_scraper import ArxivScraper
                    scraper = ArxivScraper()
                    articles = scraper.scrape(query=query if query else None, limit=limit, categories=categories)
                    _process_scraped_articles(articles)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif source == "HuggingFace":
        st.markdown("#### HuggingFace Hub")
        content_type = st.selectbox(
            "Content Type",
            ["papers", "models", "datasets", "spaces"]
        )
        query = st.text_input("Search Query (optional)", key="hf_query")
        limit = st.slider("Number of items", 5, 30, 10, key="hf_limit")

        if st.button("🔎 Fetch from HuggingFace"):
            with st.spinner("Fetching from HuggingFace..."):
                try:
                    from src.agents.huggingface_scraper import HuggingFaceScraper
                    scraper = HuggingFaceScraper()
                    articles = scraper.scrape(query=query if query else None, limit=limit, content_type=content_type)
                    _process_scraped_articles(articles)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif source == "YouTube":
        st.markdown("#### YouTube Videos")
        st.info("Enter YouTube video URLs (one per line)")
        urls = st.text_area("Video URLs", height=100, key="yt_urls")

        if st.button("🔎 Fetch YouTube Videos"):
            if urls:
                with st.spinner("Fetching YouTube videos..."):
                    try:
                        from src.agents.youtube_scraper import YouTubeScraper
                        scraper = YouTubeScraper()
                        url_list = [u.strip() for u in urls.split("\n") if u.strip()]
                        articles = scraper.scrape(video_urls=url_list)
                        _process_scraped_articles(articles)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter at least one YouTube URL")

    elif source == "RSS Feed":
        st.markdown("#### RSS/Atom Feeds")

        feed_option = st.radio(
            "Feed Selection",
            ["Popular ML/AI Feeds", "Custom Feed URL"]
        )

        if feed_option == "Popular ML/AI Feeds":
            from src.agents.rss_scraper import RSSScraper
            scraper = RSSScraper()
            feeds = scraper.list_feeds()

            selected_feeds = st.multiselect(
                "Select Feeds",
                list(feeds.keys()),
                default=["openai", "huggingface"]
            )
            limit = st.slider("Articles per feed", 3, 20, 5, key="rss_limit")

            if st.button("🔎 Fetch RSS Articles"):
                with st.spinner("Fetching from RSS feeds..."):
                    try:
                        all_articles = []
                        for feed_name in selected_feeds:
                            articles = scraper.scrape(feed_name=feed_name, limit=limit)
                            all_articles.extend(articles)
                        _process_scraped_articles(all_articles)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            feed_url = st.text_input("Feed URL", key="rss_url")
            limit = st.slider("Number of articles", 5, 30, 10, key="rss_custom_limit")

            if st.button("🔎 Fetch Custom Feed"):
                if feed_url:
                    with st.spinner("Fetching RSS feed..."):
                        try:
                            from src.agents.rss_scraper import RSSScraper
                            scraper = RSSScraper()
                            articles = scraper.scrape(feed_url=feed_url, limit=limit)
                            _process_scraped_articles(articles)
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please enter a feed URL")


def _process_scraped_articles(articles):
    """Process and save scraped articles."""
    if not articles:
        st.warning("No articles found")
        return

    db = DatabaseManager()
    vector_store = VectorStore()
    knowledge_engine = KnowledgeEngine(vector_store)

    added = 0
    skipped = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, article in enumerate(articles):
        progress_bar.progress((i + 1) / len(articles))
        status_text.text(f"Processing: {article.title[:50]}...")

        # Check if already exists
        existing = db.get_article_by_url(article.url)
        if existing:
            skipped += 1
            continue

        try:
            # Add to database
            db_article = db.add_article(article)
            article.id = db_article.id

            # Add to vector store
            vector_store.add_article(article)

            # Calculate scores
            score = knowledge_engine.calculate_growth_score(article)
            reason = knowledge_engine._generate_recommendation_reason(article, score)
            db.update_article_scores(
                article_id=article.id,
                novelty_score=score.novelty,
                growth_potential=score.total,
                recommendation_reason=reason
            )
            added += 1
        except Exception as e:
            st.warning(f"Failed to add article: {e}")

    progress_bar.empty()
    status_text.empty()

    st.success(f"✅ Added {added} new articles, skipped {skipped} duplicates")


def render_knowledge_graph_tab(knowledge_graph: KnowledgeGraph):
    """Render the knowledge graph visualization tab."""
    st.subheader("🕸️ Knowledge Graph")
    st.markdown("Visualize connections between articles and concepts")

    col1, col2 = st.columns([0.7, 0.3])

    with col2:
        include_unread = st.checkbox("Include unread articles", value=True)
        min_connections = st.slider("Min concept connections", 1, 5, 2)

        if st.button("🔄 Refresh Graph"):
            st.rerun()

        # Graph stats
        stats = knowledge_graph.get_graph_stats()
        st.markdown("### Graph Statistics")
        st.metric("Total Articles", stats.get("total_articles", 0))
        st.metric("Total Concepts", stats.get("total_concepts", 0))
        st.metric("Learned Articles", stats.get("learned_articles", 0))

    with col1:
        try:
            # Build and display graph
            net = knowledge_graph.get_pyvis_graph(include_unread=include_unread)

            if net:
                # Save to temp file and display
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                    net.save_graph(f.name)
                    with open(f.name, 'r') as html_file:
                        html_content = html_file.read()
                    components.html(html_content, height=600)
            else:
                st.info("Install pyvis for graph visualization: `pip install pyvis`")

        except ImportError:
            st.info("Install pyvis for graph visualization: `pip install pyvis`")
        except Exception as e:
            st.error(f"Error rendering graph: {e}")

    # Concept clusters
    st.markdown("### 🔗 Concept Clusters")
    clusters = knowledge_graph.get_concept_clusters()
    if clusters:
        for concept, articles in list(clusters.items())[:10]:
            with st.expander(f"{concept} ({len(articles)} articles)"):
                for article in articles[:5]:
                    st.markdown(f"• {article}")
    else:
        st.info("No concept clusters yet. Add more articles to see connections.")


def render_email_digest_tab():
    """Render the email digest configuration tab."""
    st.subheader("📧 Weekly Digest Email")
    st.markdown("Configure automatic weekly email digests with your top recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Configuration")
        recipient = st.text_input("Recipient Email", key="digest_recipient")
        sender = st.text_input("Sender Gmail", key="digest_sender",
                              help="Your Gmail address for sending")

        st.markdown("---")
        st.markdown("### Gmail Setup")
        st.markdown("""
        **Option 1: App Password (Recommended)**
        1. Enable 2-Factor Authentication on your Google Account
        2. Go to [App Passwords](https://myaccount.google.com/apppasswords)
        3. Generate a new app password for 'Mail'
        4. Set the `GMAIL_APP_PASSWORD` environment variable

        **Option 2: Gmail API (Advanced)**
        1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
        2. Enable Gmail API
        3. Create OAuth credentials
        4. Download `credentials.json`
        """)

    with col2:
        st.markdown("### Preview & Send")

        # Generate preview
        if st.button("👁️ Preview Digest"):
            with st.spinner("Generating preview..."):
                digest = EmailDigest()
                digest_data = digest.generate_digest(days=7, max_articles=10)

                st.markdown("### 📊 Digest Preview")
                st.metric("Recommendations", len(digest_data["recommendations"]))
                st.metric("New Concepts", len(digest_data["new_concepts"]))

                if digest_data["recommendations"]:
                    st.markdown("**Top Articles:**")
                    for rec in digest_data["recommendations"][:5]:
                        st.markdown(f"• {rec.article.title[:60]}...")

        st.markdown("---")

        # Send test email
        if st.button("📤 Send Test Digest"):
            if recipient and sender:
                with st.spinner("Sending email..."):
                    try:
                        config = DigestConfig(
                            recipient_email=recipient,
                            sender_email=sender,
                            max_articles=10
                        )
                        digest = EmailDigest()
                        success = digest.send_digest(config)

                        if success:
                            st.success("✅ Digest email sent!")
                        else:
                            st.error("Failed to send email. Check your Gmail credentials.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter both recipient and sender email addresses")


def main():
    """Main application entry point."""
    # Initialize services
    vector_store, knowledge_engine, categorizer, knowledge_graph = get_services()
    db = DatabaseManager()

    # Render sidebar
    summary = render_sidebar(knowledge_engine)

    # Main content area
    st.title("📚 Article Recommendations")
    st.markdown("*Curated for your optimal learning growth*")

    # View selection tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎯 For You",
        "🔧 Demos",
        "💡 Ideas",
        "🛠️ Tools",
        "📚 Tutorials",
        "✅ My Library",
        "🔍 Scraper",
        "🕸️ Knowledge Graph",
        "📧 Email Digest"
    ])

    with tab1:
        st.subheader("Recommended for Maximum Growth")
        st.markdown("These articles are selected to expand your knowledge while building on what you already know.")

        recommendations = knowledge_engine.get_recommendations(n=10)

        if not recommendations:
            st.info("No new recommendations yet. Add some articles to get started!")
            st.markdown("""
            ### Getting Started
            1. Go to the **🔍 Scraper** tab to add articles from ArXiv, HuggingFace, YouTube, or RSS feeds
            2. Rate articles with 👍 (like) or 👎 (skip) to train the system
            3. View your **🕸️ Knowledge Graph** to see connections between topics
            4. Set up **📧 Email Digest** for weekly recommendations
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

    with tab7:
        render_scraper_tab()

    with tab8:
        render_knowledge_graph_tab(knowledge_graph)

    with tab9:
        render_email_digest_tab()


if __name__ == "__main__":
    main()
