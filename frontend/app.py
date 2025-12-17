"""
Knowledge Curator - Streamlit Frontend

A reader-friendly interface for exploring and learning from curated articles.
Features:
- Research-backed article recommendations
- Like/dislike buttons to train the model
- Category-based sections with maturity assessment
- Knowledge dimensions visualization
- Settings for learning preferences
- Scraper interface with YouTube channels support
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import DatabaseManager, init_db
from src.models.article import Article, ArticleCategory
from src.services.vector_store import VectorStore
from src.services.knowledge_engine import KnowledgeEngine
from src.services.categorizer import ArticleCategorizer
from src.services.knowledge_graph import KnowledgeGraph
from src.services.email_digest import EmailDigest, DigestConfig
from src.services.user_profile import ProfileManager, LearningTopic, SkillLevel, LearningStyle
from src.services.research_growth_engine import ResearchBackedGrowthEngine
from src.services.dimension_visualizer import DimensionVisualizer


# Page configuration
st.set_page_config(
    page_title="Knowledge Curator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .growth-score {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .skill-badge {
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .novice { background-color: #ffc107; color: #333; }
    .advanced_beginner { background-color: #17a2b8; color: white; }
    .competent { background-color: #28a745; color: white; }
    .proficient { background-color: #6f42c1; color: white; }
    .expert { background-color: #dc3545; color: white; }
    .dimension-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .trend-up { color: #28a745; }
    .trend-down { color: #dc3545; }
    .trend-stable { color: #6c757d; }
    .focus-needed { border-left: 4px solid #ffc107; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Initialize and cache services."""
    init_db()
    vector_store = VectorStore()
    profile_manager = ProfileManager()
    growth_engine = ResearchBackedGrowthEngine(vector_store, profile_manager)
    dimension_visualizer = DimensionVisualizer(profile_manager)
    knowledge_graph = KnowledgeGraph()
    return vector_store, profile_manager, growth_engine, dimension_visualizer, knowledge_graph


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


def render_article_card(article: Article, growth_engine: ResearchBackedGrowthEngine, show_actions: bool = True):
    """Render a single article card."""
    category_emoji = get_category_emoji(article.category)

    with st.container():
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

        with col1:
            st.markdown(f"### {category_emoji} [{article.title}]({article.url})")

        with col2:
            if article.growth_potential > 0:
                st.metric("Growth", f"{article.growth_potential:.0%}")

        with col3:
            if article.novelty_score > 0:
                st.metric("Novelty", f"{article.novelty_score:.0%}")

        source_text = f" • {article.source}" if article.source else ""
        st.caption(f"{article.category.value.upper()}{source_text}")

        if article.summary:
            st.markdown(article.summary[:500] + "..." if len(article.summary) > 500 else article.summary)

        if article.key_insights:
            with st.expander("Key Insights"):
                for insight in article.key_insights[:3]:
                    st.markdown(f"• {insight}")

        if article.recommendation_reason:
            st.markdown(f"*💭 {article.recommendation_reason}*")

        if article.tags:
            st.markdown(" ".join([f"`{tag}`" for tag in article.tags[:5]]))

        if show_actions and article.is_liked is None:
            col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
            with col1:
                if st.button("👍 Like", key=f"like_{article.id}"):
                    growth_engine.process_rating(article.id, True)
                    st.success("Added to knowledge base!")
                    st.rerun()
            with col2:
                if st.button("👎 Skip", key=f"skip_{article.id}"):
                    growth_engine.process_rating(article.id, False)
                    st.info("Skipped")
                    st.rerun()
        elif article.is_liked is not None:
            status = "✅ Learned" if article.is_liked else "⏭️ Skipped"
            st.caption(status)

        st.divider()


def render_sidebar(growth_engine: ResearchBackedGrowthEngine, profile_manager: ProfileManager):
    """Render the sidebar with stats and maturity info."""
    st.sidebar.title("🧠 Knowledge Curator")

    # Get learning insights
    insights = growth_engine.get_learning_insights()

    st.sidebar.divider()

    # Skill Level
    level = insights["overall_level"]
    level_colors = {
        "novice": "🌱", "advanced_beginner": "🌿",
        "competent": "🌳", "proficient": "🌲", "expert": "🏔️"
    }
    st.sidebar.markdown(f"### {level_colors.get(level, '📊')} Level: {level.replace('_', ' ').title()}")

    # Stats
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Articles Read", profile_manager.profile.total_articles_read)
    with col2:
        st.metric("Streak", f"{insights['streak_days']}d 🔥" if insights['streak_days'] > 0 else "0d")

    # Weekly progress
    target = insights["target_per_week"]
    actual = insights["articles_this_week"]
    progress = min(1.0, actual / max(target, 1))
    st.sidebar.progress(progress, text=f"This week: {actual}/{target} articles")

    # Dimension summary
    if insights["dimension_scores"]:
        st.sidebar.subheader("📊 Knowledge Balance")
        for dim, score in sorted(insights["dimension_scores"].items(), key=lambda x: -x[1])[:4]:
            st.sidebar.progress(score / 100, text=f"{dim.replace('_', ' ').title()}: {score:.0f}%")

    # Learning recommendations
    recs = insights["recommendations"]
    if recs.get("next_steps"):
        st.sidebar.divider()
        st.sidebar.subheader("💡 Next Step")
        st.sidebar.info(recs["next_steps"][0])

    return insights


def render_dimensions_tab(dimension_visualizer: DimensionVisualizer):
    """Render the knowledge dimensions visualization tab."""
    st.subheader("📊 Knowledge Dimensions")
    st.markdown("Your learning progress across 6 key dimensions of ML/AI")

    chart_data = dimension_visualizer.get_streamlit_chart_data()

    # Radar chart
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        # Create radar chart with Plotly
        radar_data = chart_data["radar"]
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=radar_data["r"] + [radar_data["r"][0]],  # Close the shape
            theta=radar_data["theta"] + [radar_data["theta"][0]],
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            name='Your Knowledge'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Balance score
        balance = chart_data["balance_score"]
        st.metric("Learning Balance", f"{balance:.0f}/100")

        if balance >= 70:
            st.success("Well-balanced learning across dimensions!")
        elif balance >= 40:
            st.warning("Some dimensions need more attention")
        else:
            st.error("Consider diversifying your learning")

        # Strengths and gaps
        strengths, gaps = chart_data["strengths_gaps"]
        if strengths:
            st.markdown("**💪 Strengths:**")
            for s in strengths[:3]:
                st.markdown(f"• {s}")
        if gaps:
            st.markdown("**🎯 Focus Areas:**")
            for g in gaps[:3]:
                st.markdown(f"• {g}")

    # Progress bars
    st.subheader("📈 Dimension Details")
    progress_data = dimension_visualizer.get_progress_data()

    for dim in progress_data:
        col1, col2, col3 = st.columns([0.5, 0.3, 0.2])

        with col1:
            label = f"{dim['name']} {dim['trend_icon']}"
            st.progress(dim["score"] / 100, text=label)

        with col2:
            st.caption(f"{dim['articles']} articles • {dim['recent']} this week")

        with col3:
            if dim["needs_focus"]:
                st.warning("Focus", icon="🎯")

    # Recommendations
    st.subheader("🎯 Focus Recommendations")
    recommendations = chart_data["recommendations"]
    if recommendations:
        for rec in recommendations[:3]:
            with st.expander(f"📌 {rec['dimension']} ({rec['score']:.0f}/100)"):
                st.markdown(f"**Why:** {rec['reason']}")
                st.markdown(f"**Action:** {rec['action']}")
                st.markdown(f"**Keywords:** {', '.join(rec['suggested_keywords'])}")
    else:
        st.success("Great job! You're progressing well across all dimensions.")


def render_settings_tab(profile_manager: ProfileManager):
    """Render the settings/configuration tab."""
    st.subheader("⚙️ Learning Settings")

    profile = profile_manager.profile

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📚 Learning Topics")
        st.markdown("Configure topics you want to learn")

        # Add new topic
        with st.expander("➕ Add New Topic"):
            new_topic = st.text_input("Topic Name", key="new_topic_name")
            new_priority = st.slider("Priority", 1, 5, 3, key="new_topic_priority")
            new_keywords = st.text_input("Keywords (comma-separated)", key="new_topic_keywords")
            new_target = st.selectbox(
                "Target Level",
                options=[l.value for l in SkillLevel],
                index=2,
                key="new_topic_target"
            )

            if st.button("Add Topic"):
                if new_topic:
                    topic = LearningTopic(
                        name=new_topic,
                        priority=new_priority,
                        keywords=[k.strip() for k in new_keywords.split(",") if k.strip()],
                        target_level=SkillLevel(new_target)
                    )
                    profile_manager.add_topic(topic)
                    st.success(f"Added topic: {new_topic}")
                    st.rerun()

        # List existing topics
        if profile.topics:
            for i, topic in enumerate(profile.topics):
                with st.container():
                    col_a, col_b, col_c = st.columns([0.5, 0.3, 0.2])
                    with col_a:
                        st.markdown(f"**{topic.name}**")
                        st.caption(f"Priority: {'⭐' * topic.priority}")
                    with col_b:
                        st.caption(f"{topic.current_level.value} → {topic.target_level.value}")
                        st.caption(f"{topic.articles_read} articles")
                    with col_c:
                        if st.button("🗑️", key=f"del_topic_{i}"):
                            profile_manager.remove_topic(topic.name)
                            st.rerun()
        else:
            st.info("No topics configured. Add topics to get personalized recommendations.")

    with col2:
        st.markdown("### 🎮 Learning Preferences")

        # Learning style
        style = st.selectbox(
            "Learning Style",
            options=[s.value for s in LearningStyle],
            index=[s.value for s in LearningStyle].index(profile.learning_style.value),
            key="pref_style"
        )

        # Daily time budget
        time_budget = st.slider(
            "Daily Time Budget (minutes)",
            5, 120, profile.daily_time_budget,
            key="pref_time"
        )

        # Weekly target
        weekly_target = st.slider(
            "Articles per Week Target",
            1, 50, profile.articles_per_week_target,
            key="pref_weekly"
        )

        # Difficulty preference
        difficulty = st.selectbox(
            "Difficulty Preference",
            options=["easy", "medium", "hard", "adaptive"],
            index=["easy", "medium", "hard", "adaptive"].index(profile.difficulty_preference),
            key="pref_diff"
        )

        if st.button("Save Preferences"):
            profile.learning_style = LearningStyle(style)
            profile.daily_time_budget = time_budget
            profile.articles_per_week_target = weekly_target
            profile.difficulty_preference = difficulty
            profile_manager.save_profile()
            st.success("Preferences saved!")

        st.markdown("### 📺 YouTube Settings")

        # YouTube channels
        current_channels = "\n".join(profile.youtube_channels)
        channels_input = st.text_area(
            "Subscribed Channels (one per line, use @handle)",
            value=current_channels,
            height=100,
            key="yt_channels"
        )

        # YouTube search terms
        current_terms = "\n".join(profile.youtube_search_terms)
        terms_input = st.text_area(
            "Search Terms (one per line)",
            value=current_terms,
            height=100,
            key="yt_terms"
        )

        if st.button("Save YouTube Settings"):
            profile.youtube_channels = [c.strip() for c in channels_input.split("\n") if c.strip()]
            profile.youtube_search_terms = [t.strip() for t in terms_input.split("\n") if t.strip()]
            profile_manager.save_profile()
            st.success("YouTube settings saved!")

        st.markdown("### 📡 RSS Feeds")

        current_feeds = "\n".join(profile.rss_feeds)
        feeds_input = st.text_area(
            "Custom RSS Feeds (one URL per line)",
            value=current_feeds,
            height=100,
            key="rss_feeds"
        )

        if st.button("Save RSS Settings"):
            profile.rss_feeds = [f.strip() for f in feeds_input.split("\n") if f.strip()]
            profile_manager.save_profile()
            st.success("RSS settings saved!")


def render_scraper_tab(profile_manager: ProfileManager):
    """Render the enhanced scraper interface."""
    st.subheader("🔍 Content Scraper")

    source = st.selectbox(
        "Select Source",
        ["ArXiv", "HuggingFace", "YouTube", "RSS Feed"]
    )

    if source == "ArXiv":
        st.markdown("#### ArXiv Research Papers")
        query = st.text_input("Search Query", key="arxiv_query")
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
        content_type = st.selectbox("Content Type", ["papers", "models", "datasets", "spaces"])
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

        yt_mode = st.radio(
            "Mode",
            ["Video URLs", "Channel", "Search", "ML Channels"],
            horizontal=True
        )

        if yt_mode == "Video URLs":
            urls = st.text_area("Video URLs (one per line)", height=100)
            if st.button("🔎 Fetch Videos"):
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

        elif yt_mode == "Channel":
            from src.agents.youtube_scraper import YouTubeScraper
            scraper = YouTubeScraper()
            available = scraper.get_available_channels()

            col1, col2 = st.columns(2)
            with col1:
                channel_select = st.selectbox(
                    "Select ML/AI Channel",
                    options=["Custom"] + list(available.keys())
                )
            with col2:
                if channel_select == "Custom":
                    channel_handle = st.text_input("Channel Handle (e.g., @AndrejKarpathy)")
                else:
                    channel_handle = channel_select

            limit = st.slider("Videos to fetch", 3, 20, 5, key="yt_channel_limit")

            if st.button("🔎 Fetch from Channel"):
                if channel_handle:
                    with st.spinner(f"Fetching from {channel_handle}..."):
                        try:
                            articles = scraper.scrape(channel_handle=channel_handle, limit=limit)
                            _process_scraped_articles(articles)
                        except Exception as e:
                            st.error(f"Error: {e}")

        elif yt_mode == "Search":
            query = st.text_input("Search Query", key="yt_search")
            limit = st.slider("Results", 5, 20, 10, key="yt_search_limit")

            if st.button("🔎 Search YouTube"):
                if query:
                    with st.spinner("Searching YouTube..."):
                        try:
                            from src.agents.youtube_scraper import YouTubeScraper
                            scraper = YouTubeScraper()
                            articles = scraper.search(query, limit)
                            _process_scraped_articles(articles)
                        except Exception as e:
                            st.error(f"Error: {e}")

        elif yt_mode == "ML Channels":
            limit_per = st.slider("Videos per channel", 2, 10, 3)

            if st.button("🔎 Fetch from All ML Channels"):
                with st.spinner("Fetching from ML channels..."):
                    try:
                        from src.agents.youtube_scraper import YouTubeScraper
                        scraper = YouTubeScraper()
                        articles = scraper.scrape_ml_channels(limit_per_channel=limit_per)
                        _process_scraped_articles(articles)
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif source == "RSS Feed":
        st.markdown("#### RSS/Atom Feeds")

        feed_option = st.radio("Feed Selection", ["Popular ML/AI Feeds", "Custom Feed", "My Feeds"])

        if feed_option == "Popular ML/AI Feeds":
            from src.agents.rss_scraper import RSSScraper
            scraper = RSSScraper()
            feeds = scraper.list_feeds()

            selected = st.multiselect("Select Feeds", list(feeds.keys()), default=["openai", "huggingface"])
            limit = st.slider("Articles per feed", 3, 20, 5, key="rss_limit")

            if st.button("🔎 Fetch RSS"):
                with st.spinner("Fetching RSS..."):
                    try:
                        all_articles = []
                        for feed_name in selected:
                            articles = scraper.scrape(feed_name=feed_name, limit=limit)
                            all_articles.extend(articles)
                        _process_scraped_articles(all_articles)
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif feed_option == "Custom Feed":
            feed_url = st.text_input("Feed URL")
            limit = st.slider("Articles", 5, 30, 10, key="rss_custom")

            if st.button("🔎 Fetch Custom Feed"):
                if feed_url:
                    with st.spinner("Fetching..."):
                        try:
                            from src.agents.rss_scraper import RSSScraper
                            scraper = RSSScraper()
                            articles = scraper.scrape(feed_url=feed_url, limit=limit)
                            _process_scraped_articles(articles)
                        except Exception as e:
                            st.error(f"Error: {e}")

        elif feed_option == "My Feeds":
            if profile_manager.profile.rss_feeds:
                st.info(f"Fetching from {len(profile_manager.profile.rss_feeds)} saved feeds")
                limit = st.slider("Articles per feed", 3, 10, 5, key="rss_my")

                if st.button("🔎 Fetch My Feeds"):
                    with st.spinner("Fetching..."):
                        try:
                            from src.agents.rss_scraper import RSSScraper
                            scraper = RSSScraper()
                            all_articles = []
                            for url in profile_manager.profile.rss_feeds:
                                articles = scraper.scrape(feed_url=url, limit=limit)
                                all_articles.extend(articles)
                            _process_scraped_articles(all_articles)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.warning("No saved feeds. Configure them in Settings.")


def _process_scraped_articles(articles):
    """Process and save scraped articles."""
    if not articles:
        st.warning("No articles found")
        return

    db = DatabaseManager()
    vector_store = VectorStore()
    profile_manager = ProfileManager()
    growth_engine = ResearchBackedGrowthEngine(vector_store, profile_manager)

    added = 0
    skipped = 0

    progress = st.progress(0)
    status = st.empty()

    for i, article in enumerate(articles):
        progress.progress((i + 1) / len(articles))
        status.text(f"Processing: {article.title[:50]}...")

        existing = db.get_article_by_url(article.url)
        if existing:
            skipped += 1
            continue

        try:
            db_article = db.add_article(article)
            article.id = db_article.id
            vector_store.add_article(article)

            metrics = growth_engine.calculate_growth_metrics(article)
            reason = " ".join(metrics.reasoning[:2]) if metrics.reasoning else "Good learning opportunity"

            db.update_article_scores(
                article_id=article.id,
                novelty_score=metrics.zpd_score,
                growth_potential=metrics.total_score,
                recommendation_reason=reason
            )
            added += 1
        except Exception as e:
            st.warning(f"Failed: {e}")

    progress.empty()
    status.empty()
    st.success(f"✅ Added {added} articles, skipped {skipped} duplicates")


def render_knowledge_graph_tab(knowledge_graph: KnowledgeGraph):
    """Render the knowledge graph visualization."""
    st.subheader("🕸️ Knowledge Graph")

    col1, col2 = st.columns([0.7, 0.3])

    with col2:
        include_unread = st.checkbox("Include unread", value=True)
        if st.button("🔄 Refresh"):
            st.rerun()

        stats = knowledge_graph.get_graph_stats()
        st.metric("Articles", stats.get("total_articles", 0))
        st.metric("Concepts", stats.get("total_concepts", 0))
        st.metric("Learned", stats.get("learned_articles", 0))

    with col1:
        try:
            net = knowledge_graph.get_pyvis_graph(include_unread=include_unread)
            if net:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                    net.save_graph(f.name)
                    with open(f.name, 'r') as html_file:
                        components.html(html_file.read(), height=500)
            else:
                st.info("Install pyvis: `pip install pyvis`")
        except Exception as e:
            st.error(f"Error: {e}")


def render_email_digest_tab():
    """Render email digest configuration."""
    st.subheader("📧 Weekly Digest")

    col1, col2 = st.columns(2)

    with col1:
        recipient = st.text_input("Recipient Email")
        sender = st.text_input("Sender Gmail")

        st.markdown("---")
        st.markdown("""
        **Setup:** Create a Gmail App Password at
        [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
        """)

    with col2:
        if st.button("👁️ Preview"):
            digest = EmailDigest()
            data = digest.generate_digest(days=7, max_articles=10)
            st.metric("Recommendations", len(data["recommendations"]))
            if data["recommendations"]:
                for rec in data["recommendations"][:3]:
                    st.markdown(f"• {rec.article.title[:50]}...")

        if st.button("📤 Send Test"):
            if recipient and sender:
                config = DigestConfig(recipient_email=recipient, sender_email=sender)
                digest = EmailDigest()
                if digest.send_digest(config):
                    st.success("Sent!")
                else:
                    st.error("Failed - check credentials")


def main():
    """Main application."""
    vector_store, profile_manager, growth_engine, dimension_visualizer, knowledge_graph = get_services()
    db = DatabaseManager()

    # Sidebar
    insights = render_sidebar(growth_engine, profile_manager)

    # Title
    st.title("📚 Knowledge Curator")
    st.markdown("*Research-backed learning recommendations*")

    # Tabs
    tabs = st.tabs([
        "🎯 For You",
        "📊 Dimensions",
        "🔧 Demos",
        "💡 Ideas",
        "📚 Tutorials",
        "✅ Library",
        "🔍 Scraper",
        "🕸️ Graph",
        "⚙️ Settings",
        "📧 Digest"
    ])

    with tabs[0]:
        st.subheader("Recommended for Maximum Growth")

        # Show maturity info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Your Level:** {insights['overall_level'].replace('_', ' ').title()}")
        with col2:
            if insights["recommendations"].get("weak_areas"):
                st.warning(f"**Focus:** {insights['recommendations']['weak_areas'][0][:50]}...")
        with col3:
            balance = dimension_visualizer.get_learning_balance_score()
            st.metric("Balance Score", f"{balance:.0f}/100")

        recommendations = growth_engine.get_recommendations(n=10)

        if not recommendations:
            st.info("Add articles via the Scraper tab to get recommendations!")
        else:
            for rec in recommendations:
                render_article_card(rec.article, growth_engine)

    with tabs[1]:
        render_dimensions_tab(dimension_visualizer)

    with tabs[2]:
        st.subheader("🔧 Demos & Implementations")
        articles = db.get_all_articles(category=ArticleCategory.DEMO)
        for a in articles:
            render_article_card(db.db_article_to_pydantic(a), growth_engine)
        if not articles:
            st.info("No demos yet.")

    with tabs[3]:
        st.subheader("💡 Ideas & Research")
        articles = db.get_all_articles(category=ArticleCategory.IDEA)
        for a in articles:
            render_article_card(db.db_article_to_pydantic(a), growth_engine)
        if not articles:
            st.info("No research papers yet.")

    with tabs[4]:
        st.subheader("📚 Tutorials")
        articles = db.get_all_articles(category=ArticleCategory.TUTORIAL)
        for a in articles:
            render_article_card(db.db_article_to_pydantic(a), growth_engine)
        if not articles:
            st.info("No tutorials yet.")

    with tabs[5]:
        st.subheader("✅ Your Library")
        liked = db.get_liked_articles()
        for a in liked:
            render_article_card(db.db_article_to_pydantic(a), growth_engine, show_actions=False)
        if not liked:
            st.info("Like articles to add them here!")

    with tabs[6]:
        render_scraper_tab(profile_manager)

    with tabs[7]:
        render_knowledge_graph_tab(knowledge_graph)

    with tabs[8]:
        render_settings_tab(profile_manager)

    with tabs[9]:
        render_email_digest_tab()


if __name__ == "__main__":
    main()
