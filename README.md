# Online Knowledge Curator

A self-updating learning assistant that tracks papers, courses, and industry trends. It uses a vector database to recommend articles based on **semantic growth** rather than simple similarity - maximizing your learning efficiency by suggesting content that expands your knowledge rather than duplicating it.

## Features

- **Vector Knowledge Database**: Articles indexed by high-level summaries using ChromaDB
- **Knowledge Model**: Tracks what you've learned based on liked articles
- **Semantic Growth Recommendations**: Suggests articles that expand knowledge, not duplicate it
- **Category-based Organization**: Demos, Ideas, Tools, Tutorials, News
- **Multi-source Scrapers**: ArXiv, HuggingFace, YouTube, RSS feeds
- **Summarizer Agent**: Structured summaries with LLM support
- **Knowledge Graph Visualization**: Interactive graph showing article-concept connections
- **Weekly Email Digest**: Gmail integration for scheduled recommendations
- **Clean Frontend**: Streamlit-based reader with like/dislike feedback

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Knowledge Curator                            │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)                                           │
│  ├── Article Recommendations                                    │
│  ├── Category Sections (Demos/Ideas/Tools/Tutorials)            │
│  ├── Like/Dislike Feedback                                      │
│  ├── Content Scraper Interface                                  │
│  ├── Knowledge Graph Visualization                              │
│  └── Email Digest Configuration                                 │
├─────────────────────────────────────────────────────────────────┤
│  Scraper Agents                                                 │
│  ├── ArXiv - Research papers (cs.LG, cs.AI, cs.CL, etc.)       │
│  ├── HuggingFace - Models, datasets, spaces, daily papers      │
│  ├── YouTube - Video metadata and transcripts                   │
│  └── RSS - OpenAI, Anthropic, DeepMind, and custom feeds       │
├─────────────────────────────────────────────────────────────────┤
│  Knowledge Engine                                               │
│  ├── Growth Score Calculator                                    │
│  │   ├── Novelty (40%) - How new is this?                      │
│  │   ├── Relevance (25%) - Matches interests?                  │
│  │   ├── Foundation (20%) - Have prerequisites?                │
│  │   └── Gap-filling (15%) - Fills knowledge gaps?             │
│  ├── Summarizer Agent (OpenAI/Ollama/Rule-based)               │
│  └── Recommendation Generator                                   │
├─────────────────────────────────────────────────────────────────┤
│  Vector Store (ChromaDB)                                        │
│  ├── articles - All scraped articles                           │
│  ├── user_knowledge - Liked articles (learned)                 │
│  └── concepts - Extracted topics and concepts                  │
├─────────────────────────────────────────────────────────────────┤
│  Database (SQLite)                                              │
│  ├── Articles with metadata and scores                         │
│  └── User preferences and statistics                           │
└─────────────────────────────────────────────────────────────────┘
```

## The Growth Algorithm

Unlike simple similarity-based recommendations, our algorithm optimizes for **learning efficiency**:

1. **Novelty Score (40%)**: How different is this from what you know? We prefer articles in the "Goldilocks zone" (30-70% new) - not too familiar (boring), not too foreign (overwhelming).

2. **Relevance Score (25%)**: Does this match your interests based on previously liked articles?

3. **Foundation Score (20%)**: Do you have the prerequisite knowledge to understand this? We check if concepts from your liked articles support understanding the new article.

4. **Gap-Filling Score (15%)**: Does this article introduce concepts that connect your existing knowledge, filling gaps in your understanding?

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (optional, for enhanced features)
```

### Load Sample Data

```bash
python data/sample_articles.py
```

### Run the App

```bash
streamlit run frontend/app.py
```

Visit `http://localhost:8501` to start curating your knowledge!

## Usage

### Main Features

1. **Browse Recommendations**: The "For You" tab shows articles ranked by growth potential
2. **Explore Categories**: Filter by Demos, Ideas, Tools, Tutorials, or News
3. **Rate Articles**:
   - 👍 **Like** = "I read this and found it valuable" → Added to your knowledge base
   - 👎 **Skip** = "Not relevant or not interested" → Excluded from future recommendations
4. **Track Progress**: See your knowledge summary in the sidebar

### Scraper Tab

Add new content from multiple sources:

- **ArXiv**: Search for research papers by query and category
- **HuggingFace**: Fetch models, datasets, spaces, or daily papers
- **YouTube**: Add videos by URL (transcripts extracted automatically)
- **RSS Feeds**: Choose from popular ML/AI blogs or add custom feeds

### Knowledge Graph

Visualize your knowledge as an interactive graph:
- Articles shown as colored nodes (by category)
- Concepts shown as yellow diamonds
- Connections show which articles cover which concepts
- Learned articles highlighted in green

### Email Digest

Set up weekly email summaries:
1. Configure your Gmail credentials (App Password or OAuth)
2. Preview your digest content
3. Send test emails or schedule weekly delivery

## Scrapers

### ArXiv Scraper

```python
from src.agents import ArxivScraper

scraper = ArxivScraper()
articles = scraper.scrape(
    query="transformer attention",
    limit=10,
    categories=["cs.LG", "cs.AI", "cs.CL"]
)
```

### HuggingFace Scraper

```python
from src.agents import HuggingFaceScraper

scraper = HuggingFaceScraper()

# Fetch daily papers
papers = scraper.scrape(content_type="papers", limit=10)

# Fetch trending models
models = scraper.scrape(content_type="models", limit=10)

# Fetch spaces (demos)
spaces = scraper.scrape(content_type="spaces", limit=10)
```

### YouTube Scraper

```python
from src.agents import YouTubeScraper

scraper = YouTubeScraper()
articles = scraper.scrape(video_urls=[
    "https://www.youtube.com/watch?v=kCc8FmEb1nY",
    "https://youtu.be/example123"
])
```

### RSS Scraper

```python
from src.agents import RSSScraper

scraper = RSSScraper()

# List available feeds
print(scraper.list_feeds())

# Fetch from specific feed
articles = scraper.scrape(feed_name="openai", limit=10)

# Fetch from custom feed
articles = scraper.scrape(feed_url="https://example.com/rss", limit=10)
```

## Email Digest Setup

### Option 1: Gmail App Password (Recommended)

1. Enable 2-Factor Authentication on your Google Account
2. Go to [App Passwords](https://myaccount.google.com/apppasswords)
3. Generate a new app password for 'Mail'
4. Set environment variable: `GMAIL_APP_PASSWORD=your_app_password`

### Option 2: Gmail API (OAuth)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable Gmail API
3. Create OAuth credentials
4. Download `credentials.json` and set `GMAIL_CREDENTIALS_PATH`

## Project Structure

```
online-curator/
├── src/
│   ├── models/
│   │   ├── article.py           # Article and preference models
│   │   └── database.py          # SQLAlchemy database manager
│   ├── services/
│   │   ├── vector_store.py      # ChromaDB vector operations
│   │   ├── knowledge_engine.py  # Growth algorithm
│   │   ├── categorizer.py       # Article categorization
│   │   ├── knowledge_graph.py   # Graph visualization
│   │   └── email_digest.py      # Gmail digest service
│   └── agents/
│       ├── base.py              # Base scraper class
│       ├── arxiv_scraper.py     # ArXiv API scraper
│       ├── huggingface_scraper.py # HuggingFace Hub scraper
│       ├── youtube_scraper.py   # YouTube video scraper
│       ├── rss_scraper.py       # RSS/Atom feed scraper
│       └── summarizer.py        # LLM-based summarizer
├── frontend/
│   └── app.py                   # Streamlit application
├── data/
│   └── sample_articles.py       # Sample data loader
├── requirements.txt
└── README.md
```

## Environment Variables

```bash
# OpenAI API Key (for summarizer and enhanced embeddings)
OPENAI_API_KEY=your_key_here

# Gmail (for email digest)
GMAIL_APP_PASSWORD=your_app_password
# OR
GMAIL_CREDENTIALS_PATH=path/to/credentials.json

# Database
DATABASE_URL=sqlite:///./knowledge.db

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## License

Apache License 2.0
