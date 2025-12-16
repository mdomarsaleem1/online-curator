# Online Knowledge Curator

A self-updating learning assistant that tracks papers, courses, and industry trends. It uses a vector database to recommend articles based on **semantic growth** rather than simple similarity - maximizing your learning efficiency by suggesting content that expands your knowledge rather than duplicating it.

## Features

- **Vector Knowledge Database**: Articles indexed by high-level summaries using ChromaDB
- **Knowledge Model**: Tracks what you've learned based on liked articles
- **Semantic Growth Recommendations**: Suggests articles that expand knowledge, not duplicate it
- **Category-based Organization**: Demos, Ideas, Tools, Tutorials, News
- **Clean Frontend**: Streamlit-based reader with like/dislike feedback

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Knowledge Curator                            │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)                                           │
│  ├── Article Recommendations                                    │
│  ├── Category Sections (Demos/Ideas/Tools/Tutorials)            │
│  └── Like/Dislike Feedback                                      │
├─────────────────────────────────────────────────────────────────┤
│  Knowledge Engine                                               │
│  ├── Growth Score Calculator                                    │
│  │   ├── Novelty (40%) - How new is this?                      │
│  │   ├── Relevance (25%) - Matches interests?                  │
│  │   ├── Foundation (20%) - Have prerequisites?                │
│  │   └── Gap-filling (15%) - Fills knowledge gaps?             │
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
# Edit .env with your OpenAI API key (optional, for enhanced features)
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

1. **Browse Recommendations**: The "For You" tab shows articles ranked by growth potential
2. **Explore Categories**: Filter by Demos, Ideas, Tools, Tutorials, or News
3. **Rate Articles**:
   - 👍 **Like** = "I read this and found it valuable" → Added to your knowledge base
   - 👎 **Skip** = "Not relevant or not interested" → Excluded from future recommendations
4. **Track Progress**: See your knowledge summary in the sidebar

## Project Structure

```
online-curator/
├── src/
│   ├── models/
│   │   ├── article.py       # Article and preference models
│   │   └── database.py      # SQLAlchemy database manager
│   ├── services/
│   │   ├── vector_store.py  # ChromaDB vector operations
│   │   ├── knowledge_engine.py  # Growth algorithm
│   │   └── categorizer.py   # Article categorization
│   └── agents/              # (Future) Scraper agents
├── frontend/
│   └── app.py               # Streamlit application
├── data/
│   └── sample_articles.py   # Sample data loader
├── requirements.txt
└── README.md
```

## Future Enhancements

- [ ] Scraper Agent for ArXiv, HuggingFace, YouTube, RSS
- [ ] Summarizer Agent with structured output
- [ ] Knowledge graph visualization
- [ ] Weekly digest email/Slack integration
- [ ] API for programmatic access

## License

Apache License 2.0
