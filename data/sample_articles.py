"""
Sample data loader for testing the Knowledge Curator.

Adds sample articles across different categories to demonstrate the system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.models.database import DatabaseManager, init_db
from src.models.article import Article, ArticleCategory
from src.services.vector_store import VectorStore
from src.services.knowledge_engine import KnowledgeEngine
from src.services.categorizer import ArticleCategorizer


SAMPLE_ARTICLES = [
    # DEMO articles
    Article(
        url="https://github.com/openai/whisper",
        title="Whisper: Robust Speech Recognition via Large-Scale Weak Supervision",
        source="GitHub",
        summary="OpenAI's Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual data. It enables transcription in multiple languages and translation to English.",
        key_insights=[
            "Trained on 680k hours of web audio data",
            "Supports 99 languages for transcription",
            "Zero-shot transfer across languages and domains",
            "Open source with simple Python API"
        ],
        category=ArticleCategory.DEMO,
        tags=["speech-recognition", "whisper", "openai", "asr", "transcription"],
        related_concepts=["Speech Recognition", "Transformers", "Multilingual AI", "Audio Processing"],
        prerequisite_concepts=["Python", "Machine Learning basics"]
    ),
    Article(
        url="https://github.com/AUTOMATIC1111/stable-diffusion-webui",
        title="Stable Diffusion Web UI - A Browser Interface for Image Generation",
        source="GitHub",
        summary="A feature-rich web interface for Stable Diffusion, allowing image generation, inpainting, outpainting, and various image manipulation techniques through an intuitive browser-based UI.",
        key_insights=[
            "Supports multiple SD checkpoints and LoRAs",
            "Built-in image editing and inpainting",
            "Extensible plugin architecture",
            "Active community with regular updates"
        ],
        category=ArticleCategory.DEMO,
        tags=["stable-diffusion", "image-generation", "web-ui", "ai-art"],
        related_concepts=["Diffusion Models", "Image Generation", "LoRA", "Inpainting"],
        prerequisite_concepts=["Stable Diffusion", "Python"]
    ),

    # IDEA articles
    Article(
        url="https://arxiv.org/abs/2303.08774",
        title="GPT-4 Technical Report",
        source="ArXiv",
        summary="Technical report introducing GPT-4, a large multimodal model capable of processing image and text inputs. Demonstrates human-level performance on various professional benchmarks.",
        key_insights=[
            "Multimodal: accepts both text and image inputs",
            "Passes bar exam in 90th percentile",
            "Improved safety through RLHF",
            "128k context window capability"
        ],
        category=ArticleCategory.IDEA,
        tags=["gpt-4", "llm", "multimodal", "openai", "foundation-model"],
        related_concepts=["Large Language Models", "Multimodal AI", "RLHF", "Transformers", "Scaling Laws"],
        prerequisite_concepts=["GPT-3", "Attention Mechanism", "Neural Networks"]
    ),
    Article(
        url="https://arxiv.org/abs/2305.10601",
        title="Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        source="ArXiv",
        summary="Introduces Tree of Thoughts (ToT), a framework enabling LLMs to perform deliberate decision making by considering multiple reasoning paths and self-evaluating choices.",
        key_insights=[
            "Generalizes chain-of-thought prompting",
            "Enables lookahead and backtracking",
            "Uses LLM to evaluate intermediate steps",
            "Significant improvements on complex reasoning tasks"
        ],
        category=ArticleCategory.IDEA,
        tags=["tree-of-thoughts", "reasoning", "prompting", "llm"],
        related_concepts=["Chain of Thought", "Reasoning", "Search Algorithms", "LLM Prompting"],
        prerequisite_concepts=["Chain of Thought", "Large Language Models"]
    ),
    Article(
        url="https://arxiv.org/abs/2307.09288",
        title="Llama 2: Open Foundation and Fine-Tuned Chat Models",
        source="ArXiv",
        summary="Meta's Llama 2 family of LLMs ranging from 7B to 70B parameters, optimized for dialogue and released with open weights for research and commercial use.",
        key_insights=[
            "Open weights for commercial use",
            "RLHF trained for helpfulness and safety",
            "Competitive with closed-source models",
            "Available in 7B, 13B, and 70B sizes"
        ],
        category=ArticleCategory.IDEA,
        tags=["llama", "meta", "open-source", "llm", "foundation-model"],
        related_concepts=["Open Source AI", "RLHF", "Fine-tuning", "Model Scaling"],
        prerequisite_concepts=["Transformers", "Pre-training"]
    ),

    # TOOL articles
    Article(
        url="https://huggingface.co/docs/transformers",
        title="Hugging Face Transformers - State-of-the-Art ML for Everyone",
        source="HuggingFace",
        summary="The most popular library for working with transformer models, providing thousands of pretrained models for NLP, vision, audio, and multimodal tasks with a simple API.",
        key_insights=[
            "Supports PyTorch, TensorFlow, and JAX",
            "100,000+ community models on the Hub",
            "Simple from_pretrained() API",
            "Built-in training and optimization tools"
        ],
        category=ArticleCategory.TOOL,
        tags=["transformers", "huggingface", "nlp", "library", "python"],
        related_concepts=["Transformers", "Pre-trained Models", "NLP", "Transfer Learning"],
        prerequisite_concepts=["Python", "PyTorch or TensorFlow"]
    ),
    Article(
        url="https://langchain.com/",
        title="LangChain: Building Applications with LLMs",
        source="LangChain",
        summary="A framework for developing applications powered by language models, enabling chaining of components like prompts, LLMs, memory, and tools into complex workflows.",
        key_insights=[
            "Modular components for LLM apps",
            "Built-in memory and retrieval",
            "Agent framework for autonomous tasks",
            "Integrates with 50+ LLM providers"
        ],
        category=ArticleCategory.TOOL,
        tags=["langchain", "llm", "framework", "agents", "rag"],
        related_concepts=["LLM Applications", "RAG", "Agents", "Prompt Engineering"],
        prerequisite_concepts=["Large Language Models", "Python"]
    ),
    Article(
        url="https://www.trychroma.com/",
        title="Chroma: The AI-Native Open-Source Embedding Database",
        source="Chroma",
        summary="An open-source vector database designed for AI applications, making it easy to store, search, and retrieve embeddings for semantic search and RAG applications.",
        key_insights=[
            "Simple Python/JavaScript API",
            "Built-in embedding functions",
            "Supports metadata filtering",
            "Persistent and in-memory modes"
        ],
        category=ArticleCategory.TOOL,
        tags=["chroma", "vector-database", "embeddings", "rag", "search"],
        related_concepts=["Vector Databases", "Embeddings", "Semantic Search", "RAG"],
        prerequisite_concepts=["Embeddings", "Python"]
    ),

    # TUTORIAL articles
    Article(
        url="https://www.deeplearning.ai/short-courses/",
        title="DeepLearning.AI Short Courses - Free AI Education",
        source="DeepLearning.AI",
        summary="Free short courses on cutting-edge AI topics including LLMs, prompt engineering, LangChain, and building AI applications, taught by industry experts.",
        key_insights=[
            "Free, hands-on courses",
            "Taught by Andrew Ng and partners",
            "Topics: LLMs, RAG, Agents, Fine-tuning",
            "Jupyter notebook-based learning"
        ],
        category=ArticleCategory.TUTORIAL,
        tags=["courses", "education", "llm", "free", "deeplearning"],
        related_concepts=["AI Education", "LLM Development", "Prompt Engineering"],
        prerequisite_concepts=["Python basics", "ML fundamentals"]
    ),
    Article(
        url="https://www.youtube.com/watch?v=kCc8FmEb1nY",
        title="Let's Build GPT: From Scratch, in Code",
        source="YouTube",
        summary="Andrej Karpathy's famous walkthrough of building a GPT model from scratch in Python, explaining the transformer architecture and training process step by step.",
        key_insights=[
            "Builds GPT from first principles",
            "Explains attention mechanism clearly",
            "Shows training on Shakespeare text",
            "2+ hours of detailed coding"
        ],
        category=ArticleCategory.TUTORIAL,
        tags=["gpt", "tutorial", "karpathy", "transformers", "from-scratch"],
        related_concepts=["Transformers", "Attention", "Language Modeling", "PyTorch"],
        prerequisite_concepts=["Python", "Basic neural networks", "PyTorch basics"]
    ),

    # NEWS articles
    Article(
        url="https://openai.com/index/gpt-4o/",
        title="Hello GPT-4o - OpenAI's Latest Multimodal Model",
        source="OpenAI Blog",
        summary="OpenAI announces GPT-4o, a new flagship model that can reason across audio, vision, and text in real time, with faster response times and improved capabilities.",
        key_insights=[
            "Native audio, vision, and text capabilities",
            "2x faster than GPT-4 Turbo",
            "Available free to all ChatGPT users",
            "Improved multilingual performance"
        ],
        category=ArticleCategory.NEWS,
        tags=["gpt-4o", "openai", "announcement", "multimodal"],
        related_concepts=["GPT-4", "Multimodal AI", "Real-time AI"],
        prerequisite_concepts=["GPT-4", "ChatGPT"]
    ),
    Article(
        url="https://www.anthropic.com/news/claude-3-5-sonnet",
        title="Claude 3.5 Sonnet - Anthropic's Most Capable Model",
        source="Anthropic",
        summary="Anthropic releases Claude 3.5 Sonnet, outperforming Claude 3 Opus on most benchmarks while being faster and more cost-effective, with strong coding and vision capabilities.",
        key_insights=[
            "Outperforms GPT-4o on several benchmarks",
            "2x speed of Claude 3 Opus",
            "Improved coding and agentic capabilities",
            "Strong vision understanding"
        ],
        category=ArticleCategory.NEWS,
        tags=["claude", "anthropic", "announcement", "llm"],
        related_concepts=["Claude", "Anthropic", "AI Safety", "LLM Benchmarks"],
        prerequisite_concepts=["Large Language Models"]
    ),
]


def load_sample_data():
    """Load sample articles into the database and vector store."""
    print("Initializing database...")
    init_db()

    db = DatabaseManager()
    vector_store = VectorStore()
    knowledge_engine = KnowledgeEngine(vector_store)
    categorizer = ArticleCategorizer()

    print(f"Loading {len(SAMPLE_ARTICLES)} sample articles...")

    for article in SAMPLE_ARTICLES:
        # Check if article already exists
        existing = db.get_article_by_url(article.url)
        if existing:
            print(f"  Skipping (exists): {article.title[:50]}...")
            continue

        # Add to database
        db_article = db.add_article(article)
        article.id = db_article.id

        # Add to vector store
        vector_store.add_article(article)

        # Calculate initial scores
        score = knowledge_engine.calculate_growth_score(article)
        reason = knowledge_engine._generate_recommendation_reason(article, score)

        db.update_article_scores(
            article_id=article.id,
            novelty_score=score.novelty,
            growth_potential=score.total,
            recommendation_reason=reason
        )

        print(f"  Added: {article.title[:50]}...")

    print("\nSample data loaded successfully!")
    print(f"Vector store stats: {vector_store.get_stats()}")


if __name__ == "__main__":
    load_sample_data()
