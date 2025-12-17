"""
Weekly digest email service using Gmail.

Sends curated article recommendations via email with:
- Top recommended articles for the week
- Knowledge growth summary
- New concepts to explore
- Category breakdown
"""

import os
import base64
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.models.database import DatabaseManager
from src.models.article import Article, ArticleCategory
from src.services.knowledge_engine import KnowledgeEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DigestConfig:
    """Configuration for digest emails."""
    recipient_email: str
    sender_email: str
    sender_name: str = "Knowledge Curator"
    max_articles: int = 10
    include_stats: bool = True
    include_concepts: bool = True


class EmailDigest:
    """
    Service for sending weekly digest emails via Gmail.

    Supports two authentication methods:
    1. Gmail API with OAuth2 (recommended for production)
    2. SMTP with App Password (simpler setup)
    """

    def __init__(
        self,
        smtp_password: Optional[str] = None,
        use_gmail_api: bool = False,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize the email digest service.

        Args:
            smtp_password: Gmail App Password for SMTP auth
            use_gmail_api: Whether to use Gmail API (requires OAuth setup)
            credentials_path: Path to Gmail API credentials.json
        """
        self.smtp_password = smtp_password or os.getenv("GMAIL_APP_PASSWORD")
        self.use_gmail_api = use_gmail_api
        self.credentials_path = credentials_path or os.getenv("GMAIL_CREDENTIALS_PATH")

        self.db = DatabaseManager()
        self.knowledge_engine = KnowledgeEngine()

        self._gmail_service = None
        if use_gmail_api:
            self._init_gmail_api()

    def _init_gmail_api(self):
        """Initialize Gmail API client."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import pickle

            SCOPES = ['https://www.googleapis.com/auth/gmail.send']
            creds = None

            # Load existing credentials
            token_path = 'token.pickle'
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif self.credentials_path and os.path.exists(self.credentials_path):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save credentials
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)

            if creds:
                self._gmail_service = build('gmail', 'v1', credentials=creds)
                logger.info("Gmail API initialized successfully")

        except ImportError:
            logger.warning("Gmail API libraries not installed. Install with: pip install google-auth-oauthlib google-api-python-client")
        except Exception as e:
            logger.error(f"Failed to initialize Gmail API: {e}")

    def generate_digest(
        self,
        days: int = 7,
        max_articles: int = 10
    ) -> Dict:
        """
        Generate digest content for the specified period.

        Args:
            days: Number of days to look back
            max_articles: Maximum articles to include

        Returns:
            Dict containing digest data
        """
        # Get top recommendations
        recommendations = self.knowledge_engine.get_recommendations(n=max_articles)

        # Get knowledge summary
        knowledge_summary = self.knowledge_engine.get_knowledge_summary()

        # Get recently liked articles
        liked_articles = self.db.get_liked_articles()
        recent_liked = [
            a for a in liked_articles
            if a.read_at and a.read_at > datetime.utcnow() - timedelta(days=days)
        ]

        # New concepts from recent reading
        new_concepts = set()
        for article in recent_liked:
            if article.related_concepts:
                try:
                    import json
                    concepts = json.loads(article.related_concepts)
                    new_concepts.update(concepts)
                except:
                    pass

        # Category breakdown of recommendations
        category_breakdown = {}
        for rec in recommendations:
            cat = rec.article.category.value
            category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

        return {
            "recommendations": recommendations,
            "knowledge_summary": knowledge_summary,
            "recent_liked": recent_liked,
            "new_concepts": list(new_concepts)[:20],
            "category_breakdown": category_breakdown,
            "period_days": days,
            "generated_at": datetime.utcnow()
        }

    def format_html_digest(self, digest_data: Dict) -> str:
        """Format digest data as HTML email."""
        recommendations = digest_data["recommendations"]
        knowledge_summary = digest_data["knowledge_summary"]
        new_concepts = digest_data["new_concepts"]
        category_breakdown = digest_data["category_breakdown"]

        # Category emojis
        cat_emojis = {
            "demo": "🔧",
            "idea": "💡",
            "tool": "🛠️",
            "tutorial": "📚",
            "news": "📰",
            "other": "📎"
        }

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .article-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }}
        .article-title {{
            font-weight: bold;
            color: #333;
            text-decoration: none;
        }}
        .article-title:hover {{
            color: #667eea;
        }}
        .article-meta {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .article-reason {{
            font-style: italic;
            color: #555;
            margin-top: 10px;
            font-size: 14px;
        }}
        .growth-score {{
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
        }}
        .concepts {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .concept-tag {{
            background: #ffc107;
            color: #333;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 12px;
        }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Weekly Knowledge Digest</h1>
        <p>Your personalized learning recommendations</p>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{knowledge_summary.get('total_learned', 0)}</div>
            <div class="stat-label">Articles Read</div>
        </div>
        <div class="stat">
            <div class="stat-value">{knowledge_summary.get('concepts_count', 0)}</div>
            <div class="stat-label">Concepts Learned</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(recommendations)}</div>
            <div class="stat-label">New Recommendations</div>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Top Recommendations</h2>
        <p>Selected for maximum knowledge growth based on your learning history:</p>
"""

        # Add article cards
        for rec in recommendations[:10]:
            article = rec.article
            category = article.category.value if article.category else "other"
            emoji = cat_emojis.get(category, "📄")

            html += f"""
        <div class="article-card">
            <a href="{article.url}" class="article-title">{emoji} {article.title}</a>
            <div class="article-meta">
                <span class="growth-score">Growth: {rec.score:.0%}</span>
                • {category.upper()}
                {f'• {article.source}' if article.source else ''}
            </div>
            <div class="article-reason">💭 {rec.reason}</div>
        </div>
"""

        html += """
    </div>
"""

        # Add concepts section
        if new_concepts:
            html += """
    <div class="section">
        <h2>🌟 Concepts to Explore</h2>
        <div class="concepts">
"""
            for concept in new_concepts[:15]:
                html += f'            <span class="concept-tag">{concept}</span>\n'

            html += """
        </div>
    </div>
"""

        # Add category breakdown
        if category_breakdown:
            html += """
    <div class="section">
        <h2>📊 This Week's Mix</h2>
        <p>
"""
            parts = []
            for cat, count in sorted(category_breakdown.items(), key=lambda x: -x[1]):
                emoji = cat_emojis.get(cat, "📄")
                parts.append(f"{emoji} {count} {cat}")
            html += " • ".join(parts)

            html += """
        </p>
    </div>
"""

        # Footer
        html += f"""
    <div class="footer">
        <p>Generated by Knowledge Curator on {datetime.now().strftime('%B %d, %Y')}</p>
        <p>Keep learning, keep growing! 🚀</p>
    </div>
</body>
</html>
"""
        return html

    def format_text_digest(self, digest_data: Dict) -> str:
        """Format digest data as plain text email."""
        recommendations = digest_data["recommendations"]
        knowledge_summary = digest_data["knowledge_summary"]
        new_concepts = digest_data["new_concepts"]

        text = """
═══════════════════════════════════════════════════════
       🧠 WEEKLY KNOWLEDGE DIGEST
       Your personalized learning recommendations
═══════════════════════════════════════════════════════

📊 YOUR STATS
───────────────────────────────────────────────────────
"""
        text += f"  Articles Read: {knowledge_summary.get('total_learned', 0)}\n"
        text += f"  Concepts Learned: {knowledge_summary.get('concepts_count', 0)}\n"
        text += f"  New Recommendations: {len(recommendations)}\n"

        text += """
🎯 TOP RECOMMENDATIONS
───────────────────────────────────────────────────────
"""
        for i, rec in enumerate(recommendations[:10], 1):
            article = rec.article
            text += f"\n{i}. {article.title}\n"
            text += f"   📈 Growth Potential: {rec.score:.0%}\n"
            text += f"   💭 {rec.reason}\n"
            text += f"   🔗 {article.url}\n"

        if new_concepts:
            text += """
🌟 CONCEPTS TO EXPLORE
───────────────────────────────────────────────────────
"""
            text += ", ".join(new_concepts[:15]) + "\n"

        text += f"""
───────────────────────────────────────────────────────
Generated on {datetime.now().strftime('%B %d, %Y')}
Keep learning, keep growing! 🚀
"""
        return text

    def send_digest(
        self,
        config: DigestConfig,
        days: int = 7
    ) -> bool:
        """
        Generate and send the weekly digest email.

        Args:
            config: Email configuration
            days: Number of days to look back

        Returns:
            True if email was sent successfully
        """
        # Generate digest content
        digest_data = self.generate_digest(days=days, max_articles=config.max_articles)

        # Format email
        html_content = self.format_html_digest(digest_data)
        text_content = self.format_text_digest(digest_data)

        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = f"🧠 Your Weekly Knowledge Digest - {datetime.now().strftime('%b %d')}"
        message["From"] = f"{config.sender_name} <{config.sender_email}>"
        message["To"] = config.recipient_email

        # Attach both plain text and HTML versions
        message.attach(MIMEText(text_content, "plain"))
        message.attach(MIMEText(html_content, "html"))

        # Send via appropriate method
        if self.use_gmail_api and self._gmail_service:
            return self._send_via_api(message)
        else:
            return self._send_via_smtp(message, config.sender_email)

    def _send_via_api(self, message: MIMEMultipart) -> bool:
        """Send email via Gmail API."""
        try:
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            self._gmail_service.users().messages().send(
                userId="me",
                body={"raw": raw}
            ).execute()
            logger.info("Digest email sent successfully via Gmail API")
            return True
        except Exception as e:
            logger.error(f"Failed to send email via API: {e}")
            return False

    def _send_via_smtp(self, message: MIMEMultipart, sender_email: str) -> bool:
        """Send email via SMTP with Gmail App Password."""
        if not self.smtp_password:
            logger.error("SMTP password not configured. Set GMAIL_APP_PASSWORD env variable.")
            return False

        try:
            import smtplib

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, self.smtp_password)
                server.send_message(message)

            logger.info("Digest email sent successfully via SMTP")
            return True

        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return False

    def schedule_weekly_digest(
        self,
        config: DigestConfig,
        day_of_week: int = 0,  # 0 = Monday
        hour: int = 9
    ):
        """
        Schedule weekly digest emails.

        Note: This is a simple implementation. For production,
        use a proper scheduler like APScheduler or Celery.

        Args:
            config: Email configuration
            day_of_week: Day to send (0=Monday, 6=Sunday)
            hour: Hour to send (24-hour format)
        """
        try:
            import schedule
            import time

            days = ["monday", "tuesday", "wednesday", "thursday",
                    "friday", "saturday", "sunday"]

            def send_job():
                logger.info("Running scheduled digest email job")
                self.send_digest(config)

            # Schedule the job
            getattr(schedule.every(), days[day_of_week]).at(f"{hour:02d}:00").do(send_job)

            logger.info(f"Digest scheduled for {days[day_of_week]} at {hour:02d}:00")

            # Run the scheduler (blocking)
            while True:
                schedule.run_pending()
                time.sleep(60)

        except ImportError:
            logger.error("schedule library not installed. Install with: pip install schedule")


def send_test_digest(recipient_email: str, sender_email: str):
    """
    Send a test digest email.

    Args:
        recipient_email: Email to send to
        sender_email: Gmail address to send from
    """
    config = DigestConfig(
        recipient_email=recipient_email,
        sender_email=sender_email,
        sender_name="Knowledge Curator"
    )

    digest = EmailDigest()
    return digest.send_digest(config, days=30)
