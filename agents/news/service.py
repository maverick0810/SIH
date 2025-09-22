# server/agents/news/service.py
"""
News service for aggregating agricultural and government news from multiple sources
"""
import asyncio
import aiohttp
import feedparser
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import logging
import re
import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from agents.news.models import NewsRequest, NewsArticle, NewsDetail

logger = logging.getLogger(__name__)

class NewsService:
    """Service for fetching and analyzing agricultural news from multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize LLM for content analysis
        if self.google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=self.google_api_key
            )
        else:
            self.llm = None
            logger.warning("No GOOGLE_API_KEY - limited news analysis functionality")
        
        # News sources configuration
        self.news_sources = {
            "government": [
                {
                    "name": "PIB (Press Information Bureau)",
                    "url": "https://pib.gov.in/indexd.aspx",
                    "rss": "https://pib.gov.in/rss/livefeeds.aspx",
                    "type": "government"
                },
                {
                    "name": "Ministry of Agriculture",
                    "url": "https://agricoop.nic.in/",
                    "type": "government"
                },
                {
                    "name": "Department of Agriculture",
                    "url": "https://agricoop.nic.in/",
                    "type": "government"
                }
            ],
            "news_sites": [
                {
                    "name": "The Hindu - Agriculture",
                    "rss": "https://www.thehindu.com/agriculture/feeder/default.rss",
                    "type": "news"
                },
                {
                    "name": "Indian Express - Agriculture",
                    "rss": "https://indianexpress.com/section/india/rss/",
                    "type": "news"
                },
                {
                    "name": "Economic Times - Agriculture",
                    "rss": "https://economictimes.indiatimes.com/news/economy/agriculture/rssfeeds/13357512.cms",
                    "type": "news"
                },
                {
                    "name": "Business Standard - Agriculture",
                    "rss": "https://www.business-standard.com/rss/agriculture-103.rss",
                    "type": "news"
                }
            ],
            "specialized": [
                {
                    "name": "Krishijagran",
                    "url": "https://krishijagran.com/",
                    "type": "agricultural"
                },
                {
                    "name": "AgriProFocus India",
                    "url": "https://agriprofocus.com/",
                    "type": "agricultural"
                }
            ]
        }
        
        # Keywords for filtering relevant content
        self.agriculture_keywords = [
            "farmer", "farming", "agriculture", "crop", "irrigation", "pesticide",
            "fertilizer", "subsidy", "msp", "minimum support price", "pradhan mantri",
            "kisan", "krishi", "soil", "seed", "harvest", "monsoon", "drought",
            "food security", "rural", "agricultural policy", "farm loan", "insurance"
        ]
        
        self.government_keywords = [
            "government scheme", "policy", "budget", "ministry", "department",
            "notification", "circular", "guidelines", "implementation", "launch",
            "announcement", "initiative", "program", "yojana", "mission"
        ]
    
    async def fetch_news(self, request: NewsRequest) -> List[NewsArticle]:
        """Fetch news from multiple sources"""
        all_articles = []
        
        try:
            # Fetch from RSS feeds
            rss_articles = await self._fetch_rss_news(request)
            all_articles.extend(rss_articles)
            
            # Fetch from web scraping (if needed)
            web_articles = await self._fetch_web_news(request)
            all_articles.extend(web_articles)
            
            # Filter and rank articles
            filtered_articles = self._filter_and_rank(all_articles, request)
            
            # Limit results
            return filtered_articles[:request.limit]
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    async def _fetch_rss_news(self, request: NewsRequest) -> List[NewsArticle]:
        """Fetch news from RSS feeds"""
        articles = []
        
        # Collect RSS URLs based on category
        rss_urls = []
        for source in self.news_sources["news_sites"]:
            if "rss" in source:
                rss_urls.append((source["name"], source["rss"]))
        
        # Add government RSS if available
        for source in self.news_sources["government"]:
            if "rss" in source:
                rss_urls.append((source["name"], source["rss"]))
        
        # Fetch from each RSS feed
        for source_name, rss_url in rss_urls:
            try:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    # Parse publication date
                    pub_date = self._parse_date(entry.get('published', ''))
                    if not pub_date:
                        pub_date = datetime.now()
                    
                    # Check if article is within date range
                    if (datetime.now() - pub_date).days > request.days_back:
                        continue
                    
                    # Extract content
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    url = entry.get('link', '')
                    
                    # Calculate relevance
                    relevance = self._calculate_relevance(title + " " + summary, request)
                    
                    if relevance > 0.3:  # Only include relevant articles
                        article = NewsArticle(
                            title=title,
                            summary=self._clean_html(summary),
                            url=url,
                            source=source_name,
                            published_date=pub_date,
                            category=request.category,
                            relevance_score=relevance,
                            impact_level=self._assess_impact(title + " " + summary),
                            regions_affected=self._extract_regions(title + " " + summary),
                            tags=self._extract_tags(title + " " + summary)
                        )
                        articles.append(article)
                        
            except Exception as e:
                logger.error(f"Error fetching RSS from {source_name}: {e}")
                continue
        
        return articles
    
    async def _fetch_web_news(self, request: NewsRequest) -> List[NewsArticle]:
        """Fetch news through web search APIs or scraping"""
        articles = []
        
        # Use search API if available (placeholder for now)
        # This could integrate with NewsAPI, Google News API, etc.
        
        return articles
    
    def _calculate_relevance(self, text: str, request: NewsRequest) -> float:
        """Calculate relevance score for an article"""
        text_lower = text.lower()
        relevance = 0.0
        
        # Check for agriculture keywords
        agri_matches = sum(1 for keyword in self.agriculture_keywords if keyword in text_lower)
        relevance += (agri_matches / len(self.agriculture_keywords)) * 0.7
        
        # Check for government keywords
        gov_matches = sum(1 for keyword in self.government_keywords if keyword in text_lower)
        relevance += (gov_matches / len(self.government_keywords)) * 0.3
        
        # Boost for specific category
        if request.category == "agriculture" and agri_matches > 0:
            relevance += 0.2
        elif request.category == "government" and gov_matches > 0:
            relevance += 0.2
        
        # Boost for state-specific news
        if request.state and request.state.lower() in text_lower:
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _assess_impact(self, text: str) -> str:
        """Assess the impact level of news"""
        text_lower = text.lower()
        
        high_impact_words = ["budget", "policy", "scheme", "launch", "new", "increase", "decrease", "ban", "mandatory"]
        medium_impact_words = ["guideline", "circular", "update", "modify", "extend"]
        
        high_count = sum(1 for word in high_impact_words if word in text_lower)
        medium_count = sum(1 for word in medium_impact_words if word in text_lower)
        
        if high_count >= 2:
            return "high"
        elif high_count >= 1 or medium_count >= 2:
            return "medium"
        else:
            return "low"
    
    def _extract_regions(self, text: str) -> List[str]:
        """Extract mentioned regions/states"""
        indian_states = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
            "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
            "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
            "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
        ]
        
        found_states = []
        text_lower = text.lower()
        
        for state in indian_states:
            if state.lower() in text_lower:
                found_states.append(state)
        
        return found_states
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text"""
        tags = []
        text_lower = text.lower()
        
        tag_keywords = {
            "subsidy": ["subsidy", "financial aid", "support"],
            "policy": ["policy", "guideline", "rule"],
            "technology": ["technology", "digital", "app", "online"],
            "insurance": ["insurance", "crop insurance", "coverage"],
            "loan": ["loan", "credit", "finance"],
            "scheme": ["scheme", "yojana", "program"],
            "msp": ["msp", "minimum support price", "procurement"],
            "weather": ["weather", "monsoon", "drought", "rain"]
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _filter_and_rank(self, articles: List[NewsArticle], request: NewsRequest) -> List[NewsArticle]:
        """Filter and rank articles by relevance"""
        # Remove duplicates based on title similarity
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title_key = self._normalize_title(article.title)
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        # Sort by relevance score and date
        sorted_articles = sorted(
            unique_articles,
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True
        )
        
        return sorted_articles
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for duplicate detection"""
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        
        try:
            return date_parser.parse(date_str)
        except Exception:
            return None
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text).strip()
    
    async def get_article_details(self, url: str, article: NewsArticle) -> NewsDetail:
        """Get detailed analysis of a news article"""
        if not self.llm:
            # Return basic details without AI analysis
            return NewsDetail(
                article=article,
                key_points=[article.summary],
                impact_analysis="AI analysis not available - Google API key required",
                action_items=["Read full article for details"],
                related_schemes=[],
                deadlines=[]
            )
        
        try:
            # Try to fetch full article content
            full_content = await self._fetch_article_content(url)
            if not full_content:
                full_content = article.summary
            
            # Analyze with AI
            analysis = await self._analyze_article_with_ai(article, full_content)
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting article details: {e}")
            # Return fallback
            return NewsDetail(
                article=article,
                key_points=[article.summary],
                impact_analysis=f"Analysis failed: {str(e)}",
                action_items=["Read full article for details"],
                related_schemes=[],
                deadlines=[]
            )
    
    async def _fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch full article content from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        # Basic content extraction (could be enhanced with newspaper3k)
                        return self._extract_text_from_html(html_content)
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
        
        return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """Basic text extraction from HTML"""
        # This is a simple implementation - could be enhanced with libraries like BeautifulSoup
        import re
        
        # Remove script and style elements
        html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Return first 2000 characters
        return text[:2000] if len(text) > 2000 else text
    
    async def _analyze_article_with_ai(self, article: NewsArticle, content: str) -> NewsDetail:
        """Analyze article content with AI"""
        
        system_prompt = """You are an agricultural policy expert helping farmers understand government news and policies.
        
        Analyze the given news article and provide:
        1. Key points in simple language
        2. Impact analysis for farmers
        3. Action items farmers should take
        4. Related government schemes
        5. Important deadlines
        
        Respond in JSON format:
        {
          "key_points": ["point1", "point2"],
          "impact_analysis": "How this affects farmers",
          "action_items": ["action1", "action2"],
          "related_schemes": ["scheme1", "scheme2"],
          "deadlines": [{"task": "what", "date": "when"}]
        }"""
        
        user_message = f"""Article Title: {article.title}
        
        Article Content: {content}
        
        Source: {article.source}
        Published: {article.published_date.strftime('%Y-%m-%d')}
        
        Please analyze this article for farmers."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse JSON response
            analysis_data = self._parse_analysis_response(response_text)
            
            return NewsDetail(
                article=article,
                key_points=analysis_data.get("key_points", [article.summary]),
                impact_analysis=analysis_data.get("impact_analysis", "Impact analysis not available"),
                action_items=analysis_data.get("action_items", []),
                related_schemes=analysis_data.get("related_schemes", []),
                deadlines=analysis_data.get("deadlines", [])
            )
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return NewsDetail(
                article=article,
                key_points=[article.summary],
                impact_analysis="AI analysis failed",
                action_items=[],
                related_schemes=[],
                deadlines=[]
            )
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI analysis response"""
        try:
            # Clean response text
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError:
            logger.error("Failed to parse AI analysis as JSON")
            return {}
    
    def get_news_categories(self) -> List[Dict[str, str]]:
        """Get available news categories"""
        return [
            {"category": "agriculture", "description": "Farming, crops, agricultural technology"},
            {"category": "government", "description": "Government policies, schemes, announcements"},
            {"category": "policy", "description": "Agricultural policies and regulations"},
            {"category": "subsidies", "description": "Subsidies and financial support"},
            {"category": "technology", "description": "Agricultural technology and innovation"},
            {"category": "weather", "description": "Weather updates and agricultural advisories"},
            {"category": "market", "description": "Market prices and trade news"}
        ]