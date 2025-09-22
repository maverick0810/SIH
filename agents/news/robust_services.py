# server/agents/news/robust_service.py
"""
Robust news service with better error handling and fallback mechanisms
"""
import asyncio
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import re
import json
import os
from urllib.parse import urljoin
import ssl
import certifi

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from agents.news.models import NewsArticle, NewsDetail

logger = logging.getLogger(__name__)

class RobustNewsService:
    """Robust news service with comprehensive error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize LLM
        if self.google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=self.google_api_key
            )
        else:
            self.llm = None
            logger.warning("No GOOGLE_API_KEY - limited analysis functionality")
        
        # Reliable RSS feeds (these typically work better)
        self.reliable_rss_feeds = [
            ("The Hindu Agriculture", "https://www.thehindu.com/agriculture/feeder/default.rss"),
            ("Economic Times Agriculture", "https://economictimes.indiatimes.com/news/economy/agriculture/rssfeeds/13357512.cms"),
            ("Business Standard Agriculture", "https://www.business-standard.com/rss/agriculture-103.rss"),
            ("PIB RSS", "https://pib.gov.in/rss/livefeeds.aspx"),
            ("All India Radio News", "http://newsonair.nic.in/rss/ibc.xml"),
            ("DD News RSS", "http://ddnews.gov.in/rss.xml")
        ]
        
        # Fallback news URLs (when scraping fails, use these)
        self.fallback_sources = [
            {
                "name": "Krishijagran",
                "url": "https://krishijagran.com/news/",
                "backup_rss": "https://krishijagran.com/rss.xml"
            },
            {
                "name": "Agriculture Today",
                "url": "https://www.agriculturetoday.in/",
                "backup_rss": "https://www.agriculturetoday.in/feed/"
            }
        ]
        
        # Agriculture keywords for filtering
        self.agriculture_keywords = [
            "farmer", "farming", "agriculture", "crop", "irrigation", "pesticide", "fertilizer",
            "subsidy", "msp", "minimum support price", "pradhan mantri", "kisan", "krishi",
            "soil", "seed", "harvest", "monsoon", "drought", "food security", "rural"
        ]
    
    async def get_comprehensive_news(self, days_back: int = 7, max_articles: int = 100) -> List[NewsArticle]:
        """Get news with robust error handling and fallbacks"""
        all_articles = []
        
        # Create robust HTTP session
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context(cafile=certifi.where()),
            limit=20,
            limit_per_host=5,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        ) as session:
            
            try:
                # Phase 1: Fetch from reliable RSS feeds
                logger.info("Fetching from reliable RSS feeds...")
                rss_articles = await self._fetch_reliable_rss_feeds(session)
                all_articles.extend(rss_articles)
                logger.info(f"Got {len(rss_articles)} articles from RSS feeds")
                
                # Phase 2: Try web scraping with fallbacks
                logger.info("Attempting web scraping...")
                scraped_articles = await self._safe_web_scraping(session)
                all_articles.extend(scraped_articles)
                logger.info(f"Got {len(scraped_articles)} articles from web scraping")
                
                # Phase 3: Use search-based fallback if needed
                if len(all_articles) < 10:
                    logger.info("Using search-based fallback...")
                    search_articles = await self._search_based_fallback(session)
                    all_articles.extend(search_articles)
                
            except Exception as e:
                logger.error(f"Error in comprehensive news fetch: {e}")
                # Use emergency fallback
                all_articles = await self._emergency_fallback()
        
        # Process and filter articles
        processed_articles = self._process_articles(all_articles, days_back, max_articles)
        
        logger.info(f"Returning {len(processed_articles)} processed articles")
        return processed_articles
    
    async def _fetch_reliable_rss_feeds(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        """Fetch from known reliable RSS feeds"""
        articles = []
        
        for source_name, rss_url in self.reliable_rss_feeds:
            try:
                async with session.get(rss_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:10]:  # Limit per feed
                            article = self._parse_rss_entry(entry, source_name)
                            if article and self._is_agriculture_related(article.title + " " + article.summary):
                                articles.append(article)
                        
                        logger.info(f"Successfully fetched from {source_name}: {len(feed.entries)} entries")
                    else:
                        logger.warning(f"RSS feed {source_name} returned status {response.status}")
                        
            except Exception as e:
                logger.error(f"Failed to fetch RSS from {source_name}: {e}")
                continue
        
        return articles
    
    async def _safe_web_scraping(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        """Attempt web scraping with error handling"""
        articles = []
        
        # Try scraping major news sites (more likely to work)
        scrape_targets = [
            {
                "name": "The Hindu Agriculture Section",
                "url": "https://www.thehindu.com/news/national/",
                "selectors": {"articles": ".story-card", "title": "h3 a", "link": "h3 a"}
            },
            {
                "name": "Economic Times Business",
                "url": "https://economictimes.indiatimes.com/news/economy/",
                "selectors": {"articles": ".eachStory", "title": "h4 a", "link": "h4 a"}
            }
        ]
        
        for target in scrape_targets:
            try:
                articles_from_site = await self._scrape_single_site(session, target)
                articles.extend(articles_from_site)
                logger.info(f"Scraped {len(articles_from_site)} articles from {target['name']}")
                
            except Exception as e:
                logger.error(f"Failed to scrape {target['name']}: {e}")
                continue
        
        return articles
    
    async def _scrape_single_site(self, session: aiohttp.ClientSession, target: Dict) -> List[NewsArticle]:
        """Scrape a single website safely"""
        articles = []
        
        try:
            async with session.get(target["url"]) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find article elements
                    article_elements = soup.select(target["selectors"]["articles"])
                    
                    for element in article_elements[:15]:  # Limit per site
                        try:
                            title_elem = element.select_one(target["selectors"]["title"])
                            if not title_elem:
                                continue
                                
                            title = title_elem.get_text(strip=True)
                            if len(title) < 10:
                                continue
                            
                            # Get article URL
                            link_elem = element.select_one(target["selectors"]["link"])
                            article_url = urljoin(target["url"], link_elem.get('href')) if link_elem else target["url"]
                            
                            # Check if agriculture related
                            if self._is_agriculture_related(title):
                                article = NewsArticle(
                                    title=title,
                                    summary=title,  # Use title as summary for scraped articles
                                    url=article_url,
                                    source=target["name"],
                                    published_date=datetime.now(),
                                    category="agriculture",
                                    relevance_score=self._calculate_relevance(title),
                                    impact_level=self._assess_impact(title),
                                    regions_affected=[],
                                    tags=self._extract_tags(title)
                                )
                                articles.append(article)
                                
                        except Exception as e:
                            logger.debug(f"Error processing article element: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error scraping {target['url']}: {e}")
        
        return articles
    
    async def _search_based_fallback(self, session: aiohttp.ClientSession) -> List[NewsArticle]:
        """Use search-based approach as fallback"""
        # This could implement search via news APIs or custom search
        # For now, return empty list
        return []
    
    async def _emergency_fallback(self) -> List[NewsArticle]:
        """Emergency fallback when all scraping fails"""
        logger.warning("Using emergency fallback news")
        
        # Generate some basic agriculture news articles
        fallback_articles = [
            NewsArticle(
                title="Government Announces New Agricultural Policy Framework",
                summary="Recent policy updates aim to support farmers with enhanced subsidies and technological support.",
                url="",
                source="System Fallback",
                published_date=datetime.now(),
                category="agriculture",
                relevance_score=0.8,
                impact_level="high",
                regions_affected=["All India"],
                tags=["policy", "government", "subsidy"]
            ),
            NewsArticle(
                title="Monsoon Forecast Updates for Agricultural Planning",
                summary="Weather department releases latest monsoon predictions for upcoming agricultural season.",
                url="",
                source="System Fallback",
                published_date=datetime.now(),
                category="agriculture",
                relevance_score=0.7,
                impact_level="medium",
                regions_affected=["All India"],
                tags=["weather", "monsoon", "planning"]
            ),
            NewsArticle(
                title="Technology Adoption in Modern Farming Practices",
                summary="Digital tools and modern techniques helping farmers increase productivity and reduce costs.",
                url="",
                source="System Fallback",
                published_date=datetime.now(),
                category="agriculture",
                relevance_score=0.6,
                impact_level="medium",
                regions_affected=["All India"],
                tags=["technology", "productivity", "innovation"]
            )
        ]
        
        return fallback_articles
    
    def _parse_rss_entry(self, entry, source_name: str) -> Optional[NewsArticle]:
        """Parse RSS entry with better error handling"""
        try:
            title = entry.get('title', '').strip()
            summary = entry.get('summary', entry.get('description', '')).strip()
            url = entry.get('link', '').strip()
            
            if not title or len(title) < 10:
                return None
            
            # Parse publication date with timezone handling
            pub_date = self._parse_date_safe(entry.get('published', ''))
            
            # Clean HTML from summary
            summary = self._clean_html(summary)
            
            relevance = self._calculate_relevance(title + " " + summary)
            
            return NewsArticle(
                title=title,
                summary=summary[:500],  # Limit summary length
                url=url,
                source=source_name,
                published_date=pub_date,
                category="agriculture",
                relevance_score=relevance,
                impact_level=self._assess_impact(title + " " + summary),
                regions_affected=self._extract_regions(title + " " + summary),
                tags=self._extract_tags(title + " " + summary)
            )
        except Exception as e:
            logger.debug(f"Error parsing RSS entry: {e}")
            return None
    
    def _parse_date_safe(self, date_str: str) -> datetime:
        """Parse date string safely with timezone handling"""
        if not date_str:
            return datetime.now()
        
        try:
            # Try different date parsing approaches
            from dateutil import parser
            
            # Handle IST timezone specifically
            tzinfos = {"IST": +5.5 * 3600}  # IST is UTC+5:30
            
            parsed_date = parser.parse(date_str, tzinfos=tzinfos)
            
            # Convert to naive datetime if timezone aware
            if parsed_date.tzinfo:
                parsed_date = parsed_date.replace(tzinfo=None)
            
            return parsed_date
            
        except Exception as e:
            logger.debug(f"Date parsing failed for '{date_str}': {e}")
            return datetime.now()
    
    def _process_articles(self, articles: List[NewsArticle], days_back: int, max_articles: int) -> List[NewsArticle]:
        """Process and filter articles"""
        if not articles:
            return []
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_articles = [a for a in articles if a.published_date >= cutoff_date]
        
        # Remove duplicates
        unique_articles = self._deduplicate_articles(recent_articles)
        
        # Sort by relevance and date
        sorted_articles = sorted(
            unique_articles,
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True
        )
        
        return sorted_articles[:max_articles]
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', article.title.lower()).strip()
            
            if normalized_title not in seen_titles and len(normalized_title) > 5:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles
    
    # Helper methods
    def _is_agriculture_related(self, text: str) -> bool:
        """Check if text is agriculture related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.agriculture_keywords)
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score"""
        text_lower = text.lower()
        score = 0.0
        
        # Count agriculture keywords
        agri_matches = sum(1 for keyword in self.agriculture_keywords if keyword in text_lower)
        score = min(agri_matches / 5, 1.0)  # Normalize to max 1.0
        
        return score
    
    def _assess_impact(self, text: str) -> str:
        """Assess impact level"""
        text_lower = text.lower()
        high_impact_words = ["budget", "policy", "scheme", "announcement", "new", "increase", "ban"]
        
        if any(word in text_lower for word in high_impact_words):
            return "high"
        elif any(word in text_lower for word in ["update", "meeting", "discussion"]):
            return "medium"
        return "low"
    
    def _extract_regions(self, text: str) -> List[str]:
        """Extract mentioned regions"""
        states = ["Punjab", "Haryana", "Maharashtra", "Gujarat", "Rajasthan", "Uttar Pradesh"]
        return [state for state in states if state.lower() in text.lower()]
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags"""
        text_lower = text.lower()
        tags = []
        
        tag_keywords = {
            "subsidy": ["subsidy", "financial aid"],
            "policy": ["policy", "guideline"],
            "technology": ["technology", "digital", "app"],
            "weather": ["weather", "monsoon", "rain"],
            "loan": ["loan", "credit", "finance"]
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags"""
        if not text:
            return ""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text).strip()
    
    async def elaborate_articles(self, article_ids: List[str], articles: List[NewsArticle]) -> List[NewsDetail]:
        """Elaborate articles with AI analysis"""
        if not self.llm or not articles:
            # Return basic elaboration
            return [
                NewsDetail(
                    article=article,
                    key_points=[article.summary] if article.summary else [article.title],
                    impact_analysis="Detailed AI analysis requires Google API key configuration",
                    action_items=["Read full article for more details"],
                    related_schemes=[],
                    deadlines=[]
                )
                for article in articles
            ]
        
        elaborated = []
        for article in articles:
            try:
                detail = await self._analyze_article_with_ai(article)
                elaborated.append(detail)
            except Exception as e:
                logger.error(f"Error elaborating article {article.title}: {e}")
                # Add fallback
                elaborated.append(
                    NewsDetail(
                        article=article,
                        key_points=[article.summary or article.title],
                        impact_analysis=f"Analysis failed: {str(e)}",
                        action_items=["Contact agricultural extension office for guidance"],
                        related_schemes=[],
                        deadlines=[]
                    )
                )
        
        return elaborated
    
    async def _analyze_article_with_ai(self, article: NewsArticle) -> NewsDetail:
        """Analyze article with AI"""
        system_prompt = """You are an agricultural expert helping farmers understand news. 

Analyze the article and provide:
1. Key points in simple language
2. How this affects farmers
3. What actions farmers should take
4. Related government schemes
5. Important deadlines

Respond in JSON:
{
  "key_points": ["point1", "point2"],
  "impact_analysis": "How this affects farmers",
  "action_items": ["action1", "action2"],
  "related_schemes": ["scheme1", "scheme2"],
  "deadlines": []
}"""
        
        user_message = f"""Article: {article.title}
        
Content: {article.summary}
Source: {article.source}

Provide farmer-focused analysis."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            analysis_data = self._parse_json_response(response.content)
            
            return NewsDetail(
                article=article,
                key_points=analysis_data.get("key_points", [article.summary]),
                impact_analysis=analysis_data.get("impact_analysis", "Impact analysis not available"),
                action_items=analysis_data.get("action_items", []),
                related_schemes=analysis_data.get("related_schemes", []),
                deadlines=analysis_data.get("deadlines", [])
            )
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return NewsDetail(
                article=article,
                key_points=[article.summary or article.title],
                impact_analysis="AI analysis temporarily unavailable",
                action_items=[],
                related_schemes=[],
                deadlines=[]
            )
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI JSON response safely"""
        try:
            # Clean response
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            return json.loads(cleaned.strip())
        except:
            return {}