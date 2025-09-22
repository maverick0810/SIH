# server/agents/news/comprehensive_service.py
"""
Comprehensive news service with deep web scraping and AI analysis
"""
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import feedparser
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
import re
import json
import os
from urllib.parse import urljoin, urlparse
import hashlib

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from agents.news.models import NewsArticle, NewsDetail

logger = logging.getLogger(__name__)

class ComprehensiveNewsService:
    """Advanced news service with deep web scraping and AI analysis"""
    
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
            logger.warning("No GOOGLE_API_KEY - limited analysis functionality")
        
        # Comprehensive news sources with scraping targets
        self.news_sources = {
            "government_official": [
                {
                    "name": "PIB Press Releases",
                    "base_url": "https://pib.gov.in",
                    "rss": "https://pib.gov.in/rss/livefeeds.aspx",
                    "scrape_urls": [
                        "https://pib.gov.in/indexd.aspx",
                        "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1972468"
                    ],
                    "selectors": {
                        "articles": ".content-area",
                        "title": "h1, h2, .headline",
                        "content": ".press-release-content, .content",
                        "date": ".date, .publish-date"
                    }
                },
                {
                    "name": "Ministry of Agriculture",
                    "base_url": "https://agricoop.nic.in",
                    "scrape_urls": [
                        "https://agricoop.nic.in/en/latest-news",
                        "https://agricoop.nic.in/en/press-releases"
                    ],
                    "selectors": {
                        "articles": ".news-item, .press-release",
                        "title": "h3, h2, .title",
                        "content": ".content, .description",
                        "date": ".date"
                    }
                },
                {
                    "name": "Department of Agriculture",
                    "base_url": "https://dac.gov.in",
                    "scrape_urls": [
                        "https://dac.gov.in/latest-news",
                        "https://dac.gov.in/press-releases"
                    ]
                }
            ],
            "major_news_outlets": [
                {
                    "name": "The Hindu Agriculture",
                    "base_url": "https://www.thehindu.com",
                    "rss": "https://www.thehindu.com/agriculture/feeder/default.rss",
                    "scrape_urls": [
                        "https://www.thehindu.com/news/national/",
                        "https://www.thehindu.com/business/agri-business/"
                    ],
                    "selectors": {
                        "articles": ".story-card, article",
                        "title": "h3 a, h2 a, .title",
                        "content": ".story-content, .content",
                        "date": ".publish-time, .date"
                    }
                },
                {
                    "name": "Economic Times Agriculture",
                    "base_url": "https://economictimes.indiatimes.com",
                    "rss": "https://economictimes.indiatimes.com/news/economy/agriculture/rssfeeds/13357512.cms",
                    "scrape_urls": [
                        "https://economictimes.indiatimes.com/news/economy/agriculture",
                        "https://economictimes.indiatimes.com/news/economy/policy"
                    ],
                    "selectors": {
                        "articles": ".story-box, .news-element",
                        "title": "h4 a, h3 a",
                        "content": ".story-content",
                        "date": ".publish-date"
                    }
                },
                {
                    "name": "Business Standard Agriculture",
                    "base_url": "https://www.business-standard.com",
                    "rss": "https://www.business-standard.com/rss/agriculture-103.rss",
                    "scrape_urls": [
                        "https://www.business-standard.com/topic/agriculture",
                        "https://www.business-standard.com/topic/farming"
                    ]
                },
                {
                    "name": "Indian Express Agriculture",
                    "base_url": "https://indianexpress.com",
                    "scrape_urls": [
                        "https://indianexpress.com/section/india/",
                        "https://indianexpress.com/about/farmers/"
                    ],
                    "selectors": {
                        "articles": ".articles, .story",
                        "title": "h2 a, h3 a",
                        "content": ".story-details",
                        "date": ".date"
                    }
                }
            ],
            "specialized_agricultural": [
                {
                    "name": "Krishijagran",
                    "base_url": "https://krishijagran.com",
                    "scrape_urls": [
                        "https://krishijagran.com/news/",
                        "https://krishijagran.com/agriculture-world/"
                    ],
                    "selectors": {
                        "articles": ".news-card, .article-card",
                        "title": ".title, h3",
                        "content": ".content, .description"
                    }
                },
                {
                    "name": "Krishi Jagran Hindi",
                    "base_url": "https://hindi.krishijagran.com",
                    "scrape_urls": ["https://hindi.krishijagran.com/news/"]
                },
                {
                    "name": "Agriculture Today",
                    "base_url": "https://www.agriculturetoday.in",
                    "scrape_urls": [
                        "https://www.agriculturetoday.in/category/news/",
                        "https://www.agriculturetoday.in/category/government-schemes/"
                    ]
                }
            ],
            "regional_news": [
                {
                    "name": "Punjab Kesari Agriculture",
                    "base_url": "https://punjabkesari.in",
                    "scrape_urls": ["https://punjabkesari.in/agriculture"]
                },
                {
                    "name": "Dainik Bhaskar Agriculture",
                    "base_url": "https://www.bhaskar.com",
                    "scrape_urls": ["https://www.bhaskar.com/agriculture/"]
                },
                {
                    "name": "Times of India Agriculture",
                    "base_url": "https://timesofindia.indiatimes.com",
                    "scrape_urls": [
                        "https://timesofindia.indiatimes.com/business/india-business/agriculture",
                        "https://timesofindia.indiatimes.com/india/agriculture"
                    ]
                }
            ]
        }
        
        # Enhanced keywords for better filtering
        self.agriculture_keywords = [
            "farmer", "farming", "agriculture", "crop", "irrigation", "pesticide", "fertilizer",
            "subsidy", "msp", "minimum support price", "pradhan mantri", "kisan", "krishi",
            "soil", "seed", "harvest", "monsoon", "drought", "food security", "rural",
            "agricultural policy", "farm loan", "insurance", "organic farming", "kharif",
            "rabi", "sugarcane", "wheat", "rice", "cotton", "pulses", "horticulture",
            "dairy", "livestock", "poultry", "fisheries", "apiculture", "sericulture"
        ]
        
        self.session = None
    
    async def get_comprehensive_news(self, days_back: int = 7, max_articles: int = 100) -> List[NewsArticle]:
        """Get comprehensive news from all sources with deep scraping"""
        all_articles = []
        
        # Create aiohttp session
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        ) as session:
            self.session = session
            
            # Fetch from RSS feeds (fast)
            rss_articles = await self._fetch_all_rss_feeds()
            all_articles.extend(rss_articles)
            
            # Deep web scraping (comprehensive)
            scraped_articles = await self._deep_scrape_all_sources()
            all_articles.extend(scraped_articles)
            
            # Search-based collection (broader coverage)
            search_articles = await self._search_based_collection()
            all_articles.extend(search_articles)
        
        # Filter, deduplicate, and rank
        filtered_articles = self._process_and_rank_articles(all_articles, days_back)
        
        return filtered_articles[:max_articles]
    
    async def _fetch_all_rss_feeds(self) -> List[NewsArticle]:
        """Fetch from all available RSS feeds"""
        articles = []
        rss_urls = []
        
        # Collect all RSS URLs
        for category in self.news_sources.values():
            for source in category:
                if "rss" in source:
                    rss_urls.append((source["name"], source["rss"]))
        
        # Fetch concurrently
        tasks = [self._fetch_rss_feed(name, url) for name, url in rss_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            else:
                logger.error(f"RSS fetch error: {result}")
        
        return articles
    
    async def _fetch_rss_feed(self, source_name: str, rss_url: str) -> List[NewsArticle]:
        """Fetch articles from a single RSS feed"""
        try:
            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    for entry in feed.entries:
                        article = self._parse_rss_entry(entry, source_name)
                        if article and self._is_agriculture_related(article.title + " " + article.summary):
                            articles.append(article)
                    
                    return articles
        except Exception as e:
            logger.error(f"Error fetching RSS {rss_url}: {e}")
        
        return []
    
    async def _deep_scrape_all_sources(self) -> List[NewsArticle]:
        """Deep scrape all news sources"""
        articles = []
        
        # Scrape each category
        for category_name, sources in self.news_sources.items():
            category_articles = await self._scrape_source_category(sources, category_name)
            articles.extend(category_articles)
        
        return articles
    
    async def _scrape_source_category(self, sources: List[Dict], category: str) -> List[NewsArticle]:
        """Scrape a category of news sources"""
        articles = []
        
        tasks = []
        for source in sources:
            if "scrape_urls" in source:
                for url in source["scrape_urls"]:
                    task = self._scrape_single_page(source, url, category)
                    tasks.append(task)
        
        # Execute scraping tasks concurrently (but limited)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def limited_scrape(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_scrape(task) for task in tasks], return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
        
        return articles
    
    async def _scrape_single_page(self, source: Dict, url: str, category: str) -> List[NewsArticle]:
        """Scrape a single webpage for news articles"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    articles = []
                    selectors = source.get("selectors", {})
                    
                    # Find article containers
                    article_elements = soup.select(selectors.get("articles", "article, .news-item"))
                    
                    for element in article_elements[:20]:  # Limit per page
                        article = self._extract_article_from_element(element, source, url, category, selectors)
                        if article and self._is_agriculture_related(article.title + " " + article.summary):
                            articles.append(article)
                    
                    return articles
                    
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return []
    
    def _extract_article_from_element(self, element, source: Dict, page_url: str, category: str, selectors: Dict) -> Optional[NewsArticle]:
        """Extract article data from HTML element"""
        try:
            # Extract title
            title_elem = element.select_one(selectors.get("title", "h1, h2, h3, .title"))
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            if len(title) < 10:  # Skip if title too short
                return None
            
            # Extract content/summary
            content_elem = element.select_one(selectors.get("content", ".content, .summary, p"))
            summary = content_elem.get_text(strip=True)[:500] if content_elem else title
            
            # Extract URL
            link_elem = title_elem.find('a') or element.find('a')
            article_url = urljoin(page_url, link_elem.get('href')) if link_elem else page_url
            
            # Extract date
            date_elem = element.select_one(selectors.get("date", ".date, .publish-date, time"))
            pub_date = self._parse_date_from_element(date_elem) if date_elem else datetime.now()
            
            # Calculate relevance
            relevance = self._calculate_relevance_score(title + " " + summary)
            
            return NewsArticle(
                title=title,
                summary=self._clean_text(summary),
                url=article_url,
                source=source["name"],
                published_date=pub_date,
                category=category,
                relevance_score=relevance,
                impact_level=self._assess_impact_level(title + " " + summary),
                regions_affected=self._extract_regions(title + " " + summary),
                tags=self._extract_tags(title + " " + summary)
            )
            
        except Exception as e:
            logger.error(f"Error extracting article: {e}")
            return None
    
    async def _search_based_collection(self) -> List[NewsArticle]:
        """Collect articles using search-based approaches"""
        # This could integrate with search APIs like Google News API, Bing News API, etc.
        # For now, return empty list as placeholder
        return []
    
    def _process_and_rank_articles(self, articles: List[NewsArticle], days_back: int) -> List[NewsArticle]:
        """Process, deduplicate, and rank articles"""
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_articles = [a for a in articles if a.published_date >= cutoff_date]
        
        # Deduplicate by title similarity
        unique_articles = self._deduplicate_articles(recent_articles)
        
        # Sort by relevance and date
        sorted_articles = sorted(
            unique_articles,
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True
        )
        
        return sorted_articles
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_hashes = set()
        
        for article in articles:
            # Create a hash of normalized title
            normalized_title = re.sub(r'[^\w\s]', '', article.title.lower())
            title_hash = hashlib.md5(normalized_title.encode()).hexdigest()
            
            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique_articles.append(article)
        
        return unique_articles
    
    async def elaborate_articles(self, article_ids: List[str], articles: List[NewsArticle]) -> List[NewsDetail]:
        """Provide detailed elaboration for selected articles"""
        if not self.llm:
            # Return basic elaboration without AI
            return [
                NewsDetail(
                    article=article,
                    key_points=[article.summary],
                    impact_analysis="AI analysis not available - Google API key required",
                    action_items=["Read full article for details"],
                    related_schemes=[],
                    deadlines=[]
                )
                for article in articles
            ]
        
        # AI-powered elaboration
        elaborated_articles = []
        
        for article in articles:
            try:
                # Fetch full content if possible
                full_content = await self._fetch_full_article_content(article.url)
                
                # Generate detailed analysis
                detail = await self._generate_detailed_analysis(article, full_content)
                elaborated_articles.append(detail)
                
            except Exception as e:
                logger.error(f"Error elaborating article {article.title}: {e}")
                # Add basic fallback
                elaborated_articles.append(
                    NewsDetail(
                        article=article,
                        key_points=[article.summary],
                        impact_analysis=f"Analysis failed: {str(e)}",
                        action_items=["Read full article for details"],
                        related_schemes=[],
                        deadlines=[]
                    )
                )
        
        return elaborated_articles
    
    async def _fetch_full_article_content(self, url: str) -> str:
        """Fetch full article content from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                            element.decompose()
                        
                        # Extract main content
                        content_selectors = [
                            'article', '.article-content', '.story-content', '.post-content',
                            '.entry-content', '.content', '#content', '.main-content'
                        ]
                        
                        content = ""
                        for selector in content_selectors:
                            content_elem = soup.select_one(selector)
                            if content_elem:
                                content = content_elem.get_text(separator=' ', strip=True)
                                break
                        
                        if not content:
                            # Fallback: get all paragraphs
                            paragraphs = soup.find_all('p')
                            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                        
                        return content[:3000]  # Limit content length
                        
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {e}")
        
        return ""
    
    async def _generate_detailed_analysis(self, article: NewsArticle, full_content: str) -> NewsDetail:
        """Generate detailed AI analysis of article"""
        
        system_prompt = """You are an expert agricultural policy analyst helping farmers understand news and government policies.

Analyze the given news article thoroughly and provide:
1. Key points in simple, farmer-friendly language
2. Detailed impact analysis for farmers 
3. Specific action items farmers should take
4. Related government schemes or programs
5. Important deadlines or time-sensitive information

Focus on practical, actionable insights that help farmers make informed decisions.

Respond in JSON format:
{
  "key_points": ["point1", "point2", "point3"],
  "impact_analysis": "Detailed explanation of how this affects farmers",
  "action_items": ["specific action1", "specific action2"],
  "related_schemes": ["scheme1", "scheme2"],
  "deadlines": [{"task": "what to do", "date": "when", "importance": "high/medium/low"}],
  "farmer_benefits": ["benefit1", "benefit2"],
  "potential_challenges": ["challenge1", "challenge2"],
  "regional_impact": "Which regions/states are most affected",
  "economic_impact": "Financial implications for farmers"
}"""
        
        content_to_analyze = full_content if full_content else article.summary
        
        user_message = f"""Article Title: {article.title}

Source: {article.source}
Published: {article.published_date.strftime('%Y-%m-%d')}
Category: {article.category}

Article Content:
{content_to_analyze}

Please provide a comprehensive analysis for farmers."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            analysis_data = self._parse_analysis_json(response.content)
            
            return NewsDetail(
                article=article,
                key_points=analysis_data.get("key_points", [article.summary]),
                impact_analysis=analysis_data.get("impact_analysis", "Impact analysis not available"),
                action_items=analysis_data.get("action_items", []),
                related_schemes=analysis_data.get("related_schemes", []),
                deadlines=analysis_data.get("deadlines", [])
            )
            
        except Exception as e:
            logger.error(f"Error in detailed analysis: {e}")
            return NewsDetail(
                article=article,
                key_points=[article.summary],
                impact_analysis="Detailed analysis failed",
                action_items=[],
                related_schemes=[],
                deadlines=[]
            )
    
    def _parse_analysis_json(self, response_text: str) -> Dict[str, Any]:
        """Parse AI analysis response"""
        try:
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
    
    # Helper methods (simplified versions of previous implementations)
    def _parse_rss_entry(self, entry, source_name: str) -> Optional[NewsArticle]:
        """Parse RSS entry to NewsArticle"""
        try:
            title = entry.get('title', '')
            summary = entry.get('summary', entry.get('description', ''))
            url = entry.get('link', '')
            
            pub_date = self._parse_date_string(entry.get('published', ''))
            if not pub_date:
                pub_date = datetime.now()
            
            relevance = self._calculate_relevance_score(title + " " + summary)
            
            return NewsArticle(
                title=title,
                summary=self._clean_text(summary),
                url=url,
                source=source_name,
                published_date=pub_date,
                category="agriculture",
                relevance_score=relevance,
                impact_level=self._assess_impact_level(title + " " + summary),
                regions_affected=self._extract_regions(title + " " + summary),
                tags=self._extract_tags(title + " " + summary)
            )
        except Exception as e:
            logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    def _is_agriculture_related(self, text: str) -> bool:
        """Check if text is agriculture related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.agriculture_keywords)
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score for farmers"""
        text_lower = text.lower()
        score = 0.0
        
        # Agriculture keywords
        agri_matches = sum(1 for keyword in self.agriculture_keywords if keyword in text_lower)
        score += (agri_matches / len(self.agriculture_keywords)) * 0.8
        
        # Government/policy keywords  
        gov_keywords = ["government", "policy", "scheme", "subsidy", "ministry", "announcement"]
        gov_matches = sum(1 for keyword in gov_keywords if keyword in text_lower)
        score += (gov_matches / len(gov_keywords)) * 0.2
        
        return min(score, 1.0)
    
    def _assess_impact_level(self, text: str) -> str:
        """Assess impact level of news"""
        text_lower = text.lower()
        high_impact = ["new scheme", "budget", "policy change", "price increase", "subsidy", "loan waiver"]
        medium_impact = ["update", "extension", "guidelines", "meeting"]
        
        if any(word in text_lower for word in high_impact):
            return "high"
        elif any(word in text_lower for word in medium_impact):
            return "medium"
        return "low"
    
    def _extract_regions(self, text: str) -> List[str]:
        """Extract mentioned regions"""
        # Simplified version - could be enhanced
        states = ["Punjab", "Haryana", "Maharashtra", "Gujarat", "Rajasthan", "Uttar Pradesh"]
        return [state for state in states if state.lower() in text.lower()]
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags"""
        text_lower = text.lower()
        tag_map = {
            "subsidy": ["subsidy", "financial aid"],
            "policy": ["policy", "guideline"],
            "scheme": ["scheme", "yojana"],
            "msp": ["msp", "minimum support price"],
            "loan": ["loan", "credit"]
        }
        
        tags = []
        for tag, keywords in tag_map.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        return tags
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse date string"""
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None
    
    def _parse_date_from_element(self, element) -> datetime:
        """Parse date from HTML element"""
        try:
            date_text = element.get_text(strip=True)
            return self._parse_date_string(date_text) or datetime.now()
        except:
            return datetime.now()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    async def _generate_news_summary(self, articles: List[NewsArticle], days_back: int) -> str:
        """Generate AI-powered summary of news articles"""
        if not self.llm:
            return f"Found {len(articles)} agricultural news articles in the last {days_back} days. AI summary not available."
        
        try:
            # Prepare article summaries for AI
            article_summaries = []
            for i, article in enumerate(articles[:10], 1):  # Limit to top 10 for summary
                article_summaries.append(
                    f"{i}. {article.title} ({article.source}) - {article.summary[:100]}..."
                )
            
            system_prompt = """You are an agricultural news analyst. Create a concise, farmer-friendly summary of the provided news articles.

Focus on:
- Most important developments for farmers
- Government policy changes or new schemes
- Market trends and price impacts
- Weather and seasonal updates
- Technology or innovation news

Write in simple language that farmers can easily understand. Highlight actionable information."""
            
            user_message = f"""Please create a concise summary of these agricultural news articles from the last {days_back} days:

{chr(10).join(article_summaries)}

Total articles analyzed: {len(articles)}
High impact: {len([a for a in articles if a.impact_level == "high"])}
Medium impact: {len([a for a in articles if a.impact_level == "medium"])}

Provide a 2-3 paragraph summary highlighting the most important information for farmers."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return f"Found {len(articles)} articles in the last {days_back} days. Key topics include government schemes, market updates, and agricultural policies."