# server/agents/news/agent.py
"""
Updated news agent using comprehensive scraping service
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add server directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base import BaseAgent
from agents.news.models import NewsRequest, NewsResponse, NewsArticle
from agents.news.comprehensive_service import ComprehensiveNewsService
from core.exceptions import AgentError, AgentConfigError

class NewsAgent(BaseAgent[NewsRequest, NewsResponse]):
    """
    Comprehensive news agent with deep web scraping capabilities
    
    Features:
    - Deep web scraping across multiple sources
    - AI-powered content analysis and summarization
    - Intelligent relevance filtering
    - Impact assessment and categorization
    - Detailed article elaboration on demand
    """
    
    def __init__(self):
        super().__init__("news")
        
        # Check for Google API key for enhanced analysis
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            self.logger.warning("No GOOGLE_API_KEY found - basic aggregation only")
        else:
            self.logger.info("News agent initialized with AI analysis capabilities")
        
        self.service = ComprehensiveNewsService(config=self.config)
        self.logger.info("Comprehensive news agent initialized")
    
    def _validate_config(self) -> None:
        """Validate news agent configuration"""
        # Check scraping dependencies
        try:
            import aiohttp
            import feedparser
            from bs4 import BeautifulSoup
        except ImportError as e:
            self.logger.error(f"Missing scraping dependencies: {e}")
            raise AgentConfigError(f"Required dependencies not installed: {e}")
    
    async def process_request(self, request: NewsRequest) -> NewsResponse:
        """Process comprehensive news request"""
        
        self.logger.info(f"Processing comprehensive news request for category: {request.category}")
        
        try:
            # Use comprehensive service to get all articles
            articles = await self.service.get_comprehensive_news(
                days_back=request.days_back,
                max_articles=request.limit
            )
            
            # Filter by category and state if specified
            filtered_articles = self._filter_articles(articles, request)
            
            # Create response
            message = self._generate_response_message(filtered_articles, request)
            
            response = NewsResponse(
                success=True,
                articles=filtered_articles,
                total_found=len(filtered_articles),
                message=message,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "request_params": request.dict(),
                    "sources_scraped": self._get_scraped_sources(),
                    "scraping_method": "comprehensive_deep_scraping",
                    "ai_analysis_available": bool(os.getenv('GOOGLE_API_KEY')),
                    "impact_breakdown": self._get_impact_breakdown(filtered_articles)
                }
            )
            
            self.logger.info(f"Comprehensive news fetch completed. Found {len(filtered_articles)} relevant articles")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing comprehensive news request: {e}")
            raise AgentError(f"Failed to process news request: {e}")
    
    def _filter_articles(self, articles: List[NewsArticle], request: NewsRequest) -> List[NewsArticle]:
        """Filter articles based on request criteria"""
        filtered = articles
        
        # Filter by state if specified
        if request.state:
            filtered = [
                article for article in filtered
                if not article.regions_affected or request.state in article.regions_affected
            ]
        
        # Filter by category relevance
        if request.category != "agriculture":  # Default is agriculture
            # Could add more sophisticated category filtering here
            pass
        
        return filtered[:request.limit]
    
    def _generate_response_message(self, articles: List[NewsArticle], request: NewsRequest) -> str:
        """Generate response message"""
        if not articles:
            return f"No relevant {request.category} news found through comprehensive scraping"
        
        high_impact = sum(1 for article in articles if article.impact_level == "high")
        sources = len(set(article.source for article in articles))
        
        impact_info = f" including {high_impact} high-impact stories" if high_impact > 0 else ""
        region_info = f" for {request.state}" if request.state else ""
        
        return f"Found {len(articles)} articles from {sources} sources{region_info}{impact_info}"
    
    def _get_scraped_sources(self) -> List[str]:
        """Get list of sources being scraped"""
        return [
            "Government websites (PIB, Ministry of Agriculture)",
            "Major news outlets (The Hindu, Economic Times, etc.)",
            "Agricultural publications (Krishijagran, Agriculture Today)",
            "Regional news sources",
            "RSS feeds from all sources"
        ]
    
    def _get_impact_breakdown(self, articles: List[NewsArticle]) -> Dict[str, int]:
        """Get breakdown of articles by impact level"""
        return {
            "high": sum(1 for a in articles if a.impact_level == "high"),
            "medium": sum(1 for a in articles if a.impact_level == "medium"),
            "low": sum(1 for a in articles if a.impact_level == "low")
        }
    
    def get_fallback_response(self, request: NewsRequest, error: Exception) -> NewsResponse:
        """Get fallback response when scraping fails"""
        
        fallback_articles = [
            NewsArticle(
                title="News Service Temporarily Unavailable",
                summary="Our comprehensive news scraping service is experiencing technical difficulties. Please try again later.",
                url="",
                source="System",
                published_date=datetime.now(),
                category=request.category,
                relevance_score=0.0,
                impact_level="low"
            )
        ]
        
        return NewsResponse(
            success=False,
            articles=fallback_articles,
            total_found=0,
            message=f"Comprehensive news scraping failed: {str(error)}",
            timestamp=datetime.now().isoformat(),
            metadata={"fallback": True, "error": str(error)}
        )