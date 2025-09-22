# server/api/v1/endpoints/news.py
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List
import sys
import os

# Add the server directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.base import agent_registry

router = APIRouter()

@router.get("/all-articles")
async def get_all_agricultural_news(
    days_back: int = Query(7, ge=1, le=30, description="Number of days to look back"),
    max_articles: int = Query(50, ge=10, le=200, description="Maximum number of articles to fetch"),
    min_relevance: float = Query(0.3, ge=0.0, le=1.0, description="Minimum relevance score (0-1)")
):
    """
    Get comprehensive agricultural news from all possible sources
    
    This endpoint performs deep web scraping across:
    - Government websites (PIB, Ministry of Agriculture, etc.)
    - Major news outlets (The Hindu, Economic Times, etc.) 
    - Agricultural publications (Krishijagran, Agriculture Today)
    - Regional news sources
    - RSS feeds from all sources
    
    Returns articles ranked by relevance to farmers with impact assessment.
    """
    try:
        # Get the news agent
        news_agent = agent_registry.get("news")
        if not news_agent:
            raise HTTPException(status_code=500, detail="News agent not available")
        
        # Get comprehensive news using the robust service
        from agents.news.robust_service import RobustNewsService
        
        service = RobustNewsService(config={})
        articles = await service.get_comprehensive_news(
            days_back=days_back,
            max_articles=max_articles
        )
        
        # Filter by relevance
        filtered_articles = [
            article for article in articles 
            if article.relevance_score >= min_relevance
        ]
        
        # Group articles by impact level for better organization
        high_impact = [a for a in filtered_articles if a.impact_level == "high"]
        medium_impact = [a for a in filtered_articles if a.impact_level == "medium"]
        low_impact = [a for a in filtered_articles if a.impact_level == "low"]
        
        return {
            "success": True,
            "total_articles": len(filtered_articles),
            "articles": {
                "high_impact": [article.dict() for article in high_impact],
                "medium_impact": [article.dict() for article in medium_impact],
                "low_impact": [article.dict() for article in low_impact]
            },
            "sources_scraped": [
                "Government websites", "Major news outlets", "Agricultural publications", 
                "Regional news", "RSS feeds"
            ],
            "summary": {
                "high_impact_count": len(high_impact),
                "medium_impact_count": len(medium_impact),
                "low_impact_count": len(low_impact),
                "date_range": f"Last {days_back} days",
                "relevance_threshold": min_relevance
            },
            "timestamp": "2025-09-21T13:50:48.531643Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching comprehensive news: {str(e)}")

@router.post("/elaborate-articles")
async def elaborate_selected_articles(
    article_data: dict = Body(..., description="Selected articles data with IDs or URLs")
):
    """
    Get detailed elaboration and analysis for selected news articles
    
    Provide AI-powered detailed analysis including:
    - Key points in simple farmer-friendly language
    - Impact analysis for farmers
    - Specific action items to take
    - Related government schemes
    - Important deadlines
    - Economic implications
    - Regional impact assessment
    
    Send selected articles in request body:
    {
        "articles": [
            {"url": "article_url", "title": "article_title", ...},
            ...
        ],
        "analysis_depth": "basic|detailed|comprehensive"
    }
    """
    try:
        # Get the news agent
        news_agent = agent_registry.get("news")
        if not news_agent:
            raise HTTPException(status_code=500, detail="News agent not available")
        
        # Extract articles from request
        articles_to_elaborate = article_data.get("articles", [])
        analysis_depth = article_data.get("analysis_depth", "detailed")
        
        if not articles_to_elaborate:
            raise HTTPException(status_code=400, detail="No articles provided for elaboration")
        
        # Convert to NewsArticle objects
        from agents.news.models import NewsArticle
        from datetime import datetime
        
        news_articles = []
        for article_data in articles_to_elaborate:
            article = NewsArticle(
                title=article_data.get("title", ""),
                summary=article_data.get("summary", ""),
                url=article_data.get("url", ""),
                source=article_data.get("source", "Unknown"),
                published_date=datetime.fromisoformat(article_data.get("published_date", datetime.now().isoformat())),
                category=article_data.get("category", "agriculture"),
                relevance_score=article_data.get("relevance_score", 0.8),
                impact_level=article_data.get("impact_level", "medium"),
                regions_affected=article_data.get("regions_affected", []),
                tags=article_data.get("tags", [])
            )
            news_articles.append(article)
        
        # Get detailed elaboration
        from agents.news.robust_service import RobustNewsService
        
        service = RobustNewsService(config={})
        elaborated_articles = await service.elaborate_articles(
            article_ids=[],  # Not used in current implementation
            articles=news_articles
        )
        
        # Format response
        detailed_analysis = []
        for detail in elaborated_articles:
            detailed_analysis.append({
                "article": detail.article.dict(),
                "analysis": {
                    "key_points": detail.key_points,
                    "impact_analysis": detail.impact_analysis,
                    "action_items": detail.action_items,
                    "related_schemes": detail.related_schemes,
                    "deadlines": detail.deadlines
                }
            })
        
        return {
            "success": True,
            "elaborated_articles": detailed_analysis,
            "analysis_depth": analysis_depth,
            "total_articles_analyzed": len(elaborated_articles),
            "ai_analysis_available": bool(os.getenv('GOOGLE_API_KEY')),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error elaborating articles: {str(e)}")

@router.get("/quick-summary")
async def get_quick_news_summary(
    days_back: int = Query(3, ge=1, le=7, description="Days to summarize"),
    max_articles: int = Query(20, ge=5, le=50, description="Max articles to analyze")
):
    """
    Get a quick AI-generated summary of top agricultural news
    
    Returns a concise summary of the most important agricultural news
    for farmers who want a quick overview without reading individual articles.
    """
    try:
        # Get comprehensive news
        from agents.news.robust_service import RobustNewsService
        
        service = RobustNewsService(config={})
        articles = await service.get_comprehensive_news(
            days_back=days_back,
            max_articles=max_articles
        )
        
        if not articles:
            return {
                "success": True,
                "summary": "No significant agricultural news found in the specified period.",
                "article_count": 0,
                "period": f"Last {days_back} days"
            }
        
        # Generate AI summary if available
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            summary = await service._generate_news_summary(articles, days_back)
        else:
            # Fallback summary
            high_impact = [a for a in articles if a.impact_level == "high"]
            summary = f"Found {len(articles)} articles in the last {days_back} days. {len(high_impact)} high-impact stories including government announcements and policy changes."
        
        # Top headlines
        top_headlines = [
            {
                "title": article.title,
                "source": article.source,
                "impact": article.impact_level,
                "url": article.url
            }
            for article in articles[:5]
        ]
        
        return {
            "success": True,
            "summary": summary,
            "top_headlines": top_headlines,
            "article_count": len(articles),
            "period": f"Last {days_back} days",
            "impact_breakdown": {
                "high": len([a for a in articles if a.impact_level == "high"]),
                "medium": len([a for a in articles if a.impact_level == "medium"]),
                "low": len([a for a in articles if a.impact_level == "low"])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating news summary: {str(e)}")

@router.get("/health")
async def news_health():
    """Check news scraping service health"""
    try:
        # Check if scraping dependencies are available
        try:
            import aiohttp
            import feedparser
            from bs4 import BeautifulSoup
            dependencies_ok = True
        except ImportError as e:
            dependencies_ok = False
            missing_deps = str(e)
        
        google_api_available = bool(os.getenv('GOOGLE_API_KEY'))
        
        # Test a simple scraping operation
        scraping_test = True
        try:
            import asyncio
            from agents.news.comprehensive_service import ComprehensiveNewsService
            # Quick test of service initialization
            service = ComprehensiveNewsService(config={})
            test_result = "Service initialized successfully"
        except Exception as e:
            scraping_test = False
            test_result = f"Service test failed: {str(e)}"
        
        health_status = "healthy" if dependencies_ok and scraping_test else "unhealthy"
        
        return {
            "status": health_status,
            "dependencies_available": dependencies_ok,
            "google_api_configured": google_api_available,
            "scraping_capability": scraping_test,
            "test_result": test_result,
            "capabilities": {
                "deep_web_scraping": dependencies_ok,
                "ai_analysis": google_api_available,
                "rss_aggregation": dependencies_ok,
                "content_elaboration": google_api_available
            },
            "sources_monitored": [
                "Government websites (PIB, Ministry of Agriculture)",
                "Major news outlets (The Hindu, ET, etc.)",
                "Agricultural publications",
                "Regional news sources",
                "RSS feeds"
            ]
        }
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}