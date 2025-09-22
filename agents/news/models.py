# server/agents/news/models.py
"""
Pydantic models for news agent
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class NewsRequest(BaseModel):
    category: Optional[str] = Field("agriculture", description="News category (agriculture, government, policy, subsidies)")
    state: Optional[str] = Field(None, description="Specific state for regional news")
    language: Optional[str] = Field("english", description="Language preference")
    limit: int = Field(10, ge=1, le=50, description="Number of articles to fetch")
    days_back: int = Field(7, ge=1, le=30, description="Number of days to look back")

class NewsArticle(BaseModel):
    title: str = Field(..., description="Article title")
    summary: str = Field(..., description="Brief summary of the article")
    content: Optional[str] = Field(None, description="Full article content")
    url: str = Field(..., description="Original article URL")
    source: str = Field(..., description="News source name")
    published_date: datetime = Field(..., description="Publication date")
    category: str = Field(..., description="News category")
    tags: List[str] = Field(default_factory=list, description="Related tags")
    relevance_score: float = Field(0.0, ge=0, le=1, description="Relevance to farmers")
    impact_level: str = Field("medium", description="Impact level: low, medium, high")
    regions_affected: List[str] = Field(default_factory=list, description="Affected regions/states")

class NewsDetail(BaseModel):
    article: NewsArticle
    key_points: List[str] = Field(..., description="Key takeaways")
    impact_analysis: str = Field(..., description="Impact on farmers")
    action_items: List[str] = Field(default_factory=list, description="What farmers should do")
    related_schemes: List[str] = Field(default_factory=list, description="Related government schemes")
    deadlines: List[Dict[str, str]] = Field(default_factory=list, description="Important dates and deadlines")

class NewsResponse(BaseModel):
    success: bool
    articles: List[NewsArticle]
    total_found: int
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class NewsDetailResponse(BaseModel):
    success: bool
    data: Optional[NewsDetail] = None
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None