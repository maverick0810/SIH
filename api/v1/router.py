# server/api/v1/router.py
from fastapi import APIRouter
from .endpoints import health, mandi, irrigation, disease, news

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(mandi.router, prefix="/mandi", tags=["mandi"])
api_router.include_router(irrigation.router, prefix="/irrigation", tags=["irrigation"])
api_router.include_router(disease.router, prefix="/disease", tags=["disease"])
api_router.include_router(news.router, prefix="/news", tags=["news"])