# server/api/v1/router.py
from fastapi import APIRouter
from .endpoints import health, mandi, irrigation

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(mandi.router, prefix="/mandi", tags=["mandi"])
api_router.include_router(irrigation.router, prefix="/irrigation", tags=["irrigation"])