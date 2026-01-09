from fastapi import APIRouter
from app.api.v1.endpoints import prediction

api_router = APIRouter()


api_router.include_router(prediction.router, tags=["prediction"])