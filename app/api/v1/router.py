from fastapi import APIRouter
from app.api.v1.endpoints import prediction

api_router = APIRouter()

# We include the prediction router. 
# In a larger app, we would add prefixes like /users, /auth here.
api_router.include_router(prediction.router, tags=["prediction"])