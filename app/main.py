from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend for Saturation Vapor Pressure Estimation",
    version=settings.VERSION
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Router Registration ---
# We mount the API router at the root level to maintain compatibility
# with the frontend which calls '/predict' directly.
app.include_router(api_router)

@app.get("/")
async def root():
    return {"status": "System Operational", "service": settings.PROJECT_NAME}