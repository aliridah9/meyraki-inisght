from fastapi import APIRouter

from app.api.routes import projects, insights

api_router = APIRouter()

api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(insights.router, prefix="/insights", tags=["insights"]) 