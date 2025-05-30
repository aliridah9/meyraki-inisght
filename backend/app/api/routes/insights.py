from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Any, Optional

from app.services.analysis_service import analysis_service
from app.services.project_service import project_service
from app.services.auth_service import get_current_user

router = APIRouter()

@router.post("/{project_id}/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_insights(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate insights for a project.
    This process may take some time to complete.
    """
    try:
        # Check if the project exists and user owns it
        existing_project = await project_service.get_project(project_id)
        
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        if existing_project.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this project"
            )
            
        # Check if project has required data
        if not existing_project.floorplan:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Project must have a floorplan uploaded"
            )
            
        # Generate insights
        result = await analysis_service.generate_insights(project_id)
        
        return {
            "message": "Insights generated successfully",
            "heatmap_url": result["heatmap_url"],
            "report_url": result["report_url"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{project_id}", status_code=status.HTTP_200_OK)
async def get_insights(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get existing insights for a project"""
    try:
        # Check if the project exists and user owns it
        existing_project = await project_service.get_project(project_id)
        
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        if existing_project.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this project"
            )
            
        # Check if insights exist
        if not existing_project.insight_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No insights generated for this project yet"
            )
            
        return {
            "heatmap_url": existing_project.insight_result.heatmap_url,
            "recommendations": existing_project.insight_result.recommendations,
            "report_url": existing_project.insight_result.report_url,
            "created_at": existing_project.insight_result.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{project_id}/generate-moodboard", status_code=status.HTTP_200_OK)
async def generate_moodboard(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate a moodboard for a project"""
    try:
        # Check if the project exists and user owns it
        existing_project = await project_service.get_project(project_id)
        
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        if existing_project.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this project"
            )
            
        # Generate moodboard
        result = await analysis_service.generate_moodboard(project_id)
        
        return {
            "moodboard_url": result["moodboard_url"],
            "style_description": result["style_description"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{project_id}/download-report", status_code=status.HTTP_200_OK)
async def download_report(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a download URL for the project report"""
    try:
        # Check if the project exists and user owns it
        existing_project = await project_service.get_project(project_id)
        
        if not existing_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        if existing_project.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this project"
            )
            
        # Check if insights exist with a report
        if (not existing_project.insight_result or 
            not existing_project.insight_result.report_url):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No report available for this project"
            )
            
        # For MVP, just return the direct URL
        # In a production app, we might generate a signed URL with an expiration
        return {
            "download_url": existing_project.insight_result.report_url
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 