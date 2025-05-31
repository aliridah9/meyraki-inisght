from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import Dict, List, Any, Optional
import json

from app.models.project import (
    Project, 
    ProjectCreate, 
    ProjectUpdate, 
    FileUploadResponse, 
    ProjectObjectiveRequest
)
from app.services.project_service import project_service
from app.services.cloudinary_service import cloudinary_service
from app.services.auth_service import get_current_user
from app.utils.file_utils import convert_pdf_to_image

router = APIRouter()

@router.post("/", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new project"""
    try:
        # Set the user ID from the authenticated user
        project_data.user_id = current_user["id"]
        
        # Create the project
        project = await project_service.create_project(project_data)
        return project
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/", response_model=List[Project])
async def get_user_projects(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all projects for the authenticated user"""
    try:
        projects = await project_service.get_projects_by_user(current_user["id"])
        return projects
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a project by ID"""
    try:
        project = await project_service.get_project(project_id)
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
            
        # Check if the user owns the project
        if project.user_id != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this project"
            )
            
        return project
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.patch("/{project_id}", response_model=Project)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update a project"""
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
                detail="You don't have permission to update this project"
            )
            
        # Update the project
        updated_project = await project_service.update_project(project_id, project_data)
        return updated_project
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a project"""
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
                detail="You don't have permission to delete this project"
            )
            
        # Delete the project
        await project_service.delete_project(project_id)
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{project_id}/upload-floorplan", response_model=FileUploadResponse)
async def upload_floorplan(
    project_id: str,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload a floorplan file for a project"""
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
                detail="You don't have permission to update this project"
            )
            
        # Default values
        file_content_for_upload = None
        file_name_for_upload = file.filename.replace(" ", "_")
        resource_type_for_upload = "image" # Default for images, or after PDF conversion

        if file.filename.lower().endswith('.pdf'):
            pdf_content = await file.read()
            image_bytes = await convert_pdf_to_image(pdf_content)
            if image_bytes:
                file_content_for_upload = image_bytes
                file_name_for_upload = file.filename.rsplit('.', 1)[0].replace(" ", "_") + ".png"
                # resource_type_for_upload is already "image"
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to convert PDF to image."
                )
        else:
            # Check file extension for non-PDFs
            file_ext = file.filename.split(".")[-1].lower()
            if file_ext not in ["jpg", "jpeg", "png", "dwg"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must be an image (JPG, PNG), PDF, or DWG"
                )
            
            file_content_for_upload = await file.read()
            # For DWG, resource_type might be 'raw' or 'auto' depending on Cloudinary settings
            if file_ext == "dwg":
                resource_type_for_upload = "raw"


        if not file_content_for_upload:
             # This case should ideally not be reached if logic is correct,
             # but as a safeguard / if file.read() was empty for some reason.
            await file.seek(0) # Reset pointer just in case it was read partially
            file_content_for_upload = await file.read()
            if not file_content_for_upload:
                 raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File content is empty or could not be read."
                )

        # Upload to Cloudinary
        folder = f"meyraki/floorplans/{project_id}"
        
        cloud_response = await cloudinary_service.upload_file(
            file_content_for_upload,
            folder,
            file_name_for_upload,
            resource_type=resource_type_for_upload
        )
        
        # Update project with floorplan
        # Ensure add_floorplan can handle the correct filename and type (e.g. if PDF became PNG)
        # The cloud_response should contain the necessary details like the new URL and format.
        floorplan = await project_service.add_floorplan(project_id, cloud_response)
        
        # Return the response
        return FileUploadResponse(
            file_id=floorplan.file_id,
            filename=floorplan.filename,
            url=floorplan.url,
            file_type=floorplan.file_type,
            created_at=floorplan.created_at,
            size=floorplan.size,
            width=floorplan.width,
            height=floorplan.height
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{project_id}/upload-usage-data", response_model=FileUploadResponse)
async def upload_usage_data(
    project_id: str,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload usage data CSV for a project"""
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
                detail="You don't have permission to update this project"
            )
            
        # Check file extension
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext != "csv":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a CSV file"
            )
            
        # Read file content
        file_content = await file.read()
        
        # Upload to Cloudinary
        folder = f"meyraki/usage_data/{project_id}"
        file_name = file.filename.replace(" ", "_")
        
        cloud_response = await cloudinary_service.upload_file(file_content, folder, file_name, "raw")
        
        # Calculate basic metadata about the CSV (simple approach for MVP)
        content_str = file_content.decode("utf-8")
        lines = content_str.strip().split("\n")
        row_count = len(lines) - 1  # excluding header
        column_count = len(lines[0].split(",")) if lines else 0
        
        # Update project with usage data
        metadata = {
            "row_count": row_count,
            "column_count": column_count
        }
        
        usage_data = await project_service.add_usage_data(project_id, cloud_response, metadata)
        
        # Return the response
        return FileUploadResponse(
            file_id=usage_data.file_id,
            filename=usage_data.filename,
            url=usage_data.url,
            file_type="csv",
            created_at=usage_data.created_at,
            size=usage_data.size,
            width=None,
            height=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{project_id}/set-objectives", response_model=Project)
async def set_objectives(
    project_id: str,
    objectives: ProjectObjectiveRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Set objectives for a project"""
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
                detail="You don't have permission to update this project"
            )
            
        # Update project with objectives
        update_data = ProjectUpdate(
            space_type=objectives.space_type,
            objectives=objectives.objectives,
            custom_objectives=objectives.custom_objectives,
            additional_notes=objectives.additional_notes
        )
        
        # Update the project
        updated_project = await project_service.update_project(project_id, update_data)
        return updated_project
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 