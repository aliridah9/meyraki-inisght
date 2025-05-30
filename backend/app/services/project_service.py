from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from app.db.supabase import supabase_client
from app.models.project import (
    ProjectCreate,
    ProjectUpdate,
    Project,
    FloorplanFile,
    UsageDataFile,
    InsightResult,
    MoodboardResult
)


class ProjectService:
    @staticmethod
    async def create_project(project_data: ProjectCreate) -> Project:
        """
        Create a new project in the database
        
        Args:
            project_data: Project create data
            
        Returns:
            Created project data
        """
        now = datetime.utcnow().isoformat()
        project_id = str(uuid.uuid4())
        
        project_dict = project_data.dict()
        project_dict.update({
            "id": project_id,
            "status": "draft",
            "created_at": now,
            "updated_at": now,
            "objectives": [obj.value for obj in project_data.objectives]  # Convert enum to string
        })
        
        try:
            result = supabase_client.table("projects").insert(project_dict).execute()
            inserted = result.data[0]
            
            # Convert back to our Pydantic model
            return Project(**inserted)
        except Exception as e:
            raise Exception(f"Failed to create project: {str(e)}")
    
    @staticmethod
    async def get_project(project_id: str) -> Optional[Project]:
        """
        Get a project by ID
        
        Args:
            project_id: Project ID
            
        Returns:
            Project data or None if not found
        """
        try:
            result = supabase_client.table("projects").select("*").eq("id", project_id).execute()
            
            if not result.data:
                return None
                
            project_data = result.data[0]
            
            # Get related files and results
            floorplan = await ProjectService._get_floorplan(project_id)
            usage_data = await ProjectService._get_usage_data(project_id)
            insight_result = await ProjectService._get_insight_result(project_id)
            moodboard_result = await ProjectService._get_moodboard_result(project_id)
            
            # Add to project data
            project_data.update({
                "floorplan": floorplan,
                "usage_data": usage_data,
                "insight_result": insight_result,
                "moodboard_result": moodboard_result
            })
            
            return Project(**project_data)
        except Exception as e:
            raise Exception(f"Failed to get project: {str(e)}")
    
    @staticmethod
    async def get_projects_by_user(user_id: str) -> List[Project]:
        """
        Get all projects for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of projects
        """
        try:
            result = supabase_client.table("projects").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            
            projects = []
            for project_data in result.data:
                project_id = project_data["id"]
                
                # Get related data
                floorplan = await ProjectService._get_floorplan(project_id)
                project_data["floorplan"] = floorplan
                
                projects.append(Project(**project_data))
                
            return projects
        except Exception as e:
            raise Exception(f"Failed to get user projects: {str(e)}")
    
    @staticmethod
    async def update_project(project_id: str, project_data: ProjectUpdate) -> Project:
        """
        Update a project
        
        Args:
            project_id: Project ID
            project_data: Project update data
            
        Returns:
            Updated project data
        """
        try:
            update_dict = project_data.dict(exclude_unset=True)
            update_dict["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert enum values to strings if present
            if "objectives" in update_dict and update_dict["objectives"]:
                update_dict["objectives"] = [obj.value for obj in update_dict["objectives"]]
            
            result = supabase_client.table("projects").update(update_dict).eq("id", project_id).execute()
            
            if not result.data:
                raise Exception(f"Project with ID {project_id} not found")
                
            return await ProjectService.get_project(project_id)
        except Exception as e:
            raise Exception(f"Failed to update project: {str(e)}")
    
    @staticmethod
    async def delete_project(project_id: str) -> bool:
        """
        Delete a project
        
        Args:
            project_id: Project ID
            
        Returns:
            True if successful
        """
        try:
            # Delete related files and results
            await ProjectService._delete_floorplan(project_id)
            await ProjectService._delete_usage_data(project_id)
            await ProjectService._delete_insight_result(project_id)
            await ProjectService._delete_moodboard_result(project_id)
            
            # Delete project
            result = supabase_client.table("projects").delete().eq("id", project_id).execute()
            
            return True
        except Exception as e:
            raise Exception(f"Failed to delete project: {str(e)}")
    
    @staticmethod
    async def add_floorplan(project_id: str, floorplan_data: Dict[str, Any]) -> FloorplanFile:
        """
        Add or update a floorplan file for a project
        
        Args:
            project_id: Project ID
            floorplan_data: Floorplan data from Cloudinary
            
        Returns:
            FloorplanFile object
        """
        try:
            floorplan_dict = {
                "project_id": project_id,
                "file_id": floorplan_data["public_id"],
                "filename": floorplan_data.get("original_filename", "floorplan"),
                "url": floorplan_data["secure_url"],
                "file_type": floorplan_data.get("format", "unknown"),
                "created_at": datetime.utcnow().isoformat(),
                "size": floorplan_data.get("bytes", 0),
                "width": floorplan_data.get("width"),
                "height": floorplan_data.get("height")
            }
            
            # Check if floorplan exists
            exist_result = supabase_client.table("floorplans").select("*").eq("project_id", project_id).execute()
            
            if exist_result.data:
                # Update existing
                result = supabase_client.table("floorplans").update(floorplan_dict).eq("project_id", project_id).execute()
            else:
                # Insert new
                result = supabase_client.table("floorplans").insert(floorplan_dict).execute()
            
            # Update project status
            await ProjectService._update_project_status(project_id)
            
            return FloorplanFile(**floorplan_dict)
        except Exception as e:
            raise Exception(f"Failed to add floorplan: {str(e)}")
    
    @staticmethod
    async def add_usage_data(project_id: str, usage_data: Dict[str, Any], metadata: Dict[str, Any]) -> UsageDataFile:
        """
        Add or update usage data for a project
        
        Args:
            project_id: Project ID
            usage_data: Usage data from Cloudinary
            metadata: Additional metadata about the file
            
        Returns:
            UsageDataFile object
        """
        try:
            data_dict = {
                "project_id": project_id,
                "file_id": usage_data["public_id"],
                "filename": usage_data.get("original_filename", "usage_data.csv"),
                "url": usage_data["secure_url"],
                "created_at": datetime.utcnow().isoformat(),
                "size": usage_data.get("bytes", 0),
                "row_count": metadata.get("row_count"),
                "column_count": metadata.get("column_count")
            }
            
            # Check if usage data exists
            exist_result = supabase_client.table("usage_data").select("*").eq("project_id", project_id).execute()
            
            if exist_result.data:
                # Update existing
                result = supabase_client.table("usage_data").update(data_dict).eq("project_id", project_id).execute()
            else:
                # Insert new
                result = supabase_client.table("usage_data").insert(data_dict).execute()
                
            # Update project status
            await ProjectService._update_project_status(project_id)
            
            return UsageDataFile(**data_dict)
        except Exception as e:
            raise Exception(f"Failed to add usage data: {str(e)}")
    
    @staticmethod
    async def add_insight_result(project_id: str, insight_data: Dict[str, Any]) -> InsightResult:
        """
        Add or update insight results for a project
        
        Args:
            project_id: Project ID
            insight_data: Insight data
            
        Returns:
            InsightResult object
        """
        try:
            now = datetime.utcnow().isoformat()
            result_dict = {
                "project_id": project_id,
                "heatmap_url": insight_data.get("heatmap_url"),
                "recommendations": insight_data.get("recommendations", {}),
                "report_url": insight_data.get("report_url"),
                "created_at": now,
                "updated_at": now
            }
            
            # Check if insight result exists
            exist_result = supabase_client.table("insight_results").select("*").eq("project_id", project_id).execute()
            
            if exist_result.data:
                # Update existing
                result = supabase_client.table("insight_results").update(result_dict).eq("project_id", project_id).execute()
            else:
                # Insert new
                result = supabase_client.table("insight_results").insert(result_dict).execute()
                
            # Update project status to "completed"
            await supabase_client.table("projects").update({"status": "completed", "updated_at": now}).eq("id", project_id).execute()
            
            return InsightResult(**result_dict)
        except Exception as e:
            raise Exception(f"Failed to add insight result: {str(e)}")
    
    @staticmethod
    async def add_moodboard_result(project_id: str, moodboard_data: Dict[str, Any]) -> MoodboardResult:
        """
        Add or update moodboard result for a project
        
        Args:
            project_id: Project ID
            moodboard_data: Moodboard data
            
        Returns:
            MoodboardResult object
        """
        try:
            now = datetime.utcnow().isoformat()
            moodboard_dict = {
                "project_id": project_id,
                "moodboard_url": moodboard_data["moodboard_url"],
                "style_description": moodboard_data.get("style_description", ""),
                "created_at": now
            }
            
            # Check if moodboard result exists
            exist_result = supabase_client.table("moodboard_results").select("*").eq("project_id", project_id).execute()
            
            if exist_result.data:
                # Update existing
                result = supabase_client.table("moodboard_results").update(moodboard_dict).eq("project_id", project_id).execute()
            else:
                # Insert new
                result = supabase_client.table("moodboard_results").insert(moodboard_dict).execute()
                
            return MoodboardResult(**moodboard_dict)
        except Exception as e:
            raise Exception(f"Failed to add moodboard result: {str(e)}")
    
    # Private helper methods
    @staticmethod
    async def _get_floorplan(project_id: str) -> Optional[FloorplanFile]:
        try:
            result = supabase_client.table("floorplans").select("*").eq("project_id", project_id).execute()
            
            if not result.data:
                return None
                
            return FloorplanFile(**result.data[0])
        except Exception:
            return None
    
    @staticmethod
    async def _get_usage_data(project_id: str) -> Optional[UsageDataFile]:
        try:
            result = supabase_client.table("usage_data").select("*").eq("project_id", project_id).execute()
            
            if not result.data:
                return None
                
            return UsageDataFile(**result.data[0])
        except Exception:
            return None
    
    @staticmethod
    async def _get_insight_result(project_id: str) -> Optional[InsightResult]:
        try:
            result = supabase_client.table("insight_results").select("*").eq("project_id", project_id).execute()
            
            if not result.data:
                return None
                
            return InsightResult(**result.data[0])
        except Exception:
            return None
    
    @staticmethod
    async def _get_moodboard_result(project_id: str) -> Optional[MoodboardResult]:
        try:
            result = supabase_client.table("moodboard_results").select("*").eq("project_id", project_id).execute()
            
            if not result.data:
                return None
                
            return MoodboardResult(**result.data[0])
        except Exception:
            return None
    
    @staticmethod
    async def _delete_floorplan(project_id: str) -> bool:
        try:
            supabase_client.table("floorplans").delete().eq("project_id", project_id).execute()
            return True
        except Exception:
            return False
    
    @staticmethod
    async def _delete_usage_data(project_id: str) -> bool:
        try:
            supabase_client.table("usage_data").delete().eq("project_id", project_id).execute()
            return True
        except Exception:
            return False
    
    @staticmethod
    async def _delete_insight_result(project_id: str) -> bool:
        try:
            supabase_client.table("insight_results").delete().eq("project_id", project_id).execute()
            return True
        except Exception:
            return False
    
    @staticmethod
    async def _delete_moodboard_result(project_id: str) -> bool:
        try:
            supabase_client.table("moodboard_results").delete().eq("project_id", project_id).execute()
            return True
        except Exception:
            return False
    
    @staticmethod
    async def _update_project_status(project_id: str) -> None:
        """Update project status based on uploaded files"""
        try:
            floorplan = await ProjectService._get_floorplan(project_id)
            usage_data = await ProjectService._get_usage_data(project_id)
            
            now = datetime.utcnow().isoformat()
            status = "draft"
            
            if floorplan and usage_data:
                status = "ready_for_analysis"
            elif floorplan or usage_data:
                status = "in_progress"
                
            await supabase_client.table("projects").update({"status": status, "updated_at": now}).eq("id", project_id).execute()
        except Exception:
            pass


project_service = ProjectService() 