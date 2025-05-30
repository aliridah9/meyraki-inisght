from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

# from app.db.supabase import supabase_client # Removed
from app.db.mongodb import get_database # Added
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
        now = datetime.utcnow() # Use datetime object
        project_id = str(uuid.uuid4())
        
        project_dict = project_data.model_dump() # Pydantic V2
        project_dict.update({
            "id": project_id,    # Keep our 'id' field consistent
            "status": "draft",
            "created_at": now,
            "updated_at": now,
            "objectives": [obj.value for obj in project_data.objectives]
        })
        
        db = get_database()
        try:
            await db["projects"].insert_one(project_dict)
            # Assuming project_dict is correct and matches Project model structure
            # for returning. The Project model expects datetime objects.
            return Project(**project_dict)
        except Exception as e:
            # Log error
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
        db = get_database()
        try:
            project_data = await db["projects"].find_one({"id": project_id})
            
            if not project_data:
                return None
            
            # Related data fetching (floorplan, usage_data, etc.) will be updated later.
            floorplan = await ProjectService._get_floorplan(project_id) # Placeholder
            usage_data = await ProjectService._get_usage_data(project_id) # Placeholder
            insight_result = await ProjectService._get_insight_result(project_id) # Placeholder
            moodboard_result = await ProjectService._get_moodboard_result(project_id) # Placeholder
            
            project_data.update({
                "floorplan": floorplan,
                "usage_data": usage_data,
                "insight_result": insight_result,
                "moodboard_result": moodboard_result
            })
            
            return Project(**project_data)
        except Exception as e:
            # Log error
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
        db = get_database()
        try:
            # MongoDB sort: 1 for ascending, -1 for descending
            cursor = db["projects"].find({"user_id": user_id}).sort("created_at", -1)
            projects_data = await cursor.to_list(length=None) # Fetch all matching documents
            
            projects = []
            for project_data_item in projects_data: # renamed to avoid conflict
                project_id = project_data_item["id"]
                
                # Related data fetching (e.g., floorplan) will be updated later.
                floorplan = await ProjectService._get_floorplan(project_id) # Placeholder
                project_data_item["floorplan"] = floorplan
                # Other related data (usage_data, etc.) also needs to be fetched if included in original
                # For now, only floorplan was explicitly mentioned in original snippet for this loop
                
                projects.append(Project(**project_data_item))
                
            return projects
        except Exception as e:
            # Log error
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
        db = get_database()
        try:
            update_dict = project_data.model_dump(exclude_unset=True) # Pydantic V2
            update_dict["updated_at"] = datetime.utcnow() # Use datetime object
            
            if "objectives" in update_dict and update_dict["objectives"]:
                update_dict["objectives"] = [obj.value for obj in update_dict["objectives"]]
            
            result = await db["projects"].update_one(
                {"id": project_id},
                {"$set": update_dict}
            )
            
            if result.matched_count == 0:
                raise Exception(f"Project with ID {project_id} not found for update")
            # if result.modified_count == 0:
                # Consider logging if no modification occurred but matched.
                # pass
                
            updated_project = await ProjectService.get_project(project_id)
            if not updated_project:
                raise Exception(f"Failed to retrieve project {project_id} after update.")
            return updated_project
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
        db = get_database()
        try:
            # Delete related files and results (these methods will be migrated)
            await ProjectService._delete_floorplan(project_id)
            await ProjectService._delete_usage_data(project_id)
            await ProjectService._delete_insight_result(project_id)
            await ProjectService._delete_moodboard_result(project_id)
            
            result = await db["projects"].delete_one({"id": project_id})
            
            if result.deleted_count == 0:
                # Optionally raise an error or log if project not found for deletion
                pass
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
        db = get_database()
        try:
            floorplan_dict = {
                "project_id": project_id,
                "file_id": floorplan_data["public_id"],
                "filename": floorplan_data.get("original_filename", "floorplan"),
                "url": floorplan_data["secure_url"],
                "file_type": floorplan_data.get("format", "unknown"),
                "created_at": datetime.utcnow(), # Use datetime object
                "size": floorplan_data.get("bytes", 0),
                "width": floorplan_data.get("width"),
                "height": floorplan_data.get("height")
            }
            
            # Upsert: update if exists, insert if not
            await db["floorplans"].update_one(
                {"project_id": project_id},
                {"$set": floorplan_dict},
                upsert=True
            )
            
            await ProjectService._update_project_status(project_id) # This method also needs migration
            return FloorplanFile(**floorplan_dict)
        except Exception as e:
            raise Exception(f"Failed to add floorplan: {str(e)}")
    
    @staticmethod
    async def add_usage_data(project_id: str, usage_data: Dict[str, Any], metadata: Dict[str, Any]) -> UsageDataFile: # Parameter name 'usage_data' kept as is
        """
        Add or update usage data for a project
        
        Args:
            project_id: Project ID
            usage_data: Usage data from Cloudinary
            metadata: Additional metadata about the file
            
        Returns:
            UsageDataFile object
        """
        db = get_database()
        try:
            data_dict = {
                "project_id": project_id,
                "file_id": usage_data["public_id"], # 'usage_data' is the parameter name
                "filename": usage_data.get("original_filename", "usage_data.csv"),
                "url": usage_data["secure_url"],
                "created_at": datetime.utcnow(), # Use datetime object
                "size": usage_data.get("bytes", 0),
                "row_count": metadata.get("row_count"),
                "column_count": metadata.get("column_count")
            }
            
            await db["usage_data_files"].update_one( # Changed collection name
                {"project_id": project_id},
                {"$set": data_dict},
                upsert=True
            )
            
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
        db = get_database()
        try:
            now = datetime.utcnow() # Use datetime object
            result_dict = {
                "project_id": project_id,
                "heatmap_url": insight_data.get("heatmap_url"),
                "recommendations": insight_data.get("recommendations", {}),
                "report_url": insight_data.get("report_url"),
                "created_at": now,
                "updated_at": now
            }
            
            await db["insight_results"].update_one(
                {"project_id": project_id},
                {"$set": result_dict},
                upsert=True
            )
            
            await db["projects"].update_one(
                {"id": project_id},
                {"$set": {"status": "completed", "updated_at": now}}
            )
            
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
        db = get_database()
        try:
            now = datetime.utcnow() # Use datetime object
            moodboard_dict = {
                "project_id": project_id,
                "moodboard_url": moodboard_data["moodboard_url"],
                "style_description": moodboard_data.get("style_description", ""),
                "created_at": now
            }
            
            await db["moodboard_results"].update_one(
                {"project_id": project_id},
                {"$set": moodboard_dict},
                upsert=True
            )
            return MoodboardResult(**moodboard_dict)
        except Exception as e:
            raise Exception(f"Failed to add moodboard result: {str(e)}")
    
    # Private helper methods
    @staticmethod
    async def _get_floorplan(project_id: str) -> Optional[FloorplanFile]:
        db = get_database()
        try:
            result = await db["floorplans"].find_one({"project_id": project_id})
            if not result:
                return None
            return FloorplanFile(**result)
        except Exception: # Log error
            return None
    
    @staticmethod
    async def _get_usage_data(project_id: str) -> Optional[UsageDataFile]:
        db = get_database()
        try:
            result = await db["usage_data_files"].find_one({"project_id": project_id}) # Changed collection name
            if not result:
                return None
            return UsageDataFile(**result)
        except Exception: # Log error
            return None
    
    @staticmethod
    async def _get_insight_result(project_id: str) -> Optional[InsightResult]:
        db = get_database()
        try:
            result = await db["insight_results"].find_one({"project_id": project_id})
            if not result:
                return None
            return InsightResult(**result)
        except Exception: # Log error
            return None
    
    @staticmethod
    async def _get_moodboard_result(project_id: str) -> Optional[MoodboardResult]:
        db = get_database()
        try:
            result = await db["moodboard_results"].find_one({"project_id": project_id})
            if not result:
                return None
            return MoodboardResult(**result)
        except Exception: # Log error
            return None
    
    @staticmethod
    async def _delete_floorplan(project_id: str) -> bool:
        db = get_database()
        try:
            await db["floorplans"].delete_one({"project_id": project_id})
            return True
        except Exception: # Log error
            return False
    
    @staticmethod
    async def _delete_usage_data(project_id: str) -> bool:
        db = get_database()
        try:
            await db["usage_data_files"].delete_one({"project_id": project_id}) # Changed collection name
            return True
        except Exception: # Log error
            return False
    
    @staticmethod
    async def _delete_insight_result(project_id: str) -> bool:
        db = get_database()
        try:
            await db["insight_results"].delete_one({"project_id": project_id})
            return True
        except Exception: # Log error
            return False
    
    @staticmethod
    async def _delete_moodboard_result(project_id: str) -> bool:
        db = get_database()
        try:
            await db["moodboard_results"].delete_one({"project_id": project_id})
            return True
        except Exception: # Log error
            return False
    
    @staticmethod
    async def _update_project_status(project_id: str) -> None:
        """Update project status based on uploaded files"""
        db = get_database()
        try:
            floorplan = await ProjectService._get_floorplan(project_id)
            usage_data = await ProjectService._get_usage_data(project_id)
            
            now = datetime.utcnow() # Use datetime object
            status = "draft"
            
            if floorplan and usage_data:
                status = "ready_for_analysis"
            elif floorplan or usage_data:
                status = "in_progress"
                
            await db["projects"].update_one(
                {"id": project_id},
                {"$set": {"status": status, "updated_at": now}}
            )
        except Exception: # Log error
            pass # Original method also suppresses exceptions here


project_service = ProjectService() 