from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class SpaceType(str, Enum):
    HOTEL = "hotel"
    OFFICE = "office"
    CAFE = "cafe"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    RESIDENTIAL = "residential"
    OTHER = "other"


class ObjectiveType(str, Enum):
    FLOW = "flow"
    AMBIANCE = "ambiance"
    ZONING = "zoning"
    REVENUE = "revenue"
    EFFICIENCY = "efficiency"
    ACCESSIBILITY = "accessibility"
    SAFETY = "safety"
    SUSTAINABILITY = "sustainability"
    OTHER = "other"


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    url: str
    file_type: str
    created_at: datetime
    size: int
    width: Optional[int] = None
    height: Optional[int] = None


class ProjectObjectiveRequest(BaseModel):
    space_type: SpaceType
    objectives: List[ObjectiveType]
    custom_objectives: Optional[List[str]] = None
    additional_notes: Optional[str] = None


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    space_type: SpaceType
    objectives: List[ObjectiveType]
    custom_objectives: Optional[List[str]] = None
    additional_notes: Optional[str] = None


class ProjectCreate(ProjectBase):
    user_id: str


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    space_type: Optional[SpaceType] = None
    objectives: Optional[List[ObjectiveType]] = None
    custom_objectives: Optional[List[str]] = None
    additional_notes: Optional[str] = None
    status: Optional[str] = None


class FloorplanFile(BaseModel):
    file_id: str
    filename: str
    url: str
    file_type: str
    created_at: datetime
    size: int
    width: Optional[int] = None
    height: Optional[int] = None


class UsageDataFile(BaseModel):
    file_id: str
    filename: str
    url: str
    created_at: datetime
    size: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None


class InsightResult(BaseModel):
    heatmap_url: Optional[str] = None
    recommendations: Dict[str, Any]
    report_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class MoodboardResult(BaseModel):
    moodboard_url: str
    style_description: str
    created_at: datetime


class Project(ProjectBase):
    id: str
    user_id: str
    floorplan: Optional[FloorplanFile] = None
    usage_data: Optional[UsageDataFile] = None
    insight_result: Optional[InsightResult] = None
    moodboard_result: Optional[MoodboardResult] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}