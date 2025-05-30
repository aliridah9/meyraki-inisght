# backend/app/models/user.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class UserBase(BaseModel):
    email: EmailStr
    is_active: bool = True
    model_config = {"from_attributes": True}

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    model_config = {"from_attributes": True}

class UserInDBBase(UserBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    user_metadata: Optional[Dict[str, Any]] = {}
    app_metadata: Optional[Dict[str, Any]] = {}
    hashed_password: Optional[str] = None # To store hashed password
    model_config = {"from_attributes": True}

class User(UserInDBBase):
    """User model to be returned by API"""
    pass

class UserInDB(UserInDBBase):
    """User model as stored in DB, potentially with hashed_password"""
    pass
