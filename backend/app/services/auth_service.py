from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.core.config import settings
# from app.db.supabase import supabase_client # Removed
from app.db.mongodb import get_database # Added
# from app.models.user import User as UserModel # Optional, if returning model instance

security = HTTPBearer()

class AuthService:
    @staticmethod
    async def decode_jwt(token: str) -> Dict[str, Any]:
        """
        Decode and validate JWT token from Supabase Auth
        
        Args:
            token: JWT token
            
        Returns:
            Decoded token payload
        """
        try:
            # Decode without verification first to get the header
            unverified_header = jwt.get_unverified_header(token)
            
            # Decode and verify
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY, # Changed
                algorithms=[unverified_header["alg"]],
                # audience="authenticated", # Removed
            )
            
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid authentication credentials: {str(e)}",
            )
    
    @staticmethod
    async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """
        Validate token from HTTP Authorization header
        
        Args:
            credentials: HTTP Authorization credentials
            
        Returns:
            Decoded token payload
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
            
        token = credentials.credentials
        payload = await AuthService.decode_jwt(token)
        
        # Extract user info
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID in token",
            )
            
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "role": payload.get("role", "user"),
            "exp": payload.get("exp"),
        }
            
auth_service = AuthService()

# Define the validate_token dependency
validate_token = auth_service.validate_token

async def get_current_user(token_data: Dict[str, Any] = Depends(validate_token)) -> Dict[str, Any]:
    """
    Get current authenticated user from token data
    
    Args:
        token_data: Decoded token data
        
    Returns:
        User data
    """
    try:
        user_id = token_data["user_id"] # Extracted from 'sub' in validate_token
        
        db = get_database()
        user_doc = await db["users"].find_one({"id": user_id}) # Query MongoDB by 'id' field

        if not user_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
            
        # Construct the dictionary to be returned, similar to previous structure
        # Ensure all fields are present as expected by downstream code
        return {
            "id": user_doc.get("id"),
            "email": user_doc.get("email"),
            "app_metadata": user_doc.get("app_metadata", {}),
            "user_metadata": user_doc.get("user_metadata", {}),
            "created_at": user_doc.get("created_at"), # Motor returns datetime; FastAPI handles serialization
            "role": token_data.get("role", "user"), # Role comes from the token itself
            "is_active": user_doc.get("is_active", True) # From user document
        }
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        # Log error e if a logger is available
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 401 if more appropriate
            detail=f"Could not validate credentials or fetch user: {str(e)}",
        )