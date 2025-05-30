from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.core.config import settings
from app.db.supabase import supabase_client

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
                settings.SUPABASE_JWT_SECRET,
                algorithms=[unverified_header["alg"]],
                audience="authenticated",
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
        # Get user data from Supabase
        user_id = token_data["user_id"]
        user = supabase_client.auth.admin.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
            
        return {
            "id": user.id,
            "email": user.email,
            "app_metadata": user.app_metadata,
            "user_metadata": user.user_metadata,
            "created_at": user.created_at,
            "role": token_data.get("role", "user"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
        ) 