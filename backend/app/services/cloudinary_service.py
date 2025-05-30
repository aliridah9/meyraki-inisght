import cloudinary
import cloudinary.uploader
import cloudinary.api
from typing import Dict, Any, Optional, List
import time

from app.core.config import settings

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)

class CloudinaryService:
    @staticmethod
    async def upload_file(file_content: bytes, folder: str, file_name: str, resource_type: str = "auto") -> Dict[str, Any]:
        """
        Upload a file to Cloudinary
        
        Args:
            file_content: The binary content of the file
            folder: The folder path in Cloudinary
            file_name: The name to use for the file
            resource_type: The resource type ('image', 'raw', 'video', or 'auto')
            
        Returns:
            A dictionary with upload response data
        """
        try:
            # Upload to cloudinary
            response = cloudinary.uploader.upload(
                file_content,
                folder=folder,
                public_id=file_name,
                resource_type=resource_type,
                overwrite=True
            )
            
            return {
                "public_id": response.get("public_id"),
                "secure_url": response.get("secure_url"),
                "format": response.get("format"),
                "resource_type": response.get("resource_type"),
                "created_at": response.get("created_at"),
                "bytes": response.get("bytes"),
                "width": response.get("width"),
                "height": response.get("height"),
            }
        except Exception as e:
            # In a production app, log this error
            raise Exception(f"Failed to upload to Cloudinary: {str(e)}")
    
    @staticmethod
    async def delete_file(public_id: str, resource_type: str = "image") -> Dict[str, Any]:
        """
        Delete a file from Cloudinary
        
        Args:
            public_id: The public ID of the resource to delete
            resource_type: The resource type ('image', 'raw', 'video')
            
        Returns:
            The response from Cloudinary
        """
        try:
            response = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
            return response
        except Exception as e:
            raise Exception(f"Failed to delete from Cloudinary: {str(e)}")
    
    @staticmethod
    async def generate_download_url(public_id: str, format: Optional[str] = None, 
                            expiration: int = 3600) -> str:
        """
        Generate a secure download URL for a file
        
        Args:
            public_id: The public ID of the resource
            format: Optional format for the download
            expiration: Expiration time in seconds
            
        Returns:
            Secured download URL
        """
        try:
            options = {
                "type": "private",
                "resource_type": "image",
                "expires_at": int(time.time()) + expiration
            }
            
            if format:
                options["format"] = format
                
            url = cloudinary.utils.private_download_url(public_id, **options)
            return url
        except Exception as e:
            raise Exception(f"Failed to generate download URL: {str(e)}")
            
cloudinary_service = CloudinaryService() 