from typing import List
import os

from app.core.config import settings

def is_valid_file_extension(filename: str, allowed_extensions: List[str] = None) -> bool:
    """
    Check if a file has a valid extension
    
    Args:
        filename: The name of the file to check
        allowed_extensions: List of allowed extensions (defaults to settings.ALLOWED_EXTENSIONS)
        
    Returns:
        True if the extension is valid
    """
    if not allowed_extensions:
        allowed_extensions = settings.ALLOWED_EXTENSIONS
        
    extension = filename.split(".")[-1].lower()
    return extension in allowed_extensions


def is_valid_file_size(file_size: int, max_size: int = None) -> bool:
    """
    Check if a file size is valid
    
    Args:
        file_size: The size of the file in bytes
        max_size: Maximum allowed size in bytes (defaults to settings.MAX_UPLOAD_SIZE)
        
    Returns:
        True if the size is valid
    """
    if not max_size:
        max_size = settings.MAX_UPLOAD_SIZE
        
    return file_size <= max_size


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing potentially problematic characters
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    
    # Remove any characters that aren't alphanumeric, underscore, dash, or dot
    filename = "".join(c for c in filename if c.isalnum() or c in "_-.")
    
    return filename


def generate_unique_filename(filename: str, prefix: str = "") -> str:
    """
    Generate a unique filename using a timestamp
    
    Args:
        filename: Original filename
        prefix: Optional prefix to add
        
    Returns:
        Unique filename
    """
    import time
    import uuid
    
    # Get file extension
    extension = filename.split(".")[-1] if "." in filename else ""
    
    # Generate base name (without extension)
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    
    # Sanitize base name
    base_name = sanitize_filename(base_name)
    
    # Generate unique identifier
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    
    # Combine components
    if prefix:
        unique_filename = f"{prefix}_{base_name}_{timestamp}_{unique_id}.{extension}"
    else:
        unique_filename = f"{base_name}_{timestamp}_{unique_id}.{extension}"
        
    return unique_filename

import io
import asyncio
from typing import Optional
from pdf2image import convert_from_bytes
from PIL import Image

async def convert_pdf_to_image(pdf_bytes: bytes) -> Optional[bytes]:
    """
    Converts the first page of a PDF byte stream to a PNG image byte stream.
    Uses a thread pool executor for the blocking pdf2image call.
    """
    try:
        # convert_from_bytes is a blocking I/O-bound operation
        # Run it in a separate thread to avoid blocking the asyncio event loop
        pil_images = await asyncio.to_thread(convert_from_bytes, pdf_bytes, first_page=1, last_page=1)

        if pil_images:
            img_byte_arr = io.BytesIO()
            pil_images[0].save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue()
        return None
    except Exception as e:
        # In a real app, you'd log this error
        print(f"Error converting PDF to image: {e}") # Replace with proper logging
        return None