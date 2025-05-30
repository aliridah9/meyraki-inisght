from supabase import create_client, Client
from tenacity import retry, wait_exponential, stop_after_attempt

from app.core.config import settings

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def get_supabase_client() -> Client:
    """
    Create a Supabase client with connection retry logic.
    Retries with exponential backoff if connection fails.
    """
    try:
        url = settings.SUPABASE_URL
        key = settings.SUPABASE_KEY
        
        if not url or not key:
            raise ValueError("Supabase URL and Key must be provided")
        
        client = create_client(url, key)
        return client
    except Exception as e:
        # Log the error here if needed
        raise e

# Create a global client instance
supabase_client = get_supabase_client() 