"""
Supabase client singleton and database utilities.

Provides a centralized Supabase client for database and storage operations.
"""

import os
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Global client instances
_client: Optional[Client] = None
_service_client: Optional[Client] = None


def get_supabase_client(use_service_key: bool = False) -> Client:
    """
    Get the Supabase client singleton.
    
    Args:
        use_service_key: If True, use service role key for admin operations.
                        Otherwise, use anon key with RLS.
    
    Returns:
        Supabase client instance
        
    Raises:
        ValueError: If Supabase credentials are not configured
    """
    global _client, _service_client
    
    if not SUPABASE_URL:
        raise ValueError(
            "SUPABASE_URL not configured. "
            "Set it in your .env file or environment variables."
        )
    
    if use_service_key:
        if not SUPABASE_SERVICE_KEY:
            raise ValueError(
                "SUPABASE_SERVICE_KEY not configured. "
                "Set it in your .env file for admin operations."
            )
        
        if _service_client is None:
            _service_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        return _service_client
    else:
        if not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_ANON_KEY not configured. "
                "Set it in your .env file or environment variables."
            )
        
        if _client is None:
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _client


def get_client_for_user(user_id: str) -> Client:
    """
    Get a Supabase client configured for a specific user.
    
    For now, this returns the service client since we're not using
    full Supabase Auth yet. In production, this should set the
    auth token for the user.
    
    Args:
        user_id: The user's UUID
        
    Returns:
        Supabase client instance
    """
    # TODO: Integrate with Supabase Auth
    # For now, use service client which bypasses RLS
    return get_supabase_client(use_service_key=True)

def is_configured() -> bool:
    """
    Check if Supabase credentials are configured.
    
    Returns:
        True if SUPABASE_URL and key are set
    """
    return bool(SUPABASE_URL and SUPABASE_KEY)

if __name__ == "__main__":
    # Quick test
    print("Testing Supabase connection...")
    
    if not is_configured():
        print("❌ Supabase not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in .env")
    else:
        print(f"✓ SUPABASE_URL: {SUPABASE_URL}")
        print(f"✓ Credentials loaded")
        
        try:
            client = get_supabase_client()
            print("✓ Client created successfully")
            
            # Try a simple query
            result = client.table("users").select("id").limit(1).execute()
            print(f"✓ Database query successful (found {len(result.data)} users)")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

