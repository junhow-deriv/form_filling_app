"""
Storage service for managing session PDFs in Supabase Storage.

Handles upload, download, and deletion of form PDFs (original and filled)
from the form-pdfs bucket.
"""

import hashlib
from typing import Optional
from database.supabase_client import get_supabase_client, get_client_for_user

# Storage bucket name
FORM_PDFS_BUCKET = "form-pdfs"


# ============================================================================
# Path Generation Helpers
# ============================================================================

def get_session_pdf_path(user_id: str, session_id: str, pdf_type: str) -> str:
    """
    Generate storage path for session PDFs.
    
    Args:
        user_id: User UUID
        session_id: Session UUID
        pdf_type: 'original' or 'filled'
        
    Returns:
        Storage path string
    """
    return f"{user_id}/sessions/{session_id}/{pdf_type}.pdf"


def calculate_file_hash(file_bytes: bytes) -> str:
    """
    Calculate SHA256 hash of file bytes.
    
    Args:
        file_bytes: File content as bytes
        
    Returns:
        Hex string of SHA256 hash
    """
    return hashlib.sha256(file_bytes).hexdigest()


# ============================================================================
# Session PDF Operations
# ============================================================================

def upload_session_pdf(
    user_id: str,
    session_id: str,
    pdf_bytes: bytes,
    pdf_type: str = "filled"
) -> str:
    """
    Upload a PDF for a form-filling session.
    
    Args:
        user_id: User UUID
        session_id: Session UUID
        pdf_bytes: PDF file content as bytes
        pdf_type: 'original' or 'filled'
        
    Returns:
        Storage path where file was uploaded
        
    Raises:
        Exception: If upload fails
    """
    client = get_client_for_user(user_id)
    storage = client.storage.from_(FORM_PDFS_BUCKET)
    
    path = get_session_pdf_path(user_id, session_id, pdf_type)
    
    # Upload or update (upsert=True overwrites existing)
    result = storage.upload(
        path=path,
        file=pdf_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"}
    )
    
    print(f"[Storage] Uploaded {pdf_type} PDF to: {path}")
    return path


def download_session_pdf(
    user_id: str,
    session_id: str,
    pdf_type: str = "filled"
) -> Optional[bytes]:
    """
    Download a PDF from a session.
    
    Args:
        user_id: User UUID
        session_id: Session UUID
        pdf_type: 'original' or 'filled'
        
    Returns:
        PDF bytes or None if not found
    """
    try:
        client = get_client_for_user(user_id)
        storage = client.storage.from_(FORM_PDFS_BUCKET)
        
        path = get_session_pdf_path(user_id, session_id, pdf_type)
        
        # Download the file
        result = storage.download(path)
        print(f"[Storage] Downloaded {pdf_type} PDF from: {path}")
        return result
    except Exception as e:
        print(f"[Storage] Failed to download {pdf_type} PDF: {e}")
        return None


def delete_session_pdf(
    user_id: str,
    session_id: str,
    pdf_type: Optional[str] = None
) -> bool:
    """
    Delete session PDF(s).
    
    Args:
        user_id: User UUID
        session_id: Session UUID
        pdf_type: 'original', 'filled', or None for both
        
    Returns:
        True if deletion successful
    """
    try:
        client = get_client_for_user(user_id)
        storage = client.storage.from_(FORM_PDFS_BUCKET)
        
        if pdf_type:
            # Delete single file
            path = get_session_pdf_path(user_id, session_id, pdf_type)
            storage.remove([path])
            print(f"[Storage] Deleted {pdf_type} PDF: {path}")
        else:
            # Delete both
            paths = [
                get_session_pdf_path(user_id, session_id, "original"),
                get_session_pdf_path(user_id, session_id, "filled")
            ]
            storage.remove(paths)
            print(f"[Storage] Deleted all PDFs for session {session_id}")
        
        return True
    except Exception as e:
        print(f"[Storage] Failed to delete PDFs: {e}")
        return False


# ============================================================================
# Utility Functions
# ============================================================================

def get_public_url(bucket: str, path: str) -> str:
    """
    Get a public URL for a file (if bucket is public).
    
    For private buckets, use signed URLs instead.
    
    Args:
        bucket: Bucket name
        path: File path in bucket
        
    Returns:
        Public URL string
    """
    client = get_supabase_client()
    storage = client.storage.from_(bucket)
    return storage.get_public_url(path)


def create_signed_url(bucket: str, path: str, expires_in: int = 3600, user_id: Optional[str] = None) -> str:
    """
    Create a signed URL for temporary access to a private file.
    
    Args:
        bucket: Bucket name
        path: File path in bucket
        expires_in: Expiration time in seconds (default 1 hour)
        user_id: Optional user_id for access control
        
    Returns:
        Signed URL string
    """
    client = get_client_for_user(user_id) if user_id else get_supabase_client()
    storage = client.storage.from_(bucket)
    result = storage.create_signed_url(path, expires_in)
    return result["signedURL"]


if __name__ == "__main__":
    # Quick test
    print("Storage Service Test")
    print("=" * 50)
    
    # Test path generation
    test_user_id = "00000000-0000-0000-0000-000000000001"
    test_session_id = "11111111-1111-1111-1111-111111111111"
    
    path = get_session_pdf_path(test_user_id, test_session_id, "filled")
    print(f"Session PDF path: {path}")
    
    # Test hash
    test_bytes = b"Hello, world!"
    file_hash = calculate_file_hash(test_bytes)
    print(f"File hash: {file_hash}")

