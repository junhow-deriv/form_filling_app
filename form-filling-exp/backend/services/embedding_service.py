"""
Embedding and vector search service for document knowledge base.

This module provides:
1. Document storage and management
2. Embedding generation (using OpenAI)
3. Vector similarity search

Placeholder sections (TODO for teammate):
- Document chunking strategy

Implemented sections:
- Document text extraction (uses parser.py)
- Embedding generation
- Vector search
- Database integration
"""

import os
from typing import Optional, List
import tiktoken
from openai import AsyncOpenAI
from database.supabase_client import get_client_for_user, get_supabase_client
from storage.storage_service import calculate_file_hash
from parser import parse_file




# OpenAI client for embeddings
_openai_client: Optional[AsyncOpenAI] = None

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-ada-002"  # 1536 dimensions
EMBEDDING_DIMENSIONS = 1536


def get_openai_client() -> AsyncOpenAI:
    """Get or create OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not configured. "
                "Set it in your .env file for embedding generation."
            )
        _openai_client = AsyncOpenAI(api_key=api_key, base_url="https://litellm.deriv.ai/v1")
        _openai_client = AsyncOpenAI(api_key=api_key,base_url="https://litellm.deriv.ai/v1")
    return _openai_client


# ============================================================================
# Embedding Generation
# ============================================================================

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings using OpenAI.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (each is a list of 1536 floats)
        
    Raises:
        Exception: If OpenAI API call fails
    """
    if not texts:
        return []
    
    client = get_openai_client()
    
    # OpenAI allows batch embedding (up to ~2048 texts)
    # For very large batches, we'd need to chunk, but for now this is fine
    response = await client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    
    # Extract embeddings in order
    embeddings = [item.embedding for item in response.data]
    
    print(f"[Embeddings] Generated {len(embeddings)} embeddings using {EMBEDDING_MODEL}")
    return embeddings


async def generate_single_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector (list of 1536 floats)
    """
    embeddings = await generate_embeddings([text])
    return embeddings[0] if embeddings else []


# ============================================================================
# Document Extraction
# ============================================================================

async def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from a file using the parser module.
    
    This function first checks the file extension:
    1. If it's a simple text format (txt, md, json, etc.), it decodes and returns the text directly.
    2. If it's a complex format (PDF, DOCX, PPTX, images), it uses Docling to parse and convert the content into Markdown.
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename (for type detection)
        
    Returns:
        Extracted text content (raw text or Markdown)
    """
    print(f"[Extraction] Extracting text from {filename}...")
    try:
        # Use the parser module to extract text
        text = await parse_file(file_bytes, filename)
        print(f"[Extraction] Successfully extracted {len(text)} chars from {filename}")
        return text
    except Exception as e:
        print(f"[Extraction] Failed to extract text from {filename}: {e}")
        # Return error message so it's visible in the system that extraction failed
        return f"[FAILED TO EXTRACT TEXT: {str(e)}]"


# ============================================================================
# Document Chunking (TODO - PLACEHOLDER)
# ============================================================================

async def chunk_document(text: str, filename: str, chunk_size: int = 1000, overlap: int = 200) -> List[dict]:
    """
    Split document text into chunks for embedding using tiktoken.
    
    Args:
        text: Full document text
        filename: Original filename
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        
    Returns:
        List of chunk dicts with 'text' and 'metadata' keys
    """
    # Use cl100k_base encoding (used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    print(f"[Chunking] Document has {num_tokens} tokens")
    chunks = []
    start = 0
    
    # If the document is empty, return no chunks
    if num_tokens == 0:
        return []
        
    # Calculate step size (sliding window)
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # Ensure we always move forward

    chunk_idx = 0
    for start in range(0, num_tokens, step):
        end = min(start + chunk_size, num_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        
        chunks.append({
            "metadata": {"chunk_index": chunk_idx, "filename": filename, "chunk_token_count": len(chunk_tokens)},
            "text": chunk_text
        })
        chunk_idx += 1
        
        # If we reached the end, break
        if end == num_tokens:
            break
        
    print(f"[Chunking] Created {len(chunks)} chunks using tiktoken (size={chunk_size}, overlap={overlap})")
    return chunks


# ============================================================================
# Document Storage
# ============================================================================

async def store_document(
    user_id: str,
    filename: str,
    file_bytes: bytes,
    session_id: Optional[str] = None,  # NEW - NULL for KB, UUID for ephemeral
    metadata: Optional[dict] = None
) -> str:
    """
    Store a document with embeddings in the knowledge base.
    
    This orchestrates the full pipeline:
    1. Upload file to storage
    2. Extract text (TODO: teammate)
    3. Chunk text (TODO: teammate)
    4. Generate embeddings (IMPLEMENTED)
    5. Store in database (IMPLEMENTED)
    
    Args:
        user_id: User UUID
        filename: Original filename
        file_bytes: File content as bytes
        session_id: Optional session ID - NULL for global KB, UUID for ephemeral
        metadata: Optional custom metadata
        
    Returns:
        Document ID (UUID string)
        
    Raises:
        Exception: If any step fails
    """
    client = get_client_for_user(user_id)
    
    # Calculate file hash for deduplication
    file_hash = calculate_file_hash(file_bytes)
    
    # Check if document already exists
    existing = client.table("documents").select("id").eq("file_hash", file_hash).eq("user_id", user_id).execute()
    if existing.data:
        doc_id = existing.data[0]["id"]
        print(f"[Store] Document already exists: {doc_id}")
        return doc_id
    
    # Extract text (TODO: teammate will implement)
    raw_text = await extract_text_from_file(file_bytes, filename)
    
    # Insert document record (no storage_path - we only need text + embeddings)
    file_type = filename.split(".")[-1].lower()
    doc_data = {
        "user_id": user_id,
        "session_id": session_id,  # NULL for KB, UUID for ephemeral
        "filename": filename,
        "file_type": file_type,
        "file_hash": file_hash,
        "file_size": len(file_bytes),
        "raw_text": raw_text,
        "metadata": metadata or {}
    }
    
    doc_result = client.table("documents").insert(doc_data).execute()
    document_id = doc_result.data[0]["id"]
    
    source_type = "ephemeral" if session_id else "global KB"
    print(f"[Store] Created document: {document_id} ({source_type})")
    
    # Chunk the text (TODO: teammate will implement proper chunking)
    chunks = await chunk_document(raw_text, filename)
    
    if not chunks:
        print(f"[Store] No chunks created for document {document_id}")
        return document_id
    
    # Generate embeddings for all chunks
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = await generate_embeddings(chunk_texts)
    
    # Store chunks with embeddings (denormalize session_id for query performance)
    chunk_records = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_records.append({
            "document_id": document_id,
            "session_id": session_id,  # Denormalized for efficient filtering
            "chunk_index": i,
            "chunk_text": chunk["text"],
            "chunk_size": len(chunk["text"]),
            "embedding": embedding,
            "metadata": chunk.get("metadata", {})
        })
    
    # Batch insert chunks
    client.table("document_chunks").insert(chunk_records).execute()
    print(f"[Store] Inserted {len(chunk_records)} chunks with embeddings ({source_type})")
    
    return document_id


# ============================================================================
# Vector Search with Session Filtering
# ============================================================================

async def search_ephemeral_docs(
    user_id: str,
    session_id: str,
    query: str,
    top_k: int = 1,
    similarity_threshold: float = 0.7
) -> Optional[dict]:
    """
    Search ephemeral uploaded documents for a specific session.
    
    Args:
        user_id: User UUID
        session_id: Session UUID (for ephemeral docs)
        query: Natural language search query
        top_k: Number of results (default 1 for waterfall)
        similarity_threshold: Minimum similarity score
        
    Returns:
        Best matching chunk or None if no match:
        {
            "chunk_id": "...",
            "chunk_text": "...",
            "similarity": 0.9,
            "filename": "...",
            "source": "ephemeral"
        }
    """
    client = get_client_for_user(user_id)
    
    # Generate query embedding
    query_embedding = await generate_single_embedding(query)
    
    # Search ephemeral docs only
    result = client.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": similarity_threshold,
            "match_count": top_k,
            "filter_user_id": user_id,
            "filter_session_id": session_id,
            "search_global_kb": False
        }
    ).execute()
    
    if result.data and len(result.data) > 0:
        match = result.data[0]
        return {
            "chunk_id": match["id"],
            "chunk_text": match["chunk_text"],
            "similarity": match["similarity"],
            "filename": match["filename"],
            "metadata": match.get("metadata", {}),
            "source": "ephemeral"
        }
    
    return None


async def search_global_kb(
    user_id: str,
    query: str,
    top_k: int = 1,
    similarity_threshold: float = 0.7
) -> Optional[dict]:
    """
    Search global knowledge base (session_id = NULL).
    
    Args:
        user_id: User UUID
        query: Natural language search query
        top_k: Number of results (default 1 for waterfall)
        similarity_threshold: Minimum similarity score
        
    Returns:
        Best matching chunk or None if no match:
        {
            "chunk_id": "...",
            "chunk_text": "...",
            "similarity": 0.85,
            "filename": "...",
            "source": "knowledge_base"
        }
    """
    client = get_client_for_user(user_id)
    
    # Generate query embedding
    query_embedding = await generate_single_embedding(query)
    
    # Search global KB only
    result = client.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": similarity_threshold,
            "match_count": top_k,
            "filter_user_id": user_id,
            "filter_session_id": None,
            "search_global_kb": True
        }
    ).execute()
    
    if result.data and len(result.data) > 0:
        match = result.data[0]
        return {
            "chunk_id": match["id"],
            "chunk_text": match["chunk_text"],
            "similarity": match["similarity"],
            "filename": match["filename"],
            "metadata": match.get("metadata", {}),
            "source": "knowledge_base"
        }
    
    return None


async def waterfall_search(
    user_id: str,
    session_id: str,
    field_queries: dict,
    similarity_threshold: float = 0.7
) -> dict:
    """
    Perform waterfall retrieval: ephemeral docs → global KB.
    
    For each field query, search ephemeral docs first. If no good match,
    fall back to global knowledge base. Returns the best match from
    the first source that has one.
    
    Args:
        user_id: User UUID
        session_id: Session UUID (for ephemeral docs)
        field_queries: Output from query_generator.generate_field_queries()
        similarity_threshold: Minimum similarity for matches
        
    Returns:
        {
            "field_id": {
                "result": {"chunk_text": "...", "similarity": 0.9, ...} or None,
                "source": "ephemeral" | "knowledge_base" | "none",
                "query_used": "..."
            },
            ...
        }
    """
    results = {}
    
    # Process uploaded docs queries (ephemeral)
    for query_info in field_queries.get("uploaded_docs_queries", []):
        field_id = query_info["field_id"]
        query = query_info["query"]
        
        result = await search_ephemeral_docs(
            user_id, session_id, query,
            top_k=1, similarity_threshold=similarity_threshold
        )
        
        results[field_id] = {
            "result": result,
            "source": result["source"] if result else "none",
            "query_used": query
        }
    
    # Process knowledge base queries (global KB)
    for query_info in field_queries.get("knowledge_base_queries", []):
        field_id = query_info["field_id"]
        query = query_info["query"]
        
        result = await search_global_kb(
            user_id, query,
            top_k=1, similarity_threshold=similarity_threshold
        )
        
        results[field_id] = {
            "result": result,
            "source": result["source"] if result else "none",
            "query_used": query
        }
    
    # Log summary
    ephemeral_count = sum(1 for v in results.values() if v["source"] == "ephemeral")
    kb_count = sum(1 for v in results.values() if v["source"] == "knowledge_base")
    none_count = sum(1 for v in results.values() if v["source"] == "none")
    
    print(f"[Waterfall] Results: {ephemeral_count} from ephemeral, {kb_count} from KB, {none_count} no match")
    
    return results


def assemble_waterfall_context(waterfall_results: dict) -> str:
    """
    Assemble context from waterfall search results for agent prompt.
    
    Groups results by source (ephemeral vs knowledge base) and formats
    them clearly for the agent to use.
    
    Args:
        waterfall_results: Output from waterfall_search()
        
    Returns:
        Formatted context string ready for agent prompt
    """
    if not waterfall_results:
        return ""
    
    # Separate by source
    ephemeral_results = []
    kb_results = []
    
    for field_id, data in waterfall_results.items():
        if data["result"]:
            result = data["result"]
            result["field_id"] = field_id
            result["query"] = data["query_used"]
            
            if result["source"] == "ephemeral":
                ephemeral_results.append(result)
            elif result["source"] == "knowledge_base":
                kb_results.append(result)
    
    # Format context
    context_parts = []
    
    if ephemeral_results:
        context_parts.append("## Uploaded Document Content\n")
        context_parts.append("Priority: Use this information FIRST when filling fields.\n")
        for result in ephemeral_results:
            context_parts.append(
                f"\n**Query**: {result['query']}\n"
                f"**Source**: {result['filename']}\n"
                f"**Content**: {result['chunk_text']}\n"
            )
    
    if kb_results:
        context_parts.append("\n## Knowledge Base Context\n")
        context_parts.append("Priority: Use this only if information is missing from uploaded documents.\n")
        for result in kb_results:
            context_parts.append(
                f"\n**Query**: {result['query']}\n"
                f"**Source**: {result['filename']}\n"
                f"**Content**: {result['chunk_text']}\n"
            )
    
    if not ephemeral_results and not kb_results:
        return ""
    
    return "\n".join(context_parts)


# ============================================================================
# Document Management
# ============================================================================

async def get_user_documents(user_id: str, limit: int = 100) -> List[dict]:
    """
    Get all documents for a user.
    
    Args:
        user_id: User UUID
        limit: Max documents to return
        
    Returns:
        List of document records
    """
    client = get_client_for_user(user_id)
    result = client.table("documents").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    return result.data


async def delete_document_by_id(user_id: str, document_id: str) -> bool:
    """
    Delete a document and its chunks from the knowledge base.
    
    Args:
        user_id: User UUID
        document_id: Document UUID to delete
        
    Returns:
        True if deleted successfully
    """
    try:
        client = get_client_for_user(user_id)
        
        # Delete from database (chunks will cascade delete)
        client.table("documents").delete().eq("id", document_id).eq("user_id", user_id).execute()
        
        print(f"[Delete] Deleted document: {document_id}")
        return True
    except Exception as e:
        print(f"[Delete] Failed to delete document {document_id}: {e}")
        return False


# ============================================================================
# Statistics and Info
# ============================================================================

async def get_knowledge_base_stats(user_id: str) -> dict:
    """
    Get statistics about user's knowledge base.
    
    Args:
        user_id: User UUID
        
    Returns:
        Dict with stats (document count, chunk count, total size, etc.)
    """
    client = get_client_for_user(user_id)
    
    # Get document count and total size
    docs = client.table("documents").select("id, file_size").eq("user_id", user_id).execute()
    
    doc_count = len(docs.data)
    total_size = sum(doc["file_size"] for doc in docs.data)
    
    # Get chunk count
    if doc_count > 0:
        doc_ids = [doc["id"] for doc in docs.data]
        chunks = client.table("document_chunks").select("id", count="exact").in_("document_id", doc_ids).execute()
        chunk_count = chunks.count
    else:
        chunk_count = 0
    
    return {
        "document_count": doc_count,
        "chunk_count": chunk_count,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }


if __name__ == "__main__":
    # Quick test
    import asyncio
    
    async def test():
        # print("Embedding Service Test")
        # print("=" * 50)
        
        # # Test embedding generation
        # test_texts = ["Hello, world!", "This is a test document."]
        # embeddings = await generate_embeddings(test_texts)
        # print(f"Generated {len(embeddings)} embeddings")
        # print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Test chunking
        test_text_2 = "This is a long document. " * 100
        # Using chunk_size=100 and overlap=50 to demonstrate sliding window
        chunks = await chunk_document(test_text_2, "test_doc.txt", chunk_size=100, overlap=50)
        print(f"Created {len(chunks)} chunks")
        print(chunks)

    asyncio.run(test())
