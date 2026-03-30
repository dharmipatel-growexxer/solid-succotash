"""
Pinecone vector store operations.
Handles upserting chunks and querying.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from config import (
    CHUNKS_OUTPUT_DIR,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION,
)
from embeddings import get_embedding_model


def get_pinecone_client():
    """Initialize and return Pinecone client."""
    try:
        from pinecone import Pinecone
    except ImportError:
        raise ImportError(
            "pinecone-client not installed. "
            "Install with: pip install pinecone-client"
        )
    
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY environment variable not set. "
            "Set it with: export PINECONE_API_KEY='your-key'"
        )
    
    return Pinecone(api_key=api_key)


def get_index():
    """Get the Pinecone index."""
    pc = get_pinecone_client()
    return pc.Index(PINECONE_INDEX_NAME)


def load_chunks(chunks_file: Path = CHUNKS_OUTPUT_DIR / "chunks.json") -> List[Dict]:
    """Load chunks from JSON file."""
    print(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def prepare_vectors(
    chunks: List[Dict],
    embedding_model,
    batch_size: int = 100
) -> List[Dict]:
    """
    Prepare vectors for upserting to Pinecone.
    
    Returns list of dicts with: id, values, metadata
    """
    vectors = []
    total = len(chunks)
    
    print(f"Generating embeddings for {total} chunks...")
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        
        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Create vector records
        for chunk, embedding in zip(batch, embeddings):
            # Pinecone metadata has size limits, so we store essential fields
            metadata = {
                "scheme_id": chunk["metadata"]["scheme_id"],
                "scheme_name": chunk["metadata"]["scheme_name"][:200],  # Truncate long names
                "scheme_url": chunk["metadata"]["scheme_url"],
                "location_type": chunk["metadata"]["location_type"],
                "location_name": chunk["metadata"]["location_name"],
                "category_id": chunk["metadata"]["category_id"],
                "category_name": chunk["metadata"]["category_name"],
                "chunk_type": chunk["metadata"]["chunk_type"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "language": chunk["metadata"]["language"],
                # Store text for retrieval (Pinecone allows up to 40KB per metadata)
                "text": chunk["text"][:8000],  # Truncate very long texts
            }
            
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata
            })
        
        print(f"  Processed {min(i + batch_size, total)}/{total} chunks")
    
    return vectors


def upsert_vectors(
    vectors: List[Dict],
    batch_size: int = 100,
    namespace: str = ""
) -> int:
    """
    Upsert vectors to Pinecone index.
    
    Returns number of vectors upserted.
    """
    index = get_index()
    total = len(vectors)
    upserted = 0
    
    print(f"Upserting {total} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
    
    for i in range(0, total, batch_size):
        batch = vectors[i:i + batch_size]
        
        try:
            index.upsert(vectors=batch, namespace=namespace)
            upserted += len(batch)
            print(f"  Upserted {upserted}/{total} vectors")
        except Exception as e:
            print(f"  Error upserting batch {i//batch_size}: {e}")
            # Retry with smaller batch
            for vec in batch:
                try:
                    index.upsert(vectors=[vec], namespace=namespace)
                    upserted += 1
                except Exception as e2:
                    print(f"    Failed to upsert {vec['id']}: {e2}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return upserted


def query_vectors(
    query_text: str,
    top_k: int = 10,
    filter_dict: Optional[Dict] = None,
    namespace: str = ""
) -> List[Dict]:
    """
    Query Pinecone index.
    
    Args:
        query_text: The query string
        top_k: Number of results to return
        filter_dict: Metadata filters (e.g., {"location_name": "Gujarat"})
        namespace: Pinecone namespace
    
    Returns:
        List of matches with id, score, and metadata
    """
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query_text)
    
    index = get_index()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True,
        namespace=namespace
    )
    
    return results.matches


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the Pinecone index."""
    index = get_index()
    return index.describe_index_stats()


def delete_all_vectors(namespace: str = "") -> None:
    """Delete all vectors in the index (use with caution!)."""
    index = get_index()
    index.delete(delete_all=True, namespace=namespace)
    print(f"Deleted all vectors from namespace '{namespace or 'default'}'")


def run_upsert_pipeline(
    chunks_file: Path = CHUNKS_OUTPUT_DIR / "chunks.json",
    batch_size: int = 100,
    namespace: str = ""
) -> None:
    """
    Run the full upsert pipeline with STREAMING:
    - Embeds and upserts in batches (not all at once)
    - Progress visible on Pinecone immediately
    - Memory efficient
    - Resumable (tracks progress)
    """
    # Load chunks
    chunks = load_chunks(chunks_file)
    total = len(chunks)
    
    # Get embedding model and index
    embedding_model = get_embedding_model()
    index = get_index()
    
    # Progress tracking
    progress_file = CHUNKS_OUTPUT_DIR / "upsert_progress.json"
    start_idx = 0
    
    # Check for existing progress
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
            start_idx = progress.get("last_completed_idx", 0)
            if start_idx > 0:
                print(f"Resuming from index {start_idx} ({start_idx}/{total} already done)")
    
    upserted = start_idx
    failed = 0
    
    print(f"Streaming upsert: {total} chunks, batch_size={batch_size}")
    print(f"Progress will be visible on Pinecone immediately.\n")
    
    for i in range(start_idx, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch_chunks]
        
        try:
            # Step 1: Generate embeddings for this batch
            embeddings = embedding_model.embed_documents(texts)
            
            # Step 2: Prepare vectors
            vectors = []
            for chunk, embedding in zip(batch_chunks, embeddings):
                metadata = {
                    "scheme_id": chunk["metadata"]["scheme_id"],
                    "scheme_name": chunk["metadata"]["scheme_name"][:200],
                    "scheme_url": chunk["metadata"]["scheme_url"],
                    "location_type": chunk["metadata"]["location_type"],
                    "location_name": chunk["metadata"]["location_name"],
                    "category_id": chunk["metadata"]["category_id"],
                    "category_name": chunk["metadata"]["category_name"],
                    "chunk_type": chunk["metadata"]["chunk_type"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "language": chunk["metadata"]["language"],
                    "text": chunk["text"][:8000],
                }
                vectors.append({
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Step 3: Upsert this batch to Pinecone
            index.upsert(vectors=vectors, namespace=namespace)
            upserted += len(vectors)
            
            # Save progress
            with open(progress_file, "w") as f:
                json.dump({"last_completed_idx": i + batch_size}, f)
            
            print(f"  [{upserted}/{total}] Embedded & upserted batch {i//batch_size + 1}")
            
        except Exception as e:
            print(f"  Error at batch {i//batch_size + 1}: {e}")
            failed += len(batch_chunks)
            # Continue with next batch instead of crashing
            continue
        
        # Small delay to avoid rate limiting
        time.sleep(0.05)
    
    # Cleanup progress file on completion
    if progress_file.exists() and upserted >= total:
        progress_file.unlink()
    
    print(f"\n{'='*50}")
    print(f"Upsert complete!")
    print(f"  Successful: {upserted}")
    print(f"  Failed: {failed}")
    print(f"{'='*50}")
    
    # Verify
    time.sleep(2)
    stats = get_index_stats()
    print(f"\nPinecone index stats:")
    print(f"  Total vectors: {stats.get('total_vector_count', 'N/A')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pinecone vector store operations")
    parser.add_argument(
        "--action",
        choices=["upsert", "stats", "query", "delete-all"],
        default="stats",
        help="Action to perform"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query text (for query action)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results for query"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for upserting"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="Pinecone namespace"
    )
    
    args = parser.parse_args()
    
    if args.action == "upsert":
        run_upsert_pipeline(batch_size=args.batch_size, namespace=args.namespace)
    
    elif args.action == "stats":
        stats = get_index_stats()
        print(f"Index stats:\n{json.dumps(stats, indent=2, default=str)}")
    
    elif args.action == "query":
        if not args.query:
            print("Error: --query is required for query action")
        else:
            results = query_vectors(args.query, top_k=args.top_k, namespace=args.namespace)
            print(f"\nQuery: {args.query}")
            print(f"Results ({len(results)}):\n")
            for i, match in enumerate(results):
                print(f"{i+1}. Score: {match.score:.4f}")
                print(f"   Scheme: {match.metadata.get('scheme_name', 'N/A')}")
                print(f"   Type: {match.metadata.get('chunk_type', 'N/A')}")
                print(f"   Location: {match.metadata.get('location_name', 'N/A')}")
                print(f"   Text: {match.metadata.get('text', '')[:200]}...")
                print()
    
    elif args.action == "delete-all":
        confirm = input("Are you sure you want to delete ALL vectors? (yes/no): ")
        if confirm.lower() == "yes":
            delete_all_vectors(namespace=args.namespace)
        else:
            print("Cancelled.")
