"""
Field-aware chunking logic for government scheme data.
Creates semantically coherent chunks with proper metadata.
"""
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback: simple implementation
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=100, 
                         length_function=len, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
                self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
            
            def split_text(self, text: str) -> list:
                """Simple recursive splitting implementation."""
                if self.length_function(text) <= self.chunk_size:
                    return [text]
                
                chunks = []
                for sep in self.separators:
                    if sep in text:
                        parts = text.split(sep)
                        current_chunk = ""
                        for part in parts:
                            if self.length_function(current_chunk + sep + part) <= self.chunk_size:
                                current_chunk = current_chunk + sep + part if current_chunk else part
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = part
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        return chunks if chunks else [text]
                
                # Fallback: hard split
                return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

from config import (
    CHARS_PER_TOKEN,
    CHUNK_CONFIG,
    CHUNK_TYPES,
    CHUNKS_OUTPUT_DIR,
)
from data_loader import SchemeData, load_all_schemes


@dataclass
class Chunk:
    """A single chunk ready for embedding and storage."""
    
    id: str                     # Unique chunk ID
    text: str                   # The actual text content
    metadata: Dict[str, Any]    # All metadata for filtering/retrieval
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"],
        )


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def create_text_splitter(chunk_type: str) -> RecursiveCharacterTextSplitter:
    """Create a text splitter configured for the given chunk type."""
    config = CHUNK_CONFIG.get(chunk_type, {"max_tokens": 500, "overlap": 50})
    
    # Convert tokens to characters
    chunk_size = config["max_tokens"] * CHARS_PER_TOKEN
    chunk_overlap = config["overlap"] * CHARS_PER_TOKEN
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )


def format_section_text(section_data: Any, section_name: str) -> str:
    """
    Format section data into readable text.
    Handles both list and dict formats.
    """
    if isinstance(section_data, list):
        # Join list items with newlines
        return "\n".join(str(item) for item in section_data if item)
    
    elif isinstance(section_data, dict):
        # For application_process with online/offline subsections
        parts = []
        for key, value in section_data.items():
            if isinstance(value, list) and value:
                parts.append(f"{key.upper()}:")
                parts.extend(str(item) for item in value if item)
        return "\n".join(parts)
    
    elif isinstance(section_data, str):
        return section_data
    
    return ""


def format_faq_text(faq: Dict[str, str]) -> str:
    """Format a single FAQ into readable text."""
    question = faq.get("question", "").strip()
    answer = faq.get("answer", "").strip()
    
    if not question:
        return ""
    
    if answer and answer != "Answer not captured.":
        return f"Q: {question}\nA: {answer}"
    else:
        return f"Q: {question}"


def create_chunk_id(scheme_id: str, category_id: int, chunk_type: str, index: int) -> str:
    """Generate a unique chunk ID including category to handle cross-category schemes."""
    return f"{scheme_id}_cat{category_id}_{chunk_type}_{index}"


def chunk_scheme(scheme: SchemeData) -> List[Chunk]:
    """
    Create chunks from a single scheme.
    
    Strategy:
    - Each section becomes one or more chunks
    - Each FAQ becomes a separate chunk
    - Large sections are split using RecursiveCharacterTextSplitter
    """
    chunks = []
    base_metadata = scheme.to_metadata_base()
    
    # Process each section
    section_order = [
        "details",
        "benefits", 
        "eligibility",
        "application_process",
        "documents_required",
        "sources_and_references",
    ]
    
    for section_name in section_order:
        section_data = scheme.sections.get(section_name)
        if not section_data:
            continue
        
        # Format section text
        text = format_section_text(section_data, section_name)
        if not text.strip():
            continue
        
        # Add section header for context
        header = section_name.replace("_", " ").title()
        full_text = f"[{scheme.scheme_name}]\n[{header}]\n\n{text}"
        
        # Check if we need to split
        config = CHUNK_CONFIG.get(section_name, {"max_tokens": 500})
        max_chars = config["max_tokens"] * CHARS_PER_TOKEN
        
        if len(full_text) <= max_chars:
            # Single chunk for this section
            chunk_id = create_chunk_id(scheme.scheme_id, scheme.category_id, section_name, 0)
            metadata = {
                **base_metadata,
                "chunk_type": section_name,
                "chunk_index": 0,
            }
            chunks.append(Chunk(id=chunk_id, text=full_text, metadata=metadata))
        else:
            # Split into multiple chunks
            splitter = create_text_splitter(section_name)
            split_texts = splitter.split_text(full_text)
            
            for idx, split_text in enumerate(split_texts):
                chunk_id = create_chunk_id(scheme.scheme_id, scheme.category_id, section_name, idx)
                metadata = {
                    **base_metadata,
                    "chunk_type": section_name,
                    "chunk_index": idx,
                }
                chunks.append(Chunk(id=chunk_id, text=split_text, metadata=metadata))
    
    # Process FAQs - each FAQ is a separate chunk
    for idx, faq in enumerate(scheme.faqs):
        faq_text = format_faq_text(faq)
        if not faq_text.strip():
            continue
        
        # Add scheme context to FAQ
        full_text = f"[{scheme.scheme_name}]\n[FAQ]\n\n{faq_text}"
        
        chunk_id = create_chunk_id(scheme.scheme_id, scheme.category_id, "faq", idx)
        metadata = {
            **base_metadata,
            "chunk_type": "faq",
            "chunk_index": idx,
        }
        chunks.append(Chunk(id=chunk_id, text=full_text, metadata=metadata))
    
    return chunks


def chunk_all_schemes(
    schemes: List[SchemeData],
    verbose: bool = True
) -> List[Chunk]:
    """
    Create chunks from all schemes.
    
    Args:
        schemes: List of SchemeData objects
        verbose: Print progress information
    
    Returns:
        List of Chunk objects (deduplicated by ID)
    """
    all_chunks = []
    seen_ids = set()
    duplicates_skipped = 0
    total = len(schemes)
    
    if verbose:
        print(f"Chunking {total} schemes...")
    
    for idx, scheme in enumerate(schemes):
        if verbose and (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} schemes...")
        
        chunks = chunk_scheme(scheme)
        
        # Deduplicate by chunk ID
        for chunk in chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                all_chunks.append(chunk)
            else:
                duplicates_skipped += 1
    
    if verbose:
        print(f"\nCreated {len(all_chunks)} unique chunks from {total} schemes.")
        print(f"Duplicates skipped: {duplicates_skipped}")
        print(f"Average chunks per scheme: {len(all_chunks) / total:.1f}")
    
    return all_chunks


def get_chunk_stats(chunks: List[Chunk]) -> Dict[str, Any]:
    """Generate statistics about chunks."""
    from collections import Counter
    
    chunk_types = Counter(c.metadata["chunk_type"] for c in chunks)
    text_lengths = [len(c.text) for c in chunks]
    token_estimates = [estimate_tokens(c.text) for c in chunks]
    
    return {
        "total_chunks": len(chunks),
        "chunk_types": dict(chunk_types),
        "text_length": {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "avg": sum(text_lengths) / len(text_lengths),
        },
        "token_estimate": {
            "min": min(token_estimates),
            "max": max(token_estimates),
            "avg": sum(token_estimates) / len(token_estimates),
        },
    }


def save_chunks(chunks: List[Chunk], output_dir: Path = CHUNKS_OUTPUT_DIR) -> Path:
    """Save chunks to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(chunks)} chunks to {output_file}")
    return output_file


def load_chunks(input_file: Path = CHUNKS_OUTPUT_DIR / "chunks.json") -> List[Chunk]:
    """Load chunks from a JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [Chunk.from_dict(c) for c in data]


def print_sample_chunks(chunks: List[Chunk], n: int = 3) -> None:
    """Print sample chunks for validation."""
    print(f"\n=== Sample Chunks (showing {n}) ===\n")
    
    # Get one of each type
    seen_types = set()
    samples = []
    
    for chunk in chunks:
        chunk_type = chunk.metadata["chunk_type"]
        if chunk_type not in seen_types and len(samples) < n:
            samples.append(chunk)
            seen_types.add(chunk_type)
    
    for i, chunk in enumerate(samples):
        print(f"--- Chunk {i + 1} ---")
        print(f"ID: {chunk.id}")
        print(f"Type: {chunk.metadata['chunk_type']}")
        print(f"Scheme: {chunk.metadata['scheme_name'][:50]}...")
        print(f"Location: {chunk.metadata['location_name']} ({chunk.metadata['location_type']})")
        print(f"Category: {chunk.metadata['category_name']}")
        print(f"Text length: {len(chunk.text)} chars (~{estimate_tokens(chunk.text)} tokens)")
        print(f"Text preview:\n{chunk.text[:300]}...")
        print()


if __name__ == "__main__":
    # Load schemes
    schemes = load_all_schemes(verbose=True)
    
    # Create chunks
    chunks = chunk_all_schemes(schemes, verbose=True)
    
    # Print statistics
    stats = get_chunk_stats(chunks)
    print("\n=== Chunk Statistics ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"\nChunk types: {stats['chunk_types']}")
    print(f"\nText length (chars): min={stats['text_length']['min']}, "
          f"max={stats['text_length']['max']}, avg={stats['text_length']['avg']:.0f}")
    print(f"Token estimate: min={stats['token_estimate']['min']}, "
          f"max={stats['token_estimate']['max']}, avg={stats['token_estimate']['avg']:.0f}")
    
    # Print samples
    print_sample_chunks(chunks, n=5)
    
    # Save chunks
    save_chunks(chunks)
