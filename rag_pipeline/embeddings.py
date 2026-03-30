"""
Embedding model setup using HuggingFace sentence-transformers.
"""
import os
from typing import List

from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingModel:
    """Wrapper for HuggingFace sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self._model = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                print(f"Model loaded. Dimension: {self.dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed multiple documents."""
        model = self._load_model()
        embeddings = model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_documents for compatibility."""
        return self.embed_documents(texts)


# Singleton instance
_embedding_model = None

def get_embedding_model() -> EmbeddingModel:
    """Get or create the singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


if __name__ == "__main__":
    # Test the embedding model
    model = get_embedding_model()
    
    # Test single query
    query = "What schemes are available for farmers in Gujarat?"
    embedding = model.embed_query(query)
    print(f"Query: {query}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch
    docs = [
        "PM Kisan provides financial support to farmers",
        "Scholarship for students from rural areas",
        "Healthcare scheme for senior citizens"
    ]
    embeddings = model.embed_documents(docs)
    print(f"\nEmbedded {len(docs)} documents")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
