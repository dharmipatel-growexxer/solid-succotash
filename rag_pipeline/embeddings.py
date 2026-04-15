"""
Embedding model setup using HuggingFace sentence-transformers.
"""
import os
import logging
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import List

from config import (
    EMBEDDING_CACHE_DIR,
    EMBEDDING_DIMENSION,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL,
)


class EmbeddingModelLoadError(RuntimeError):
    """Raised when embedding model cannot be loaded."""


class EmbeddingModel:
    """Wrapper for HuggingFace sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self._model = None
        self.cache_dir = Path(EMBEDDING_CACHE_DIR)
    
    def _prepare_cache(self):
        """Prepare local cache directories and env vars for offline reuse."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(self.cache_dir))
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        for logger_name in ("sentence_transformers", "transformers", "huggingface_hub"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._prepare_cache()
                sink = StringIO()
                with redirect_stdout(sink), redirect_stderr(sink):
                    self._model = SentenceTransformer(
                        self.model_name,
                        cache_folder=str(self.cache_dir),
                        local_files_only=EMBEDDING_LOCAL_FILES_ONLY,
                    )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as exc:
                message = str(exc)
                network_issue = any(
                    token in message.lower()
                    for token in [
                        "temporary failure in name resolution",
                        "failed to establish a new connection",
                        "connection error",
                        "max retries exceeded",
                        "httpsconnectionpool",
                    ]
                )

                if EMBEDDING_LOCAL_FILES_ONLY:
                    raise EmbeddingModelLoadError(
                        "Embedding model load failed in local-files-only mode. "
                        f"Expected model in cache at {self.cache_dir}. "
                        "Disable EMBEDDING_LOCAL_FILES_ONLY or pre-download the model. "
                        f"Original error: {exc}"
                    ) from exc

                if network_issue:
                    raise EmbeddingModelLoadError(
                        "Embedding model download failed due to network/DNS issue while contacting Hugging Face. "
                        f"Cache directory: {self.cache_dir}. "
                        "Fix internet/DNS or pre-download the model and run with EMBEDDING_LOCAL_FILES_ONLY=true. "
                        f"Original error: {exc}"
                    ) from exc

                raise EmbeddingModelLoadError(
                    f"Embedding model failed to load: {exc}"
                ) from exc
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
            show_progress_bar=False,
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


def warmup_embedding_cache(model_name: str = EMBEDDING_MODEL) -> str:
    """Preload embedding model into local cache directory."""
    model = EmbeddingModel(model_name=model_name)
    model._load_model()
    return str(model.cache_dir)


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
