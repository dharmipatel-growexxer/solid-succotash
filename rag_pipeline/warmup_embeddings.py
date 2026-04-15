"""Pre-download embedding model into local cache for offline-safe startup."""

from embeddings import warmup_embedding_cache


def main() -> None:
    cache_dir = warmup_embedding_cache()
    print(f"Embedding model cached at: {cache_dir}")


if __name__ == "__main__":
    main()
