"""
Configuration constants for the RAG pipeline.
"""
from pathlib import Path

# === PATHS ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "schemes_data_json"
CHUNKS_OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"

# === CATEGORY MAPPING ===
CATEGORY_MAP = {
    1: "agriculture_rural_environment",
    2: "banking_financial_services_insurance",
    3: "business_entrepreneurship",
    4: "education_learning",
    5: "health_wellness",
    6: "housing_shelter",
    7: "public_safety_law_justice",
    8: "science_it_communications",
    9: "skills_employment",
    10: "social_welfare_empowerment",
    11: "sports_culture",
    12: "transport_infrastructure",
    13: "travel_tourism",
    14: "utility_sanitation",
    15: "women_child",
}

# Reverse mapping for lookup
CATEGORY_NAME_TO_ID = {v: k for k, v in CATEGORY_MAP.items()}

# === LOCATION NORMALIZATION ===
CENTRAL_GOVT_LABEL = "India"
LOCATION_TYPE_CENTRAL = "central"
LOCATION_TYPE_STATE = "state"
LOCATION_TYPE_UT = "union_territory"

# === CHUNKING CONFIG ===
CHUNK_CONFIG = {
    "details": {"max_tokens": 500, "overlap": 50},
    "benefits": {"max_tokens": 500, "overlap": 50},
    "eligibility": {"max_tokens": 500, "overlap": 50},
    "application_process": {"max_tokens": 500, "overlap": 50},
    "documents_required": {"max_tokens": 300, "overlap": 30},
    "sources_and_references": {"max_tokens": 200, "overlap": 0},
    "faq": {"max_tokens": 300, "overlap": 0},
}

# Approximate tokens per character (for estimation)
CHARS_PER_TOKEN = 4

# === EMBEDDING CONFIG ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# === PINECONE CONFIG ===
PINECONE_INDEX_NAME = "govt-scheme"
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# === RETRIEVAL CONFIG ===
RETRIEVAL_K = 10  # Number of documents to retrieve
MMR_FETCH_K = 20  # Fetch more for MMR diversity
MMR_LAMBDA = 0.7  # Balance between relevance (1.0) and diversity (0.0)

# === LLM CONFIG ===
# Using Google Gemini (free tier)
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-1.5-flash"  # Free tier model, fast and capable
# Alternative models:
# - "gemini-1.5-pro" (more capable, also free with limits)
# - "gemini-2.0-flash-exp" (experimental, latest)
LLM_TEMPERATURE = 0.3
MEMORY_WINDOW_SIZE = 5  # Number of conversation turns to remember

# === CHUNK TYPES ===
CHUNK_TYPES = [
    "details",
    "benefits", 
    "eligibility",
    "application_process",
    "documents_required",
    "sources_and_references",
    "faq",
]

# === METADATA KEYS ===
# These are the metadata fields that will be stored in Pinecone
METADATA_KEYS = [
    "scheme_id",
    "scheme_name",
    "scheme_url",
    "location_type",
    "location_name",
    "category_id",
    "category_name",
    "chunk_type",
    "chunk_index",
    "language",
    "scraped_at",
]

# === SUPPORTED LANGUAGES (for future multilingual support) ===
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
}

DEFAULT_LANGUAGE = "en"
