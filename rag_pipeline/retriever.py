"""
Retrieval layer with MMR (Maximal Marginal Relevance) and dynamic metadata filtering.
"""
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from config import (
    CATEGORY_MAP,
    CATEGORY_NAME_TO_ID,
    CENTRAL_GOVT_LABEL,
    LOCATION_TYPE_CENTRAL,
    LOCATION_TYPE_STATE,
    LOCATION_TYPE_UT,
    MMR_FETCH_K,
    MMR_LAMBDA,
    PINECONE_INDEX_NAME,
    RETRIEVAL_K,
)
from embeddings import EmbeddingModelLoadError, get_embedding_model


class RetrievalError(RuntimeError):
    """Raised when retrieval pipeline fails."""


# === State/UT Recognition ===
STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
}

UNION_TERRITORIES = {
    "andaman and nicobar islands", "chandigarh", 
    "dadra and nagar haveli and daman and diu", "delhi", "jammu and kashmir",
    "ladakh", "lakshadweep", "puducherry",
}

# Common aliases
STATE_ALIASES = {
    "up": "uttar pradesh",
    "mp": "madhya pradesh",
    "hp": "himachal pradesh",
    "ap": "andhra pradesh",
    "wb": "west bengal",
    "tn": "tamil nadu",
    "jk": "jammu and kashmir",
    "j&k": "jammu and kashmir",
    "uk": "uttarakhand",
    "cg": "chhattisgarh",
    "jh": "jharkhand",
}

# Category keywords for detection
CATEGORY_KEYWORDS = {
    "agriculture_rural_environment": [
        "farmer", "farming", "agriculture", "crop", "kisan", "rural", "land",
        "irrigation", "seed", "fertilizer", "animal husbandry", "dairy", "fishery",
    ],
    "banking_financial_services_insurance": [
        "loan", "bank", "insurance", "credit", "subsidy", "finance", "mudra",
    ],
    "business_entrepreneurship": [
        "business", "startup", "entrepreneur", "msme", "enterprise", "self-employment",
        "shop", "industry", "manufacturing",
    ],
    "education_learning": [
        "education", "student", "school", "college", "university", "scholarship",
        "study", "degree", "exam", "tuition", "hostel",
    ],
    "health_wellness": [
        "health", "medical", "hospital", "treatment", "disease", "medicine",
        "doctor", "ayushman", "healthcare", "disability", "disabled",
    ],
    "housing_shelter": [
        "house", "housing", "home", "shelter", "awas", "building", "construction",
        "plot", "flat", "apartment",
    ],
    "public_safety_law_justice": [
        "legal", "court", "law", "justice", "victim", "crime", "police",
    ],
    "science_it_communications": [
        "science", "research", "technology", "it", "computer", "innovation",
        "scientist", "laboratory",
    ],
    "skills_employment": [
        "skill", "training", "employment", "job", "vocational", "apprentice",
        "placement", "unemployed", "labour", "worker",
    ],
    "social_welfare_empowerment": [
        "sc", "st", "obc", "minority", "backward", "tribal", "caste",
        "welfare", "pension", "widow", "destitute", "bpl", "poor",
    ],
    "sports_culture": [
        "sport", "sports", "athlete", "player", "game", "culture", "art",
        "music", "dance", "theatre",
    ],
    "transport_infrastructure": [
        "transport", "vehicle", "road", "highway", "bus", "taxi", "auto",
    ],
    "travel_tourism": [
        "travel", "tourism", "pilgrimage", "yatra", "tourist",
    ],
    "utility_sanitation": [
        "water", "electricity", "gas", "sanitation", "toilet", "sewage", "lpg",
    ],
    "women_child": [
        "woman", "women", "girl", "female", "mother", "pregnant", "maternity",
        "child", "children", "daughter", "ladki", "mahila",
    ],
}

# Gender keywords
GENDER_KEYWORDS = {
    "female": ["woman", "women", "girl", "female", "mother", "widow", "mahila", "ladki", "daughter"],
    "male": ["man", "men", "boy", "male", "father"],
}

# Occupation keywords
OCCUPATION_KEYWORDS = {
    "farmer": ["farmer", "kisan", "agriculture", "farming", "cultivator"],
    "student": ["student", "studying", "school", "college", "university"],
    "labourer": ["labour", "labourer", "worker", "construction", "daily wage", "mazdoor"],
    "artisan": ["artisan", "craftsman", "handicraft", "weaver", "potter"],
    "fisherman": ["fisherman", "fishing", "fisher", "matsya"],
    "senior_citizen": ["senior citizen", "old age", "elderly", "retired", "60 year", "65 year"],
    "entrepreneur": ["entrepreneur", "business", "startup", "self-employed", "shop owner"],
}


@dataclass
class UserProfile:
    """Extracted user profile from query for filtering."""
    
    state: Optional[str] = None
    location_type: Optional[str] = None  # state, union_territory, or None for central
    category_hints: List[str] = field(default_factory=list)
    gender: Optional[str] = None  # male, female, or None
    occupation: Optional[str] = None
    age: Optional[int] = None
    income: Optional[int] = None
    caste_category: Optional[str] = None  # sc, st, obc, general
    is_bpl: bool = False
    is_disabled: bool = False
    raw_query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state,
            "location_type": self.location_type,
            "category_hints": self.category_hints,
            "gender": self.gender,
            "occupation": self.occupation,
            "age": self.age,
            "income": self.income,
            "caste_category": self.caste_category,
            "is_bpl": self.is_bpl,
            "is_disabled": self.is_disabled,
        }


def extract_user_profile(query: str) -> UserProfile:
    """
    Extract user profile information from natural language query.
    
    Examples:
        "I am a 45-year-old farmer in Gujarat" 
        -> UserProfile(state="Gujarat", occupation="farmer", age=45)
        
        "scholarship for SC girl students in UP"
        -> UserProfile(state="Uttar Pradesh", gender="female", caste_category="sc", category_hints=["education"])
    """
    query_lower = query.lower()
    profile = UserProfile(raw_query=query)
    
    # === Extract State ===
    # Check aliases first
    for alias, full_name in STATE_ALIASES.items():
        # Match whole word
        if re.search(rf'\b{re.escape(alias)}\b', query_lower):
            profile.state = full_name.title()
            profile.location_type = LOCATION_TYPE_STATE
            break
    
    # Check full state names
    if not profile.state:
        for state in STATES:
            if state in query_lower:
                profile.state = state.title()
                profile.location_type = LOCATION_TYPE_STATE
                break
    
    # Check UTs
    if not profile.state:
        for ut in UNION_TERRITORIES:
            if ut in query_lower:
                profile.state = ut.title()
                profile.location_type = LOCATION_TYPE_UT
                break
    
    # === Extract Age ===
    age_patterns = [
        r'(\d+)\s*(?:year|yr)s?\s*old',
        r'age\s*(?:is|:)?\s*(\d+)',
        r'(\d+)\s*(?:year|yr)s?\s*of\s*age',
        r"i(?:'m| am)\s*(\d+)",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, query_lower)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:  # Sanity check
                profile.age = age
                break
    
    # === Extract Gender ===
    for gender, keywords in GENDER_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                profile.gender = gender
                break
        if profile.gender:
            break
    
    # === Extract Occupation ===
    for occupation, keywords in OCCUPATION_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                profile.occupation = occupation
                break
        if profile.occupation:
            break
    
    # === Extract Caste Category ===
    if re.search(r'\bsc\b|scheduled caste', query_lower):
        profile.caste_category = "sc"
    elif re.search(r'\bst\b|scheduled tribe|tribal', query_lower):
        profile.caste_category = "st"
    elif re.search(r'\bobc\b|other backward', query_lower):
        profile.caste_category = "obc"
    elif "general" in query_lower:
        profile.caste_category = "general"
    
    # === Extract BPL Status ===
    if re.search(r'\bbpl\b|below poverty|poor family', query_lower):
        profile.is_bpl = True
    
    # === Extract Disability Status ===
    if re.search(r'disab|handicap|divyang|pwd|differently.?abled', query_lower):
        profile.is_disabled = True
    
    # === Extract Income ===
    income_patterns = [
        r'income\s*(?:is|:)?\s*(?:rs\.?|₹)?\s*(\d+(?:,\d+)*)',
        r'(?:rs\.?|₹)\s*(\d+(?:,\d+)*)\s*(?:per\s*(?:month|year)|income)',
        r'earn(?:ing|s)?\s*(?:rs\.?|₹)?\s*(\d+(?:,\d+)*)',
    ]
    for pattern in income_patterns:
        match = re.search(pattern, query_lower)
        if match:
            income_str = match.group(1).replace(',', '')
            profile.income = int(income_str)
            break
    
    # === Extract Category Hints ===
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                if category not in profile.category_hints:
                    profile.category_hints.append(category)
                break
    
    return profile


def build_metadata_filter(profile: UserProfile) -> Optional[Dict[str, Any]]:
    """
    Build Pinecone metadata filter from user profile.
    
    Pinecone filter syntax:
    - {"field": "value"} for exact match
    - {"field": {"$in": [values]}} for OR match
    - {"$and": [filters]} for AND
    - {"$or": [filters]} for OR
    """
    filters = []
    
    # === Location Filter ===
    if profile.state:
        # Include both state-specific AND central schemes
        filters.append({
            "$or": [
                {"location_name": profile.state},
                {"location_type": LOCATION_TYPE_CENTRAL},
            ]
        })
    
    # === Category Filter ===
    if profile.category_hints:
        # Map category names to IDs
        category_ids = [
            CATEGORY_NAME_TO_ID.get(cat)
            for cat in profile.category_hints
            if cat in CATEGORY_NAME_TO_ID
        ]
        if category_ids:
            filters.append({
                "category_id": {"$in": category_ids}
            })
    
    # Combine filters
    if not filters:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}


def mmr_rerank(
    query_embedding: List[float],
    candidates: List[Dict],
    k: int = 10,
    lambda_mult: float = 0.7
) -> List[Dict]:
    """
    Maximal Marginal Relevance reranking for diversity.
    
    MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected_docs))
    
    Args:
        query_embedding: Query vector
        candidates: List of candidate documents with 'values' (embedding) and 'score'
        k: Number of documents to return
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)
    
    Returns:
        Reranked list of k documents
    """
    import numpy as np
    
    if len(candidates) <= k:
        return candidates
    
    # Convert to numpy for efficient computation
    query_vec = np.array(query_embedding)
    doc_vecs = np.array([c.get('values', c.get('embedding', [])) for c in candidates])
    
    # If no embeddings available, return top-k by score
    if doc_vecs.size == 0 or len(doc_vecs[0]) == 0:
        return candidates[:k]
    
    # Normalize vectors
    query_vec = query_vec / np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_vecs = doc_vecs / np.where(doc_norms > 0, doc_norms, 1)
    
    # Compute query-document similarities
    query_sims = doc_vecs @ query_vec
    
    selected_indices = []
    remaining_indices = list(range(len(candidates)))
    
    for _ in range(min(k, len(candidates))):
        if not remaining_indices:
            break
        
        mmr_scores = []
        for idx in remaining_indices:
            # Relevance to query
            relevance = query_sims[idx]
            
            # Maximum similarity to already selected docs
            if selected_indices:
                selected_vecs = doc_vecs[selected_indices]
                doc_sims = selected_vecs @ doc_vecs[idx]
                max_sim_to_selected = np.max(doc_sims)
            else:
                max_sim_to_selected = 0
            
            # MMR score
            mmr = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
            mmr_scores.append((idx, mmr))
        
        # Select document with highest MMR
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return [candidates[i] for i in selected_indices]


class SchemeRetriever:
    """
    Retriever for government schemes with MMR and metadata filtering.
    """
    
    def __init__(
        self,
        k: int = RETRIEVAL_K,
        fetch_k: int = MMR_FETCH_K,
        lambda_mult: float = MMR_LAMBDA,
    ):
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.embedding_model = get_embedding_model()
        self._index = None
    
    def _get_index(self):
        """Lazy load Pinecone index."""
        if self._index is None:
            from pinecone import Pinecone
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not set")
            pc = Pinecone(api_key=api_key)
            self._index = pc.Index(PINECONE_INDEX_NAME)
        return self._index
    
    def retrieve(
        self,
        query: str,
        profile: Optional[UserProfile] = None,
        k: Optional[int] = None,
        use_mmr: bool = True,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant scheme chunks.
        
        Args:
            query: Search query
            profile: Optional user profile for filtering
            k: Number of results (overrides default)
            use_mmr: Whether to apply MMR reranking
            filter_dict: Optional explicit filter (overrides profile-based filter)
        
        Returns:
            List of retrieved documents with metadata
        """
        k = k or self.k
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
        except EmbeddingModelLoadError as exc:
            raise RetrievalError(str(exc)) from exc
        except Exception as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        
        if filter_dict is None and profile:
            filter_dict = build_metadata_filter(profile)
        
        # Fetch more candidates for MMR
        fetch_k = self.fetch_k if use_mmr else k
        
        try:
            index = self._get_index()
            results = index.query(
                vector=query_embedding,
                top_k=fetch_k,
                filter=filter_dict,
                include_metadata=True,
                include_values=use_mmr,
            )
        except Exception as exc:
            raise RetrievalError(f"Pinecone query failed: {exc}") from exc
        
        candidates = results.matches
        
        # Apply MMR reranking
        if use_mmr and len(candidates) > k:
            # Convert Pinecone results to dict format for MMR
            candidate_dicts = [
                {
                    "id": m.id,
                    "score": m.score,
                    "values": m.values if hasattr(m, 'values') else [],
                    "metadata": m.metadata,
                }
                for m in candidates
            ]
            reranked = mmr_rerank(query_embedding, candidate_dicts, k, self.lambda_mult)
            return reranked
        
        # Return top-k without MMR
        return [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata,
            }
            for m in candidates[:k]
        ]

    def retrieve_with_debug(
        self,
        query: str,
        profile: Optional[UserProfile] = None,
        k: Optional[int] = None,
        use_mmr: bool = True,
        filter_dict: Optional[Dict] = None,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Retrieve documents and return debug metadata."""
        k = k or self.k
        resolved_filter = filter_dict
        if resolved_filter is None and profile:
            resolved_filter = build_metadata_filter(profile)

        start_time = time.perf_counter()
        results = self.retrieve(
            query=query,
            profile=profile,
            k=k,
            use_mmr=use_mmr,
            filter_dict=resolved_filter,
        )
        retrieval_ms = round((time.perf_counter() - start_time) * 1000, 2)

        debug = {
            "query": query,
            "k": k,
            "fetch_k": self.fetch_k if use_mmr else k,
            "use_mmr": use_mmr,
            "lambda_mult": self.lambda_mult if use_mmr else None,
            "metadata_filter": resolved_filter,
            "retrieved_count": len(results),
            "latency_ms": retrieval_ms,
        }
        return results, debug
    
    def retrieve_with_profile_extraction(
        self,
        query: str,
        k: Optional[int] = None,
        use_mmr: bool = True,
    ) -> Tuple[List[Dict], UserProfile]:
        """
        Extract profile from query and retrieve.
        
        Returns:
            Tuple of (results, extracted_profile)
        """
        profile = extract_user_profile(query)
        results = self.retrieve(query, profile=profile, k=k, use_mmr=use_mmr)
        return results, profile


def format_retrieved_docs(results: List[Dict], show_scores: bool = True) -> str:
    """Format retrieved documents for display or LLM context."""
    formatted = []
    
    for i, doc in enumerate(results):
        meta = doc.get("metadata", {})
        score = doc.get("score", 0)
        
        header = f"[{i+1}] {meta.get('scheme_name', 'Unknown Scheme')}"
        if show_scores:
            header += f" (score: {score:.3f})"
        
        lines = [
            header,
            f"    Location: {meta.get('location_name', 'India')} ({meta.get('location_type', 'central')})",
            f"    Category: {meta.get('category_name', 'N/A')}",
            f"    Section: {meta.get('chunk_type', 'N/A')}",
            f"    URL: {meta.get('scheme_url', 'N/A')}",
            f"    Content: {meta.get('text', '')[:300]}...",
        ]
        formatted.append("\n".join(lines))
    
    return "\n\n".join(formatted)


# === Singleton Retriever ===
_retriever = None

def get_retriever() -> SchemeRetriever:
    """Get or create singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = SchemeRetriever()
    return _retriever


if __name__ == "__main__":
    # Test the retriever
    print("=" * 60)
    print("Government Schemes Retriever - Test Suite")
    print("=" * 60)
    
    retriever = get_retriever()
    
    # Test queries
    test_queries = [
        "I am a 45-year-old farmer in Gujarat",
        "scholarship for SC girl students in Maharashtra",
        "housing scheme for poor families in Delhi",
        "skill training for unemployed youth",
        "health insurance for senior citizens",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        
        # Extract profile
        profile = extract_user_profile(query)
        print(f"\nExtracted Profile:")
        for key, value in profile.to_dict().items():
            if value:
                print(f"  {key}: {value}")
        
        # Build filter
        filter_dict = build_metadata_filter(profile)
        print(f"\nMetadata Filter: {filter_dict}")
        
        # Retrieve
        results = retriever.retrieve(query, profile=profile, k=5)
        print(f"\nRetrieved {len(results)} results:")
        print(format_retrieved_docs(results))
