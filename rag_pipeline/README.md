# Government Schemes RAG Pipeline

## Data Overview
- **Total Schemes**: 6,427
- **Central Government**: 1,137 (location_name/type empty)
- **State Schemes**: 4,675
- **Union Territory Schemes**: 615

### Category Distribution
| Category | Count |
|----------|-------|
| Social Welfare & Empowerment | 1,415 |
| Agriculture, Rural & Environment | 1,133 |
| Education & Learning | 1,028 |
| Business & Entrepreneurship | 691 |
| Women & Child | 463 |
| Skills & Employment | 363 |
| Banking, Financial Services | 308 |
| Health & Wellness | 263 |
| Sports & Culture | 255 |
| Housing & Shelter | 130 |
| Science, IT & Communications | 100 |
| Transport & Infrastructure | 98 |
| Travel & Tourism | 93 |
| Utility & Sanitation | 58 |
| Public Safety, Law & Justice | 29 |

---

## Phase 1: Data Preparation

### 1.1 Metadata Schema

```python
METADATA_SCHEMA = {
    # === IDENTIFICATION ===
    "scheme_id": str,           # Unique ID (hash of scheme_name + scheme_url)
    "scheme_name": str,         # Full scheme name
    "scheme_url": str,          # Original myscheme.gov.in URL
    
    # === LOCATION ===
    "location_type": str,       # "central" | "state" | "union_territory"
    "location_name": str,       # State/UT name or "India" for central
    
    # === CATEGORIZATION ===
    "category_id": int,         # 1-15 (folder number)
    "category_name": str,       # e.g., "education_learning"
    
    # === CHUNK INFO ===
    "chunk_type": str,          # "details" | "benefits" | "eligibility" | 
                                # "application_process" | "documents_required" | 
                                # "sources" | "faq"
    "chunk_index": int,         # Position within chunk_type (0, 1, 2...)
    
    # === FOR FUTURE FEATURES ===
    "language": str,            # "en" (default), future: "hi", "gu", "ta", etc.
    "scraped_at": str,          # Timestamp for freshness tracking
}
```

### 1.2 Chunking Strategy

**Field-Aware Chunking** - Each section becomes separate chunks with section-specific metadata:

```
Scheme: "PM Kisan Samman Nidhi"
├── Chunk 1: details (chunk_type="details")
├── Chunk 2: benefits (chunk_type="benefits")  
├── Chunk 3: eligibility (chunk_type="eligibility")
├── Chunk 4: application_process (chunk_type="application_process")
├── Chunk 5: documents_required (chunk_type="documents_required")
├── Chunk 6: sources (chunk_type="sources")
├── Chunk 7-N: faqs (chunk_type="faq", one per Q&A pair)
```

**Why this approach?**
1. **Precise retrieval**: Query "eligibility for farmers" → retrieves eligibility chunks
2. **Metadata filtering**: Filter by `chunk_type="eligibility"` for eligibility queries
3. **Better context**: Each chunk is semantically coherent
4. **FAQ handling**: Each FAQ as separate chunk improves Q&A retrieval

### 1.3 Chunk Size Guidelines

| Section | Strategy | Max Tokens |
|---------|----------|------------|
| details | Keep whole if <500 tokens, else split | 500 |
| benefits | Keep whole (usually structured lists) | 500 |
| eligibility | Keep whole (critical for matching) | 500 |
| application_process | Split by online/offline | 500 |
| documents_required | Keep whole (usually lists) | 300 |
| sources | Keep whole | 200 |
| faqs | One chunk per Q&A pair | 300 |

**Secondary Splitter**: Use `RecursiveCharacterTextSplitter` only when section exceeds max tokens.

---

## Metadata Design for Future Features

### For Multilingual Support (Feature #2)
```python
"language": "en",  # Future: "hi", "gu", "ta", "bn", etc.
```
- Store translated summaries as separate chunks with `language` metadata
- Query: Filter by user's preferred language

### For Document Verification (Feature #3)
Eligibility chunks will be structured to enable:
```python
# Extract structured eligibility criteria
{
    "age_min": 18,
    "age_max": 60,
    "income_max": 250000,
    "required_docs": ["aadhaar", "income_certificate"],
    "gender": "any",  # or "female", "male"
    "occupation": ["farmer", "labourer"],
    "state": "Gujarat"
}
```

### For Scheme Comparison (Feature #6)
```python
"comparable_fields": ["benefit_amount", "eligibility", "application_mode"]
```

---

## File Structure

```
rag_pipeline/
├── README.md                 # This file
├── config.py                 # Configuration constants
├── data_loader.py            # Load JSONs, validate, normalize
├── chunker.py                # Field-aware chunking logic
├── embeddings.py             # HuggingFace embedding setup
├── vector_store.py           # Pinecone operations
├── retriever.py              # MMR retriever + metadata filters
├── llm.py                    # Google Gemini LLM wrapper
├── chain.py                  # RAG chain with conversation memory
├── chat.py                   # Interactive CLI chatbot
├── requirements.txt          # Python dependencies
├── setup.sh                  # Environment setup script
└── .env                      # API keys (not in git)
```

---

## Chunking Results

After running `chunker.py`:
- **Total Chunks**: 109,595
- **Average chunks per scheme**: 17.1

| Chunk Type | Count |
|------------|-------|
| faq | 69,670 |
| application_process | 7,120 |
| documents_required | 6,808 |
| details | 6,746 |
| benefits | 6,531 |
| eligibility | 6,483 |
| sources_and_references | 6,237 |

**Chunk Sizes**:
- Min: 23 chars (~5 tokens)
- Max: 15,820 chars (~3,955 tokens)
- Average: 413 chars (~103 tokens)

---

## Usage

### Quick Start

```bash
# 1. Setup environment
cd rag_pipeline
./setup.sh

# 2. Configure API keys
cp .env.example .env
# Edit .env with your PINECONE_API_KEY and GOOGLE_API_KEY

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run the chatbot
python chat.py
```

### Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/profile` | Show detected user profile |
| `/reset` | Start new conversation |
| `/debug` | Toggle retrieval debug info |
| `/quit` | Exit chatbot |

### Single Query Mode

```bash
python chat.py --query "I'm a farmer in Gujarat. What schemes can help me?"
```

---

## Next Steps

1. Run `python data_loader.py` - Load and validate all JSONs
2. Run `python chunker.py` - Create chunks with metadata
3. Run `python vector_store.py --upsert` - Upload to Pinecone
4. Run `python chat.py` - Start chatting!
