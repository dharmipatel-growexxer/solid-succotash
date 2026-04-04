#!/bin/bash
# Setup script for Government Schemes RAG Pipeline

echo "=== Government Schemes RAG Pipeline Setup ==="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip

# Core dependencies
pip install \
    sentence-transformers \
    pinecone \
    google-genai \
    python-dotenv \
    tqdm

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your API keys:"
echo "   cp .env.example .env"
echo "   nano .env  # or use any editor"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the upsert pipeline:"
echo "   cd rag_pipeline"
echo "   python vector_store.py --action upsert"
echo ""
