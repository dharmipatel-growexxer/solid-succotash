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
    groq \
    python-dotenv \
    tqdm \
    numpy \
    streamlit

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the upsert pipeline:"
echo "   cd rag_pipeline"
echo "   python vector_store.py --action upsert"
echo ""
