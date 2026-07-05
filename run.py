"""
Quick-start script for the Financial Document Intelligence System.

Usage:
    python run.py streamlit   - Launch the Streamlit dashboard
    python run.py api         - Launch the FastAPI server
    python run.py test        - Run the test suite
    python run.py index       - Index sample documents
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_streamlit():
    """Launch the Streamlit dashboard."""
    os.system("streamlit run app.py")


def run_api():
    """Launch the FastAPI server."""
    from api import app
    import uvicorn
    from config.settings import settings
    uvicorn.run("api:app", host=settings.api_host, port=settings.api_port, reload=True)


def run_tests():
    """Run the test suite."""
    import pytest
    pytest.main(["tests/", "-v", "--tb=short"])


def run_index():
    """Index sample documents into the vector store."""
    from config.settings import ensure_directories
    ensure_directories()
    
    from src.data.sample_data import get_all_documents
    from src.data.document_parser import get_parser
    from src.embeddings.dense_embedder import get_dense_embedder
    from src.embeddings.sparse_embedder import get_sparse_embedder
    
    print("📄 Loading sample documents...")
    docs = get_all_documents()
    parser = get_parser()
    processed = parser.process_sample_documents(docs)
    print(f"   Loaded {len(processed)} documents")
    
    print("🔢 Building dense index (ChromaDB)...")
    dense = get_dense_embedder()
    dense.add_documents(processed)
    stats = dense.get_collection_stats()
    print(f"   Indexed {stats['total_documents']} documents")
    
    print("📝 Building sparse index (BM25)...")
    sparse = get_sparse_embedder()
    sparse.build_index(processed)
    bm25_stats = sparse.get_stats()
    print(f"   Indexed {bm25_stats['total_documents']} documents")
    
    print("✅ Indexing complete!")


COMMANDS = {
    "streamlit": run_streamlit,
    "api": run_api,
    "test": run_tests,
    "index": run_index,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"Available commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)
    
    COMMANDS[sys.argv[1]]()
