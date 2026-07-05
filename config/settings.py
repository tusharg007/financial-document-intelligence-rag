"""
Centralized configuration for the Financial Document Intelligence System.
Uses pydantic-settings for environment variable management.
"""
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except Exception:
    class BaseSettings:
        model_config = {}

        def __init__(self, **kwargs):
            for key, value in self.__class__.__dict__.items():
                if key.startswith("_") or callable(value):
                    continue
                if not isinstance(value, (str, int, float, bool)):
                    continue
                setattr(self, key, kwargs.get(key, value))

    def Field(default=None, description=""):
        return default


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- HuggingFace Configuration ---
    huggingface_api_token: str = Field(
        default="",
        description="HuggingFace API token for inference"
    )

    # --- Model Configuration ---
    llm_model_id: str = Field(
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="HuggingFace model ID for text generation"
    )
    embedding_model_id: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model for dense embeddings"
    )
    reranker_model_id: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )

    # --- Fine-Tuned Model (LoRA) ---
    use_finetuned_model: bool = Field(
        default=False,
        description="Use local fine-tuned LoRA model instead of HuggingFace API"
    )
    finetuned_base_model: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Base model for the LoRA adapter"
    )
    finetuned_adapter_path: str = Field(
        default="./lora_findoc",
        description="Path or HuggingFace repo ID for the LoRA adapter"
    )
    finetuned_load_in_4bit: bool = Field(
        default=True,
        description="Load fine-tuned model in 4-bit quantization"
    )

    # --- SEC EDGAR ---
    sec_edgar_user_agent: str = Field(
        default="FinDocIntel research@example.com",
        description="User-Agent for SEC EDGAR API (name + email required)"
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "indexes" / "chroma"),
        description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="sec_filings",
        description="ChromaDB collection name"
    )

    # --- Document Processing ---
    chunk_size: int = Field(default=1000, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Chunk overlap in characters")

    # --- Retrieval ---
    dense_top_k: int = Field(default=20, description="Top-K for dense retrieval")
    sparse_top_k: int = Field(default=20, description="Top-K for sparse retrieval")
    rerank_top_k: int = Field(default=5, description="Top-K after reranking")
    hybrid_alpha: float = Field(
        default=0.5,
        description="Weight for dense vs sparse (0=sparse, 1=dense)"
    )

    # --- Multi-Query ---
    num_generated_queries: int = Field(
        default=3, description="Number of alternative queries to generate"
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(
        default=str(PROJECT_ROOT / "logs"),
        description="Log file directory"
    )

    # --- Data Paths ---
    raw_data_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "raw"),
        description="Raw SEC filings directory"
    )
    processed_data_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "processed"),
        description="Processed data directory"
    )

    # --- API ---
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Global settings instance
settings = Settings()


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        settings.chroma_persist_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.log_dir,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
