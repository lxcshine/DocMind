"""
DocMind Backend Configuration

Refactored to use RAG-Anything architecture:
  PDF -> MinerU parsing -> LightRAG (KG + Vector DB) -> Multi-mode query
"""
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Settings:
    """DocMind settings based on RAG-Anything architecture"""

    # ===== LLM Configuration =====
    LLM_MODE: str = os.getenv("LLM_MODE", "online")

    # Gemini Model Configuration (for KG extraction, summarization, chat)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Vision Model Configuration (for multimodal image/table/equation analysis)
    VISION_API_KEY: str = os.getenv("VISION_API_KEY", "")
    VISION_BASE_URL: str = os.getenv("VISION_BASE_URL", "https://api.openai.com/v1/")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")

    # Embedding Configuration (for vectorization)
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY",
        os.getenv("GEMINI_API_KEY", ""))
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL",
        os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_MAX_LENGTH: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "3072"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))

    # ===== RAG-Anything / LightRAG Configuration =====
    # Storage
    RAG_WORKING_DIR: str = os.getenv("RAG_WORKING_DIR",
        str(Path(__file__).resolve().parent.parent / "rag_storage"))
    RAG_OUTPUT_DIR: str = os.getenv("RAG_OUTPUT_DIR",
        str(Path(__file__).resolve().parent.parent / "output"))

    # Chunking
    CHUNK_TOKEN_SIZE: int = int(os.getenv("CHUNK_TOKEN_SIZE", "1200"))
    CHUNK_OVERLAP_TOKEN_SIZE: int = int(os.getenv("CHUNK_OVERLAP_TOKEN_SIZE", "100"))

    # KG Extraction
    MAX_ENTITY_TOKENS: int = int(os.getenv("MAX_ENTITY_TOKENS", "4096"))
    MAX_RELATION_TOKENS: int = int(os.getenv("MAX_RELATION_TOKENS", "4096"))
    MAX_TOTAL_TOKENS: int = int(os.getenv("MAX_TOTAL_TOKENS", "8192"))
    MAX_GRAPH_NODES: int = int(os.getenv("MAX_GRAPH_NODES", "1000"))

    # Retrieval
    COSINE_THRESHOLD: float = float(os.getenv("COSINE_THRESHOLD", "0.4"))
    TOP_K: int = int(os.getenv("TOP_K", "60"))
    RELATED_CHUNK_NUMBER: int = int(os.getenv("RELATED_CHUNK_NUMBER", "5"))

    # ===== MinerU Parser Configuration =====
    PARSER: str = os.getenv("PARSER", "mineru")
    PARSE_METHOD: str = os.getenv("PARSE_METHOD", "auto")
    # MinerU backend. `pipeline` is the most reliable across hardware;
    # `hybrid-auto-engine` is higher quality but slower and prone to
    # internal per-task timeouts on CPU-only or low-VRAM machines.
    MINERU_BACKEND: str = os.getenv("MINERU_BACKEND", "pipeline")
    # Outer timeout for the entire MinerU subprocess call (seconds).
    # MinerU's *internal* per-page timeout is hard-coded and not user-tunable,
    # so we bound the whole call from the outside and retry on TimeoutError.
    MINERU_TIMEOUT: int = int(os.getenv("MINERU_TIMEOUT", "1800"))  # 30 min
    # Number of retries when MinerU times out (e.g. busy CPU/GPU). Each retry
    # uses a fresh MinerU subprocess so transient resource pressure is shaken off.
    MINERU_MAX_RETRIES: int = int(os.getenv("MINERU_MAX_RETRIES", "2"))
    DISPLAY_CONTENT_STATS: bool = True

    # ===== Multimodal Processing Configuration =====
    ENABLE_IMAGE_PROCESSING: bool = os.getenv("ENABLE_IMAGE_PROCESSING", "true").lower() == "true"
    ENABLE_TABLE_PROCESSING: bool = os.getenv("ENABLE_TABLE_PROCESSING", "true").lower() == "true"
    ENABLE_EQUATION_PROCESSING: bool = os.getenv("ENABLE_EQUATION_PROCESSING", "true").lower() == "true"

    # Context Extraction
    CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "1"))
    CONTEXT_MODE: str = os.getenv("CONTEXT_MODE", "page")
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
    INCLUDE_HEADERS: bool = True
    INCLUDE_CAPTIONS: bool = True
    CONTEXT_FILTER_CONTENT_TYPES: list = ["text"]

    # ===== Storage Configuration (legacy, kept for compatibility) =====
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    TREE_INDEX_DIR: Path = BASE_DIR / "tree_index"  # deprecated, kept for migration
    LOG_DIR: Path = BASE_DIR / "logs"
    # Persistent state DB (SQLite) - replaces JSON file & in-memory state.
    STATE_DB_PATH: Path = BASE_DIR / "docmind_state.db"

    # ===== MySQL Configuration =====
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "docmind")

    # ===== Server Configuration =====
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ===== Reranker Configuration =====
    # Cohere Rerank API for Lost-in-the-Middle mitigation.
    # When empty, falls back to local model or heuristic reordering.
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    # Top-N chunks to keep after reranking (0 = keep all, just reorder)
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "0"))

    # Local reranker model (second priority after Cohere).
    # Default: jinaai/jina-reranker-v3 (listwise, 0.6B, auto-download ~1.2GB)
    # Alternatives:
    #   BAAI/bge-reranker-v2-m3   (~560MB, pairwise cross-encoder, Chinese optimized)
    #   Alibaba-NLP/gte-reranker-modernbert-base  (149M, efficient)
    # Requires: pip install transformers torch (Jina v3) or sentence-transformers (BGE)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v3")
    RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "")  # "cuda", "cpu", or "" for auto
    RERANKER_MAX_LENGTH: int = int(os.getenv("RERANKER_MAX_LENGTH", "512"))

    # ===== Application Configuration =====
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # ===== Logging Configuration =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "plain")  # "plain" or "json"
    LOG_FILE_ENABLED: bool = os.getenv("LOG_FILE_ENABLED", "false").lower() == "true"

    # ===== CORS Configuration =====
    # Use a list literal as the env-driven default. `or` correctly handles
    # the case where CORS_ORIGINS is set to an empty string in the environment.
    _raw_cors = os.getenv("CORS_ORIGINS")
    CORS_ORIGINS: list = (
        [o.strip() for o in _raw_cors.split(",") if o.strip()]
        if _raw_cors
        else ["http://localhost:3000", "http://localhost:5173"]
    )
    # When true, the wildcard "*" is rejected by the CORS layer if credentials
    # are also enabled (browser spec violation). Set to "true" to allow "*"
    # only in trusted single-tenant deployments.
    CORS_ALLOW_WILDCARD: bool = os.getenv("CORS_ALLOW_WILDCARD", "false").lower() == "true"

    # ===== Upload Configuration =====
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
    ALLOWED_UPLOAD_TYPES: list = [
        ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".gif", ".webp", ".doc", ".docx", ".ppt", ".pptx",
        ".xls", ".xlsx", ".txt", ".md"
    ]
    # Body size limit for non-upload JSON requests. Must NOT conflict with
    # MAX_UPLOAD_SIZE_MB: uploads go through multipart/form-data and are
    # validated per-file in `validate_upload_file`; the body middleware
    # intentionally skips multipart requests.
    MAX_BODY_SIZE_MB: int = int(os.getenv("MAX_BODY_SIZE_MB", "10"))

    # ===== API Key Authentication =====
    # Comma-separated list. When empty, auth is disabled (dev mode only).
    # In production, set API_KEYS to one or more non-empty values to require
    # `X-API-Key` header on all protected routes.
    _raw_api_keys = os.getenv("API_KEYS", "")
    API_KEYS: list = (
        [k.strip() for k in _raw_api_keys.split(",") if k.strip()]
        if _raw_api_keys
        else []
    )
    # Endpoints that are always public (no API key required) - keep the
    # health/info/docs endpoints open so monitoring & browser access still
    # work when auth is enabled.
    PUBLIC_PATHS: list = [
        "/", "/api/health", "/api/info",
        "/api/docs", "/api/docs/oauth2-redirect",
        "/api/redoc", "/openapi.json",
    ]

    # ===== OCR Configuration =====
    TESSERACT_PATH: str = os.getenv("TESSERACT_PATH", "")
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "chi_sim+eng")
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    OCR_ALLOWED_TYPES: list = [
        ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".ppt", ".pptx",
    ]
    OCR_MAX_FILE_SIZE_MB: int = int(os.getenv("OCR_MAX_FILE_SIZE_MB", "100"))

    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.RAG_WORKING_DIR, exist_ok=True)
        os.makedirs(self.RAG_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in .env file")


settings = Settings()
