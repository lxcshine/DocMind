"""
ResearchFlow Backend Configuration

Uses the same environment variables as research_rag project
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
    """Backend settings based on research_rag .env configuration"""
    
    # LLM Configuration (from research_rag .env)
    LLM_MODE: str = os.getenv("LLM_MODE", "online")
    
    # Gemini Model Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Vision Model Configuration (for PDF page image understanding)
    VISION_API_KEY: str = os.getenv("VISION_API_KEY", "")
    VISION_BASE_URL: str = os.getenv("VISION_BASE_URL", "https://api.openai.com/v1/")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o")
    
    # Retrieval Configuration
    RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "balanced")
    
    # PageIndex Document Processing Configuration
    MAX_TOKENS_PER_CHUNK: int = 20000
    MAX_PAGES_PER_CHUNK: int = 10
    TOC_CHECK_PAGE_NUM: int = 20
    ENABLE_TOC_EXTRACTION: bool = True
    ENABLE_VERIFICATION: bool = True
    ENABLE_RECURSIVE_SPLIT: bool = True
    
    # Storage Configuration
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    TREE_INDEX_DIR: Path = BASE_DIR / "tree_index"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # MySQL Configuration
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "20010609")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "researchflow")
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Upload Configuration
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_UPLOAD_TYPES: list = [".pdf", ".txt", ".md"]
    
    # OCR Configuration
    TESSERACT_PATH: str = os.getenv("TESSERACT_PATH", r"D:\software\Tesseract-OCR\tesseract.exe")
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "chi_sim+eng")
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    OCR_ALLOWED_TYPES: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".ppt", ".pptx"]
    OCR_MAX_FILE_SIZE_MB: int = int(os.getenv("OCR_MAX_FILE_SIZE_MB", "100"))
    
    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.TREE_INDEX_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        # Validate Gemini configuration
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in .env file")


settings = Settings()
