"""Application configuration for the RAG pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RAGConfig:
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "chroma_db_ollama"))
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_results: int = 4
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    max_tokens: int = 1000
    temperature: float = 0


DEFAULT_CONFIG = RAGConfig()
