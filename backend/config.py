import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.ollama_model: str = os.getenv(
            "OLLAMA_MODEL", "SpeakLeash/bielik-7b-instruct-v0.1-gguf"
        )
        self.ollama_base_url: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.vault_path: str = os.getenv("VAULT_PATH", "./obsidian-vault")
        self.chroma_persist_dir: str = os.getenv(
            "CHROMA_PERSIST_DIR", "./obsidian-vault/.rag-index"
        )
        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))

    @property
    def vault_abs_path(self) -> Path:
        return Path(self.vault_path).resolve()

    @property
    def chroma_abs_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()
