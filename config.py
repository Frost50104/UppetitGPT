import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Load .env from current dir or by searching up the tree to make startup robust in IDEs/alt CWDs
load_dotenv(find_dotenv(), override=False)


@dataclass
class Settings:
    # Bot
    telegram_token: str = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("telegram_token")
        or os.getenv("BOT_TOKEN")
        or ""
    )
    allowed_chat_ids: List[int] = None

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # RAG/index
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", "index"))
    top_k: int = int(os.getenv("TOP_K", "8"))
    max_ctx_chars: int = int(os.getenv("MAX_CTX_CHARS", "6000"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))

    # Org/policy
    org_name: str = os.getenv("ORG_NAME", "Компания")

    # Logs
    logs_dir: Path = Path(os.getenv("LOGS_DIR", "logs"))
    queries_log: Path = logs_dir / "queries.csv"
    feedback_log: Path = logs_dir / "feedback.csv"

    def ensure_dirs(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


def parse_allowed() -> List[int]:
    raw = os.getenv("ALLOWED_CHAT_IDS", "")
    ids = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            pass
    return ids


def get_settings() -> Settings:
    s = Settings()
    s.allowed_chat_ids = parse_allowed()
    s.ensure_dirs()
    return s
