"""Đấu nối API key, Model name, đường dẫn dữ liệu và Configs cho hệ thống LifeMate Agent Backend."""
from __future__ import annotations

import os
from pathlib import Path

# Đường dẫn thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent

# Đường dẫn Dataset & Runtime của BTC / Local LLM
_CANDIDATE_ROOTS = [
    Path(r"D:\BTTL\AI\train+public_test"),
    Path(r"D:\train+public_test\train+public_test"),
]
DEFAULT_ROOT = Path(
    os.environ.get("LIFEMATE_DATA_ROOT", str(next((p for p in _CANDIDATE_ROOTS if p.exists()), _CANDIDATE_ROOTS[0])))
)
RUNTIME_DIR = Path(os.environ.get("LIFEMATE_RUNTIME_DIR", str(WORKSPACE_ROOT / "work" / "llm")))
ASR_HANDOFF_DIR = Path(os.environ.get("LIFEMATE_ASR_DIR", str(WORKSPACE_ROOT / "outputs" / "lifemate_asr" / "handoff")))
DEV_REFERENCES_PATH = Path(
    os.environ.get(
        "LIFEMATE_DEV_REFERENCES",
        str(PROJECT_ROOT.parent / "lifemate_data_eval" / "splits" / "dev_references.jsonl"),
    )
)

# Cấu hình thư mục nội bộ theo Architecture Blueprint
CONFIG_DIR = PROJECT_ROOT / "config"
TOOL_SCHEMAS_PATH = CONFIG_DIR / "tool_schemas.json"

DATA_DIR = PROJECT_ROOT / "data"
USER_PROFILES_DIR = DATA_DIR / "user_profiles"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
SELECTOR_MODEL_PATH = VECTOR_DB_DIR / "selector_model.json"

# Cấu hình LLM / MCP Agent Engine
API_KEY = os.environ.get("LIFEMATE_API_KEY", "local")
LLM_BASE_URL = os.environ.get("LIFEMATE_LLM_URL", "http://127.0.0.1:8765/v1/chat/completions")
LLM_HEALTH_URL = os.environ.get("LIFEMATE_LLM_HEALTH_URL", "http://127.0.0.1:8765/health")
MODEL_NAME = os.environ.get("LIFEMATE_MODEL_NAME", "local")
DEFAULT_GGUF_MODEL = RUNTIME_DIR / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# Tham số suy luận & Guardrails
MAX_TOKENS = 640
MAX_ROUNDS = 9
TEMPERATURE = 0
SEED = 42
REQUEST_TIMEOUT_SECONDS = 240
DEFAULT_WORKERS = 2
DEFAULT_TIMEZONE = "+07:00"
FUZZY_MATCH_THRESHOLD = 75
PUBLIC_TEST_EXPECTED_COUNT = 634
FAILED_STATUSES = {"runtime_error", "parse_error", "invalid_action"}

