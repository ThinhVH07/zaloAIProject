"""Đóng gói APIs và môi trường Simulator của BTC theo chuẩn Model Context Protocol (MCP)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from config.settings import DEFAULT_ROOT, USER_PROFILES_DIR
from src.mcp_agent.tool_registry import ToolRegistry


def load_btc_runtime(root: str | Path = DEFAULT_ROOT, split: str = "public_test"):
    """Load zaic_tools của từng split trong namespace cô lập mà không chạm vào code BTC."""
    folder = Path(root).resolve() / split / "zaic_tools"
    alias = "_lifemate_" + hashlib.sha256(str(folder).encode()).hexdigest()[:16]
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(
        alias, folder / "__init__.py", submodule_search_locations=[str(folder)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


class MCPServer:
    """
    Tầng giao tiếp chuẩn MCP bọc quanh Simulator (ToolEnv) của BTC:
    - Quản lý state_hash & user_profile
    - Cung cấp `load_state()`, `validate()`, và `call_tool()`
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        split: str = "public_test",
        registry: ToolRegistry | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.registry = registry or ToolRegistry()
        self._engine = None
        self._envs: dict[str, object] = {}

    @property
    def engine(self):
        if self._engine is None:
            self._engine = load_btc_runtime(self.root, self.split)
        return self._engine

    def get_env(self, sample_id: str):
        """Khởi tạo hoặc lấy lại instance ToolEnv cho một sample_id."""
        if sample_id not in self._envs:
            self._envs[sample_id] = self.engine.ToolEnv(self.split, sample_id, root=self.root)
        return self._envs[sample_id]

    def reset_env(self, sample_id: str):
        """Tạo mới instance ToolEnv cho sample_id."""
        self._envs[sample_id] = self.engine.ToolEnv(self.split, sample_id, root=self.root)
        return self._envs[sample_id]

    def load_state(self, state_hash_or_id: str) -> dict:
        """
        Tải user_profile / state từ local cache (`data/user_profiles`) hoặc từ thư mục environment của BTC.
        """
        local_profile = USER_PROFILES_DIR / f"{state_hash_or_id}.json"
        if local_profile.exists():
            return json.loads(local_profile.read_text(encoding="utf-8"))

        env_state = self.root / self.split / f"{self.split}_environment" / state_hash_or_id / "state.json"
        if env_state.exists():
            return json.loads(env_state.read_text(encoding="utf-8"))

        return {"contacts": [], "devices": []}

    def validate(self, sample_id: str, tool_calls: list[dict]) -> tuple[bool, str | None]:
        """Kiểm tra danh sách tool_calls thông qua Simulator `env.validate(name, args)` của BTC."""
        try:
            env = self.get_env(sample_id)
        except Exception:
            # Nếu chạy ở chế độ offline unit test không có dataset BTC trên ổ D:
            for call in tool_calls:
                if not isinstance(call, dict) or "name" not in call:
                    return False, "Each tool call must be a dict with 'name'"
                if call["name"] not in self.registry.tools:
                    return False, f"Unknown tool: {call['name']}"
            return True, None

        for call in tool_calls:
            if not isinstance(call, dict):
                return False, "Tool call must be an object"
            name = call.get("name")
            args = call.get("arguments", {})
            if not isinstance(name, str) or name not in env.allowed:
                return False, f"Tool '{name}' is not installed"
            res = env.validate(name, args)
            if not res.get("ok"):
                return False, res.get("error", "Validation failed")
        return True, None

