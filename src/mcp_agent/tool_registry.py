"""Đăng ký danh sách Tools và quản lý JSON Schema chuẩn Model Context Protocol (MCP)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from config.settings import TOOL_SCHEMAS_PATH


def schema_with_kind(tool: dict) -> dict:
    """Rút gọn metadata mô tả thừa trong JSON Schema để tối ưu context cho LLM."""
    params = copy.deepcopy(tool["parameters"])
    for name, field in params.get("properties", {}).items():
        field.pop("x-match", None)
        description = field.get("description", "")
        if field.get("x-ref") or description.startswith(("Số nguyên", "ISO-8601")):
            field.pop("description", None)
        elif name not in params.get("required", []) and not description:
            field.pop("description", None)
    return {
        "name": tool["name"],
        "kind": tool.get("kind"),
        "description": tool.get("description", ""),
        "parameters": params,
    }


class ToolRegistry:
    """Quản lý toàn bộ danh sách Tools/APIs từ config/tool_schemas.json hoặc catalog của BTC."""

    def __init__(self, catalog: dict | None = None, schema_path: str | Path = TOOL_SCHEMAS_PATH):
        if catalog is None:
            catalog = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.catalog = catalog
        self.tools: dict[str, dict] = {
            tool["name"]: tool
            for app in catalog.get("apps", [])
            for tool in app.get("tools", [])
        }
        self.schemas: dict[str, dict] = {
            name: tool["parameters"] for name, tool in self.tools.items()
        }
        self.kinds: dict[str, str | None] = {
            name: tool.get("kind") for name, tool in self.tools.items()
        }

    def get_specs(self, allowed: set[str] | None = None) -> list[dict]:
        """Lấy danh sách MCP Tool Specs cho các ứng dụng đã cài đặt."""
        if allowed is None:
            return [schema_with_kind(tool) for tool in self.tools.values()]
        return [schema_with_kind(tool) for name, tool in self.tools.items() if name in allowed]

    @staticmethod
    def allowed_from_context(context: dict) -> set[str]:
        """Trích xuất tập tên tool được phép gọi từ context.installed_apps."""
        installed_apps = context.get("installed_apps", {})
        allowed = {
            tool_name
            for app in installed_apps.values()
            if isinstance(app, dict)
            for tool_name in app.get("tools", [])
        }
        allowed.add("sys.ask_missing_info")
        return allowed

