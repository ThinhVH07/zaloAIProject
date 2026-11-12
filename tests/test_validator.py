"""Test suite kiểm thử Tầng 3 (MCP Tool Registry) & Tầng 4 (Guardrails, ISO 8601, Self-Correction)."""
from __future__ import annotations

import unittest
from datetime import datetime

from main import process_sample
from src.guardrails.iso_formatter import normalize_arguments
from src.guardrails.self_correction import OutputGuardrail
from src.mcp_agent.mcp_server import MCPServer
from src.mcp_agent.tool_registry import ToolRegistry


class TestGuardrailsAndValidator(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    def test_tool_registry_loaded_all_schemas(self):
        self.assertIn("alarm.create", self.registry.schemas)
        self.assertIn("banking.transfer", self.registry.schemas)
        self.assertIn("sys.ask_missing_info", self.registry.schemas)

    def test_fix_iso_time_adds_vietnam_timezone(self):
        calls = [
            {"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00"}},
            {"name": "calendar.create_event", "arguments": {"title": "Họp", "start": "2026-11-11T09:00"}},
        ]
        fixed = OutputGuardrail.fix_iso_time(calls)
        self.assertEqual(fixed[0]["arguments"]["time"], "2026-11-11T05:30:00+07:00")
        self.assertEqual(fixed[1]["arguments"]["start"], "2026-11-11T09:00:00+07:00")

    def test_remove_redundant_missing_info(self):
        calls = [
            {"name": "contacts.search", "arguments": {"query": "Minh"}},
            {"name": "phone.call", "arguments": {"contact_id": "ct_01"}},
            {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}},
        ]
        cleaned = OutputGuardrail.remove_redundant_missing_info(calls)
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(all(c["name"] != "sys.ask_missing_info" for c in cleaned))

    def test_filter_self_cancellation(self):
        calls = [{"name": "ride_hailing.book", "arguments": {"pickup": "A", "dropoff": "B"}}]
        # Khi người dùng đổi ý tự hủy
        filtered = OutputGuardrail.filter_self_cancellation(
            "Book xe cho em qua trường... À thôi khỏi, anh Long chở đi rồi", calls
        )
        self.assertEqual(filtered, [])

        # Khi người dùng không tự hủy
        kept = OutputGuardrail.filter_self_cancellation("Đặt xe cho em qua trường nha", calls)
        self.assertEqual(len(kept), 1)

    def test_sanitize_tool_calls(self):
        calls = [
            {"name": "phone.call", "arguments": {"contact_id": "c1"}},
            {"name": "parking.reserve", "arguments": {"lot_id": "lot_1", "hours": 180}},
        ]
        sanitized = OutputGuardrail.sanitize_tool_calls(calls, allowed_tools={"chat_ott.call", "parking.reserve"})
        self.assertEqual(sanitized[0]["name"], "chat_ott.call")
        self.assertEqual(sanitized[1]["arguments"]["hours"], 3)

    def test_keep_sole_missing_info(self):
        calls = [
            {"name": "sys.ask_missing_info", "arguments": {"tool_name": "alarm.create", "params": ["time"]}},
        ]
        cleaned = OutputGuardrail.remove_redundant_missing_info(calls)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["name"], "sys.ask_missing_info")

    def test_normalize_arguments_coerces_integer_and_iso(self):
        now = datetime.fromisoformat("2026-11-10T10:00:00+07:00")
        transfer_schema = self.registry.schemas["banking.transfer"]
        norm_transfer = normalize_arguments(
            {"from_account_id": "acc_1", "beneficiary_id": "ben_1", "amount": "1.200.000"},
            transfer_schema,
            now,
        )
        self.assertEqual(norm_transfer["amount"], 1200000)

        alarm_schema = self.registry.schemas["alarm.create"]
        norm_alarm = normalize_arguments({"time": "2026-11-11T06:00:00"}, alarm_schema, now)
        self.assertEqual(norm_alarm["time"], "2026-11-11T06:00:00+07:00")

    def test_process_sample_fast_path_greeting(self):
        server = MCPServer()
        sample = {"id": "test_001", "state_hash": "hash_123", "text": "Xin chào bạn nhé!"}
        out = process_sample(sample, server)
        self.assertEqual(out, {"id": "test_001", "tool_calls": [], "state_hash": "hash_123"})


    def test_inject_dependency_tools(self):
        """Kiểm tra tự động chèn contacts.search khi gửi tin nhắn."""
        calls = [{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "alo"}}]
        allowed = {"sms.send", "contacts.search"}
        state = {"contacts": {"contacts": [{"id": "c_01", "name": "Mẹ"}]}}
        res = OutputGuardrail.inject_dependency_tools(calls, allowed_tools=allowed, state=state)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["name"], "contacts.search")
        self.assertEqual(res[0]["arguments"]["query"], "Mẹ")
        self.assertEqual(res[1]["name"], "sms.send")


if __name__ == "__main__":
    unittest.main()

