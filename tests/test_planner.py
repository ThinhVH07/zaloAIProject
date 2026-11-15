"""Test suite kiểm thử Agentic Multi-Step Task Planner."""
from __future__ import annotations

import unittest

from src.mcp_agent.task_planner import AgenticPlanner


class TestAgenticPlanner(unittest.TestCase):
    def test_extract_subgoals(self):
        text = "Tôi sắp về nhà, sau đó bật điều hòa phòng khách và đặt xe đón tôi"
        subgoals = AgenticPlanner.extract_subgoals(text)
        self.assertTrue(len(subgoals) >= 2)

    def test_plan_read_dependencies_smart_home(self):
        allowed = {"smart_home.light_set", "smart_home.get_status"}
        planned = AgenticPlanner.plan_read_dependencies(
            "Bật đèn phòng khách giúp tôi",
            allowed_tools=allowed,
            existing_calls=[{"name": "smart_home.light_set", "arguments": {"power": "on"}}],
        )
        self.assertEqual(len(planned), 2)
        self.assertEqual(planned[0]["name"], "smart_home.get_status")
        self.assertEqual(planned[1]["name"], "smart_home.light_set")

    def test_plan_read_dependencies_contacts(self):
        allowed = {"phone.call", "contacts.search"}
        planned = AgenticPlanner.plan_read_dependencies(
            "Gọi điện cho mẹ giúp tôi",
            allowed_tools=allowed,
            existing_calls=[{"name": "phone.call", "arguments": {"contact_id": "c1"}}],
        )
        self.assertEqual(len(planned), 2)
        self.assertEqual(planned[0]["name"], "contacts.search")
        self.assertEqual(planned[1]["name"], "phone.call")

    def test_missing_info_safeguard(self):
        allowed = {"banking.transfer", "sys.ask_missing_info"}
        calls = [{"name": "banking.transfer", "arguments": {"beneficiary_id": "b1"}}]
        sanitized = AgenticPlanner.plan_missing_info_safeguard(
            "Chuyển khoản cho Nam",
            planned_calls=calls,
            allowed_tools=allowed,
        )
        self.assertEqual(len(sanitized), 1)
        self.assertEqual(sanitized[0]["name"], "sys.ask_missing_info")
        self.assertEqual(sanitized[0]["arguments"]["tool_name"], "banking.transfer")
        self.assertIn("amount", sanitized[0]["arguments"]["params"])


if __name__ == "__main__":
    unittest.main()
