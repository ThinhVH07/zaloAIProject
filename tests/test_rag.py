"""Test suite kiểm thử Tầng 1 (Entity RAG & ASR Text Cleaner) và Tầng 2 (Y/N Router Gateway)."""
from __future__ import annotations

import unittest
from datetime import datetime

from src.asr_normalizer.entity_retriever import (
    DynamicFewShotRAG,
    EntityRAG,
    ToolKnowledgeRAG,
)
from src.asr_normalizer.text_cleaner import clean_asr_text, fold, hints
from src.router.intent_router import IntentRouter, timestamp_for_alarm


class TestEntityRAGAndCleaner(unittest.TestCase):
    def setUp(self):
        self.user_profile = {
            "contacts": [
                {"id": "c1", "name": "Minh", "phone": "0912345678"},
                {"id": "c2", "name": "Nguyễn Thị Lan", "phone": "0987654321"},
                {"id": "c3", "name": "Bố", "phone": "0900000000"},
            ],
            "devices": [
                {"id": "d1", "name": "Đèn phòng khách", "type": "light", "room": "phòng khách"},
                {"id": "d2", "name": "Điều hòa phòng ngủ", "type": "climate", "room": "phòng ngủ"},
            ],
        }
        self.rag = EntityRAG(self.user_profile)

    def test_normalize_contact_fuzzy(self):
        # Ví dụ nghe nhầm "Minh đồng nghiệp thúng" hoặc "Minh" -> map về "Minh"
        self.assertEqual(self.rag.normalize_contact("Minh"), "Minh")
        self.assertEqual(self.rag.normalize_contact("Nguyen Thi Lan"), "Nguyễn Thị Lan")

    def test_normalize_device_fuzzy(self):
        self.assertEqual(self.rag.normalize_device("Den phong khach"), "Đèn phòng khách")

    def test_grounded_entity_retrieval(self):
        contact = self.rag.get_grounded_contact("Nguyen Thi Lan")
        self.assertIsNotNone(contact)
        self.assertEqual(contact["id"], "c2")
        self.assertEqual(contact["phone"], "0987654321")

        device = self.rag.get_grounded_device("Den phong khach")
        self.assertIsNotNone(device)
        self.assertEqual(device["id"], "d1")
        self.assertEqual(device["room"], "phòng khách")

    def test_retrieve_entities_from_sentence(self):
        query = "Trợ lý ơi gọi điện cho Nguyễn Thị Lan và tắt Đèn phòng khách giúp tôi"
        retrieved = self.rag.retrieve_entities(query)
        self.assertTrue(len(retrieved["contacts"]) >= 1)
        self.assertEqual(retrieved["contacts"][0]["name"], "Nguyễn Thị Lan")
        self.assertTrue(len(retrieved["devices"]) >= 1)
        self.assertEqual(retrieved["devices"][0]["name"], "Đèn phòng khách")

    def test_normalize_entities_in_text(self):
        raw = "gọi cho nguyen thi lan và bật den phong khach giup minh"
        normalized = self.rag.normalize_entities_in_text(raw)
        self.assertIn("Nguyễn Thị Lan", normalized)
        self.assertIn("Đèn phòng khách", normalized)

    def test_clean_asr_text_and_fold(self):
        raw = "  Trợ lý ơi,   bật đèn phòng khách giúp mình nhé  "
        self.assertEqual(
            clean_asr_text(raw, strip_prefix=True),
            "bật đèn phòng khách giúp mình nhé",
        )
        self.assertEqual(fold("Đặt báo thức 6 giờ rưỡi sáng mai"), "dat bao thuc 6 gio ruoi sang mai")

    def test_clock_hints_extraction(self):
        now = datetime.fromisoformat("2026-11-10T21:00:00+07:00")
        extracted = hints("Đặt báo thức 6 giờ rưỡi sáng mai", now)
        self.assertIn("clocks", extracted)
        self.assertEqual(extracted["clocks"][0]["clock"], "06:30")
        self.assertEqual(extracted["clocks"][0]["timestamp"], "2026-11-11T06:30:00+07:00")

    def test_intent_router_yn_gateway(self):
        router = IntentRouter()
        self.assertFalse(router.needs_tool("Xin chào bạn"))
        self.assertFalse(router.needs_tool("Trợ lý ơi, cảm ơn nhiều nhé!"))
        self.assertFalse(router.needs_tool("Tạm biệt nha"))
        self.assertTrue(router.needs_tool("Đặt báo thức 7 giờ sáng mai"))
        self.assertTrue(router.needs_tool("Gọi điện cho Minh giúp tôi"))

    def test_deterministic_alarm_route(self):
        now = datetime.fromisoformat("2026-11-10T21:00:00+07:00")
        ts = timestamp_for_alarm("Đặt báo thức 6 giờ rưỡi sáng mai giúp tôi nhé", now)
        self.assertEqual(ts, "2026-11-11T06:30:00+07:00")


class TestToolKnowledgeAndFewShotRAG(unittest.TestCase):
    def setUp(self):
        self.tool_rag = ToolKnowledgeRAG()

    def test_tool_knowledge_retrieval(self):
        # Thử tìm kiếm công cụ liên quan đến đèn
        tools = self.tool_rag.retrieve_tools("bật đèn phòng khách", top_k=3)
        self.assertTrue(len(tools) > 0)
        tool_names = [t["name"] for t in tools]
        self.assertIn("smart_home.light_set", tool_names)

        # Kiểm tra lấy schema và required params của tool
        schema = self.tool_rag.get_tool_schema("smart_home.light_set")
        self.assertIsNotNone(schema)
        req_params = self.tool_rag.get_required_params("smart_home.light_set")
        self.assertIn("device_id", req_params)

    def test_fewshot_rag_dynamic(self):
        demo_samples = [
            {
                "id": "sample_1",
                "audio_text": "bật đèn phòng khách",
                "label": {
                    "tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "d1", "state": "on"}}]
                },
            },
            {
                "id": "sample_2",
                "audio_text": "gọi điện cho mẹ",
                "label": {
                    "tool_calls": [{"name": "phone.call", "arguments": {"contact_id": "c1"}}]
                },
            },
        ]
        rag = DynamicFewShotRAG(demo_samples)
        guidance = rag.retrieve_guidance({"transcript": "bật đèn phòng khách"}, k=1)
        self.assertEqual(len(guidance), 1)
        self.assertEqual(guidance[0]["id"], "sample_1")

        prompt_str = rag.format_for_prompt(guidance)
        self.assertIn("smart_home.light_set", prompt_str)


if __name__ == "__main__":
    unittest.main()
