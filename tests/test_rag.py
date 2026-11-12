"""Test suite kiểm thử Tầng 1 (Entity RAG & ASR Text Cleaner) và Tầng 2 (Y/N Router Gateway)."""
from __future__ import annotations

import unittest
from datetime import datetime

from src.asr_normalizer.entity_retriever import EntityRAG
from src.asr_normalizer.text_cleaner import clean_asr_text, fold, hints
from src.router.intent_router import IntentRouter, timestamp_for_alarm


class TestEntityRAGAndCleaner(unittest.TestCase):
    def setUp(self):
        self.user_profile = {
            "contacts": [
                {"id": "c1", "name": "Minh"},
                {"id": "c2", "name": "Nguyễn Thị Lan"},
                {"id": "c3", "name": "Bố"},
            ],
            "devices": [
                {"id": "d1", "name": "Đèn phòng khách", "type": "light"},
                {"id": "d2", "name": "Điều hòa phòng ngủ", "type": "climate"},
            ],
        }
        self.rag = EntityRAG(self.user_profile)

    def test_normalize_contact_fuzzy(self):
        # Ví dụ nghe nhầm "Minh đồng nghiệp thúng" hoặc "Minh" -> map về "Minh"
        self.assertEqual(self.rag.normalize_contact("Minh"), "Minh")
        self.assertEqual(self.rag.normalize_contact("Nguyen Thi Lan"), "Nguyễn Thị Lan")

    def test_normalize_device_fuzzy(self):
        self.assertEqual(self.rag.normalize_device("Den phong khach"), "Đèn phòng khách")

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


if __name__ == "__main__":
    unittest.main()

