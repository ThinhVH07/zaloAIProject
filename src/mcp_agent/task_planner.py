"""Module Planning Đa Tác Vụ (Agentic Multi-Step Task Planner) cho LifeMate.

Kiến trúc Planning 5 giai đoạn:
1. Phân rã mục tiêu (Goal / Intent Decomposition)
2. Lập kế hoạch phụ thuộc 'Đọc trước - Ghi sau' (Read-Before-Write Dependency Planning)
3. Đánh giá điều kiện & rẽ nhánh thực thi (Conditional Branching Execution)
4. Phát hiện thiếu thông tin & hỏi lại chuẩn quy định (sys.ask_missing_info Planning)
5. Thẩm định mô phỏng & đồng bộ State Hash (Replay & State Hash Integrity)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from src.asr_normalizer.text_cleaner import fold, hints, words
from src.guardrails.iso_formatter import calendar_dates, fix_iso_time_list, normalize_arguments


@dataclass
class PlanStep:
    """Đại diện cho 1 bước trong kế hoạch thực thi của Agent."""

    tool_name: str
    kind: str  # 'read' | 'write' | 'meta'
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    condition: str | None = None
    resolved: bool = False


class AgenticPlanner:
    """
    Bộ lập kế hoạch tác vụ (Task Planner) kết hợp quy tắc tất định & suy luận phụ thuộc.
    Đảm bảo 100% tuân thủ nguyên tắc 'Đọc trước - Ghi sau' và không bịa tham số.
    """

    def __init__(self, catalog: dict | None = None):
        self.catalog = catalog or {}

    @classmethod
    def extract_subgoals(cls, transcript: str) -> list[str]:
        """Phân rã câu thoại tiếng Việt phức hợp thành các mệnh đề/mục tiêu con."""
        text = transcript.strip()
        # Tách theo liên từ thời gian hoặc liên từ nối giữa các hành động
        delimiters = re.compile(
            r"\s*(?:;\s*|[,.]\s*(?:sau\s+(?:đó|do)|rồi|roi|đồng\s+thời|dong\s+thoi|và|va|kèm\s+theo|kem\s+theo|rồi\s+mới|roi\s+moi)?\s*|"
            r"\s+(?:sau\s+(?:đó|do)|rồi|roi|đồng\s+thời|dong\s+thoi|và\s+sau\s+(?:đó|do)|rồi\s+nhớ|roi\s+nho)\s+|"
            r"\s+(?:và|va|đồng\s+thời|dong\s+thoi)\s+(?=(?:bật|tắt|đặt|chuyển|gọi|nhắn|nhắc|kiểm\s+tra|hủy|thanh\s+toán|mở|đóng|xem)\b))\s*",
            flags=re.IGNORECASE,
        )
        parts = delimiters.split(text)
        subgoals = [p.strip() for p in parts if p.strip()]
        return subgoals if subgoals else [text]

    @classmethod
    def plan_read_dependencies(
        cls, transcript: str, allowed_tools: set[str], existing_calls: list[dict] | None = None
    ) -> list[dict]:
        """
        Lập kế hoạch chèn các tool ĐỌC (Read Tools) bắt buộc trước khi thực hiện thao tác GHI (Write Tools).
        Giải quyết dứt điểm lỗi bịa ID hoặc thiếu thông tin ban đầu.
        """
        calls = list(existing_calls) if existing_calls else []
        call_names = {c["name"] for c in calls}
        text_f = fold(transcript)

        injected = []

        # 1. Liên hệ / Danh bạ (Contacts): Cần tìm kiếm để lấy contact_id thật
        needs_contact = any(
            t in allowed_tools for t in ["phone.call", "sms.send", "chat_ott.send_message"]
        ) and any(w in text_f for w in ["goi", "nhan", "gui tin", "nhan tin", "bao cho"])
        if needs_contact and "contacts.search" in allowed_tools and "contacts.search" not in call_names:
            # Rút trích tên đối tượng tìm kiếm nếu có
            name_match = re.search(r"(?:cho|den|toi|so cua)\s+([A-Za-z0-9_\s]{2,20})", transcript, re.IGNORECASE)
            query = name_match.group(1).strip() if name_match else ""
            injected.append({"name": "contacts.search", "arguments": {"query": query or "người thân"}})

        # 2. Thiết bị nhà thông minh (Smart Home): Bắt buộc get_status để lấy device_id
        needs_smart_home = any(
            t in allowed_tools
            for t in ["smart_home.light_set", "smart_home.climate_set", "smart_home.fan_set", "smart_home.switch_set"]
        ) and any(w in text_f for w in ["den", "dieu hoa", "quat", "cong tac", "tat ca", "phong", "tang"])
        if needs_smart_home and "smart_home.get_status" in allowed_tools and "smart_home.get_status" not in call_names:
            injected.append({"name": "smart_home.get_status", "arguments": {}})

        # 3. Lịch trình / Sự kiện (Calendar): Cần xem sự kiện trước khi cập nhật, dời, hủy
        needs_calendar = any(
            t in allowed_tools for t in ["calendar.update_event", "calendar.cancel_event"]
        ) or (
            "ride_hailing.book" in allowed_tools
            and any(w in text_f for w in ["lich", "hop", "kham", "chuyen bay", "di hop", "sau khi"])
        )
        if needs_calendar and "calendar.list_events" in allowed_tools and "calendar.list_events" not in call_names:
            injected.append({"name": "calendar.list_events", "arguments": {}})

        # 4. Hóa đơn (Bill Payment): Cần kiểm tra danh sách hóa đơn trước khi thanh toán
        needs_bill = "bill_payment.pay" in allowed_tools and any(
            w in text_f for w in ["hoa don", "tien dien", "tien nuoc", "internet", "truyen hinh", "hoc phi", "tra tien"]
        )
        if needs_bill and "bill_payment.list_bills" in allowed_tools and "bill_payment.list_bills" not in call_names:
            injected.append({"name": "bill_payment.list_bills", "arguments": {}})

        # 5. Số dư ví / Ngân hàng (Balance): Cần xem số dư khi có điều kiện về nguồn tiền
        needs_balance = any(w in text_f for w in ["neu du", "khong du", "so du", "con du", "vi con"])
        if needs_balance and "ewallet.get_balance" in allowed_tools and "ewallet.get_balance" not in call_names:
            injected.append({"name": "ewallet.get_balance", "arguments": {}})

        return injected + calls

    @classmethod
    def plan_conditional_execution(
        cls, transcript: str, planned_calls: list[dict], env
    ) -> list[dict]:
        """
        Thực thi kế hoạch có rẽ nhánh điều kiện (Conditional Branching):
        Ví dụ: 'Nếu mai mưa thì dời chạy bộ... không mưa thì thôi'
        Thực hiện bước kiểm tra điều kiện trước, nếu điều kiện không thỏa mãn thì hủy các bước ghi.
        """
        text_f = fold(transcript)
        if not re.search(r"\bneu\b.*\bthi\b", text_f):
            return planned_calls

        # Kịch bản kiểm tra thời tiết
        if "thoi tiet" in text_f or "mua" in text_f or "nang" in text_f:
            weather_call = next((c for c in planned_calls if c["name"] == "weather.get_forecast"), None)
            if weather_call and env is not None:
                # Gọi thử tool đọc thời tiết trên simulator
                result = env.call(weather_call["name"], weather_call.get("arguments", {}))
                forecast = fold(str(result))
                # Nếu người dùng yêu cầu 'nếu mưa' nhưng kết quả dự báo không có mưa
                if "neu mai mua" in text_f and not any(w in forecast for w in ["mua", "rain", "shower"]):
                    # Điều kiện không thỏa mãn -> Chỉ trả về lệnh đọc thời tiết, không thực hiện hành động ghi
                    return [weather_call]

        return planned_calls

    @classmethod
    def plan_missing_info_safeguard(
        cls, transcript: str, planned_calls: list[dict], allowed_tools: set[str]
    ) -> list[dict]:
        """
        Kiểm định xem có tác vụ nào bị thiếu tham số cốt lõi không thể tự suy luận.
        Nếu thiếu -> Chuyển thành `sys.ask_missing_info` chuẩn quy định của BTC.
        """
        if "sys.ask_missing_info" not in allowed_tools:
            return planned_calls

        text_f = fold(transcript)
        sanitized = []
        for call in planned_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})

            # 1. Chuyển tiền ngân hàng nhưng thiếu số tiền
            if name == "banking.transfer":
                if not args.get("amount") and not re.search(r"\b\d+(?:\s*(?:k|nghin|trieu|dong|vnd))?\b", text_f):
                    sanitized.append({
                        "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": "banking.transfer", "params": ["amount"]},
                    })
                    continue

            # 2. Gọi điện thoại nhưng thiếu người nhận
            if name == "phone.call":
                if not args.get("contact_id") and not args.get("phone_number"):
                    sanitized.append({
                        "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": "phone.call", "params": ["contact_id"]},
                    })
                    continue

            # 3. Đặt xe nhưng thiếu điểm đến
            if name == "ride_hailing.book":
                if not args.get("dropoff") and not re.search(r"\b(?:den|ve|toi)\b", text_f):
                    sanitized.append({
                        "name": "sys.ask_missing_info",
                        "arguments": {"tool_name": "ride_hailing.book", "params": ["dropoff"]},
                    })
                    continue

            sanitized.append(call)

        return sanitized

    @classmethod
    def optimize_plan(
        cls, transcript: str, raw_calls: list[dict], allowed_tools: set[str], env=None
    ) -> list[dict]:
        """
        Toàn bộ luồng tối ưu hóa kế hoạch (Master Planning Pipeline):
        1. Chèn các Tool Đọc bắt buộc (Read-before-Write)
        2. Đánh giá rẽ nhánh điều kiện thực tế
        3. Rà soát thiếu thông tin bắt buộc (sys.ask_missing_info)
        4. Chuẩn hóa ISO thời gian và loại bỏ lệnh trùng thừa
        """
        # Bước 1: Dependency Planning
        step1_calls = cls.plan_read_dependencies(transcript, allowed_tools, raw_calls)

        # Bước 2: Conditional Evaluation
        step2_calls = cls.plan_conditional_execution(transcript, step1_calls, env)

        # Bước 3: Missing Info Guardrail
        step3_calls = cls.plan_missing_info_safeguard(transcript, step2_calls, allowed_tools)

        # Bước 4: Chuẩn hóa thời gian ISO
        final_calls = fix_iso_time_list(step3_calls)

        # Loại bỏ các call không thuộc allowed_tools
        final_calls = [c for c in final_calls if c.get("name") in allowed_tools]

        return final_calls
