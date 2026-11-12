"""Phân loại nhanh câu thoại (Y/N Gate) và điều hướng Fast-Path cho các tác vụ tất định."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from src.asr_normalizer.text_cleaner import FILLER, NUMBER, START, fold, hints, number

COMPOUND = re.compile(r"\b(?:nhac|nhan|gui|tra|thanh toan|chuyen|dat xe|roi|dong thoi|neu|huy|doi)\b")

# Các mẫu câu chào hỏi / xã giao thuần túy không cần gọi Tool (Fast-Path -> [])
CHITCHAT_PATTERNS = re.compile(
    r"^(?:(?:xin )?chao(?: ban| em| anh| chi| tro ly| lifemate| nhe| nha)*|"
    r"cam on(?: ban| em| anh| chi| tro ly| nhieu| nhe| nha)*|"
    r"tam biet(?: nhe| nha)*|hen gap lai|khong co gi(?: dau)?|"
    r"ok(?: cam on)?|duoc roi(?: cam on)?|thoi khong can nua)[ .!?]*$"
)

DEFINITIONS = {
    "workflow.pay_due_bills": (
        {"bill_payment.list_bills", "bill_payment.pay"},
        "Pay all unpaid bills due from today through days_ahead, each at most max_vnd, by bank. "
        "Use only when the request supplies both a due window and an upper limit.",
        {"days_ahead": "integer", "max_vnd": "integer"},
    ),
    "workflow.pay_wallet_fallback": (
        {"bill_payment.list_bills", "bill_payment.pay", "ewallet.get_balance"},
        "Pay a single explicit bill type. Read wallet balance and use bank only if wallet cannot cover it. "
        "Use only when the user explicitly allows both sources in that order.",
        {"type": "electricity | water | internet | tv | phone | tuition"},
    ),
    "workflow.turn_off_floor_lights": (
        {"smart_home.get_status", "smart_home.light_set"},
        "Turn off all lights on a floor except in rooms with an active fan or air conditioner. "
        "Use only for explicit floor-wide scope and a room exception.",
        {"floor": "integer", "keep_rooms_with_active": "fan | climate"},
    ),
}


def available_operations(request: str, allowed: set[str]) -> dict:
    """Kiểm tra các workflow đa bước tất định khả dụng theo câu thoại và danh sách tool đã cài."""
    text = fold(request)
    available = {}
    cues = {
        "workflow.pay_due_bills": all(
            re.search(pattern, text)
            for pattern in (r"ngay toi", r"ngan hang", r"\b(han|hen)\b", r"\b(vuot|qua|tren|hon)\b")
        )
        and not re.search(r"\b(?:hoa don|tien) (?:dien|nuoc|mang|internet|truyen hinh|dien thoai)\b", text),
        "workflow.pay_wallet_fallback": bool(
            re.search(r"\bvi\b.*\b(?:khong du|thieu|khong con du)\b.*\bngan hang\b", text)
        ),
        "workflow.turn_off_floor_lights": all(
            re.search(pattern, text)
            for pattern in (
                r"\b(den|dien)\b",
                r"\b(tang|than|tham)\b",
                r"\b(phong nao|tru|rieng)\b",
                r"\b(de|giu)\b",
                r"\b(het|tat ca|ca tang|toan bo)\b",
            )
        ),
    }
    for name, (required_tools, description, arguments) in DEFINITIONS.items():
        if cues[name] and required_tools <= allowed:
            available[name] = {"description": description, "arguments": arguments}
    return available


def execute(name: str, args: dict, now: datetime | None, call) -> None:
    """Thực thi workflow tất định. Hàm `call` kiểm tra hợp lệ từng tool invocation."""
    if not isinstance(args, dict):
        raise ValueError("Workflow arguments must be an object")
    if name == "workflow.pay_due_bills":
        if now is None or set(args) != {"days_ahead", "max_vnd"}:
            raise ValueError("A due window and amount cap are required")
        days, maximum = args["days_ahead"], args["max_vnd"]
        if type(days) is not int or not 0 <= days <= 365 or type(maximum) is not int or maximum < 0:
            raise ValueError("Invalid due window or amount cap")
        start, end = now.date().isoformat(), (now.date() + timedelta(days=days)).isoformat()
        bills = call("bill_payment.list_bills", {}).get("bills", [])
        for bill in bills:
            due = bill.get("due_date", bill.get("due"))
            amount = bill.get("amount")
            if (
                isinstance(due, str)
                and isinstance(amount, (int, float))
                and start <= due[:10] <= end
                and amount <= maximum
            ):
                call("bill_payment.pay", {"bill_id": bill["id"], "source": "bank"})
        return
    if name == "workflow.pay_wallet_fallback":
        if set(args) != {"type"} or args["type"] not in (
            "electricity",
            "water",
            "internet",
            "tv",
            "phone",
            "tuition",
        ):
            raise ValueError("Specify a supported bill type")
        bills = call("bill_payment.list_bills", {}).get("bills", [])
        choices = [bill for bill in bills if bill.get("type") == args["type"]]
        if len(choices) != 1:
            raise ValueError("The bill type did not resolve to one unpaid bill")
        bill = choices[0]
        balance = call("ewallet.get_balance", {}).get("balance")
        amount = bill.get("amount")
        if not isinstance(balance, (int, float)) or not isinstance(amount, (int, float)):
            raise ValueError("No observed balance or bill amount")
        call("bill_payment.pay", {"bill_id": bill["id"], "source": "ewallet" if balance >= amount else "bank"})
        return
    if name != "workflow.turn_off_floor_lights":
        raise ValueError("Unknown workflow")
    if set(args) != {"floor", "keep_rooms_with_active"} or type(args["floor"]) is not int:
        raise ValueError("Specify a floor and exception device type")
    floor, kind = args["floor"], args["keep_rooms_with_active"]
    if kind not in ("fan", "climate"):
        raise ValueError("Exception must be fan or climate")
    devices = call("smart_home.get_status", {}).get("devices", [])
    excluded = set()
    for device in devices:
        if device.get("floor") != floor or device.get("type") != kind:
            continue
        state = device.get("state", {})
        active = state.get("power") == "on" if kind == "fan" else state.get("mode") not in (None, "off")
        if active:
            excluded.add(device.get("room"))
    for device in devices:
        if device.get("type") == "light" and device.get("floor") == floor and device.get("room") not in excluded:
            call("smart_home.light_set", {"device_id": device["id"], "power": "off"})


def direct_missing_request(row: dict, allowed: set[str]) -> dict | None:
    """Phát hiện yêu cầu độc lập bị thiếu tham số thời gian bắt buộc."""
    if "sys.ask_missing_info" not in allowed:
        return None
    text = fold(row["transcript"])
    text = re.sub(START, "", text)
    if re.search(
        r"\b(?:gio|luc|mai|nay|tuan|thang|sang|trua|chieu|toi nay|phut|tieng|"
        r"thu hai|thu ba|chu nhat|truoc|sau khi|khi|neu|roi|xong|va|dong thoi|"
        r"hang ngay|moi ngay|dung|huy|doi|lai do|cai do)\b|\d",
        text,
    ):
        return None
    if "alarm.create" in allowed and re.fullmatch(
        r"(?:dat|tao|hen) (?:giup |gium )?bao thuc(?: (?:cho|giup|gium|toi|minh|anh|chi|em|nhe|nha|voi|di|goi|day))*[.!? ]*",
        text,
    ):
        return {"tool_name": "alarm.create", "params": ["time"]}
    if "reminders.create" in allowed and re.fullmatch(
        r"nhac(?: nho)? (?:toi|minh|anh|chi|em) (?:goi(?: dien)?|mua|mang|don|do|nop|gui|uongs?|uong|tra) [\w .!?]+",
        text,
    ):
        return {"tool_name": "reminders.create", "params": ["time"]}
    return None


def timestamp_for_alarm(text: str, now: datetime) -> str | None:
    cleaned = re.sub(START, "", fold(text))
    if COMPOUND.search(cleaned):
        return None
    if not re.match(r"(?:(?:ngay )?mai )?(?:dat|tao|hen) (?:giup |gium )?bao thuc\b", cleaned):
        return None
    times = hints(text, now).get("clocks", [])
    if len(times) != 1 or not times[0].get("timestamp"):
        return None
    phrase = times[0]["phrase"]
    tail = cleaned.partition(phrase)[2]
    if not re.fullmatch(FILLER, tail):
        return None
    if re.search(r"\b(?:ten|nhan|hang ngay|moi ngay|thu|truoc|sau khi|som hon|muon hon)\b", cleaned):
        return None
    return times[0]["timestamp"]


def contact_name(text: str) -> str | None:
    match = re.search(r"\b(?:gọi điện|gọi)\s+cho\s+(.+)$", text.strip(), re.IGNORECASE)
    if not match:
        return None
    query = re.split(
        r"\s+(?:giúp|giùm|hộ|dùm|luôn|nhé|nha|nghe|đi|với)\b",
        match[1],
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.!?")
    if not re.fullmatch(r"[^\W\d_]+(?:\s+[^\W\d_]+){0,3}", query):
        return None
    if (
        COMPOUND.search(fold(match[1]))
        or fold(query) in {"toi", "minh", "anh", "chi", "em", "ban", "nguoi do", "nguoi ay"}
        and query != "Minh"
    ):
        return None
    prefix = fold(text[: match.start()])
    prefix = re.sub(START, "", prefix).strip(" ,")
    if prefix and not re.fullmatch(r"(?:giup|gium|ho|dum) (?:toi|minh|anh|chi|em)", prefix):
        return None
    return query


def call_contact(query: str, call) -> None:
    contacts = call("contacts.search", {"query": query}).get("contacts", [])
    exact = [contact for contact in contacts if fold(contact.get("name", "")) == fold(query)]
    selected = (
        contacts[0]
        if len(contacts) == 1
        else exact[0]
        if len(query.split()) > 1 and len(exact) == 1
        else None
    )
    if selected:
        call("phone.call", {"contact_id": selected["id"]})
    else:
        call("sys.ask_missing_info", {"tool_name": "phone.call", "params": ["contact_id"]})


def route(row: dict, allowed: set[str]):
    """Định tuyến nhanh (Fast-Path) cho các yêu cầu rõ ràng không cần qua vòng lặp LLM."""
    request = row["transcript"]
    try:
        now = datetime.fromisoformat(row["context"]["current_time"])
    except (KeyError, TypeError, ValueError):
        return None
    phone = {"contacts.search", "phone.call", "sys.ask_missing_info"} <= allowed
    alarm = "alarm.create" in allowed
    if phone and alarm:
        parts = re.split(
            r"\s*[,;]?\s+\b(?:rồi|với|và)\s+(?=gọi(?: điện)? cho\b)",
            request,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        if len(parts) == 2:
            timestamp, query = timestamp_for_alarm(parts[0], now), contact_name(parts[1])
            if timestamp and query:
                def alarm_and_phone(call):
                    call("alarm.create", {"time": timestamp})
                    call_contact(query, call)

                return "alarm_and_phone", alarm_and_phone
    if alarm:
        timestamp = timestamp_for_alarm(request, now)
        if timestamp:
            return "alarm", lambda call: call("alarm.create", {"time": timestamp})
    if phone and not any(
        c.get("name") == "contacts.search"
        for turn in row.get("history", [])
        for c in turn.get("tool_calls", [])
    ):
        query = contact_name(request)
        if query:
            return "contact_call", lambda call: call_contact(query, call)
    text = fold(request)
    workflows = available_operations(request, allowed)
    if "workflow.pay_due_bills" in workflows and not re.search(
        r"\b(?:nhan|nhac|gui|goi|chuyen|roi|vi dien tu)\b", text
    ):
        window = re.search(rf"\b({NUMBER}) ngay toi\b", text)
        caps = re.search(r"\b(?:tren|hon|vuot|qua) (\d+)(?: (nghin|ngan|trieu)|[.,](\d{3}))\b", text)
        if window and caps and re.search(r"\b(?:chua tra|de lai|de nguyen|khoi tra|dung tra)\b", text):
            maximum = (
                int(caps[1]) * 1000 + int(caps[3])
                if caps[3]
                else int(caps[1]) * (1_000_000 if caps[2] == "trieu" else 1000)
            )
            days = number(window[1])
            if days is not None:
                return (
                    "due_bills",
                    lambda call: execute(
                        "workflow.pay_due_bills",
                        {"days_ahead": days, "max_vnd": maximum},
                        now,
                        call,
                    ),
                )
    return None


class IntentRouter:
    """Tầng 2: Router Gateway (Y/N Classifier) phân loại nhanh câu thoại có cần gọi Tool không."""

    def needs_tool(self, clean_text: str) -> bool:
        """
        Trả về False nếu câu thoại rỗng hoặc chỉ là câu chào hỏi/cảm ơn/xã giao thuần túy
        (giúp đi thẳng vào Fast-Path Node trả về `{"tool_calls": []}`).
        """
        folded = fold(clean_text)
        folded = re.sub(START, "", folded).strip(" ,.!?")
        if not folded:
            return False
        if CHITCHAT_PATTERNS.fullmatch(folded):
            return False
        return True

    def match_fast_route(self, row: dict, allowed: set[str]):
        """Kiểm tra xem mẫu có thể xử lý bằng đường dẫn tất định (deterministic route) không."""
        return route(row, allowed)

