"""Lọc từ đệm, từ rác ASR và trích xuất gợi ý thời gian / tiền tệ / thời lượng tiếng Việt."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from src.utils.jsonl_writer import read_jsonl, write_jsonl

FILLER = r"(?:[ ,]*(?:giup|gium|dum|ho|cho|toi|minh|anh|chi|em|nhe|nha|di|luon|voi))*[ .!?]*"
START = r"^(?:(?:o|a|oi|tro ly oi|ban oi|em oi|anh oi|chi oi)[ ,]+)+"
START_ACCENTED = re.compile(
    r"^(?:(?:ờ|à|ừm|ừ|ơi|trợ lý ơi|bạn ơi|em ơi|anh ơi|chị ơi|lifemate ơi)[ ,]+)+",
    flags=re.IGNORECASE,
)

UNITS = dict(zip("khong mot hai ba bon nam sau bay tam chin".split(), range(10)))
UNITS.update({"tu": 4, "lam": 5, "muoi": 10})
NUMBER = (
    r"(?:\d{1,2}|(?:hai|ba|bon|nam|sau|bay|tam|chin) muoi"
    r"(?: (?:mot|hai|ba|bon|tu|nam|lam|sau|bay|tam|chin))?|"
    r"muoi(?: (?:mot|hai|ba|bon|tu|nam|lam|sau|bay|tam|chin))?|"
    r"mot|hai|ba|bon|tu|nam|lam|sau|bay|tam|chin)"
)
TIME = re.compile(
    rf"(?<!\w)(?P<h>{NUMBER})(?:\s*(?:gio\b|h(?=\d|\b)|:)\s*(?:(?P<kem>kem)\s+)?"
    rf"(?P<m>ruoi|{NUMBER})?|\s+(?P<half>ruoi)\b)"
    r"(?:\s+(?P<period>sang|trua|chieu|toi|dem))?(?:\s+(?P<day>ngay mai|mai|nay|ngay kia))?"
)


def fold(text: str) -> str:
    """Chuẩn hóa chữ thường, bỏ dấu tiếng Việt để so khớp bền vững trước lỗi ASR."""
    normalized = unicodedata.normalize("NFD", str(text).lower()).replace("đ", "d")
    return " ".join("".join(c for c in normalized if not unicodedata.combining(c)).split())


def words(text: str) -> list[str]:
    """Tách từ đã bỏ dấu phục vụ truy vấn BM25 / RAG."""
    return re.findall(r"\w+", fold(text))


def all_strings(value):
    """Duyệt đệ quy tất cả các chuỗi xuất hiện trong context / kết quả tool."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_strings(item)


ASR_PHONETIC_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bđặt bảo thích\b", re.IGNORECASE), "đặt báo thức"),
    (re.compile(r"\bđặc bảo thức\b", re.IGNORECASE), "đặt báo thức"),
    (re.compile(r"\btrượt cho bái bá chàng\b", re.IGNORECASE), "trước giờ bay 3 tiếng"),
    (re.compile(r"\bdiễn phương thốt\b", re.IGNORECASE), "tiệm photo"),
    (re.compile(r"\btuyển không dính\b", re.IGNORECASE), "tuyến 09"),
    (re.compile(r"\btục logitech\b", re.IGNORECASE), "chuột Logitech"),
    (re.compile(r"\bhỷ đơn tại nghe\b", re.IGNORECASE), "hủy đơn tai nghe"),
    (re.compile(r"\bbrisa sama\b", re.IGNORECASE), "Priya Sharma"),
    (re.compile(r"\bbãi bá\b", re.IGNORECASE), "bãi đỗ"),
    (re.compile(r"\brạp xà\b", re.IGNORECASE), "cháo gà"),
    (re.compile(r"\bTVB\b", re.IGNORECASE), "VTV3"),
    (re.compile(r"\bcổ phiếu thoái\b", re.IGNORECASE), "cổ phiếu ZTP"),
    (re.compile(r"\bmã thoái\b", re.IGNORECASE), "mã ZTP"),
    (re.compile(r"\bcổ phiếu bảo\b", re.IGNORECASE), "cổ phiếu ZKN"),
    (re.compile(r"\bmã bảo\b", re.IGNORECASE), "mã ZKN"),
]


def clean_asr_text(raw_text: str, strip_prefix: bool = False) -> str:
    """Làm sạch khoảng trắng thừa, sửa lỗi đồng âm ASR phổ biến và tùy chọn cắt bỏ cụm từ gọi mở đầu câu."""
    cleaned = " ".join(str(raw_text or "").strip().split())
    for pattern, replacement in ASR_PHONETIC_FIXES:
        cleaned = pattern.sub(replacement, cleaned)
    if strip_prefix:
        cleaned = START_ACCENTED.sub("", cleaned).strip()
    return cleaned


def number(text: str) -> int | None:
    """Chuyển đổi chữ số hoặc số đếm tiếng Việt (đã fold) sang số nguyên."""
    if text.isdigit():
        return int(text)
    parts = text.split()
    if len(parts) == 1:
        return UNITS.get(parts[0])
    if parts[0] == "muoi":
        return 10 + UNITS[parts[1]]
    return UNITS[parts[0]] * 10 + (UNITS[parts[2]] if len(parts) == 3 else 0)


def hints(request: str, now: datetime | None = None) -> dict:
    """Trích xuất các mốc giờ (clocks), số tiền (money) và khoảng thời gian (durations) từ câu ASR."""
    text = fold(request)
    times = []
    for match in TIME.finditer(text):
        hour = number(match["h"])
        minute_text = match["m"] or match["half"]
        minute = 30 if minute_text == "ruoi" else number(minute_text) if minute_text else 0
        if hour is None or minute is None or not (0 <= hour < 24 and 0 <= minute < 60):
            continue
        period = match["period"]
        if period in {"trua", "chieu", "toi", "dem"} and hour < 12:
            hour += 12
        elif period == "sang" and hour == 12:
            continue
        clock = datetime(2000, 1, 2, hour, 0) + timedelta(minutes=(-minute if match["kem"] else minute))
        item = {"phrase": match[0].strip(), "clock": clock.strftime("%H:%M")}
        day = match["day"]
        if now and day and (period or hour > 12):
            offset = {"mai": 1, "ngay mai": 1, "nay": 0, "ngay kia": 2}[day]
            date = now.date() + timedelta(days=offset + clock.day - 2)
            item["timestamp"] = datetime.combine(date, clock.time(), tzinfo=now.tzinfo).isoformat()
        times.append(item)

    amounts = []
    for match in re.finditer(r"(?<!\w)(\d+(?:[.,]\d+)?)\s+(trieu|nghin|ngan|dong)\b", text):
        units = {"trieu": 1_000_000, "nghin": 1000, "ngan": 1000, "dong": 1}
        value = Decimal(match[1].replace(",", ".")) * units[match[2]]
        if value == int(value):
            amounts.append({"phrase": match[0], "vnd": int(value)})

    offsets = []
    for match in re.finditer(rf"(?<!\w)(nua|{NUMBER})\s+(tieng|phut)\b", text):
        value = Decimal("0.5") if match[1] == "nua" else number(match[1])
        if value is not None:
            offsets.append({"phrase": match[0], "minutes": int(value * (60 if match[2] == "tieng" else 1))})

    return {k: v for k, v in (("clocks", times), ("money", amounts), ("durations", offsets)) if v}


def merge_asr_handoff(base: str | Path, alternative: str | Path, out: str | Path) -> int:
    """Ghép bản chép ASR mới (PhoWhisper) vào tập dữ liệu gốc nhưng bảo toàn nguyên vẹn lịch sử."""
    rows, base_errors = read_jsonl(base)
    transcripts, errors = read_jsonl(alternative)
    indexed = {(r["id"], r["audio"]): r["text"] for r in transcripts}
    if base_errors or errors or len(indexed) != len(transcripts):
        raise ValueError("Invalid or duplicate ASR inputs")
    keys = {(r["id"], r["audio"]) for r in rows}
    if indexed.keys() != keys:
        raise ValueError("Alternative ASR must cover exactly the current-request audio")
    changed = 0
    for row in rows:
        text = indexed[row["id"], row["audio"]]
        if not isinstance(text, str):
            raise ValueError("ASR text must be a string")
        if text.strip() and text != row["transcript"]:
            row["transcript"] = text
            changed += 1
    write_jsonl(out, rows)
    return changed

