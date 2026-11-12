"""Regex ép múi giờ chuẩn +07:00, ISO 8601 và ép kiểu số nguyên cho các tham số Tool."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from config.settings import DEFAULT_TIMEZONE

TIME_KEYS = ("time", "start", "end", "schedule_time", "from_time", "delivery_time", "due")


def calendar_dates(now: datetime | None) -> dict[str, str]:
    """Tạo bảng tra nhanh ngày hôm nay / ngày mai / ngày kia theo chuẩn YYYY-MM-DD."""
    if now is None:
        return {}
    return {
        name: (now.date() + timedelta(days=delta)).isoformat()
        for name, delta in (("hôm nay", 0), ("ngày mai", 1), ("ngày kia", 2))
    }


def normalize_arguments(args: dict, schema: dict, now: datetime | None = None) -> dict:
    """Chuẩn hóa kiểu dữ liệu integer và timestamp ISO-8601 (+07:00) dựa trên JSON Schema."""
    cleaned = dict(args)
    for key, value in args.items():
        spec = schema.get("properties", {}).get(key, {})
        if isinstance(value, str) and spec.get("type") == "integer":
            if re.fullmatch(r"[+-]?\d+|[+-]?\d{1,3}(?:[.,]\d{3})+", value.strip()):
                cleaned[key] = int(re.sub(r"[.,]", "", value.strip()))
        if isinstance(value, str) and now is not None:
            if re.fullmatch(r"\d{4}-\d\d-\d\d[ T]\d\d:\d\d(?::\d\d)?(?:Z|[+-]\d\d:\d\d)?", value):
                description = spec.get("description", "")
                if spec.get("format") == "date-time" or "ISO-8601" in description:
                    try:
                        parsed = datetime.fromisoformat(value)
                    except ValueError:
                        continue
                    cleaned[key] = (
                        parsed if parsed.tzinfo else parsed.replace(tzinfo=now.tzinfo)
                    ).isoformat(timespec="seconds")
    return cleaned


def fix_iso_time_list(
    tool_calls: list[dict],
    default_tz: str = DEFAULT_TIMEZONE,
    default_date: str | None = None,
) -> list[dict]:
    """Tự động bù múi giờ +07:00, giây :00 và ngày nếu LLM chỉ trả về định dạng thiếu."""
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for key in TIME_KEYS:
            if key in args and isinstance(args[key], str):
                val = args[key].strip()
                # Trường hợp chỉ có HH:MM hoặc HH:MM:SS và có default_date
                if default_date and re.match(r"^\d{2}:\d{2}(?::\d{2})?$", val):
                    if len(val) == 5:
                        val = f"{val}:00"
                    val = f"{default_date}T{val}{default_tz}"
                # Trường hợp YYYY-MM-DDTHH:MM (thiếu giây và timezone)
                elif re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}$", val):
                    val = val.replace(" ", "T") + f":00{default_tz}"
                # Trường hợp YYYY-MM-DDTHH:MM:SS (thiếu timezone +07:00)
                elif re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}$", val):
                    val = val.replace(" ", "T") + default_tz
                args[key] = val
    return tool_calls

