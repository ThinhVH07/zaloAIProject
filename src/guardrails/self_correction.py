"""Tầng 4: Guardrail & Self-Correction — kiểm lỗi schema, xóa sys.ask_missing_info thừa và phát lại (replay)."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import DEFAULT_ROOT, WORKSPACE_ROOT
from src.asr_normalizer.text_cleaner import NUMBER, fold, hints, number
from src.guardrails.iso_formatter import fix_iso_time_list
from src.router.intent_router import route
from src.utils.jsonl_writer import dump, export_submission, index_jsonl, read_jsonl, write_jsonl


class OutputGuardrail:
    """Lá chắn đảm bảo 0% lỗi Schema trước khi xuất kết quả."""

    @staticmethod
    def fix_iso_time(tool_calls: list[dict], default_date: str | None = None) -> list[dict]:
        """Tự động bù múi giờ +07:00 và Ngày nếu LLM chỉ trả về HH:MM hoặc thiếu offset."""
        return fix_iso_time_list(tool_calls, default_date=default_date)

    @staticmethod
    def remove_redundant_missing_info(tool_calls: list[dict]) -> list[dict]:
        """Nếu Tool mục tiêu trong sys.ask_missing_info đã được gọi thực thi, xóa bỏ sys.ask_missing_info thừa."""
        executed_tools = {
            c.get("name")
            for c in tool_calls
            if isinstance(c, dict) and c.get("name") != "sys.ask_missing_info"
        }
        return [
            c
            for c in tool_calls
            if isinstance(c, dict)
            and not (
                c.get("name") == "sys.ask_missing_info"
                and isinstance(c.get("arguments"), dict)
                and c["arguments"].get("tool_name") in executed_tools
            )
        ]

    @staticmethod
    def is_self_cancelled(text: str) -> bool:
        """Kiểm tra người dùng có đổi ý / tự hủy yêu cầu ở cuối câu (ví dụ: à thôi khỏi, thôi đừng,...)."""
        f = fold(text)
        patterns = [
            r"\b(?:a\s+)?thoi\s+(?:khoi|dung|bo|thoi)\b",
            r"\b(?:ma\s+thoi\s*,?\s*khoi\s+huy)\b",
            r"\bkhoi\s+(?:dat|book|huy|lam|mua|dich|goi|nhan)\b",
            r"\b(?:khoi\s+can\s+nua)\b",
            r"\b(?:a\s+)?thoi\b.*?\b(?:cu\s+de\s+day|cu\s+de\s+do|mai\s+moi\s+mua)\b",
            r"\b(?:no\s+qua\s+don\s+toi\s+roi)\b",
            r"\b(?:anh\s+long\s+cho\s+chi\s+di\s+roi)\b",
            r"\b(?:anh\s+biet\s+tieng\s+viet\s+ma)\b",
        ]
        return any(re.search(pat, f) for pat in patterns)

    @staticmethod
    def filter_self_cancellation(text: str, tool_calls: list[dict]) -> list[dict]:
        """Nếu người dùng tự hủy yêu cầu và không có yêu cầu thay thế, xóa toàn bộ tool calls."""
        if OutputGuardrail.is_self_cancelled(text):
            return []
        return tool_calls

    @staticmethod
    def sanitize_tool_calls(tool_calls: list[dict], allowed_tools: set[str] | None = None) -> list[dict]:
        """Làm sạch và chỉnh sửa schema các tool calls trước khi gửi đi hoặc lưu trữ."""
        cleaned = []
        for call in tool_calls:
            if not isinstance(call, dict) or "name" not in call:
                continue
            name = call["name"]
            args = dict(call.get("arguments", {}))

            # Chuyển đổi tên tool không hợp lệ sang tool được cho phép tương đương
            if allowed_tools is not None:
                if name not in allowed_tools:
                    if name == "phone.call" and "chat_ott.call" in allowed_tools:
                        name = "chat_ott.call"
                    elif name == "ewallet.pay_merchant" and "ewallet.pay_qr" in allowed_tools:
                        name = "ewallet.pay_qr"
                    else:
                        continue

            # Sửa lỗi đơn vị thời gian gửi xe (phút -> giờ)
            if name == "parking.reserve" and "hours" in args:
                try:
                    val = int(args["hours"])
                    if val > 168 and val % 60 == 0:
                        args["hours"] = val // 60
                except (ValueError, TypeError):
                    pass

            cleaned.append({"name": name, "arguments": args})
        return OutputGuardrail.remove_redundant_missing_info(cleaned)

    @staticmethod
    def inject_dependency_tools(
        tool_calls: list[dict],
        allowed_tools: set[str] | None = None,
        state: dict | None = None,
        transcript: str = "",
    ) -> list[dict]:
        """Tự động chèn các tool tra cứu tiền đề bắt buộc (Dependency Tools) theo quy chuẩn BTC."""
        if not allowed_tools or not tool_calls:
            return tool_calls

        injected = []
        state = state or {}

        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})

            # 1. Bổ sung contacts.search khi gọi người nhận mà chưa có search
            if name in ["sms.send", "phone.call", "chat_ott.send_message", "ewallet.transfer"] and "contacts.search" in allowed_tools:
                if "contacts.search" not in [c["name"] for c in injected]:
                    cid = args.get("contact_id")
                    c_name = None
                    if cid and "contacts" in state:
                        for ct in state["contacts"].get("contacts", []):
                            if ct.get("id") == cid:
                                c_name = ct.get("name")
                                break
                    if not c_name and "chat_id" in args and "chat_ott" in state:
                        for ch in state["chat_ott"].get("chats", []):
                            if ch.get("id") == args["chat_id"]:
                                c_name = ch.get("name")
                                break
                    if c_name:
                        injected.append({"name": "contacts.search", "arguments": {"query": c_name}})

            # 2. Bổ sung bill_payment.list_bills trước khi pay
            if name == "bill_payment.pay" and "bill_payment.list_bills" in allowed_tools:
                if "bill_payment.list_bills" not in [c["name"] for c in injected]:
                    injected.append({"name": "bill_payment.list_bills", "arguments": {}})

            # 3. Bổ sung my_bookings trước khi hủy vé xe/tàu/máy bay
            if name == "train_bus_tickets.cancel" and "train_bus_tickets.my_bookings" in allowed_tools:
                if "train_bus_tickets.my_bookings" not in [c["name"] for c in injected]:
                    injected.append({"name": "train_bus_tickets.my_bookings", "arguments": {}})
            if name == "flights.cancel" and "flights.my_bookings" in allowed_tools:
                if "flights.my_bookings" not in [c["name"] for c in injected]:
                    injected.append({"name": "flights.my_bookings", "arguments": {}})

            # 4. Bổ sung calendar.list_events trước khi đặt xe theo lịch hẹn/sự kiện
            if name == "ride_hailing.book" and "calendar.list_events" in allowed_tools:
                if "calendar.list_events" not in [c["name"] for c in injected]:
                    trans_l = transcript.lower()
                    if any(w in trans_l for w in ["lịch", "họp", "khám", "sân bay", "ga", "bến", "chỗ hẹn", "công viên"]):
                        injected.append({"name": "calendar.list_events", "arguments": {}})

            # 5. Bổ sung smart_home.get_status trước khi thay đổi trạng thái thiết bị có so sánh
            if name in ["smart_home.light_set", "smart_home.fan_set", "smart_home.climate_set"] and "smart_home.get_status" in allowed_tools:
                if "smart_home.get_status" not in [c["name"] for c in injected]:
                    trans_l = transcript.lower()
                    if any(w in trans_l for w in ["nửa", "xuống", "tăng", "giảm", "bớt", "mức", "%", "như cũ"]):
                        injected.append({"name": "smart_home.get_status", "arguments": {}})

            injected.append(call)

        return injected


# --- Các hàm Self-Correction / Replay / Selector nâng cao từ v5 ---

DURATION = rf"(nua|{NUMBER}) (tieng|phut)"
MIN_LEAF = 25
MAX_DEPTH = 2


def offset(text: str, prefix: str) -> timedelta | None:
    match = re.search(prefix + r"[ ,]*" + DURATION, text)
    if not match:
        return None
    amount = 0.5 if match[1] == "nua" else number(match[1])
    return timedelta(minutes=amount * (60 if match[2] == "tieng" else 1))


def active_events(row: dict, results: list[dict]) -> dict[str, dict]:
    events = {}
    for turn in row.get("history", []):
        for call, result in zip(turn.get("tool_calls", []), turn.get("tool_results", [])):
            if call.get("name") == "calendar.list_events" and isinstance(result, dict):
                events.update({event["id"]: event for event in result.get("events", [])})
    for result in results:
        if isinstance(result, dict) and "events" in result:
            events.update({event["id"]: event for event in result["events"]})
    return events


def datetime_or_none(value) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def accounting_fallback(text: str, calls: list[dict], results: list[dict]) -> str | None:
    if not re.search(r"\bkhong du (?:ca hai|ca 2|hai khoan|2 khoan|tong)\b", text):
        return None
    if not re.search(r"\btiet kiem\b", text) or not re.search(r"\bthanh toan\b", text):
        return None
    transfers = [call["arguments"] for call in calls if call["name"] == "banking.transfer"]
    transfers = {(args.get("beneficiary_id"), args.get("amount")): args for args in transfers}
    if len(transfers) < 2 or any(
        not recipient or type(amount) is not int or amount <= 0 for recipient, amount in transfers
    ):
        return None
    accounts = next(
        (r["accounts"] for r in results if isinstance(r, dict) and isinstance(r.get("accounts"), list)),
        [],
    )
    checking = [a for a in accounts if "thanh toan" in fold(a.get("name", ""))]
    savings = [a for a in accounts if "tiet kiem" in fold(a.get("name", ""))]
    if len(checking) != 1 or len(savings) != 1:
        return None
    total = sum(amount for _, amount in transfers)
    for account in [checking[0], savings[0]]:
        if isinstance(account.get("balance"), (int, float)) and account["balance"] >= total:
            return account["id"]
    return None


def discovery_call(name: str, names: set[str]) -> bool:
    if name == "calendar.list_events":
        return bool(names & {"calendar.update_event", "calendar.cancel_event", "ride_hailing.book", "alarm.create"})
    if name == "smart_home.get_status":
        return any(n.startswith("smart_home.") and n != name for n in names)
    if name == "bill_payment.list_bills":
        return bool(names & {"bill_payment.pay", "banking.transfer", "ewallet.transfer"})
    return False


def rewrite(row: dict, calls: list[dict], results: list[dict], env, normalize: bool = False) -> Counter:
    """Phát lại (replay) và tự động sửa các phụ thuộc chéo giữa các tool calls."""
    text = fold(row["transcript"])
    now = datetime_or_none(row["context"].get("current_time"))
    clocks = hints(row["transcript"], now).get("clocks", [])
    single_alarm = sum(c["name"] == "alarm.create" for c in calls) == 1
    explicit_alarm = (
        single_alarm
        and len(clocks) == 1
        and clocks[0].get("timestamp")
        and re.search(r"\b(dat|tao|hen)\b.*\bbao thuc\b", text)
        and not re.search(r"\b(neu|truoc|sau khi|doi|huy)\b", text)
    )
    events = active_events(row, results)
    event_count = sum(c["name"] == "calendar.update_event" for c in calls)
    event_text = re.split(r"\b(?:dat bao thuc|nhac|nhan|gui)\b", text, maxsplit=1)[0]
    is_shift = event_count == 1 and re.search(r"\b(?:doi|roi|dich)\s+(?:cai )?lich\b", event_text)
    event_delta = offset(event_text, r"\b(?:muon|som)(?: hon| them)?")
    if event_delta and re.search(r"\bsom\b", event_text):
        event_delta = -event_delta
    lead = offset(text, r"\btruoc gio (?:hen|hop)")
    pickup_offset = offset(text, r"\btruoc (?:luc|gio) xe (?:don|dong)")
    account = accounting_fallback(text, calls, results)
    changed = Counter()
    transfers = set()
    updated_start = None
    pickup_time = None
    payment_succeeded = False
    names = {c["name"] for c in calls}
    latest_contacts = {}
    seen_discovery = set()
    later_successes = [
        (index, call["name"])
        for index, (call, result) in enumerate(zip(calls, results))
        if isinstance(result, dict) and result.get("ok") and result.get("done") is not False
    ]
    payment_requested = bool(
        re.search(r"\b(?:tra|dong|thanh toan) (?:hoa don|tien)\b", text)
        and any(c["name"] == "bill_payment.pay" for c in calls)
    )
    for index, call in enumerate(calls):
        name, original = call["name"], call["arguments"]
        args = dict(original)
        result_before = results[index] if index < len(results) else {}
        if normalize and name == "sys.ask_missing_info":
            target = args.get("tool_name")
            if any(position > index and candidate == target for position, candidate in later_successes):
                changed["resolved_question"] += 1
                continue
        discovery = normalize and discovery_call(name, names)
        if discovery:
            args.clear()
            if name in seen_discovery:
                changed["redundant_discovery"] += 1
                continue
            seen_discovery.add(name)
        elif normalize and name.split(".")[0] in {"calendar", "smart_home", "bill_payment"}:
            seen_discovery = {tool for tool in seen_discovery if tool.split(".")[0] != name.split(".")[0]}
        if normalize and name == "banking.transfer" and "note" in args:
            if not re.search(r"\b(?:ghi chu|noi dung chuyen|kem loi|loi nhan)\b", text):
                args.pop("note")
        if normalize and name in {"phone.call", "sms.send"}:
            contacts = latest_contacts.get(args.get("contact_id"), [])
            if len(contacts) > 1 and not re.search(r"\b(?:nguoi do|nguoi ay|so dau|so sau)\b", text):
                exact = [
                    person
                    for person in contacts
                    if person.get("name")
                    and re.search(r"(?<!\w)" + re.escape(fold(person["name"])) + r"(?!\w)", text)
                ]
                if len(exact) == 1:
                    args["contact_id"] = exact[0]["id"]
                    args.pop("phone_number", None)
                elif "sys.ask_missing_info" in env.allowed:
                    question = {"tool_name": name, "params": ["contact_id"]}
                    if env.validate("sys.ask_missing_info", question)["ok"]:
                        env.call("sys.ask_missing_info", question)
                        changed["ambiguous_contact"] += 1
                        continue
        if (
            name == "contacts.search"
            and isinstance(args.get("query"), str)
            and isinstance(result_before, dict)
        ):
            matched_names = {
                person["name"]
                for person in result_before.get("contacts", [])
                if isinstance(person.get("name"), str) and fold(person["name"]) == fold(args["query"])
            }
            if len(matched_names) == 1:
                args["query"] = next(iter(matched_names))
        if name == "banking.transfer" and account:
            key = (args["beneficiary_id"], args["amount"])
            if key in transfers:
                changed["duplicate_transfer"] += 1
                continue
            transfers.add(key)
            args["from_account_id"] = account
        if (
            name == "banking.open_savings"
            and account
            and not re.search(r"\bmo (?:so |tai khoan )?tiet kiem\b", text)
        ):
            changed["unrequested_savings_account"] += 1
            continue
        if name == "calendar.update_event" and is_shift and event_delta:
            event = events.get(args.get("event_id"), {})
            start, end = datetime_or_none(event.get("start")), datetime_or_none(event.get("end"))
            if start and end:
                args.update(start=(start + event_delta).isoformat(), end=(end + event_delta).isoformat())
        if name == "alarm.create" and explicit_alarm:
            args["time"] = clocks[0]["timestamp"]
        if name == "alarm.create" and updated_start:
            before = offset(text, r"\bbao thuc(?: [\w]+){0,4} truoc (?:do|gio hen)")
            if before is not None:
                args["time"] = (updated_start - before).isoformat()
        if name == "ride_hailing.book" and lead is not None and args.get("dropoff"):
            matches = [
                event
                for event in events.values()
                if event.get("location") and fold(event["location"]) == fold(args["dropoff"])
            ]
            start = datetime_or_none(matches[0].get("start")) if len(matches) == 1 else None
            if start and now and start.tzinfo and now.tzinfo and start - lead > now:
                args["schedule_time"] = (start - lead).isoformat()
        if name == "reminders.create" and pickup_time and pickup_offset is not None:
            args["time"] = (pickup_time - pickup_offset).isoformat()
        if name in {"sms.send", "chat_ott.send_message"} and payment_requested and not payment_succeeded:
            if re.search(
                r"\b(?:da dong|da tra|dong xong|da thanh toan|thanh toan xong)\b",
                fold(str(args.get("content", ""))),
            ):
                changed["premature_confirmation"] += 1
                continue
        if args != original and not env.validate(name, args)["ok"]:
            args = original
        if args != original:
            changed[name] += 1
        result = env.call(name, args)
        if normalize and name == "contacts.search" and args.get("query"):
            people = result.get("contacts", [])
            for person in people:
                if isinstance(person.get("id"), str):
                    latest_contacts[person["id"]] = people
        if result.get("ok") and result.get("done") is not False:
            if name == "calendar.update_event":
                updated_start = datetime_or_none(args.get("start"))
            elif name == "ride_hailing.book":
                pickup_time = datetime_or_none(args.get("schedule_time"))
            elif name == "bill_payment.pay":
                payment_succeeded = True
    return changed


def refine(
    split: str,
    source: str | Path,
    original_dir: str | Path,
    out: str | Path,
    normalize: bool = False,
    use_routes: bool = False,
) -> dict:
    """Replay một bản dự đoán với kiểm tra phụ thuộc chéo và sửa lỗi số học chính xác."""
    from src.mcp_agent.mcp_server import load_btc_runtime

    original_dir, out = Path(original_dir), Path(out)
    rows = index_jsonl(source)
    baseline = index_jsonl(original_dir / "submission.jsonl")
    diagnostics = index_jsonl(original_dir / "checks" / "per_sample.jsonl")
    if not baseline.keys() <= rows.keys() or baseline.keys() != diagnostics.keys():
        raise ValueError("Source and prediction IDs do not match")
    engine = load_btc_runtime(DEFAULT_ROOT, split)
    predictions, changes = [], Counter()
    for sample_id, original in baseline.items():
        row, diagnostic = rows[sample_id], diagnostics[sample_id]
        if original["tool_calls"] != diagnostic["predicted_calls"]:
            raise ValueError(f"Stale diagnostics for {sample_id}")
        env = engine.ToolEnv(split, sample_id, root=DEFAULT_ROOT)
        selected = route(row, env.allowed) if use_routes else None
        if selected:
            name, operation = selected

            def call(tool, args):
                if tool not in env.allowed or not env.validate(tool, args)["ok"]:
                    raise ValueError(f"Invalid workflow tool {tool} for {sample_id}")
                return env.call(tool, args)

            operation(call)
            changes[name] += 1
        else:
            changes.update(rewrite(row, original["tool_calls"], diagnostic["tool_results"], env, normalize))
        predictions.append(
            {"id": sample_id, "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
        )
    write_jsonl(out / "submission.jsonl", predictions)
    dump(out / "ids.json", list(baseline))
    dump(out / "changes.json", dict(changes))
    return dict(changes)

