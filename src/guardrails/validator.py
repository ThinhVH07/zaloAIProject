"""Kiểm tra tính hợp lệ của Tool Calls, đối chiếu ID thực thể đã quan sát và gọi env.validate()."""
from __future__ import annotations

import operator
import re
from collections import defaultdict
from datetime import datetime, timedelta

from src.asr_normalizer.text_cleaner import all_strings, fold
from src.utils.jsonl_writer import history_public

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def unknown_ids(value, observed: set[str]):
    """Tìm các tham số *_id chưa từng xuất hiện trong lịch sử hoặc kết quả quan sát."""
    if isinstance(value, dict):
        for name, item in value.items():
            if name.endswith("_id") and isinstance(item, str) and item not in observed:
                yield name, item
            else:
                yield from unknown_ids(item, observed)
    elif isinstance(value, list):
        for item in value:
            yield from unknown_ids(item, observed)


class Observations:
    """Index typed IDs only from actual tool results."""

    def __init__(self, row: dict):
        history = history_public(row.get("history", []))
        self.strings = set(all_strings(row.get("context", {}))) | set(all_strings(history))
        self.refs = defaultdict(dict)
        self.contact_query = {}
        for turn in history:
            for call, result in zip(turn.get("tool_calls", []), turn.get("tool_results", [])):
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    self.add(call["name"], call.get("arguments", {}), result)

    def add(self, name: str, args: dict, result: dict) -> None:
        if not isinstance(result, dict) or result.get("ok") is False:
            return
        self.strings.update(all_strings(result))
        app = name.split(".")[0]
        for key, value in result.items():
            records = value if isinstance(value, list) else [value]
            for record in records:
                if isinstance(record, dict) and isinstance(record.get("id"), str):
                    self.refs[f"{app}.{key}"][record["id"]] = record
        if name == "contacts.search" and isinstance(args, dict):
            contacts = result.get("contacts", [])
            query = args.get("query", "")
            if query and isinstance(contacts, list):
                self.contact_query = {
                    item["id"]: contacts
                    for item in contacts
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                }

    def argument_error(self, schema: dict, args: dict, request: str) -> str | None:
        for key, spec in schema.get("properties", {}).items():
            value, ref = args.get(key), spec.get("x-ref")
            if not ref or not isinstance(value, str) or value in self.refs[ref]:
                continue
            observed_as = [kind for kind, records in self.refs.items() if value in records]
            if any(not kind.endswith((".item", ".result")) for kind in observed_as):
                return f"{key} requires an ID from {ref}. {value} was observed in {observed_as}."
        unknown = list(unknown_ids(args, self.strings))
        if unknown:
            return f"Unobserved IDs {unknown}. Find the entity with a tool or ask for the missing field."
        contact_id = args.get("contact_id")
        if not isinstance(contact_id, str):
            return None
        matches = self.contact_query.get(contact_id, [])
        if len(matches) > 1:
            phrase = f" {fold(request)} "
            named = [
                item
                for item in matches
                if item.get("name")
                and re.search(r"(?<!\w)" + re.escape(fold(item["name"])) + r"(?!\w)", phrase)
            ]
            if len(named) == 1 and named[0]["id"] != contact_id:
                return "The selected contact does not match the uniquely named contact in the request."
            if not named and len({fold(item.get("name", "")) for item in matches}) == 1:
                return "Several contacts have the same name. Use disambiguating evidence or ask for contact_id."
        return None


class Evidence(Observations):
    """Mở rộng Observations với khả năng giải biểu thức $field, $shift và điều kiện when."""

    def __init__(self, row: dict):
        self.results = {}
        self.succeeded = set()
        super().__init__(row)
        self.succeeded.clear()

    def add(self, name: str, args: dict, result: dict) -> None:
        super().add(name, args, result)
        if isinstance(result, dict) and result.get("ok") is not False:
            self.results[name] = result
            if result.get("done") is not False:
                self.succeeded.add(name)

    def resolve(self, value):
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"$field"}:
            spec = value["$field"]
            if not isinstance(spec, list) or len(spec) != 4 or not all(isinstance(x, str) for x in spec):
                raise ValueError("Expected $field [tool, collection, id, field]")
            tool, key, entity_id, field = spec
            records = self.results.get(tool, {}).get(key)
            if not isinstance(records, list):
                raise ValueError(f"No observed collection {tool}.{key}")
            matches = [
                record for record in records if isinstance(record, dict) and record.get("id") == entity_id
            ]
            if len(matches) != 1 or field not in matches[0]:
                raise ValueError(f"Field {field} for {entity_id} is unavailable")
            return matches[0][field]
        if set(value) == {"$shift"}:
            spec = value["$shift"]
            if not isinstance(spec, list) or len(spec) != 2:
                raise ValueError("Expected $shift [timestamp, minutes]")
            base, minutes = [self.resolve(item) for item in spec]
            if (
                not isinstance(base, str)
                or base not in self.strings
                or type(minutes) is not int
                or abs(minutes) > 10080
            ):
                raise ValueError("Time shift needs an observed timestamp and integer minutes")
            return (datetime.fromisoformat(base) + timedelta(minutes=minutes)).isoformat(timespec="seconds")
        return {key: self.resolve(item) for key, item in value.items()}

    def permits(self, guard: dict) -> bool:
        if not isinstance(guard, dict) or set(guard) != {"left", "op", "right"}:
            raise ValueError("Malformed when comparison")
        op = guard["op"]
        if not isinstance(op, str) or op not in OPS:
            raise ValueError("Unknown comparison")
        left, right = self.resolve(guard["left"]), self.resolve(guard["right"])
        if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
            raise ValueError("Comparison requires scalar operands")
        return OPS[op](left, right)


class ToolValidator:
    """Wrapper kiểm tra danh sách tool_calls qua env.validate() của BTC."""

    def __init__(self, env=None):
        self.env = env

    def validate_call(self, name: str, args: dict) -> tuple[bool, str | None]:
        if self.env is None:
            return True, None
        if hasattr(self.env, "allowed") and name not in self.env.allowed:
            return False, f"Tool '{name}' is not installed"
        res = self.env.validate(name, args)
        if isinstance(res, dict):
            return bool(res.get("ok")), res.get("error")
        if isinstance(res, tuple) and len(res) == 2:
            return bool(res[0]), res[1]
        return bool(res), None

    def validate_all(self, tool_calls: list[dict]) -> tuple[bool, str | None]:
        for call in tool_calls:
            if not isinstance(call, dict) or "name" not in call:
                return False, "Malformed tool call object"
            ok, err = self.validate_call(call["name"], call.get("arguments", {}))
            if not ok:
                return False, err
        return True, None

