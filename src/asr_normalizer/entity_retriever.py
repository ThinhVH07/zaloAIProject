"""Fuzzy match / Entity RAG với danh bạ, thiết bị smart home và truy xuất ví dụ mẫu (BM25 / Phrase RAG)."""
from __future__ import annotations

import difflib
import json
import math
import re
from collections import Counter
from pathlib import Path

from config.settings import FUZZY_MATCH_THRESHOLD, TOOL_SCHEMAS_PATH
from src.asr_normalizer.text_cleaner import clean_asr_text, fold, words
from src.utils.jsonl_writer import read_jsonl

try:
    from rapidfuzz import process as _rapidfuzz_process
except ImportError:  # Fallback an toàn nếu môi trường chưa cài rapidfuzz
    _rapidfuzz_process = None


def _extract_one(query: str, choices: list[str]) -> tuple[str, float, int]:
    """So khớp mờ (fuzzy match) dùng rapidfuzz nếu có, hoặc fallback chuẩn bằng difflib + token overlap."""
    if not choices:
        return query, 0.0, 0
    if _rapidfuzz_process is not None:
        match, score, idx = _rapidfuzz_process.extractOne(query, choices)
        return match, float(score), int(idx)

    folded_query = fold(query)
    query_tokens = set(folded_query.split())
    best_choice, best_score, best_idx = choices[0], 0.0, 0
    for idx, choice in enumerate(choices):
        folded_choice = fold(choice)
        if not folded_choice:
            continue
        seq_score = difflib.SequenceMatcher(None, folded_query, folded_choice).ratio() * 100.0
        choice_tokens = set(folded_choice.split())
        if choice_tokens and choice_tokens <= query_tokens:
            token_score = 90.0
        elif choice_tokens & query_tokens:
            token_score = (len(choice_tokens & query_tokens) / len(choice_tokens)) * 85.0
        else:
            token_score = 0.0
        score = max(seq_score, token_score)
        if score > best_score:
            best_choice, best_score, best_idx = choice, score, idx
    return best_choice, best_score, best_idx


class EntityRAG:
    """Tầng 1: Chuẩn hóa thực thể danh bạ / thiết bị nhà thông minh từ user_profile."""

    def __init__(self, user_profile: dict | None = None, threshold: float = FUZZY_MATCH_THRESHOLD):
        profile = user_profile or {}
        contacts_raw = profile.get("contacts", [])
        if isinstance(contacts_raw, dict):
            contacts_raw = contacts_raw.get("contacts", [])
        devices_raw = profile.get("devices", [])
        if not devices_raw and isinstance(profile.get("smart_home"), dict):
            devices_raw = profile["smart_home"].get("devices", [])
        self.contacts = contacts_raw if isinstance(contacts_raw, list) else []
        self.smart_devices = devices_raw if isinstance(devices_raw, list) else []
        self.threshold = threshold

        # Lưu ánh xạ name -> entity object để truy xuất nhanh
        self.contact_names = [c["name"] for c in self.contacts if isinstance(c, dict) and c.get("name")]
        self.device_names = [d["name"] for d in self.smart_devices if isinstance(d, dict) and d.get("name")]

    def normalize_contact(self, raw_query: str) -> str:
        """Ví dụ nghe nhầm 'Minh đồng nghiệp thúng' -> map về 'Minh'."""
        if not self.contact_names or not raw_query:
            return raw_query
        match, score, _ = _extract_one(raw_query, self.contact_names)
        return match if score > self.threshold else raw_query

    def normalize_device(self, raw_query: str) -> str:
        """Map tên thiết bị nghe nhầm về tên chuẩn trong smart_home."""
        if not self.device_names or not raw_query:
            return raw_query
        match, score, _ = _extract_one(raw_query, self.device_names)
        return match if score > self.threshold else raw_query

    def get_grounded_contact(self, raw_query: str) -> dict | None:
        """Truy xuất toàn bộ thông tin contact (id, name, phone...) nếu khớp mờ."""
        if not self.contact_names or not raw_query:
            return None
        match, score, idx = _extract_one(raw_query, self.contact_names)
        if score > self.threshold and idx < len(self.contacts):
            return self.contacts[idx]
        return None

    def get_grounded_device(self, raw_query: str) -> dict | None:
        """Truy xuất toàn bộ thông tin thiết bị (id, name, type, room...) nếu khớp mờ."""
        if not self.device_names or not raw_query:
            return None
        match, score, idx = _extract_one(raw_query, self.device_names)
        if score > self.threshold and idx < len(self.smart_devices):
            return self.smart_devices[idx]
        return None

    def retrieve_entities(self, text: str) -> dict:
        """
        Quét câu văn bản và truy xuất tất cả các entity (contacts, devices) có liên quan.
        Trả về danh sách thực thể kèm độ khớp (score) để grounding vào tham số tool.
        """
        matched_contacts = []
        for c in self.contacts:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            name = c["name"]
            folded_name = fold(name)
            # Kiểm tra xem tên có xuất hiện hoặc gần giống trong text không
            if folded_name in fold(text):
                matched_contacts.append({**c, "score": 100.0, "match_type": "exact"})
            else:
                _, score, _ = _extract_one(name, [text])
                if score >= self.threshold:
                    matched_contacts.append({**c, "score": score, "match_type": "fuzzy"})

        matched_devices = []
        for d in self.smart_devices:
            if not isinstance(d, dict) or not d.get("name"):
                continue
            name = d["name"]
            folded_name = fold(name)
            if folded_name in fold(text):
                matched_devices.append({**d, "score": 100.0, "match_type": "exact"})
            else:
                _, score, _ = _extract_one(name, [text])
                if score >= self.threshold:
                    matched_devices.append({**d, "score": score, "match_type": "fuzzy"})

        return {
            "contacts": sorted(matched_contacts, key=lambda x: x["score"], reverse=True),
            "devices": sorted(matched_devices, key=lambda x: x["score"], reverse=True),
        }

    def normalize_entities_in_text(self, text: str) -> str:
        """
        Tự động tìm kiếm các thực thể (contacts, devices) trong câu văn bản và thay thế
        bằng tên chuẩn chính xác (canonical name) từ profile người dùng.
        """
        if not text:
            return text
        all_choices = sorted(self.contact_names + self.device_names, key=lambda x: len(x.split()), reverse=True)
        res = text
        for choice in all_choices:
            folded_choice = fold(choice)
            if not folded_choice:
                continue
            c_words = choice.split()
            n_words = len(c_words)

            word_spans = [m.span() for m in re.finditer(r"\S+", res)]
            if len(word_spans) < n_words:
                continue

            i = 0
            while i <= len(word_spans) - n_words:
                start_idx = word_spans[i][0]
                end_idx = word_spans[i + n_words - 1][1]
                sub_text = res[start_idx:end_idx]
                if fold(sub_text) == folded_choice:
                    # Tránh nhầm đại từ nhân xưng
                    if n_words == 1:
                        raw_lower = sub_text.lower()
                        choice_lower = choice.lower()
                        if raw_lower != choice_lower and raw_lower in {"mình", "tôi", "ta", "bạn"}:
                            i += 1
                            continue
                        if fold(choice) == "minh" and i > 0:
                            prev_word = res[word_spans[i - 1][0]:word_spans[i - 1][1]]
                            if fold(prev_word) in {"giup", "ho", "dum", "voi"}:
                                i += 1
                                continue
                    res = res[:start_idx] + choice + res[end_idx:]
                    word_spans = [m.span() for m in re.finditer(r"\S+", res)]
                    i += n_words
                else:
                    i += 1
        return res

    def normalize(self, raw_audio_text: str, normalize_entities: bool = False) -> str:
        """Làm sạch văn bản ASR thô và tùy chọn chuẩn hóa thực thể trong câu."""
        cleaned = clean_asr_text(raw_audio_text)
        if normalize_entities:
            return self.normalize_entities_in_text(cleaned)
        return cleaned


class Examples:
    """Retrieve demonstrations from development records (BM25), never evaluation labels."""

    def __init__(self, source: str | Path | list[dict] | None = None):
        self.rows: list[dict] = []
        if source is not None:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if path.exists():
                    if path.suffix == ".jsonl":
                        rows, errors = read_jsonl(path)
                        if not errors and rows:
                            self.rows = rows
                    elif path.suffix == ".json":
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                self.rows = data if isinstance(data, list) else []
                        except Exception:
                            self.rows = []
            elif isinstance(source, list):
                self.rows = source

        if not self.rows:
            self.counts = []
            self.denominators = []
            self.idf = {}
            self.tool_sets = []
            return

        self.counts = [
            Counter(words(row.get("audio_text") or row.get("transcript", "")))
            for row in self.rows
        ]
        lengths = [sum(counts.values()) for counts in self.counts]
        average = sum(lengths) / len(lengths) if lengths and sum(lengths) > 0 else 1.0
        self.denominators = [1.2 * (0.25 + 0.75 * length / (average or 1.0)) for length in lengths]
        frequencies = Counter(word for counts in self.counts for word in counts)
        self.idf = {
            word: math.log(1 + (len(self.rows) - n + 0.5) / (n + 0.5))
            for word, n in frequencies.items()
        }
        self.tool_sets = []
        for row in self.rows:
            label = row.get("label") or {}
            calls = label.get("tool_calls") or row.get("tool_calls", [])
            self.tool_sets.append({
                call.get("name")
                for call in calls
                if isinstance(call, dict) and "name" in call
            })

    def find(self, row: dict, k: int = 2) -> list[dict]:
        if not self.rows:
            return []
        query_text = row.get("transcript") or row.get("audio_text", "")
        query = set(words(query_text))
        installed_apps = (row.get("context") or {}).get("installed_apps", {})
        if isinstance(installed_apps, dict) and installed_apps:
            allowed = {
                name
                for app in installed_apps.values()
                if isinstance(app, dict)
                for name in app.get("tools", [])
            }
        else:
            allowed = None
        if allowed is not None:
            allowed.add("sys.ask_missing_info")

        scores = []
        for i, counts in enumerate(self.counts):
            if self.rows[i].get("id") == row.get("id"):
                continue
            if allowed is not None and not self.tool_sets[i] <= allowed:
                continue
            score = sum(
                self.idf.get(word, 0.0) * counts[word] * 2.2 / (counts[word] + self.denominators[i])
                for word in query
                if counts[word]
            )
            scores.append((score, i))
        return [self.rows[i] for _, i in sorted(scores, reverse=True)[:k]]


def _phrase_features(text: str) -> list[str]:
    tokens = words(text)
    return tokens + ["+".join(pair) for pair in zip(tokens, tokens[1:])]


class PhraseExamples(Examples):
    """Truy xuất ví dụ mẫu kết hợp unigram + bigram (phrase features)."""

    def __init__(self, source: str | Path | list[dict] | None = None):
        super().__init__(source)
        if not self.rows:
            self.phrases = []
            self.weights = {}
            return
        self.phrases = [
            set(_phrase_features(row.get("audio_text") or row.get("transcript", "")))
            for row in self.rows
        ]
        frequencies = Counter(feature for example in self.phrases for feature in example)
        self.weights = {
            feature: math.log1p(len(self.rows) / frequency)
            for feature, frequency in frequencies.items()
        }

    def find(self, row: dict, k: int = 2) -> list[dict]:
        if not self.rows or not self.phrases:
            return []
        query_text = row.get("transcript") or row.get("audio_text", "")
        query = set(_phrase_features(query_text))
        installed_apps = (row.get("context") or {}).get("installed_apps", {})
        if isinstance(installed_apps, dict) and installed_apps:
            installed = {
                tool
                for app in installed_apps.values()
                if isinstance(app, dict)
                for tool in app.get("tools", [])
            }
        else:
            installed = None
        if installed is not None:
            installed.add("sys.ask_missing_info")

        candidates = []
        for index, observed in enumerate(self.phrases):
            if self.rows[index].get("id") == row.get("id"):
                continue
            if installed is not None and not self.tool_sets[index] <= installed:
                continue
            score = sum(
                self.weights.get(feature, 0.0) * (1.3 if "+" in feature else 1.0)
                for feature in sorted(query & observed)
            ) / math.sqrt(len(observed) or 1)
            candidates.append((score, index))
        return [self.rows[index] for _, index in sorted(candidates, reverse=True)[:k]]


class ToolKnowledgeRAG:
    """
    RAG tra cứu đặc tả công cụ (Tool Schemas & Requirements) từ danh mục chính thức của BTC.
    Giúp Router, Agent và Guardrail tra cứu nhanh thông tin schema, mô tả và tham số bắt buộc.
    """

    def __init__(self, catalog_path: str | Path | None = TOOL_SCHEMAS_PATH):
        self.tools: dict[str, dict] = {}
        self.tool_apps: dict[str, dict] = {}
        self.tool_tokens: dict[str, Counter] = {}
        self._load_catalog(catalog_path)

    def _load_catalog(self, path: str | Path | None) -> None:
        if path is None:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for app in data.get("apps", []):
                for tool in app.get("tools", []):
                    name = tool.get("name")
                    if not name:
                        continue
                    self.tools[name] = tool
                    self.tool_apps[name] = {
                        "app_id": app.get("id", ""),
                        "app_name_vi": app.get("name_vi", ""),
                        "app_description": app.get("description", ""),
                    }
                    doc_text = f"{name} {app.get('name_vi', '')} {app.get('description', '')} {tool.get('description', '')}"
                    self.tool_tokens[name] = Counter(words(doc_text))
        except Exception:
            pass

    def retrieve_tools(
        self, query: str, allowed_tools: set[str] | None = None, top_k: int = 5
    ) -> list[dict]:
        """
        Truy xuất các công cụ liên quan nhất đến câu thoại của người dùng bằng BM25 / Token Overlap.
        Đảm bảo không vượt quá phạm vi `allowed_tools` nếu được cung cấp.
        """
        q_tokens = words(query)
        if not q_tokens or not self.tools:
            return []
        scores = []
        for name, tool in self.tools.items():
            if allowed_tools is not None and name not in allowed_tools:
                continue
            t_counts = self.tool_tokens.get(name, Counter())
            score = sum(t_counts[w] for w in q_tokens if w in t_counts)
            if score > 0:
                scores.append((score, tool))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scores[:top_k]]

    def get_tool_schema(self, tool_name: str) -> dict | None:
        """Lấy schema chi tiết của một tool."""
        return self.tools.get(tool_name)

    def get_required_params(self, tool_name: str) -> list[str]:
        """Lấy danh sách các tham số bắt buộc của một tool."""
        tool = self.tools.get(tool_name)
        if not tool:
            return []
        params = tool.get("parameters", {})
        return params.get("required", [])

    def format_tools_context(self, tools: list[dict]) -> str:
        """Định dạng danh sách tool thành ngữ cảnh gọn gàng để tiêm vào Prompt."""
        lines = []
        for t in tools:
            name = t.get("name", "")
            desc = t.get("description", "")
            req = t.get("parameters", {}).get("required", [])
            lines.append(f"- `{name}`: {desc} (Tham số bắt buộc: {req or 'Không có'})")
        return "\n".join(lines)


class DynamicFewShotRAG:
    """
    Quản lý và truy xuất ví dụ mẫu (Few-Shot Demonstrations) động cho MCP Agent.
    Hỗ trợ nạp kịch bản mẫu từ danh sách hoặc file dữ liệu BTC.
    """

    def __init__(self, examples: list[dict] | None = None):
        self.engine = PhraseExamples(examples or [])

    def add_examples(self, samples: list[dict]) -> None:
        """Nạp thêm các kịch bản mẫu chuẩn vào bộ nhớ RAG."""
        current_rows = self.engine.rows + samples
        self.engine = PhraseExamples(current_rows)

    def retrieve_guidance(self, row: dict, k: int = 2) -> list[dict]:
        """Truy xuất các ví dụ mẫu tương đồng nhất với câu thoại hiện tại."""
        return self.engine.find(row, k=k)

    def format_for_prompt(self, demonstrations: list[dict]) -> str:
        """Định dạng các kịch bản mẫu thành hướng dẫn Prompt chuẩn cho LLM."""
        if not demonstrations:
            return ""
        guide = []
        for demo in demonstrations:
            label = demo.get("label") or {}
            calls = label.get("tool_calls") or demo.get("tool_calls", [])
            guide.append({
                "request": demo.get("audio_text") or demo.get("transcript", ""),
                "tool_calls_demonstration_only": calls,
            })
        return json.dumps(guide, ensure_ascii=False, indent=2)
