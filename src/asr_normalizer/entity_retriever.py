"""Fuzzy match / Entity RAG với danh bạ, thiết bị smart home và truy xuất ví dụ mẫu (BM25 / Phrase RAG)."""
from __future__ import annotations

import difflib
import math
from collections import Counter
from pathlib import Path

from config.settings import FUZZY_MATCH_THRESHOLD
from src.asr_normalizer.text_cleaner import clean_asr_text, fold, words
from src.utils.jsonl_writer import read_jsonl

try:
    from rapidfuzz import process as _rapidfuzz_process
except ImportError:  # Fallback an toàn nếu môi trường chưa cài rapidfuzz
    _rapidfuzz_process = None


def _extract_one(query: str, choices: list[str]) -> tuple[str, float, int]:
    """So khớp mờ (fuzzy match) dùng rapidfuzz nếu có, hoặc fallback chuẩn bằng difflib + token overlap."""
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

    def normalize_contact(self, raw_query: str) -> str:
        """Ví dụ nghe nhầm 'Minh đồng nghiệp thúng' -> map về 'Minh'."""
        choices = [c["name"] for c in self.contacts if isinstance(c, dict) and c.get("name")]
        if not choices or not raw_query:
            return raw_query
        match, score, _ = _extract_one(raw_query, choices)
        return match if score > self.threshold else raw_query

    def normalize_device(self, raw_query: str) -> str:
        """Map tên thiết bị nghe nhầm về tên chuẩn trong smart_home."""
        choices = [d["name"] for d in self.smart_devices if isinstance(d, dict) and d.get("name")]
        if not choices or not raw_query:
            return raw_query
        match, score, _ = _extract_one(raw_query, choices)
        return match if score > self.threshold else raw_query

    def normalize(self, raw_audio_text: str) -> str:
        """Làm sạch văn bản ASR thô trước khi chuyển sang tầng Router & MCP Agent."""
        return clean_asr_text(raw_audio_text)


class Examples:
    """Retrieve demonstrations from development records (BM25), never evaluation labels."""

    def __init__(self, path: str | Path):
        self.rows, errors = read_jsonl(path)
        if errors or not self.rows:
            raise ValueError(f"Invalid development examples: {errors}")
        self.counts = [Counter(words(row["audio_text"])) for row in self.rows]
        lengths = [sum(counts.values()) for counts in self.counts]
        average = sum(lengths) / len(lengths)
        self.denominators = [1.2 * (0.25 + 0.75 * length / average) for length in lengths]
        frequencies = Counter(word for counts in self.counts for word in counts)
        self.idf = {
            word: math.log(1 + (len(self.rows) - n + 0.5) / (n + 0.5))
            for word, n in frequencies.items()
        }
        self.tool_sets = [
            {call["name"] for call in row["label"]["tool_calls"]}
            for row in self.rows
        ]

    def find(self, row: dict, k: int = 2) -> list[dict]:
        query = set(words(row["transcript"]))
        allowed = {
            name
            for app in row["context"]["installed_apps"].values()
            for name in app["tools"]
        }
        allowed.add("sys.ask_missing_info")
        scores = []
        for i, counts in enumerate(self.counts):
            if self.rows[i]["id"] == row["id"] or not self.tool_sets[i] <= allowed:
                continue
            score = sum(
                self.idf[word] * counts[word] * 2.2 / (counts[word] + self.denominators[i])
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

    def __init__(self, path: str | Path):
        super().__init__(path)
        self.phrases = [set(_phrase_features(row["audio_text"])) for row in self.rows]
        frequencies = Counter(feature for example in self.phrases for feature in example)
        self.weights = {
            feature: math.log1p(len(self.rows) / frequency)
            for feature, frequency in frequencies.items()
        }

    def find(self, row: dict, k: int = 2) -> list[dict]:
        query = set(_phrase_features(row["transcript"]))
        installed = {
            tool
            for app in row["context"]["installed_apps"].values()
            for tool in app["tools"]
        }
        installed.add("sys.ask_missing_info")
        candidates = []
        for index, observed in enumerate(self.phrases):
            if self.rows[index]["id"] == row["id"] or not self.tool_sets[index] <= installed:
                continue
            score = sum(
                self.weights[feature] * (1.3 if "+" in feature else 1)
                for feature in sorted(query & observed)
            ) / math.sqrt(len(observed) or 1)
            candidates.append((score, index))
        return [self.rows[index] for _, index in sorted(candidates, reverse=True)[:k]]

