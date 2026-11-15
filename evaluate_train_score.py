"""Bộ đánh giá độ chính xác offline (Offline Scorer) chuẩn 100% công thức toán học của BTC Zalo AI Challenge."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_ROOT

# Nạp zaic_tools của public_test / train
btc_dir = DEFAULT_ROOT / "public_test" if (DEFAULT_ROOT / "public_test").exists() else DEFAULT_ROOT / "train"
sys.path.insert(0, str(btc_dir))
from zaic_tools import ToolEnv, catalog_path
from zaic_tools.utils import FREE_TEXT_KEYS, _tokens, canon, state_hash


def _token_f1(pred_str: str, gold_str: str) -> float:
    """Tính token-F1 giữa 2 chuỗi văn bản tự do (fuzzy argument)."""
    p_tokens = re.findall(r"\w+", unicodedata.normalize("NFC", str(pred_str).lower()))
    g_tokens = re.findall(r"\w+", unicodedata.normalize("NFC", str(gold_str).lower()))
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(set(p_tokens))
    recall = len(common) / len(set(g_tokens))
    return 2 * precision * recall / (precision + recall)


def call_matches(pred_call: dict, gold_call: dict, catalog_params: dict) -> bool:
    """So khớp 1 tool call dự đoán với 1 tool call nhãn theo luật BTC."""
    if pred_call.get("name") != gold_call.get("name"):
        return False
    pred_args = pred_call.get("arguments") or {}
    gold_args = gold_call.get("arguments") or {}

    tool_name = pred_call.get("name")
    param_meta = catalog_params.get(tool_name, {}).get("properties", {})

    all_keys = set(pred_args.keys()) | set(gold_args.keys())
    for k in all_keys:
        p_val = pred_args.get(k)
        g_val = gold_args.get(k)
        if p_val is None and g_val is None:
            continue
        if p_val is None or g_val is None:
            return False

        meta = param_meta.get(k, {})
        is_fuzzy = meta.get("x-match") == "fuzzy" or k in FREE_TEXT_KEYS

        if is_fuzzy:
            if _token_f1(str(p_val), str(g_val)) < 0.6:
                return False
        else:
            # Tham số cứng (non-fuzzy): phải trùng khớp tuyệt đối
            if str(p_val).strip() != str(g_val).strip():
                return False

    return True


def compute_sample_score(
    pred_calls: list[dict],
    gold_calls: list[dict],
    pred_hash: str,
    gold_hash: str,
    catalog_params: dict,
) -> tuple[float, int, int, int, bool]:
    """
    Tính điểm cho 1 mẫu theo công thức chính thức của BTC:
    Score = 0 nếu pred_hash != gold_hash hoặc M == 0
    Score = 2*P*R / (P+R) với P = M/(M+U), R = M/|G|
    """
    # Trường hợp out-of-scope: nhãn rỗng
    if not gold_calls:
        is_correct = len(pred_calls) == 0
        return (1.0 if is_correct else 0.0), 0, 0, len(pred_calls), pred_hash == gold_hash

    # Kiểm tra điều kiện tiên quyết: Hash phải khớp tuyệt đối
    hash_matched = pred_hash == gold_hash
    if not hash_matched:
        return 0.0, 0, len(gold_calls), len(pred_calls), False

    # So khớp tool calls
    matched_gold_indices = set()
    M = 0
    for p_call in pred_calls:
        for g_idx, g_call in enumerate(gold_calls):
            if g_idx not in matched_gold_indices and call_matches(p_call, g_call, catalog_params):
                matched_gold_indices.add(g_idx)
                M += 1
                break

    if M == 0:
        return 0.0, 0, len(gold_calls), len(pred_calls), hash_matched

    U = len(pred_calls) - M
    G = len(gold_calls)

    P = M / (M + U)
    R = M / G
    f1 = 2 * P * R / (P + R)
    return f1, M, G, U, hash_matched


def run_benchmark(num_samples: int = 50):
    """Chạy chấm điểm thử nghiệm trên tập train (có nhãn ground truth)."""
    print("=" * 65)
    print(f"  ĐÁNH GIÁ ĐIỂM SỐ OFFLINE CHUẨN BTC TRÊN {num_samples} MẪU TRAIN")
    print("=" * 65)

    train_file = DEFAULT_ROOT / "train" / "train.jsonl"
    if not train_file.exists():
        print(f"❌ Không tìm thấy file {train_file}")
        return

    cat_file = DEFAULT_ROOT / "train" / "train_tools_catalog.json"
    catalog_params = {}
    if cat_file.exists():
        cat_data = json.load(open(cat_file, encoding="utf-8"))
        for app in cat_data.get("apps", []):
            for t in app.get("tools", []):
                catalog_params[t["name"]] = t.get("parameters", {})

    from main import process_sample
    from src.mcp_agent.mcp_server import MCPServer

    server = MCPServer(DEFAULT_ROOT, "train")

    total_score = 0.0
    matched_hashes = 0
    total_M = 0
    total_G = 0
    total_U = 0

    with open(train_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break
            sample = json.loads(line)
            sid = sample["id"]
            gold = sample.get("label", {})
            gold_calls = gold.get("tool_calls", [])
            gold_hash = state_hash(canon(gold.get("final_state", {})))

            # Chạy qua Pipeline
            result = process_sample(sample, server, local_only=True)
            pred_calls = result.get("tool_calls", [])
            pred_hash = result.get("state_hash", "")

            score, m, g, u, h_match = compute_sample_score(
                pred_calls, gold_calls, pred_hash, gold_hash, catalog_params
            )
            total_score += score
            if h_match:
                matched_hashes += 1
            total_M += m
            total_G += g
            total_U += u

            status = "✅" if score > 0 else "❌"
            print(f"[{idx+1:02d}/{num_samples}] {sid}: Score={score*100:.1f} | Hash={'OK' if h_match else 'FAIL'} | Calls: Pred={len(pred_calls)} Gold={len(gold_calls)} {status}")

    final_score = (total_score / num_samples) * 100.0
    hash_rate = (matched_hashes / num_samples) * 100.0
    print("-" * 65)
    print(f"📊 TỶ LỆ KHỚP HASH: {hash_rate:.2f}% ({matched_hashes}/{num_samples})")
    print(f"🏆 ĐIỂM SỐ CUỐI CÙNG (FINAL SCORE): {final_score:.2f} / 100")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30, help="Số lượng mẫu train kiểm thử")
    args = parser.parse_args()
    run_benchmark(args.samples)
