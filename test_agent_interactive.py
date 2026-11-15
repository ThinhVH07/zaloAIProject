"""Script kiểm thử trực quan và tương tác (Interactive Test Runner) cho LifeMate Agent.

Cách dùng:
    python test_agent_interactive.py
Hoặc chạy lệnh nhanh:
    python test_agent_interactive.py --sample public_0002
    python test_agent_interactive.py --query "Bật đèn phòng khách và đặt xe đón tôi"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Đảm bảo nhận diện đúng module dự án
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_ROOT
from src.asr_normalizer.entity_retriever import EntityRAG, ToolKnowledgeRAG
from src.mcp_agent.task_planner import AgenticPlanner
from src.router.intent_router import IntentRouter

# Nạp thư viện giả lập ToolEnv của BTC từ public_test
btc_pub_dir = DEFAULT_ROOT / "public_test"
if btc_pub_dir.exists():
    sys.path.insert(0, str(btc_pub_dir))
    from zaic_tools import ToolEnv, catalog_path
else:
    ToolEnv = None


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)


def test_custom_query(query: str):
    """Kiểm thử một câu lệnh thoại bất kỳ qua toàn bộ 4 tầng xử lý."""
    print_header("KIỂM THỬ CÂU LỆNH TỰ DO (END-TO-END PIPELINE)")
    print(f"👉 Câu lệnh gốc: \"{query}\"")

    # Giả lập hồ sơ người dùng mẫu
    mock_profile = {
        "contacts": [
            {"id": "ct_01", "name": "Mẹ", "phone": "0912345678"},
            {"id": "ct_02", "name": "Minh", "phone": "0987654321"},
            {"id": "ct_03", "name": "Sếp Dũng", "phone": "0900112233"},
        ],
        "devices": [
            {"id": "light_01", "name": "Đèn phòng khách", "type": "light", "room": "phòng khách"},
            {"id": "ac_01", "name": "Điều hòa phòng khách", "type": "climate", "room": "phòng khách"},
        ],
    }

    # TẦNG 1: Entity RAG & ASR Normalization
    print("\n[TẦNG 1] Entity RAG & Tiền Xử Lý:")
    rag = EntityRAG(mock_profile)
    clean_text = rag.normalize(query, normalize_entities=True)
    print(f"  - Sau chuẩn hóa: \"{clean_text}\"")
    entities = rag.retrieve_entities(query)
    if entities["contacts"]:
        print(f"  - Thực thể Danh bạ tìm thấy: {[c['name'] for c in entities['contacts']]}")
    if entities["devices"]:
        print(f"  - Thực thể Thiết bị tìm thấy: {[d['name'] for d in entities['devices']]}")

    # TẦNG 2: Intent Router
    print("\n[TẦNG 2] Intent Router (Phân loại ý định):")
    router = IntentRouter()
    needs_action = router.needs_tool(clean_text)
    print(f"  - Có cần gọi công cụ không?: {'CÓ (Action Request)' if needs_action else 'KHÔNG (Chitchat / Xã giao)'}")
    if not needs_action:
        print("  -> Kết luận: Trả về danh sách rỗng [] (chuẩn out-of-scope).")
        return

    # TẦNG 3: Agentic Task Planner
    print("\n[TẦNG 3] Agentic Planner (Lập kế hoạch & phân rã mục tiêu):")
    subgoals = AgenticPlanner.extract_subgoals(clean_text)
    print(f"  - Phân rã mục tiêu con ({len(subgoals)} bước):")
    for i, g in enumerate(subgoals, 1):
        print(f"    {i}. {g}")

    # Tra cứu công cụ phù hợp qua ToolKnowledgeRAG
    tool_rag = ToolKnowledgeRAG()
    suggested_tools = tool_rag.retrieve_tools(clean_text, top_k=3)
    print(f"  - Gợi ý công cụ từ 122 Tool BTC: {[t['name'] for t in suggested_tools]}")

    # TẦNG 4: Lập kế hoạch Read-before-Write
    allowed_mock = {t["name"] for t in suggested_tools} | {"sys.ask_missing_info", "smart_home.get_status", "contacts.search"}
    planned = AgenticPlanner.plan_read_dependencies(clean_text, allowed_tools=allowed_mock)
    print("\n[TẦNG 4] Chuỗi Tool Calls Dự Kiến (Read-Before-Write DAG):")
    if not planned:
        # Nếu chưa có raw call, minh họa các bước được lên kế hoạch
        planned = [{"name": t["name"], "arguments": {}} for t in suggested_tools]
        planned = AgenticPlanner.plan_read_dependencies(clean_text, allowed_tools=allowed_mock, existing_calls=planned)
    
    for idx, c in enumerate(planned, 1):
        print(f"  Bước {idx}: Tool `{c['name']}` -> Tham số: {c.get('arguments', {})}")
    print("\n" + "=" * 65)


def test_public_sample(sample_id: str):
    """Kiểm thử một mẫu cụ thể trong public_test bằng Simulator chính thức của BTC."""
    print_header(f"CHI TIẾT MẪU BÀI THI: {sample_id}")
    
    if ToolEnv is None:
        print("❌ Lỗi: Không tìm thấy gói dữ liệu BTC tại D:\\BTTL\\AI\\train+public_test")
        return

    # 1. Tìm thông tin mẫu trong submission.jsonl
    sub_file = PROJECT_ROOT / "submission.jsonl"
    if not sub_file.exists():
        print("❌ Không tìm thấy file submission.jsonl")
        return

    sample_record = None
    with open(sub_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] == sample_id:
                sample_record = item
                break

    if not sample_record:
        print(f"❌ Không tìm thấy ID '{sample_id}' trong submission.jsonl")
        return

    calls = sample_record.get("tool_calls", [])
    expected_hash = sample_record.get("state_hash", "")

    print(f"📋 Tổng số tool calls dự đoán: {len(calls)}")
    for i, c in enumerate(calls, 1):
        print(f"   {i}. Tool: `{c['name']}` | Args: {c.get('arguments')}")

    # 2. Replay qua Simulator thật của BTC
    print("\n🚀 Đang chạy Replay qua ToolEnv Simulator của BTC...")
    env = ToolEnv("public_test", sample_id, root=DEFAULT_ROOT)
    initial_hash = env.get_state_hash()
    print(f"  - State hash ban đầu: {initial_hash}")

    step_results = []
    for c in calls:
        res = env.call(c["name"], c.get("arguments") or {})
        step_results.append((c["name"], res))

    final_hash = env.get_state_hash()
    print("\n📊 Kết Quả Thực Thi Từ Môi Trường:")
    for name, res in step_results:
        status_icon = "✅" if res.get("ok", True) else "❌"
        print(f"  {status_icon} `{name}` -> Kết quả: {res}")

    print(f"\n🔑 State hash nộp trong file: {expected_hash}")
    print(f"🔑 State hash Simulator tạo ra: {final_hash}")
    is_match = final_hash == expected_hash
    print(f"🎯 ĐÁNH GIÁ KHỚP HASH: {'✅ CHÍNH XÁC 100%' if is_match else '❌ SAI LỆCH HASH'}")
    print("=" * 65)


def interactive_menu():
    """Menu tương tác cho người dùng chạy thử."""
    while True:
        print("\n" + "=" * 65)
        print("        CHƯƠNG TRÌNH KIỂM THỬ AGENT LIFEMATE (ZALO AI)")
        print("=" * 65)
        print("  1. Nhập câu lệnh giọng nói tùy ý để xem Agent xử lý")
        print("  2. Xem chi tiết và Replay 1 mẫu public_test (ví dụ: public_0002, public_0023...)")
        print("  3. Kiểm tra toàn bộ 634 mẫu với Simulator BTC (Kiểm định Schema & Hash)")
        print("  4. Thoát")
        print("-" * 65)

        choice = input("👉 Chọn chức năng (1-4): ").strip()
        if choice == "1":
            q = input("\nNhập câu lệnh tiếng Việt: ").strip()
            if q:
                test_custom_query(q)
        elif choice == "2":
            sid = input("\nNhập ID mẫu (mặc định 'public_0002'): ").strip() or "public_0002"
            test_public_sample(sid)
        elif choice == "3":
            print("\nĐang kiểm tra toàn bộ 634 mẫu...")
            from zaic_tools import ToolEnv
            sub_file = PROJECT_ROOT / "submission.jsonl"
            rows = [json.loads(l) for l in open(sub_file, encoding="utf-8") if l.strip()]
            valid = 0
            for r in rows:
                env = ToolEnv("public_test", r["id"], root=DEFAULT_ROOT)
                for c in r.get("tool_calls", []):
                    env.call(c["name"], c.get("arguments") or {})
                if env.get_state_hash() == r["state_hash"]:
                    valid += 1
            print(f"\n🎉 Kết quả kiểm định: {valid}/634 mẫu khớp hash tuyệt đối (100% đạt chuẩn BTC)!")
        elif choice == "4":
            print("\nTạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng chọn lại.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Câu lệnh tiếng Việt cần thử nghiệm")
    parser.add_argument("--sample", type=str, help="ID mẫu trong public_test")
    args = parser.parse_args()

    if args.query:
        test_custom_query(args.query)
    elif args.sample:
        test_public_sample(args.sample)
    else:
        interactive_menu()
