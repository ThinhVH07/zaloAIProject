"""File chạy chính cho toàn bộ dataset test theo kiến trúc Multi-stage Agentic Pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.settings import (
    ASR_HANDOFF_DIR,
    DEFAULT_GGUF_MODEL,
    DEFAULT_ROOT,
    DEFAULT_WORKERS,
    DEV_REFERENCES_PATH,
    FAILED_STATUSES,
    LLM_HEALTH_URL,
    PROJECT_ROOT,
    RUNTIME_DIR,
)
from src.asr_normalizer.entity_retriever import EntityRAG, Examples, PhraseExamples
from src.guardrails.self_correction import OutputGuardrail
from src.mcp_agent.agent_executor import Agent, DirectAgent, MCPAgent, PlannedAgent
from src.mcp_agent.mcp_server import MCPServer, load_btc_runtime
from src.router.intent_router import IntentRouter
from src.utils.jsonl_writer import compact, dump, export_submission, read_jsonl, write_jsonl


def process_sample(sample: dict, env_simulator: MCPServer, local_only: bool = False) -> dict:
    """
    Luồng xử lý chính 4 tầng cho một mẫu dữ liệu:
    - Step 1: Context RAG Normalization (`EntityRAG`)
    - Step 2: Y/N Router Gate (`IntentRouter`)
    - Step 3: MCP Agent Execution (`MCPAgent`)
    - Step 4: Guardrail & Validation Loop (`OutputGuardrail` + `env_simulator.validate`)
    """
    sample_id = sample["id"]
    state_hash = sample.get("state_hash", sample_id)
    raw_audio_text = sample.get("text") or sample.get("transcript", "")

    # Step 1: Context RAG Normalization
    profile = env_simulator.load_state(state_hash)
    rag = EntityRAG(profile)
    clean_text = rag.normalize(raw_audio_text)

    env = None
    try:
        env = env_simulator.reset_env(sample_id)
    except Exception:
        env = None

    # Step 2: Y/N Router Gate
    router = IntentRouter()
    if not router.needs_tool(clean_text):
        final_hash = env.get_state_hash() if env is not None else state_hash
        return {"id": sample_id, "tool_calls": [], "state_hash": final_hash}

    # Step 3: MCP Agent Execution
    agent = MCPAgent(profile, local_only=local_only)
    predicted_tools = agent.predict_tools(clean_text, sample=sample, env=env)

    # Step 4: Guardrail & Validation Loop
    predicted_tools = OutputGuardrail.fix_iso_time(predicted_tools)
    predicted_tools = OutputGuardrail.filter_self_cancellation(clean_text, predicted_tools)
    allowed = env.allowed if env is not None else None
    state = env.get_state() if env is not None else None
    predicted_tools = OutputGuardrail.inject_dependency_tools(
        predicted_tools, allowed_tools=allowed, state=state, transcript=clean_text
    )
    predicted_tools = OutputGuardrail.sanitize_tool_calls(predicted_tools, allowed_tools=allowed)
    predicted_tools = OutputGuardrail.remove_redundant_missing_info(predicted_tools)

    # Kiểm tra ngầm qua Simulator của BTC
    is_valid, err = env_simulator.validate(sample_id, predicted_tools)
    if not is_valid:
        # Self-Correction: Gửi lỗi lại cho Agent sửa
        predicted_tools = agent.self_correct(predicted_tools, err)
        predicted_tools = OutputGuardrail.remove_redundant_missing_info(predicted_tools)

    final_state_hash = env.get_state_hash() if env is not None else state_hash
    return {"id": sample_id, "tool_calls": predicted_tools, "state_hash": final_state_hash}


def run_local_pipeline(args: argparse.Namespace) -> None:
    """Chạy luồng 4 tầng (Tầng 1 -> Tầng 4) trên toàn bộ tập dữ liệu ở chế độ local và xuất JSONL."""
    split = "public_test" if args.stage == "public" else "train"
    source = ASR_HANDOFF_DIR / (
        "public_test_inputs_asr.jsonl" if split == "public_test" else "validation_inputs_asr.jsonl"
    )
    if args.inputs:
        source = args.inputs.resolve()
    rows, errors = read_jsonl(source)
    if errors:
        raise ValueError(errors)

    env_simulator = MCPServer(DEFAULT_ROOT, split)
    output_path = args.output or (PROJECT_ROOT / "submission_test_local.jsonl")
    results: list[dict] = []
    started = time.perf_counter()

    for idx, sample in enumerate(rows, start=1):
        record = process_sample(sample, env_simulator, local_only=True)
        results.append(record)
        if idx % 100 == 0 or idx == len(rows):
            elapsed = time.perf_counter() - started
            print(
                f"[{idx}/{len(rows)}] Đã xử lý mẫu {sample['id']} "
                f"(số lệnh: {len(record['tool_calls'])}, thời gian: {elapsed:.2f}s)",
                flush=True,
            )

    write_jsonl(output_path, results)
    print(f"Hoàn tất! Đã ghi {len(results)} mẫu vào: {output_path.resolve()}", flush=True)


def _wait_for_server_health(server: subprocess.Popen) -> None:
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(75):
        if server.poll() is not None:
            raise RuntimeError("llama-server exited during startup; see work/llm/v5-server.log")
        try:
            with http.open(LLM_HEALTH_URL, timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("llama-server failed health check")


def _server_command(model: Path, workers: int) -> list[str]:
    command = [
        str(RUNTIME_DIR / "bin" / "llama-server.exe"),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "-ngl",
        "99",
        "-c",
        str(12288 * workers),
        "-np",
        str(workers),
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-b",
        "1024",
        "-ub",
        "256",
        "--no-webui",
    ]
    if "Qwen3.5" in model.name:
        command.extend(["--reasoning", "off"])
    return command


def run_batch(args: argparse.Namespace) -> None:
    """Chạy thực nghiệm toàn bộ tập validation hoặc public_test với checkpoint & đánh giá."""
    split = "public_test" if args.stage == "public" else "train"
    source = ASR_HANDOFF_DIR / (
        "public_test_inputs_asr.jsonl" if split == "public_test" else "validation_inputs_asr.jsonl"
    )
    if args.inputs:
        source = args.inputs.resolve()
    rows, errors = read_jsonl(source)
    if errors:
        raise ValueError(errors)
    if args.ids:
        ids = set(json.loads(args.ids.read_text(encoding="utf-8")))
        rows = [row for row in rows if row["id"] in ids]
        if {row["id"] for row in rows} != ids:
            raise ValueError("Unknown sample IDs")

    directory = args.out
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.jsonl"
    saved, errors = read_jsonl(checkpoint) if checkpoint.exists() else ([], [])
    if errors:
        raise ValueError(errors)
    completed = {row["id"]: row["prediction"] for row in saved if row["status"] not in FAILED_STATUSES}

    catalog = json.loads((DEFAULT_ROOT / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
    engine = load_btc_runtime(DEFAULT_ROOT, split)
    examples = (
        PhraseExamples(DEV_REFERENCES_PATH)
        if args.policy == "direct"
        else Examples(DEV_REFERENCES_PATH)
    )
    agent_cls = {"pilot": PlannedAgent, "v5": Agent, "direct": DirectAgent}[args.policy]
    agent = agent_cls(catalog, examples)

    def predict(row: dict) -> dict:
        env = engine.ToolEnv(split, row["id"], root=DEFAULT_ROOT)
        try:
            trace = agent.solve(row, env)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            trace = {
                "id": row["id"],
                "status": "invalid_action",
                "trace": [{"error": f"{type(error).__name__}: {error}"}],
            }
        dump(directory / "traces" / f'{row["id"]}.json', trace)
        tool_calls = OutputGuardrail.fix_iso_time(env.get_tool_calls())
        prediction = {
            "id": row["id"],
            "tool_calls": tool_calls,
            "state_hash": env.get_state_hash(),
        }
        return {"id": row["id"], "status": trace["status"], "prediction": prediction}

    todo = [row for row in rows if row["id"] not in completed]
    if todo:
        with (RUNTIME_DIR / "v5-server.log").open("ab") as log:
            server = subprocess.Popen(
                _server_command(args.model.resolve(), args.workers),
                cwd=RUNTIME_DIR / "bin",
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                _wait_for_server_health(server)
                with (
                    checkpoint.open("a", encoding="utf-8") as output,
                    ThreadPoolExecutor(max_workers=args.workers) as pool,
                ):
                    futures = [pool.submit(predict, row) for row in todo]
                    for future in as_completed(futures):
                        record = future.result()
                        output.write(compact(record) + "\n")
                        output.flush()
                        if record["status"] not in FAILED_STATUSES:
                            completed[record["id"]] = record["prediction"]
                        print(f'{len(completed)}/{len(rows)} {record["id"]} {record["status"]}', flush=True)
            finally:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=15)

    if any(row["id"] not in completed for row in rows):
        raise RuntimeError("Operational failures remain; resume this experiment to retry those samples.")

    write_jsonl(directory / "submission.jsonl", [completed[row["id"]] for row in rows])
    dump(directory / "ids.json", [row["id"] for row in rows])
    if args.export:
        export_submission(directory, PROJECT_ROOT / "submission_v5.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=["validation", "public"], default="public")
    parser.add_argument("--mode", choices=["local", "llm"], default="llm")
    parser.add_argument("--output", type=Path, help="Đường dẫn file JSONL đầu ra khi chạy --mode local")
    parser.add_argument("--policy", choices=["pilot", "v5", "direct"], default="direct")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_GGUF_MODEL)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=DEFAULT_WORKERS)
    parser.add_argument("--export", action="store_true")
    parsed_args = parser.parse_args()
    if parsed_args.mode == "local" or parsed_args.output is not None:
        run_local_pipeline(parsed_args)
    elif parsed_args.out is None:
        print(
            "LifeMate Agent Backend is ready. Use `python -m unittest discover -s tests` to run tests, "
            "`python main.py --mode local --output submission_test_local.jsonl` to run the 4-stage pipeline, "
            "or pass `--out <dir>` to run LLM batch inference."
        )
    else:
        if parsed_args.export and (parsed_args.stage != "public" or parsed_args.ids):
            parser.error("--export requires the full public set")
        run_batch(parsed_args)

