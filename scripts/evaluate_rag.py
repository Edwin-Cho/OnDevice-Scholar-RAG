"""
evaluate_rag.py — RAGAS-based RAG evaluation pipeline for OnDevice Scholar RAG

Usage:
  python scripts/evaluate_rag.py
  python scripts/evaluate_rag.py --config config/eval_config.yaml
  python scripts/evaluate_rag.py --mode openai   # requires OPENAI_API_KEY
  python scripts/evaluate_rag.py --mode basic    # heuristic only, no LLM needed
  python scripts/evaluate_rag.py --output reports/

Modes:
  basic  (default) — heuristic metrics without external LLM
                     • retrieval_score_avg, citation_rate, no_context_rate,
                       answer_length_avg, chunk_overlap_ratio
  openai           — full RAGAS metrics via OpenAI judge
                     • faithfulness, answer_relevancy, context_precision,
                       context_recall (if ground_truth provided)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml

# ── Helpers ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "eval_config.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_token(base_url: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{base_url}/auth/token",
        json={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def query_backend(
    base_url: str,
    token: str,
    question: str,
    top_k: int = 5,
) -> dict:
    resp = requests.post(
        f"{base_url}/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": question, "top_k": top_k, "include_chunks": True},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Basic (heuristic) metrics ─────────────────────────────────────────────────

_NO_CONTEXT_PATTERN = re.compile(
    r"no relevant information found", re.IGNORECASE
)
_CITATION_PATTERN = re.compile(r"\[Source:", re.IGNORECASE)


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def _chunk_overlap_ratio(answer: str, chunks: list[str]) -> float:
    """Fraction of answer words that appear in at least one retrieved chunk."""
    if not chunks:
        return 0.0
    answer_words = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
    if not answer_words:
        return 0.0
    chunk_words: set[str] = set()
    for c in chunks:
        chunk_words.update(re.findall(r"[a-zA-Z]{4,}", c.lower()))
    overlap = answer_words & chunk_words
    return round(len(overlap) / len(answer_words), 4)


def compute_basic_metrics(results: list[dict]) -> dict[str, Any]:
    retrieval_scores: list[float] = []
    citation_rates: list[float] = []
    overlap_ratios: list[float] = []
    answer_lengths: list[int] = []
    no_context_count = 0
    warning_count = 0

    for r in results:
        answer = r.get("answer", "")
        chunks = r.get("retrieved_chunks", [])
        citations = r.get("citations", [])
        warnings = r.get("warnings", [])

        if _NO_CONTEXT_PATTERN.search(answer):
            no_context_count += 1

        if citations:
            scores = [c.get("score") for c in citations if c.get("score") is not None]
            if scores:
                retrieval_scores.append(sum(scores) / len(scores))

        sentences = _sentence_count(answer)
        citation_matches = len(_CITATION_PATTERN.findall(answer))
        citation_rates.append(min(1.0, citation_matches / sentences))

        overlap_ratios.append(_chunk_overlap_ratio(answer, chunks))
        answer_lengths.append(len(answer.split()))
        warning_count += len(warnings)

    n = len(results)
    return {
        "num_questions": n,
        "no_context_rate": round(no_context_count / n, 4) if n else 0,
        "retrieval_score_avg": round(sum(retrieval_scores) / len(retrieval_scores), 4) if retrieval_scores else 0,
        "citation_rate_avg": round(sum(citation_rates) / len(citation_rates), 4) if citation_rates else 0,
        "chunk_overlap_ratio_avg": round(sum(overlap_ratios) / len(overlap_ratios), 4) if overlap_ratios else 0,
        "answer_length_avg_words": round(sum(answer_lengths) / len(answer_lengths), 1) if answer_lengths else 0,
        "total_warnings": warning_count,
        "warnings_per_query": round(warning_count / n, 2) if n else 0,
    }


def compute_basic_per_category(results: list[dict]) -> dict[str, Any]:
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r.get("category", "uncategorized")].append(r)
    return {
        cat: compute_basic_metrics(items)
        for cat, items in sorted(by_cat.items())
    }


# ── RAGAS (OpenAI) metrics ─────────────────────────────────────────────────────

def compute_ragas_metrics(results: list[dict], openai_model: str) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        print(f"[ERROR] ragas/datasets not installed: {exc}")
        print("        Run: pip install ragas datasets")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(model=openai_model, api_key=api_key)
        embeddings = OpenAIEmbeddings(api_key=api_key)
    except ImportError:
        print("[ERROR] langchain-openai not installed. Run: pip install langchain-openai")
        sys.exit(1)

    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for r in results:
        data["question"].append(r["question"])
        data["answer"].append(r.get("answer", ""))
        data["contexts"].append(r.get("retrieved_chunks", [""]))
        data["ground_truth"].append(r.get("ground_truth", ""))

    has_ground_truth = any(gt.strip() for gt in data["ground_truth"])
    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_ground_truth:
        metrics.append(context_recall)

    dataset = Dataset.from_dict(data)
    score = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    return {k: round(float(v), 4) for k, v in score.items()}


# ── Report ─────────────────────────────────────────────────────────────────────

def save_report(
    output_dir: Path,
    mode: str,
    metrics: dict,
    per_category: dict,
    raw_results: list[dict],
    elapsed: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "mode": mode,
        "elapsed_seconds": round(elapsed, 1),
        "summary_metrics": metrics,
        "per_category_metrics": per_category,
        "raw_results": raw_results,
    }
    json_path = output_dir / f"eval_{mode}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {json_path}")
    return json_path


def print_summary(mode: str, metrics: dict, per_category: dict, elapsed: float) -> None:
    print("\n" + "═" * 60)
    print(f"  OnDevice Scholar RAG — Evaluation Report  [{mode.upper()} mode]")
    print("═" * 60)
    print(f"  Elapsed: {elapsed:.1f}s")
    print()
    print("  ── Summary Metrics ──────────────────────────────────────")
    for k, v in metrics.items():
        bar = ""
        if isinstance(v, float) and 0.0 <= v <= 1.0:
            filled = int(v * 20)
            bar = "  [" + "█" * filled + "░" * (20 - filled) + f"]  {v:.4f}"
        else:
            bar = f"  {v}"
        print(f"  {k:<35}{bar}")
    print()
    print("  ── Per-Category Breakdown ───────────────────────────────")
    for cat, cat_metrics in per_category.items():
        print(f"\n  [{cat}]")
        for k, v in cat_metrics.items():
            print(f"    {k:<33}  {v}")
    print("═" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OnDevice Scholar RAG — RAGAS Evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["basic", "openai"], default=None,
                        help="Override judge.mode from config")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_cfg = cfg.get("api", {})
    judge_cfg = cfg.get("judge", {})
    out_cfg = cfg.get("output", {})

    mode = args.mode or judge_cfg.get("mode", "basic")
    output_dir = args.output or PROJECT_ROOT / out_cfg.get("dir", "reports")
    questions_cfg: list[dict] = cfg.get("questions", [])

    if not questions_cfg:
        print("[ERROR] No questions found in config.")
        sys.exit(1)

    base_url = api_cfg.get("base_url", "http://localhost:8000")
    top_k = api_cfg.get("top_k", 5)

    print(f"[INFO] Mode: {mode.upper()}")
    print(f"[INFO] Questions: {len(questions_cfg)}")
    print(f"[INFO] Backend: {base_url}")

    print("\n[INFO] Acquiring auth token...")
    try:
        token = get_token(base_url, api_cfg["username"], api_cfg["password"])
    except Exception as exc:
        print(f"[ERROR] Auth failed: {exc}")
        sys.exit(1)

    print("[INFO] Running queries...\n")
    raw_results: list[dict] = []
    start = time.time()

    for i, q_item in enumerate(questions_cfg, 1):
        question = q_item["question"]
        category = q_item.get("category", "uncategorized")
        ground_truth = q_item.get("ground_truth", "")

        print(f"  [{i:02d}/{len(questions_cfg)}] {question[:70]}...")
        try:
            resp = query_backend(base_url, token, question, top_k=top_k)
            raw_results.append({
                "question": question,
                "category": category,
                "ground_truth": ground_truth,
                "answer": resp.get("answer", ""),
                "citations": resp.get("citations", []),
                "retrieved_chunks": resp.get("retrieved_chunks", []),
                "warnings": resp.get("warnings", []),
                "status": resp.get("status", ""),
            })
            print(f"         status={resp.get('status')}  chunks={len(resp.get('retrieved_chunks', []))}  warnings={len(resp.get('warnings', []))}")
        except Exception as exc:
            print(f"         [FAIL] {exc}")
            raw_results.append({
                "question": question,
                "category": category,
                "ground_truth": ground_truth,
                "answer": "",
                "citations": [],
                "retrieved_chunks": [],
                "warnings": [str(exc)],
                "status": "error",
            })

    elapsed = time.time() - start

    print("\n[INFO] Computing metrics...")
    if mode == "openai":
        summary = compute_ragas_metrics(raw_results, judge_cfg.get("openai_model", "gpt-4o-mini"))
        per_category: dict = {}
    else:
        summary = compute_basic_metrics(raw_results)
        per_category = compute_basic_per_category(raw_results)

    print_summary(mode, summary, per_category, elapsed)
    save_report(output_dir, mode, summary, per_category, raw_results, elapsed)


if __name__ == "__main__":
    main()
