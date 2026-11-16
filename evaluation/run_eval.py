"""CLI for the end-to-end evaluation.

    python -m evaluation.run_eval --suite all                 # real LLM (needs Ollama)
    python -m evaluation.run_eval --suite all --llm stub      # no GPU: pipeline + retrieval only
    python -m evaluation.run_eval --suite multi --llm stub --no-rag   # no vector index either
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.catalog import catalog_version  # noqa: E402
from agents.sales_graph import build_sales_graph  # noqa: E402
from config import settings  # noqa: E402
from evaluation.evaluator import DIMENSIONS, EchoLLM, Evaluator, summarize  # noqa: E402
from evaluation.test_cases import MULTI_TURN_CASES, TEST_CASES, TEST_CASES_V2  # noqa: E402

SUITES = {
    "v1": TEST_CASES,
    "v2": TEST_CASES_V2,
    "multi": MULTI_TURN_CASES,
    "all": TEST_CASES + TEST_CASES_V2 + MULTI_TURN_CASES,
}


def run_evaluation(suite: str = "v1", llm_mode: str = "real", use_rag: bool = True,
                   test_ids: list = None, output_file: str = None, verbose: bool = True) -> dict:
    cases = SUITES[suite]
    if test_ids:
        cases = [c for c in cases if c["id"] in test_ids]

    graph = build_sales_graph(retriever="auto" if use_rag else None)
    if llm_mode == "real":
        from chat_service import build_llm
        llm = build_llm()
    else:
        llm = EchoLLM()

    print("=" * 64)
    print("Lenovo AI Sales Chatbot — evaluation")
    print(f"suite={suite} cases={len(cases)} llm={llm_mode}"
          f"{' (' + settings.ollama_model + ')' if llm_mode == 'real' else ''} "
          f"rag={'on' if use_rag else 'off'} catalog={catalog_version()}")
    print("=" * 64)

    started = time.time()
    results = asyncio.run(Evaluator(graph, llm, llm_mode).run_all(cases, verbose=verbose))
    summary = summarize(results)
    summary["total_time_s"] = round(time.time() - started, 1)

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Cases passed:   {summary['cases_passed']}/{summary['cases']} ({summary['pass_rate_pct']}%)")
    print(f"Avg score:      {summary['avg_score']:.2f} / 1.00")
    if llm_mode != "real":
        # Nothing is generated in stub mode, so grounding cannot be judged.
        summary["grounding_pass"], summary["grounding_pass_pct"] = "n/a", None
    for d in DIMENSIONS:
        if summary[d + "_pass_pct"] is None:
            print(f"{d.capitalize() + ':':<15} n/a (needs --llm real)")
        else:
            print(f"{d.capitalize() + ':':<15} {summary[d + '_pass']} ({summary[d + '_pass_pct']}%)")
    print(f"Actions:        {summary['actions']}")
    print(f"LLM call rate:  {summary['llm_call_rate_pct']}% of turns, fallbacks: {summary['fallbacks']}")
    if llm_mode == "real" and "llm_latency_avg_s" in summary:
        print(f"LLM latency:    avg {summary['llm_latency_avg_s']}s | p50 {summary['llm_latency_p50_s']}s | "
              f"p95 {summary['llm_latency_p95_s']}s | TTFT avg {summary.get('ttft_avg_s', 'n/a')}s")
    print(f"Total time:     {summary['total_time_s']}s")

    if output_file is None:
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"results_{suite}_{llm_mode}_{int(time.time())}.json")
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"suite": suite, "llm_mode": llm_mode, "rag": use_rag,
                   "model": settings.ollama_model if llm_mode == "real" else "EchoLLM",
                   "temperature": settings.temperature, "reasoning": settings.reasoning,
                   "catalog_version": catalog_version()},
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=list(SUITES), default="v1")
    parser.add_argument("--llm", choices=["real", "stub"], default="real", dest="llm_mode")
    parser.add_argument("--no-rag", action="store_true", help="lexical ranking only (no vector index)")
    parser.add_argument("--ids", type=int, nargs="*", help="run only these case ids")
    parser.add_argument("--out", help="path for the JSON results file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    out = run_evaluation(args.suite, args.llm_mode, not args.no_rag, args.ids, args.out, not args.quiet)
    sys.exit(0 if out["summary"]["cases_passed"] == out["summary"]["cases"] else 1)
