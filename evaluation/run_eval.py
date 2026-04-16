import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_cases_mod = load_module("test_cases", os.path.join(base, "evaluation", "test_cases.py"))
evaluator_mod = load_module("evaluator_module", os.path.join(base, "evaluation", "evaluator.py"))

Evaluator = evaluator_mod.Evaluator


def run_evaluation(test_cases_override: list = None, test_ids: list = None, output_file: str = None):
    print("=" * 60)
    print("Lenovo AI Sales Chatbot — Automated Evaluation")
    print("=" * 60)

    evaluator = Evaluator()

    cases = test_cases_override if test_cases_override else test_cases_mod.TEST_CASES
    if test_ids:
        cases = [tc for tc in cases if tc["id"] in test_ids]

    print(f"Running {len(cases)} test cases...\n")
    start_total = time.time()
    results = evaluator.run_all(cases, verbose=True)
    total_time = round(time.time() - start_total, 1)

    scores = [r.score() for r in results]
    avg_score = sum(scores) / len(scores)
    avg_latency = sum(r.latency for r in results) / len(results)
    budget_pass = sum(1 for r in results if r.budget_ok)
    keyword_pass = sum(1 for r in results if r.keyword_ok)
    category_pass = sum(1 for r in results if r.category_ok)
    llm_calls = sum(1 for r in results if r.llm_called)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total tests:          {len(results)}")
    print(f"Avg score:            {avg_score:.2f} / 1.00")
    print(f"Budget compliance:    {budget_pass}/{len(results)} ({budget_pass/len(results)*100:.0f}%)")
    print(f"Keyword match:        {keyword_pass}/{len(results)} ({keyword_pass/len(results)*100:.0f}%)")
    print(f"Category accuracy:    {category_pass}/{len(results)} ({category_pass/len(results)*100:.0f}%)")
    print(f"Avg latency:          {avg_latency:.1f}s")
    print(f"LLM call rate:        {llm_calls}/{len(results)} ({llm_calls/len(results)*100:.0f}%)")
    print(f"Total eval time:      {total_time}s")

    failures = [r for r in results if r.score() < 0.7]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures:
            print(f"  #{r.test_id} {r.description}")
            if not r.budget_ok:
                print(f"    - Budget violation: {r.over_budget_prices}")
            if not r.keyword_ok:
                print(f"    - Missing keywords: {r.missing_keywords}")
                if r.found_forbidden:
                    print(f"    - Forbidden found: {r.found_forbidden}")
            if not r.category_ok:
                print(f"    - Wrong category: {r.category}")

    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"results_{int(time.time())}.json"
        )

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(results),
            "avg_score": round(avg_score, 3),
            "budget_compliance": f"{budget_pass}/{len(results)}",
            "keyword_match": f"{keyword_pass}/{len(results)}",
            "category_accuracy": f"{category_pass}/{len(results)}",
            "avg_latency": round(avg_latency, 1),
            "llm_call_rate": f"{llm_calls}/{len(results)}",
        },
        "results": [r.to_dict() for r in results]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["v1", "v2", "all"], default="v1")
    args = parser.parse_args()

    v1 = test_cases_mod.TEST_CASES
    v2 = test_cases_mod.TEST_CASES_V2

    if args.suite == "v1":
        run_evaluation(test_cases_override=v1)
    elif args.suite == "v2":
        run_evaluation(test_cases_override=v2)
    else:
        run_evaluation(test_cases_override=v1 + v2)