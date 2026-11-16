"""Re-run the text checks on a saved results file without calling the LLM again.

    python -m evaluation.recheck                      # newest results_*_real_*.json
    python -m evaluation.recheck path\\to\\results.json --show 34

Useful after a checker fix, and for reading the answer behind a failure.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.catalog import get_by_id  # noqa: E402
from evaluation import checks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?")
    parser.add_argument("--show", type=int, nargs="*", default=[], help="print the full answers of these case ids")
    args = parser.parse_args()

    path = args.path
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        files = sorted(glob.glob(os.path.join(here, "results_*_real_*.json")), key=os.path.getmtime)
        if not files:
            sys.exit("No results_*_real_*.json found; pass a path.")
        path = files[-1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Rechecking grounding in {path}")

    cases = data["results"]
    failed = 0
    for case in cases:
        case_failed = False
        for turn in case["turns"]:
            if not turn["llm_called"] or turn["fallback"]:
                continue
            provided = [p for p in (get_by_id(i) for i in turn["product_ids"]) if p]
            out = checks.check_grounding(turn["response"], provided)
            if not out["ok"]:
                case_failed = True
                print(f"  #{case['id']} {case['description']}: "
                      f"ungrounded={out['ungrounded']} wrong_prices={out['wrong_prices']}")
        failed += case_failed
        if case["id"] in args.show:
            for turn in case["turns"]:
                print(f"\n----- #{case['id']} > {turn['user']}\n{turn['response']}\n-----")
    print(f"Grounding: {len(cases) - failed}/{len(cases)}")


if __name__ == "__main__":
    main()
