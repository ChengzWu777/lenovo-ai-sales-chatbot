import re
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from agents.session_manager import Session
from agents.sales_graph import build_sales_graph


class EvalResult:
    def __init__(self, test_id: int, description: str):
        self.test_id = test_id
        self.description = description
        self.response = ""
        self.category = ""
        self.latency = 0.0
        self.products_count = 0
        self.llm_called = False
        self.budget_ok = True
        self.keyword_ok = True
        self.category_ok = True
        self.missing_keywords = []
        self.found_forbidden = []
        self.over_budget_prices = []

    def score(self) -> float:
        s = 1.0
        if not self.budget_ok: s -= 0.4
        if not self.keyword_ok: s -= 0.3
        if not self.category_ok: s -= 0.3
        return max(0.0, s)

    def to_dict(self) -> dict:
        return {
            "id": self.test_id,
            "description": self.description,
            "category": self.category,
            "latency": self.latency,
            "products": self.products_count,
            "llm_called": self.llm_called,
            "budget_ok": self.budget_ok,
            "keyword_ok": self.keyword_ok,
            "category_ok": self.category_ok,
            "score": self.score(),
            "issues": {
                "missing_keywords": self.missing_keywords,
                "found_forbidden": self.found_forbidden,
                "over_budget_prices": self.over_budget_prices,
            }
        }


class Evaluator:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen3.5:27b",
            base_url="http://localhost:11434",
            temperature=0.3,
        )
        self.graph = build_sales_graph()

    def run_single(self, test_case: dict) -> EvalResult:
        result = EvalResult(test_case["id"], test_case["description"])

        session = Session(session_id=f"eval_{test_case['id']}")
        session.turn_count = 1

        if test_case.get("budget_max"):
            session.profile.budget_max = test_case["budget_max"]
            session.profile.budget_firm = test_case.get("budget_firm", False)
        if test_case.get("weight_max"):
            session.profile.weight_max = test_case["weight_max"]

        start = time.time()

        state = self.graph.invoke({
            "session": session,
            "user_msg": test_case["query"],
            "action": None,
            "response": None,
            "category": None,
            "matched_products": None,
            "system_prompt": None,
            "skip": False,
        })

        action = state.get("action", "direct")
        category = state.get("category") or "all"
        matched = state.get("matched_products") or []

        response = state.get("response", "")
        if action == "llm":
            system_prompt = state.get("system_prompt", "")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=test_case["query"])
            ]
            llm_result = self.llm.invoke(messages)
            response = llm_result.content
            result.llm_called = True

        result.latency = round(time.time() - start, 2)
        result.response = response
        result.category = category
        result.products_count = len(matched)

        # Budget compliance
        if test_case.get("budget_max") and test_case.get("budget_firm"):
            prices = re.findall(r'\$(\d{3,4})', response)
            for price_str in prices:
                price = int(price_str)
                if price > test_case["budget_max"]:
                    result.budget_ok = False
                    result.over_budget_prices.append(price)

        # Keyword presence
        response_lower = response.lower()
        must_include = test_case.get("must_include", [])
        if must_include:
            found_any = any(kw.lower() in response_lower for kw in must_include)
            if not found_any:
                result.keyword_ok = False
                result.missing_keywords = must_include

        # Forbidden keywords
        must_not = test_case.get("must_not_include", [])
        for kw in must_not:
            if kw.lower() in response_lower:
                result.found_forbidden.append(kw)
        if result.found_forbidden:
            result.keyword_ok = False

        # Category routing
        expected = test_case.get("expected_category", "all")
        if expected != "all" and category != expected and category != "all":
            result.category_ok = False

        return result

    def run_all(self, test_cases: list, verbose: bool = True) -> list:
        results = []
        for i, tc in enumerate(test_cases):
            if verbose:
                print(f"[{i+1:02d}/{len(test_cases)}] Running: {tc['description']}...")
            result = self.run_single(tc)
            results.append(result)
            if verbose:
                status = "✅" if result.score() >= 0.7 else "⚠️" if result.score() >= 0.4 else "❌"
                print(f"       {status} score={result.score():.1f} | lat={result.latency}s | "
                      f"cat={result.category} | budget_ok={result.budget_ok} | kw_ok={result.keyword_ok}")
        return results