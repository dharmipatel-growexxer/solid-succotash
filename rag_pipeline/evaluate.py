"""
Baseline evaluator for the Government Schemes RAG pipeline.

Metrics included:
- Retrieval success rate
- Citation coverage (URL presence)
- Heuristic groundedness score
- Latency statistics

Optional dataset format (JSON list):
[
  {
    "query": "scholarship for girls in Gujarat",
    "expected_scheme_keywords": ["scholarship", "education"],
    "expected_location": "Gujarat"
  }
]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List

from service import answer_query


DEFAULT_QUERIES = [
    "I am a 19 year old college student in Gujarat from general category. Suggest schemes.",
    "What housing schemes are available for poor families in Delhi?",
    "I am a farmer in Maharashtra. Any crop insurance schemes?",
    "Are there scholarship schemes for engineering students in Karnataka?",
    "What pension schemes are there for senior citizens?",
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def heuristic_groundedness(answer: str, citations: List[Dict[str, Any]]) -> float:
    """
    Heuristic groundedness baseline.
    A sentence is considered grounded if >=2 non-trivial tokens overlap with citation text.
    """
    if not answer:
        return 0.0

    citation_corpus = " ".join((c.get("snippet") or "") for c in citations)
    citation_tokens = set(_tokenize(citation_corpus))
    if not citation_tokens:
        return 0.0

    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        tokens = [t for t in _tokenize(sentence) if len(t) > 2]
        overlap = sum(1 for token in tokens if token in citation_tokens)
        if overlap >= 2:
            supported += 1

    return supported / len(sentences)


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = math.ceil(0.95 * len(sorted_vals)) - 1
    index = max(0, min(index, len(sorted_vals) - 1))
    return sorted_vals[index]


def load_dataset(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return [{"query": q} for q in DEFAULT_QUERIES]

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array.")
    return data


def run_evaluation(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []

    for item in dataset:
        query = item["query"]
        response = answer_query(query, session_id="eval", include_debug=True)

        success = bool(response.get("success"))
        citations = response.get("citations", [])
        scheme_links = response.get("cited_schemes", [])
        debug = response.get("debug", {})
        timings = debug.get("timings", {})

        result = {
            "query": query,
            "success": success,
            "retrieved_count": len(citations),
            "scheme_links_count": len(scheme_links),
            "has_url": len(scheme_links) > 0,
            "groundedness": heuristic_groundedness(response.get("answer", ""), citations),
            "latency_ms": timings.get("total_ms", 0.0),
            "retrieval_ms": timings.get("retrieval_ms", 0.0),
            "llm_ms": timings.get("llm_ms", 0.0),
        }

        expected_keywords = [kw.lower() for kw in item.get("expected_scheme_keywords", [])]
        if expected_keywords:
            names_text = " ".join((c.get("scheme_name") or "").lower() for c in citations)
            result["keyword_hit"] = any(keyword in names_text for keyword in expected_keywords)

        expected_location = item.get("expected_location")
        if expected_location:
            locations_text = " ".join((c.get("location_name") or "").lower() for c in citations)
            result["location_hit"] = expected_location.lower() in locations_text

        results.append(result)

    success_flags = [r["success"] for r in results]
    url_flags = [r["has_url"] for r in results]
    groundedness_scores = [r["groundedness"] for r in results]
    latency_values = [float(r["latency_ms"] or 0.0) for r in results]
    retrieval_values = [float(r["retrieval_ms"] or 0.0) for r in results]
    llm_values = [float(r["llm_ms"] or 0.0) for r in results]

    summary = {
        "total_queries": len(results),
        "success_rate": sum(success_flags) / len(results) if results else 0.0,
        "retrieval_success_rate": sum(1 for r in results if r["retrieved_count"] > 0) / len(results) if results else 0.0,
        "url_coverage_rate": sum(url_flags) / len(results) if results else 0.0,
        "avg_groundedness": statistics.mean(groundedness_scores) if groundedness_scores else 0.0,
        "latency_ms": {
            "avg": statistics.mean(latency_values) if latency_values else 0.0,
            "p95": p95(latency_values),
        },
        "retrieval_ms": {
            "avg": statistics.mean(retrieval_values) if retrieval_values else 0.0,
            "p95": p95(retrieval_values),
        },
        "llm_ms": {
            "avg": statistics.mean(llm_values) if llm_values else 0.0,
            "p95": p95(llm_values),
        },
    }

    if any("keyword_hit" in r for r in results):
        keyword_hits = [bool(r.get("keyword_hit")) for r in results if "keyword_hit" in r]
        summary["keyword_hit_rate"] = sum(keyword_hits) / len(keyword_hits) if keyword_hits else 0.0

    if any("location_hit" in r for r in results):
        location_hits = [bool(r.get("location_hit")) for r in results if "location_hit" in r]
        summary["location_hit_rate"] = sum(location_hits) / len(location_hits) if location_hits else 0.0

    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation for RAG service")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve() if args.dataset else None
    dataset = load_dataset(dataset_path)
    evaluation = run_evaluation(dataset)

    print(json.dumps(evaluation["summary"], indent=2))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(evaluation, file, indent=2)
        print(f"Detailed report written to: {output_path}")


if __name__ == "__main__":
    main()
