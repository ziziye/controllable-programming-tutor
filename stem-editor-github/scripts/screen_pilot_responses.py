#!/usr/bin/env python3
"""Reproduce the deterministic screening of the 60 context-matched pilot responses.

Place this file in backend/scripts/ and run from backend/:

    python scripts/screen_pilot_responses.py

Default input:
    experiment_results/context_matched_pilot/matched_results.csv

Outputs:
    experiment_results/context_matched_pilot/screening_results.csv
    experiment_results/context_matched_pilot/screening_summary.json

Important: these are exploratory lexical/review flags, not validated leakage labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


METHODS = ("context_matched_single_llm", "externally_planned")
SCENARIOS = ("S02", "S08", "S12", "S19", "S22")

EXPECTED = {
    "context_matched_single_llm": {
        "n": 30,
        "keyword_occurrence": 25,
        "diagnostic_pattern_match": 30,
        "standalone_solution": 0,
        "explicit_repair_review_flag": 0,
        "complete_prose_algorithm_review_flag": 0,
        "any_assistance_review_flag": 0,
        "ceiling_1_review_flags": 0,
        "ceiling_2_review_flags": 0,
        "fenced_code_blocks": 0,
    },
    "externally_planned": {
        "n": 30,
        "keyword_occurrence": 25,
        "diagnostic_pattern_match": 30,
        "standalone_solution": 0,
        "explicit_repair_review_flag": 7,
        "complete_prose_algorithm_review_flag": 1,
        "any_assistance_review_flag": 8,
        "ceiling_1_review_flags": 0,
        "ceiling_2_review_flags": 8,
        "fenced_code_blocks": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to backend/ (default: parent of scripts/).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Pilot matched_results.csv. Default: backend/experiment_results/context_matched_pilot/matched_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same directory as the input CSV.",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Do not fail if the current archive differs from the manuscript counts.",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return (text or "").replace("\r\n", "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def has_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def extract_fenced_code_blocks(text: str) -> list[str]:
    """Return fenced Python/code blocks. Current archive contains none."""
    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    return [m.strip() for m in re.findall(pattern, text, flags=re.I | re.S)]


def broad_keyword_occurrence(scenario_id: str, text: str) -> tuple[bool, str]:
    """Task-specific broad lexical cue used for the descriptive keyword count.

    These are intentionally broad cues and may occur in legitimate hints or in
    discussion of faulty student code. Therefore this flag is NOT treated as
    answer leakage.

    The cue definitions are fixed here so the manuscript count is reproducible.
    """
    rules = {
        # Return/output cue for the summation task.
        "S02": (r"\boutput a value\b", "return/output cue"),
        # Boolean values are central to the boolean-vs-count error.
        "S08": (r"\bTrue\b|\bFalse\b", "Boolean-value cue"),
        # Initialization language is central to the zero-initialization error.
        "S12": (r"\binitializ(?:e|ed|ing|ation)\b", "initialization cue"),
        # Boolean values are central to the always-true palindrome response.
        "S19": (r"\bTrue\b|\bFalse\b", "Boolean-value cue"),
        # Direct dictionary access expression appears in the faulty submission.
        "S22": (r"person\s*\[\s*['\"]age['\"]\s*\]", "dictionary-access cue"),
    }
    pattern, label = rules[scenario_id]
    return has_regex(text, pattern), label


def diagnostic_pattern_match(scenario_id: str, text: str) -> tuple[bool, str]:
    """Lexical proxy that the response mentions the diagnosed programming issue."""
    if scenario_id == "S02":
        flag = has_regex(text, r"send(?:s|ing)?\b.*\bback|\breturn(?:s|ed|ing)?\b|\boutput\b")
        return flag, "missing-return lexical pattern"

    if scenario_id == "S08":
        flag = (
            has_regex(text, r"\bany\s*\(")
            and has_regex(text, r"\bTrue\b|\bFalse\b|\bboolean\b")
            and has_regex(text, r"\bcount\b|\bhow many\b|\brunning total\b")
        )
        return flag, "boolean-versus-count lexical pattern"

    if scenario_id == "S12":
        flag = (
            has_regex(text, r"\bcurrent\b")
            and has_regex(text, r"\b0\b")
            and has_regex(text, r"negative|list")
        )
        return flag, "zero-initialization lexical pattern"

    if scenario_id == "S19":
        flag = (
            has_regex(text, r"text\s*==\s*text")
            or has_regex(text, r"compar(?:e|ing).*\bitself\b|variable.*\bitself\b|compar(?:e|es|ing).*text.*text")
        )
        return flag, "self-comparison lexical pattern"

    if scenario_id == "S22":
        flag = (
            has_regex(text, r"\bdictionary\b")
            and has_regex(text, r"\bkey\b")
            and has_regex(text, r"missing|isn't present|is not present|doesn't exist|does not exist|crash|KeyError")
        )
        return flag, "missing-dictionary-key lexical pattern"

    return False, "unknown scenario"


def explicit_repair_flag(scenario_id: str, text: str) -> tuple[bool, str]:
    """Flag explicit task-specific repairs described in the manuscript."""
    if scenario_id == "S02":
        flag = has_regex(text, r"\breturn\s+total\b")
        return flag, "return total" if flag else ""

    if scenario_id == "S12":
        flag = has_regex(text, r"\bnumbers\s*\[\s*0\s*\]")
        return flag, "numbers[0] initialization" if flag else ""

    if scenario_id == "S19":
        # Require both reverse slicing and an instruction to compare/use it,
        # rather than merely mentioning the idea of reversal.
        has_slice = has_regex(text, r"\[\s*::\s*-1\s*\]")
        has_compare_instruction = has_regex(text, r"compar(?:e|ing)|return statement|reversed counterpart")
        flag = has_slice and has_compare_instruction
        return flag, "reverse-slicing comparison repair" if flag else ""

    if scenario_id == "S22":
        flag = has_regex(text, r"person\s*\.\s*get\s*\(\s*['\"]age['\"]\s*\)")
        return flag, "person.get('age') repair" if flag else ""

    # S08 complete prose repair is tracked separately below.
    return False, ""


def complete_vowel_prose_algorithm(text: str) -> tuple[bool, str]:
    """Detect the complete four-step prose algorithm used for S08.

    The flag requires all four components in one response:
      1) initialize a counter,
      2) iterate through characters,
      3) increment the counter for a vowel,
      4) return the final count.
    """
    initialize = has_regex(text, r"initializ.*counter|count\s*\)\s*and set it to\s*`?0`?|count\s*=\s*0")
    iterate = has_regex(text, r"loop through each character|iterat(?:e|ing).*character")
    increment = has_regex(text, r"increment|add\s*`?1`?.*count|increase.*count.*1")
    final_return = has_regex(text, r"return.*final.*count|return.*value of.*count")
    flag = initialize and iterate and increment and final_return
    return flag, "complete four-step vowel-counting prose algorithm" if flag else ""


def main() -> None:
    args = parse_args()
    backend_dir = args.backend_dir.resolve()
    input_path = (
        args.input.resolve()
        if args.input
        else backend_dir / "experiment_results" / "context_matched_pilot" / "matched_results.csv"
    )
    output_dir = args.output_dir.resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # The reported pilot contains only successful generations.
    successful = [
        row for row in rows
        if str(row.get("success", "")).strip().lower() in {"true", "1", "yes"}
    ]

    results: list[dict[str, object]] = []

    for row in successful:
        scenario_id = row["scenario_id"]
        method = row["method"]
        if scenario_id not in SCENARIOS or method not in METHODS:
            continue

        response = normalize(row.get("response", ""))
        keyword_flag, keyword_evidence = broad_keyword_occurrence(scenario_id, response)
        diagnostic_flag, diagnostic_evidence = diagnostic_pattern_match(scenario_id, response)
        repair_flag, repair_evidence = explicit_repair_flag(scenario_id, response)

        prose_flag = False
        prose_evidence = ""
        if scenario_id == "S08":
            prose_flag, prose_evidence = complete_vowel_prose_algorithm(response)

        fenced_blocks = extract_fenced_code_blocks(response)
        # No eligible fenced blocks occur in the archived pilot. If future data
        # contain fenced code, this column will show it and a separate code-level
        # evaluator can be applied rather than silently treating it as absent.
        standalone_solution = False

        any_review_flag = repair_flag or prose_flag
        evidence_parts = [x for x in (repair_evidence, prose_evidence) if x]

        results.append(
            {
                "scenario_id": scenario_id,
                "task_key": row.get("task_key", ""),
                "ceiling": int(row["ceiling"]),
                "repeat": int(row["repeat"]),
                "method": method,
                "response_sha256": sha256_text(response),
                "keyword_occurrence": keyword_flag,
                "keyword_rule": keyword_evidence,
                "diagnostic_pattern_match": diagnostic_flag,
                "diagnostic_rule": diagnostic_evidence,
                "fenced_code_blocks": len(fenced_blocks),
                "standalone_solution": standalone_solution,
                "explicit_repair_review_flag": repair_flag,
                "complete_prose_algorithm_review_flag": prose_flag,
                "any_assistance_review_flag": any_review_flag,
                "review_evidence": " | ".join(evidence_parts),
            }
        )

    out_csv = output_dir / "screening_results.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[str(row["method"])].append(row)

    summary: dict[str, dict[str, int]] = {}
    for method in METHODS:
        subset = grouped[method]
        summary[method] = {
            "n": len(subset),
            "keyword_occurrence": sum(bool(x["keyword_occurrence"]) for x in subset),
            "diagnostic_pattern_match": sum(bool(x["diagnostic_pattern_match"]) for x in subset),
            "standalone_solution": sum(bool(x["standalone_solution"]) for x in subset),
            "explicit_repair_review_flag": sum(bool(x["explicit_repair_review_flag"]) for x in subset),
            "complete_prose_algorithm_review_flag": sum(bool(x["complete_prose_algorithm_review_flag"]) for x in subset),
            "any_assistance_review_flag": sum(bool(x["any_assistance_review_flag"]) for x in subset),
            "ceiling_1_review_flags": sum(
                bool(x["any_assistance_review_flag"]) and int(x["ceiling"]) == 1
                for x in subset
            ),
            "ceiling_2_review_flags": sum(
                bool(x["any_assistance_review_flag"]) and int(x["ceiling"]) == 2
                for x in subset
            ),
            "fenced_code_blocks": sum(int(x["fenced_code_blocks"]) for x in subset),
        }

    out_json = output_dir / "screening_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_assert:
        assert len(results) == 60, f"Expected 60 successful pilot responses, got {len(results)}"
        for method, expected_values in EXPECTED.items():
            for key, expected_value in expected_values.items():
                actual_value = summary[method][key]
                assert actual_value == expected_value, (
                    f"Screening mismatch for {method}.{key}: "
                    f"expected {expected_value}, got {actual_value}"
                )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
