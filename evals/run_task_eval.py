"""Run Contribution 2 evaluation: Task Identification.

Evaluates whether the agent correctly generates a recovery task for
RPA failures. Uses the JUDGE_PROMPT-based evaluator to assess validity
and extensiveness of the generated task.

Reports agent token usage and timing metrics per case and in aggregate.

Usage:
    uv run python -m evals.run_task_eval
    uv run python -m evals.run_task_eval --model gpt-4o --repetitions 3
    uv run python -m evals.run_task_eval --vision --judge-iterations 5
    uv run python -m evals.run_task_eval --api-url http://localhost:8080/v1 --output results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from strands_evals import Case
from strands_evals.types.evaluation import EvaluationData

import settings
from evals.agents import build_model, handler_task
from evals.dataset import load_cases
from evals.evaluators import TaskIdentificationEvaluator, evaluation_summary

OUTPUT_DIR = Path("experiment_files")
DATA_DIR = Path(__file__).parent / "data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task Identification evaluation (Contribution 2).",
    )
    _ = parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="OpenAI-compatible API endpoint for inference (default: settings.PROVIDER_API_BASE).",
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for inference — NOT the judge model (default: settings.PROVIDER_MODEL).",
    )
    _ = parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results JSON (default: experiment_files/task_identification_eval.json).",
    )
    _ = parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to run the full experiment (default: 1).",
    )
    _ = parser.add_argument(
        "--judge-iterations",
        type=int,
        default=1,
        help="Number of judge evaluations per case; final result is majority consensus (default: 1).",
    )
    _ = parser.add_argument(
        "--vision",
        action="store_true",
        default=False,
        help="Include screenshot image (data/images/{index}.png) during inference.",
    )
    return parser.parse_args()


def _get_output_path(custom: str | None) -> Path:
    if custom:
        return Path(custom)
    return OUTPUT_DIR / "task_identification_eval.json"


EXTENSIVENESS_LABELS = ("INSUFFICIENT", "OPTIMAL", "EXTENSIVE", "INVALID")


def _run_case(
    case: Case[str, str],
    evaluator: TaskIdentificationEvaluator,
    *,
    api_url: str | None,
    model_name: str | None,
    image_path: Path | None,
    judge_iterations: int,
) -> dict[str, Any]:
    """Run the agent on one case, judge it, print its verdict and return its result entry."""
    task_output = handler_task(
        case,
        api_url=api_url,
        model_name=model_name,
        image_path=image_path,
    )

    eval_data = EvaluationData[str, str](
        input=case.input,
        actual_output=json.dumps(task_output),
        expected_output=case.expected_output,
        name=case.name,
        metadata=case.metadata,
    )

    eval_results, judge_votes = evaluator.evaluate_with_consensus(
        eval_data,
        iterations=judge_iterations,
    )
    eval_result = eval_results[0] if eval_results else None

    agent_m = task_output.get("agent_metrics", {})

    status = "PASS" if (eval_result and eval_result.test_pass) else "FAIL"
    tokens = agent_m.get("total_tokens", 0)
    dur = agent_m.get("total_duration_s", 0.0)
    f2l = agent_m.get("first_to_last_token_s", 0.0)
    print(
        f"  [{status}] label={eval_result.label if eval_result else '—'}  "
        + f"task='{task_output.get('generated_task', 'NONE')}' "
        + f"tokens={tokens}  duration={dur:.1f}s  first-to-last={f2l:.1f}s"
    )

    return {
        "case_id": case.name,
        "task": (case.metadata or {}).get("task_name", ""),
        "ground_truth": json.loads(case.expected_output or "{}"),
        "agent_output": {
            "generated_task": task_output.get("generated_task"),
            "raw_instruction": task_output.get("raw_instruction", ""),
            "model_messages": task_output.get("model_messages", []),
        },
        "agent_metrics": {
            "input_tokens": agent_m.get("input_tokens", 0),
            "output_tokens": agent_m.get("output_tokens", 0),
            "total_tokens": tokens,
            "total_duration_s": dur,
            "first_to_last_token_s": f2l,
        },
        "evaluation": evaluation_summary(eval_result),
        "judge_iterations": judge_iterations,
        "judge_votes": judge_votes,
        "vision_enabled": image_path is not None,
    }


def _quality_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """The valid rate and extensiveness distribution of the cases' evaluations."""
    passes = [
        r["evaluation"]["pass"]
        for r in case_results
        if r["evaluation"]["pass"] is not None
    ]
    labels = [
        r["evaluation"]["label"]
        for r in case_results
        if r["evaluation"]["label"] is not None
    ]

    valid_count = sum(1 for p in passes if p)
    total_count = len(passes)
    return {
        "valid_rate": valid_count / total_count if total_count > 0 else 0.0,
        "total_cases": total_count,
        "valid_count": valid_count,
        "extensiveness_distribution": {
            label: labels.count(label) for label in EXTENSIVENESS_LABELS
        },
    }


def _sum(entries: list[dict[str, Any]], group: str, metric: str) -> Any:
    """The sum of ``entry[group][metric]`` over ``entries``."""
    return sum(entry[group][metric] for entry in entries)


def _resource_metrics(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    duration_s: float,
    first_to_last_token_s: float,
    case_count: int,
) -> dict[str, Any]:
    """Token and timing totals, with their averages over ``case_count`` cases (at least 1)."""
    n = case_count or 1
    return {
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens": round(input_tokens / n),
        "avg_output_tokens": round(output_tokens / n),
        "avg_tokens": round(total_tokens / n),
        "total_duration_s": round(duration_s, 3),
        "avg_duration_s": round(duration_s / n, 3),
        "total_first_to_last_token_s": round(first_to_last_token_s, 3),
        "avg_first_to_last_token_s": round(first_to_last_token_s / n, 3),
    }


def run_evaluation(
    *,
    api_url: str | None = None,
    model_name: str | None = None,
    vision: bool = False,
    judge_iterations: int = 1,
) -> dict[str, Any]:
    """Run the task identification evaluation and return full results."""
    cases = load_cases("task_identification")
    print(f"Loaded {len(cases)} test cases for task identification evaluation.\n")

    judge_model = build_model()
    evaluator = TaskIdentificationEvaluator(judge_model=judge_model)

    case_results: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] Running {case.name}...")
        case_results.append(
            _run_case(
                case,
                evaluator,
                api_url=api_url,
                model_name=model_name,
                image_path=DATA_DIR / "images" / f"{i}.png" if vision else None,
                judge_iterations=judge_iterations,
            )
        )

    return {
        "experiment": "task_identification_eval",
        "timestamp": datetime.now(UTC).isoformat(),
        "model_id": model_name or settings.PROVIDER_MODEL,
        "api_url": api_url or settings.PROVIDER_API_BASE,
        "config": {
            "repetitions": 1,
            "judge_iterations": judge_iterations,
            "vision": vision,
        },
        "metrics": {
            "task_identification": _quality_metrics(case_results),
            "agent_resources": _resource_metrics(
                input_tokens=_sum(case_results, "agent_metrics", "input_tokens"),
                output_tokens=_sum(case_results, "agent_metrics", "output_tokens"),
                total_tokens=_sum(case_results, "agent_metrics", "total_tokens"),
                duration_s=_sum(case_results, "agent_metrics", "total_duration_s"),
                first_to_last_token_s=_sum(
                    case_results, "agent_metrics", "first_to_last_token_s"
                ),
                case_count=len(case_results),
            ),
        },
        "case_results": case_results,
    }


def _merge_repetition_results(all_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple repetition runs into a single aggregate result."""
    if len(all_runs) == 1:
        return all_runs[0]

    base = all_runs[0]
    n_runs = len(all_runs)

    # Collect per-case labels, passes, tasks, and judge votes across runs
    case_labels: dict[str, list[str | None]] = {}
    case_passes: dict[str, list[bool | None]] = {}
    case_tasks: dict[str, list[str | None]] = {}
    case_judge_votes: dict[str, list[str]] = {}
    for run in all_runs:
        for cr in run["case_results"]:
            cid = cr["case_id"]
            case_labels.setdefault(cid, []).append(cr["evaluation"]["label"])
            case_passes.setdefault(cid, []).append(cr["evaluation"]["pass"])
            case_tasks.setdefault(cid, []).append(
                cr["agent_output"].get("generated_task")
            )
            case_judge_votes.setdefault(cid, []).extend(cr.get("judge_votes", []))

    merged_case_results: list[dict[str, Any]] = []
    for case_id, labels in case_labels.items():
        last_entry = next(cr for cr in base["case_results"] if cr["case_id"] == case_id)
        votes = [v for v in labels if v is not None]
        pass_votes = [v for v in case_passes[case_id] if v is not None]

        majority_label = Counter(votes).most_common(1)[0][0] if votes else "UNKNOWN"
        majority_pass = (
            Counter(pass_votes).most_common(1)[0][0] if pass_votes else False
        )

        merged_entry = dict(last_entry)
        merged_entry["evaluation"] = {
            "score": 1.0 if majority_pass else 0.0,
            "pass": majority_pass,
            "label": majority_label,
            "reason": (f"Majority vote over {n_runs} runs: {dict(Counter(votes))}"),
        }
        merged_entry["repetition_labels"] = labels
        merged_entry["generated_tasks"] = case_tasks[case_id]
        merged_entry["judge_votes"] = case_judge_votes[case_id]
        merged_case_results.append(merged_entry)

    merged = dict(base)
    merged["config"]["repetitions"] = n_runs
    # Rebuild quality metrics from the merged labels; sum resources over all runs
    run_metrics = [run["metrics"] for run in all_runs]
    merged["metrics"] = {
        "task_identification": _quality_metrics(merged_case_results),
        "agent_resources": _resource_metrics(
            input_tokens=_sum(run_metrics, "agent_resources", "total_input_tokens"),
            output_tokens=_sum(run_metrics, "agent_resources", "total_output_tokens"),
            total_tokens=_sum(run_metrics, "agent_resources", "total_tokens"),
            duration_s=_sum(run_metrics, "agent_resources", "total_duration_s"),
            first_to_last_token_s=_sum(
                run_metrics, "agent_resources", "total_first_to_last_token_s"
            ),
            case_count=len(merged_case_results),
        ),
    }
    merged["case_results"] = merged_case_results
    merged["repetitions_summary"] = {
        "total_runs": n_runs,
        "per_run_valid_rates": [
            run["metrics"]["task_identification"]["valid_rate"] for run in all_runs
        ],
    }

    return merged


def main() -> None:
    args = _parse_args()

    all_runs: list[dict[str, Any]] = []
    for rep in range(args.repetitions):
        if args.repetitions > 1:
            print(f"\n{'=' * 60}")
            print(f"  REPETITION {rep + 1}/{args.repetitions}")
            print(f"{'=' * 60}\n")

        results = run_evaluation(
            api_url=args.api_url,
            model_name=args.model,
            vision=args.vision,
            judge_iterations=args.judge_iterations,
        )
        all_runs.append(results)

    final = _merge_repetition_results(all_runs)

    m = final["metrics"]["task_identification"]
    ar = final["metrics"]["agent_resources"]
    ed = m["extensiveness_distribution"]
    print("\n" + "=" * 60)
    print("  Task Identification Evaluation")
    print("=" * 60 + "\n")

    # Per-case results
    for r in final["case_results"]:
        status = "PASS" if r["evaluation"]["pass"] else "FAIL"
        generated = r["agent_output"].get("generated_task") or "(none)"
        label = r["evaluation"].get("label", "—")
        am = r["agent_metrics"]
        print(f"  [{status}] [{label}] {r['case_id']}")
        print(f"         Generated: {generated[:100]}")
        print(
            f"         Tokens: in={am['input_tokens']} out={am['output_tokens']}  "
            + f"Duration: {am['total_duration_s']:.1f}s  First-to-last: {am['first_to_last_token_s']:.1f}s"
        )
        print()

    print("  --- Quality Metrics ---")
    print(
        f"  Valid rate:     {m['valid_rate']:.2%}  ({m['valid_count']}/{m['total_cases']})"
    )
    print(
        f"  Extensiveness:  insufficient={ed['INSUFFICIENT']}  "
        + f"optimal={ed['OPTIMAL']}  extensive={ed['EXTENSIVE']}  "
        + f"invalid={ed['INVALID']}"
    )
    print()
    print("  --- Agent Resource Metrics ---")
    print(f"  Total tokens:   {ar['total_tokens']:,}  (avg {ar['avg_tokens']:,}/case)")
    print(
        f"    Input:        {ar['total_input_tokens']:,}  (avg {ar['avg_input_tokens']:,}/case)"
    )
    print(
        f"    Output:       {ar['total_output_tokens']:,}  (avg {ar['avg_output_tokens']:,}/case)"
    )
    print(
        f"  Total duration: {ar['total_duration_s']:.1f}s  (avg {ar['avg_duration_s']:.1f}s/case)"
    )
    print(
        f"  First-to-last:  {ar['total_first_to_last_token_s']:.1f}s  (avg {ar['avg_first_to_last_token_s']:.1f}s/case)"
    )
    if final.get("repetitions_summary"):
        rs = final["repetitions_summary"]
        print()
        print(f"  --- Repetitions ({rs['total_runs']} runs) ---")
        print(
            f"  Per-run valid rates: {[f'{v:.0%}' for v in rs['per_run_valid_rates']]}"
        )
    print()

    output_path = _get_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=str)
    print(f"Full results (with model responses) saved to {output_path}")


if __name__ == "__main__":
    logging.getLogger("strands").setLevel(logging.WARNING)
    logging.getLogger("strands").propagate = False
    logging.getLogger("strands").handlers = []
    main()
