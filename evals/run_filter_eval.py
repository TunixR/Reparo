"""Run Contribution 1 evaluation: Error Filtering.

Evaluates whether the Gateway Orchestrator correctly routes UI errors
to the UI handler and non-UI errors to human escalation.

Usage:
    uv run python -m evals.run_filter_eval
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strands_evals import Case
from strands_evals.evaluators import OutputEvaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput

import settings
from evals.agents import build_model, gateway_task
from evals.dataset import load_cases
from evals.evaluators import RoutingEvaluator, evaluation_summary
from evals.mocks import GatewayMocks

OUTPUT_DIR = Path("experiment_files")


RUBRIC = """
        Evaluate whether the Gateway Orchestrator correctly classified this error.

        Score 1.0 if:
        - The agent correctly determined whether this is a UI-related error or not
        - The routing decision (to UI handler vs human escalation) is appropriate

        Score 0.5 if:
        - The agent's reasoning is sound but the final routing decision is debatable
          given the ambiguity of the error message

        Score 0.0 if:
        - The agent clearly misclassified a clear-cut error
        - The agent could not make a routing decision

        Consider the error message, activity context, and application involved.
        """


def _first(results: list[EvaluationOutput]) -> EvaluationOutput | None:
    return results[0] if results else None


def _run_case(
    case: Case[str, str],
    mocks: GatewayMocks,
    routing_eval: RoutingEvaluator,
    rubric_eval: OutputEvaluator[str, str],
) -> dict[str, Any]:
    """Run the gateway on one case, evaluate it, print its verdict and return its result entry."""
    # Run task function to get model response
    task_output = gateway_task(case, mocks)

    eval_data = EvaluationData[str, str](
        input=case.input,
        actual_output=json.dumps(task_output),
        expected_output=case.expected_output,
        name=case.name,
        metadata=case.metadata,
    )
    routing_result = _first(routing_eval.evaluate(eval_data))
    rubric_result = _first(rubric_eval.evaluate(eval_data))

    gt = json.loads(case.expected_output or "{}")

    status = "PASS" if (routing_result and routing_result.test_pass) else "FAIL"
    print(
        f"  [{status}] routing={task_output.get('is_ui_error')}, expected={gt.get('is_ui_error')}"
    )

    return {
        "case_id": case.name,
        "task": (case.metadata or {}).get("task", ""),
        "ground_truth": gt,
        "agent_output": {
            "is_ui_error": task_output.get("is_ui_error"),
            "tool_calls": deepcopy(
                task_output.get("tool_calls", {})
            ),  # Deepcopy to avoid mutation issues
            "raw_instruction": task_output.get("raw_instruction", ""),
            "model_messages": task_output.get("model_messages", []),
        },
        "evaluations": {
            "routing": evaluation_summary(routing_result),
            "rubric": evaluation_summary(rubric_result, with_label=False),
        },
    }


def _avg_score(case_results: list[dict[str, Any]], evaluation: str) -> float:
    """The mean score ``evaluation`` gave, over the cases it scored."""
    scores = [
        r["evaluations"][evaluation]["score"]
        for r in case_results
        if r["evaluations"][evaluation]["score"] is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _routing_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy, precision, recall, F1 and the confusion matrix of the routing decisions."""
    confusion = Counter(
        (
            bool(r["ground_truth"].get("is_ui_error", False)),
            r["evaluations"]["routing"]["pass"] is True,
        )
        for r in case_results
    )
    # Keyed by (is a UI error, routed correctly): a non-UI error routed wrongly is a false positive.
    tp = confusion[True, True]
    fn = confusion[True, False]
    tn = confusion[False, True]
    fp = confusion[False, False]

    total = tp + fp + tn + fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "accuracy": _ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "avg_score": _avg_score(case_results, "routing"),
    }


def run_evaluation() -> dict[str, Any]:
    """Run the error filtering evaluation and return full results."""
    cases = load_cases("error_filtering")
    print(f"Loaded {len(cases)} test cases for error filtering evaluation.\n")

    judge_model = build_model()
    mocks = GatewayMocks()
    routing_eval = RoutingEvaluator()
    rubric_eval = OutputEvaluator(
        model=judge_model,
        rubric=RUBRIC,
        include_inputs=True,
    )

    case_results: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] Running {case.name}...")
        case_results.append(_run_case(case, mocks, routing_eval, rubric_eval))

    return {
        "experiment": "error_filtering_eval",
        "timestamp": datetime.now(UTC).isoformat(),
        "model_id": settings.PROVIDER_MODEL,
        "metrics": {
            "routing": _routing_metrics(case_results),
            "rubric": {"avg_score": _avg_score(case_results, "rubric")},
        },
        "case_results": case_results,
    }


def main() -> None:
    results = run_evaluation()

    m = results["metrics"]["routing"]
    print("\n" + "=" * 60)
    print("  CONTRIBUTION 1: Error Filtering Evaluation")
    print("=" * 60 + "\n")

    # Per-case results
    for r in results["case_results"]:
        gt = r["ground_truth"].get("is_ui_error", False)
        pred = r["agent_output"].get("is_ui_error")
        correct = r["evaluations"]["routing"]["pass"]
        status = "PASS" if correct else "FAIL"
        gt_label = "UI" if gt else "NON-UI"
        pred_label = "UI" if pred else ("NON-UI" if pred is not None else "?")
        print(f"  [{status}] {r['case_id']}")
        print(f"         GT={gt_label}  Pred={pred_label}  | {r['task']}")

    print()
    print("  --- Classification Metrics ---")
    print(
        f"  Accuracy:  {m['accuracy']:.2%}  ({m['tp'] + m['tn']}/{m['tp'] + m['fp'] + m['tn'] + m['fn']})"
    )
    print(
        f"  Precision: {m['precision']:.2%}  ({m['tp']}/{m['tp'] + m['fp']})"
        if (m["tp"] + m["fp"]) > 0
        else "  Precision: N/A"
    )
    print(
        f"  Recall:    {m['recall']:.2%}  ({m['tp']}/{m['tp'] + m['fn']})"
        if (m["tp"] + m["fn"]) > 0
        else "  Recall: N/A"
    )
    print(f"  F1 Score:  {m['f1']:.2%}")
    print(f"  Confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"  Rubric avg: {results['metrics']['rubric']['avg_score']:.2f}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "error_filtering_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Full results (with model responses) saved to {output_path}")


if __name__ == "__main__":
    main()
