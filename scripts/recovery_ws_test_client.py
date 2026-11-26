from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from scripts.recovery_ws_client import ClientConfig, RecoveryWsRunner
from scripts.recovery_ws_client.payloads import load_payload_by_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone recovery websocket test client"
    )
    _ = parser.add_argument("--ws-url", required=True)
    _ = parser.add_argument("--robot-key", required=True)
    _ = parser.add_argument("--agent-id", default=None)
    _ = parser.add_argument("--payload-id", type=int, required=True)
    _ = parser.add_argument(
        "--payload-file",
        default="evals/data/payloads.json",
    )
    _ = parser.add_argument("--action-delay-seconds", type=float, default=1.0)
    _ = parser.add_argument("--max-actions", type=int, default=None)
    _ = parser.add_argument("--dry-run-actions", action="store_true")
    _ = parser.add_argument("--log-json", default=None)
    _ = parser.add_argument("--save-screenshots-dir", default=None)
    _ = parser.add_argument(
        "--capabilities",
        default=None,
        help=(
            "JSON object sent as the payload's capabilities, e.g. "
            '\'{"modalities": ["vision"], "ocr": false}\'; replaces any in the payload'
        ),
    )
    _ = parser.add_argument(
        "--eval", action="store_true", help="Enable evaluation mode"
    )
    _ = parser.add_argument(
        "--ground-truth-file",
        default="evals/data/ground_truth.json",
        help="Ground truth JSON file for eval mode",
    )
    return parser


def load_recovery_task(ground_truth_file: Path, payload_id: int) -> str:
    if not ground_truth_file.exists():
        return ""
    try:
        raw = json.loads(ground_truth_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    if payload_id != -1:
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list) or payload_id < 1 or payload_id > len(raw):
            return ""
        entry = raw[payload_id - 1]
    else:
        entry = raw
    levels = entry.get("ground_truth_levels", {})
    return levels.get("moderate", "")


def _with_capabilities(
    parser: argparse.ArgumentParser, payload: dict[str, Any], capabilities_json: str
) -> dict[str, Any]:
    """`payload` with its capabilities replaced by the `--capabilities` object."""
    try:
        capabilities = json.loads(capabilities_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--capabilities is not valid JSON: {exc.msg}")
    if not isinstance(capabilities, dict):
        parser.error("--capabilities must be a JSON object")
    return payload | {"capabilities": capabilities}


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def build_config(args: argparse.Namespace) -> ClientConfig:
    ground_truth_file = Path(args.ground_truth_file)
    return ClientConfig(
        ws_url=args.ws_url,
        robot_key=args.robot_key,
        agent_id=args.agent_id,
        payload_file=Path(args.payload_file),
        payload_id=args.payload_id,
        action_delay_seconds=args.action_delay_seconds,
        max_actions=args.max_actions,
        dry_run_actions=args.dry_run_actions,
        log_json=_optional_path(args.log_json),
        save_screenshots_dir=_optional_path(args.save_screenshots_dir),
        recovery_task=load_recovery_task(ground_truth_file, args.payload_id),
        eval=args.eval,
        ground_truth_file=ground_truth_file,
    )


def run_output(runner: RecoveryWsRunner) -> dict[str, Any]:
    """What the run prints: its summary and how it ended."""
    summary = runner.transcript.summary
    output: dict[str, Any] = {
        "terminal": summary.terminal_type,
        "actions_received": summary.actions_received,
        "actions_success": summary.actions_success,
        "actions_failed": summary.actions_failed,
        "recovery_id": summary.recovery_id,
    }
    if summary.terminal_type == "done" and runner.eval_result is None:
        # The `done` message's content: success, process_finished,
        # continue_from_model_event_id, loop_iteration and variables.
        output["done"] = runner.terminal_content
    if summary.terminal_type == "error":
        output["error"] = runner.terminal_content
    if runner.eval_result is not None:
        output["eval_result"] = runner.eval_result
    return output


def write_log(log_json: Path, runner: RecoveryWsRunner) -> None:
    log_transcript = runner.transcript.to_dict()
    log_transcript["eval_result"] = runner.eval_result
    log_json.parent.mkdir(parents=True, exist_ok=True)
    _ = log_json.write_text(json.dumps(log_transcript, indent=2), encoding="utf-8")


def run_client() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.action_delay_seconds < 0:
        parser.error("--action-delay-seconds must be >= 0")

    try:
        payload = load_payload_by_id(
            payload_file=Path(args.payload_file), payload_id=args.payload_id
        )
    except ValueError as exc:
        print(f"Payload error: {exc}", file=sys.stderr)
        return 2

    if args.capabilities is not None:
        payload = _with_capabilities(parser, payload, args.capabilities)

    config = build_config(args)
    runner = RecoveryWsRunner(config=config, payload=payload)
    try:
        status = asyncio.run(runner.run())
    except KeyboardInterrupt:
        # The runner already sent `abort`; the server stops the recovery.
        status = 130

    print(json.dumps(run_output(runner), indent=2))
    if config.log_json is not None:
        write_log(config.log_json, runner)
    return status


def main() -> int:
    try:
        return run_client()
    except Exception as _:  # noqa: BLE001
        print(f"Error: {traceback.format_exc()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    status = main()
    print(f"Program exited with status: {status}")
