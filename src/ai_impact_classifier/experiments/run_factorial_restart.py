"""Run the frozen factorial manifest through the Responses API with checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd


LABELS = {"E0", "E1", "E2", "E3"}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reasoning_key", "truncated_reason", "label"], "properties": {"reasoning_key": {"type": "string"}, "truncated_reason": {"type": "string"}, "label": {"type": "string", "enum": sorted(LABELS)}}}


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def digest(system: str, user: str) -> str:
    return hashlib.sha256(json.dumps({"system": system, "user": user}, sort_keys=True).encode()).hexdigest()


def output_text(body: dict) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content["text"]
    # Historical GPT-4 snapshots use the legacy Chat Completions response shape.
    choices = body.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content
    raise ValueError("No output text in Responses API response")


def call(api_key: str, base_url: str, model: str, system: str, user: str, temperature: float, api_mode: str, seed: int | None, legacy_json: bool) -> tuple[dict, str]:
    if api_mode == "legacy_chat":
        payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": temperature}
        if seed is not None:
            payload["seed"] = seed
        if legacy_json:
            payload["response_format"] = {"type": "json_object"}
        endpoint = "/chat/completions"
    elif api_mode == "responses_single_user":
        payload = {"model": model, "input": [{"role": "user", "content": system + "\n\n" + user}], "temperature": temperature}
        endpoint = "/responses"
    else:
        payload = {"model": model, "input": [{"role": "system", "content": system}, {"role": "user", "content": user}], "text": {"format": {"type": "json_schema", "name": "ai_impact_score", "strict": True, "schema": SCHEMA}}, "temperature": temperature}
        endpoint = "/responses"
    request = Request(base_url.rstrip("/") + endpoint, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode()), response.headers.get("x-request-id", "")
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error


def successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["call_id", "status"])
    return set(frame.loc[frame.status.eq("success"), "call_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paper-system", required=True)
    parser.add_argument("--paper-user", required=True)
    parser.add_argument("--production-system", required=True)
    parser.add_argument("--production-user", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--api-mode", choices=["responses", "responses_single_user", "legacy_chat"], default="responses")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--legacy-json", action="store_true")
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing ${args.api_key_env}")
    manifest = pd.read_csv(args.manifest)
    if manifest.call_id.duplicated().any():
        raise ValueError("Manifest has duplicate call IDs")
    legacy_prompt = (read(args.paper_system), read(args.paper_user))
    production_prompt = (read(args.production_system), read(args.production_user))
    prompts = {"paper_2023": legacy_prompt, "legacy": legacy_prompt, "new_production": production_prompt, "production_legacy_anchors": production_prompt}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = successful_ids(path)
    pending = manifest.loc[~manifest.call_id.astype(str).isin(done)].copy()
    if args.limit:
        pending = pending.head(args.limit)

    def run(row: object) -> dict[str, object]:
        system, template = prompts[row.prompt_id]
        user = template.format(occupation=row.occupation, task=row.task_text)
        result: dict[str, object] = {"call_id": row.call_id, "status": "error", "label": "", "resolved_model": "", "gateway_request_id": "", "request_timestamp_utc": datetime.now(timezone.utc).isoformat(), "latency_ms": None, "attempt_count": 0, "prompt_sha256": digest(system, template), "rendered_request_sha256": digest(system, user), "error_type": "", "error_message": ""}
        started = perf_counter()
        for attempt in range(1, args.max_attempts + 1):
            result["attempt_count"] = attempt
            try:
                body, request_id = call(api_key, args.base_url, row.requested_model, system, user, args.temperature, args.api_mode, args.seed, args.legacy_json)
                raw_output = output_text(body)
                try:
                    parsed = json.loads(raw_output)
                except json.JSONDecodeError:
                    parsed = {}
                if "label" not in parsed:
                    label_match = re.search(r"LABEL\s*\(E0/E1/E2/E3\)\s*:\s*\**\s*(E[0-3])\b", raw_output.upper())
                    labels = [label_match.group(1)] if label_match else re.findall(r"^\s*\**\s*(E[0-3])\b", raw_output.upper(), flags=re.MULTILINE)
                    if not labels:
                        raise ValueError(f"Could not parse label from legacy output: {raw_output[:240]}")
                    parsed = {"label": labels[0], "reasoning_key": "", "truncated_reason": raw_output[:500]}
                parsed.setdefault("reasoning_key", "")
                parsed.setdefault("truncated_reason", raw_output[:500])
                if parsed["label"] not in LABELS:
                    raise ValueError(f"Invalid label {parsed['label']}")
                result.update({"status": "success", "label": parsed["label"], "resolved_model": body.get("model", ""), "gateway_request_id": request_id or body.get("id", ""), "reasoning_key": parsed["reasoning_key"], "truncated_reason": parsed["truncated_reason"], "error_type": "", "error_message": ""})
                break
            except Exception as error:
                result["error_type"], result["error_message"] = type(error).__name__, str(error)
                if attempt < args.max_attempts:
                    delay = args.retry_base_seconds * (2 ** (attempt - 1))
                    time.sleep(delay + random.random() * delay / 4)
        result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return result

    rows = ThreadPoolExecutor(max_workers=args.max_workers).map(run, pending.itertuples(index=False))
    for result in rows:
        pd.DataFrame([result]).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(json.dumps({"call_id": result["call_id"], "status": result["status"], "attempt_count": result["attempt_count"]}), flush=True)


if __name__ == "__main__":
    main()
