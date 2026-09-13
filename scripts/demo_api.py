#!/usr/bin/env python3
"""Call the deployed prediction API using an API key from the environment."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("AI_IMPACT_API_URL", "https://morning-beach-71824-d79c695b33eb.herokuapp.com"))
    parser.add_argument("--api-key-env", default="AI_IMPACT_API_KEY")
    parser.add_argument("--keytask-content")
    parser.add_argument("--action")
    parser.add_argument("--object", dest="object_")
    parser.add_argument("--purpose")
    parser.add_argument("--jobrole-title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set ${args.api_key_env} before calling the API; the key is intentionally not accepted as a CLI argument.")
    payload: dict[str, Any] = {}
    if args.keytask_content:
        payload["keytaskContent"] = args.keytask_content
    else:
        if not args.action or not args.object_:
            raise SystemExit("Provide --keytask-content, or both --action and --object.")
        payload.update({"action": args.action, "object": args.object_})
        if args.purpose is not None:
            payload["purpose"] = args.purpose
    if args.jobrole_title is not None:
        payload["jobroleTitle"] = args.jobrole_title
    request = Request(
        args.base_url.rstrip("/") + "/v1/predict",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            print(json.dumps(json.loads(response.read()), indent=2))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API returned HTTP {error.code}: {detail}") from error


if __name__ == "__main__":
    main()
