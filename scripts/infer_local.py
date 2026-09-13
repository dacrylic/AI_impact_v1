#!/usr/bin/env python3
"""Run one prediction against a local release artifact."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from ai_impact_classifier.production.predictor import ProductionPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("AI_IMPACT_MODEL_PATH", "artifacts/current/model.joblib"))
    parser.add_argument("--keytask-content", help="Already-composed task text.")
    parser.add_argument("--action")
    parser.add_argument("--object", dest="object_")
    parser.add_argument("--purpose")
    parser.add_argument("--jobrole-title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.keytask_content
    if not task:
        if not args.action or not args.object_:
            raise SystemExit("Provide --keytask-content, or both --action and --object.")
        task = " ".join(part for part in (args.action, args.object_, args.purpose) if part)
    predictor = ProductionPredictor.from_path(Path(args.model))
    prediction = predictor.predict(task, args.jobrole_title, args.action, args.object_, args.purpose)
    print(json.dumps({
        "predictedLabel": prediction.predicted_label,
        "reviewScore": prediction.review_score,
        "modelVersion": prediction.model_version,
    }, indent=2))


if __name__ == "__main__":
    main()
