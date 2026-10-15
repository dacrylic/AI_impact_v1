"""Package the approved classifier artifact for SageMaker.

The SageMaker model archive must contain ``model.joblib`` at its root. This
utility validates the optional release manifest and creates that archive
without including source data, reports, or repository files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/gpt52-current-basic-v2.joblib")
    parser.add_argument("--manifest", default="models/gpt52-current-basic-v2.manifest.json")
    parser.add_argument("--output", default="/tmp/ai-impact-model.tar.gz")
    args = parser.parse_args()

    model = Path(args.model)
    manifest = Path(args.manifest)
    output = Path(args.output)
    if not model.is_file():
        raise SystemExit(f"Model artifact not found: {model}")
    if not manifest.is_file():
        raise SystemExit(f"Release manifest not found: {manifest}")

    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    actual_hash = sha256(model)
    expected_hash = metadata.get("sha256")
    if actual_hash != expected_hash:
        raise SystemExit(f"Artifact checksum mismatch: expected {expected_hash}, got {actual_hash}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        archive.add(model, arcname="model.joblib")
    print(json.dumps({"archive": str(output), "modelVersion": metadata.get("modelVersion"), "sha256": actual_hash}, indent=2))


if __name__ == "__main__":
    main()
