"""Deploy a packaged production artifact to a SageMaker scikit-learn endpoint."""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-data", required=True, help="S3 URI to model.tar.gz containing model.joblib at archive root.")
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--region")
    parser.add_argument("--framework-version", default="1.5-1", help="Approved SageMaker scikit-learn container tag; verify it matches the artifact runtime.")
    args = parser.parse_args()

    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel

    session = sagemaker.Session(boto_session=None if args.region is None else __import__("boto3").Session(region_name=args.region))
    # The SageMaker sklearn image does not know this repository package. Stage
    # the exact production package beside the handler so unpickling is reliable.
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="ai-impact-sagemaker-") as staging:
        staging_path = Path(staging)
        shutil.copy2(Path(__file__).with_name("inference.py"), staging_path / "inference.py")
        shutil.copytree(repo_root / "src" / "ai_impact_classifier", staging_path / "ai_impact_classifier")
        model = SKLearnModel(
            model_data=args.model_data,
            role=args.role,
            entry_point="inference.py",
            source_dir=str(staging_path),
            framework_version=args.framework_version,
            py_version="py3",
            sagemaker_session=session,
        )
        predictor = model.deploy(initial_instance_count=1, instance_type=args.instance_type, endpoint_name=args.endpoint_name)
    print(f"Deployed endpoint: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
