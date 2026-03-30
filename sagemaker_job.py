"""
sagemaker_job.py — AWS SageMaker Training Job Launcher
AI-Based Accent Transcribing System | University of Sharjah SDP
Author: Ammar

Launches a SageMaker training job that runs train.py on the cloud.
Run this script locally (or from SageMaker Studio) to kick off training.

Prerequisites:
    pip install sagemaker boto3

    Your SageMaker execution role must have:
      - AmazonSageMakerFullAccess
      - AmazonS3FullAccess (for your data bucket)

Usage:
    python sagemaker_job.py \
        --s3_data  s3://your-bucket/accent-data/manifests/ \
        --s3_output s3://your-bucket/accent-data/checkpoints/ \
        --role      arn:aws:iam::123456789:role/SageMakerRole \
        --debug

Tip: In SageMaker Studio, role and region are auto-detected.
    Just run:  python sagemaker_job.py --s3_data s3://... --s3_output s3://...
"""

import argparse
import logging
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────
# INSTANCE SELECTION
# ──────────────────────────────────────────────
# University SageMaker provides GPU instances.
# ml.g4dn.xlarge = 1x T4 GPU, 16 GB VRAM — cheapest GPU option, good for Whisper-small.
# ml.g4dn.2xlarge = 1x T4 GPU, 32 GB RAM — use if OOM on xlarge.
TRAIN_INSTANCE  = "ml.g4dn.xlarge"
DEBUG_INSTANCE  = "ml.t3.medium"   # CPU only — for smoke tests, very cheap


# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
# These are passed as CLI args to train.py inside the container.
DEFAULT_HYPERPARAMS = {
    "model":       "openai/whisper-small",
    "epochs":      "10",
    "batch_size":  "8",
    "lr":          "1e-5",
    "warmup_steps":"100",
    "grad_accum":  "4",
    "num_workers": "2",
}

DEBUG_HYPERPARAMS = {
    **DEFAULT_HYPERPARAMS,
    "epochs":      "2",
    "batch_size":  "4",
    "debug":       "",   # flag — presence enables debug mode in train.py
}


# ──────────────────────────────────────────────
# LAUNCH JOB
# ──────────────────────────────────────────────
def launch_job(args):
    # ── Session and role ───────────────────────
    session = sagemaker.Session()
    region  = boto3.Session().region_name

    if args.role:
        role = args.role
    else:
        role = get_execution_role()   # auto-detects when running inside SageMaker Studio
        log.info(f"Auto-detected role: {role}")

    log.info(f"Region  : {region}")
    log.info(f"Role    : {role}")
    log.info(f"S3 data : {args.s3_data}")
    log.info(f"S3 out  : {args.s3_output}")
    log.info(f"Debug   : {args.debug}")

    # ── Hyperparameters ────────────────────────
    hyperparams = DEBUG_HYPERPARAMS if args.debug else DEFAULT_HYPERPARAMS

    # ── Instance type ──────────────────────────
    instance_type = DEBUG_INSTANCE if args.debug else TRAIN_INSTANCE

    # ── Job name ──────────────────────────────
    from datetime import datetime
    suffix    = datetime.now().strftime("%Y%m%d-%H%M")
    mode      = "debug" if args.debug else "full"
    job_name  = f"accent-whisper-{mode}-{suffix}"

    log.info(f"Job name: {job_name}")
    log.info(f"Instance: {instance_type}")

    # ── PyTorch Estimator ──────────────────────
    estimator = PyTorch(
        entry_point="train.py",
        source_dir=".",                         # uploads the entire phase3/ folder
        role=role,
        framework_version="2.1",
        py_version="py310",
        instance_count=1,
        instance_type=instance_type,
        output_path=args.s3_output,
        hyperparameters=hyperparams,
        base_job_name=job_name,
        sagemaker_session=session,

        # Environment variables — useful for HuggingFace cache
        environment={
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
            "HF_DATASETS_CACHE":  "/tmp/hf_datasets",
        },

        # Keep training logs visible in CloudWatch
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {"Name": "train:loss",   "Regex": r"Train loss: ([0-9\.]+)"},
            {"Name": "val:loss",     "Regex": r"Val loss\s+: ([0-9\.]+)"},
            {"Name": "val:wer",      "Regex": r"WER \(overall\): ([0-9\.]+)"},
        ],
    )

    # ── Data channels ─────────────────────────
    # SageMaker downloads S3 data to /opt/ml/input/data/<channel>/
    # train.py reads SM_CHANNEL_MANIFEST env var → points to this folder.
    data_channels = {
        "manifest": sagemaker.inputs.TrainingInput(
            s3_data=args.s3_data,
            content_type="text/csv",
        ),
    }

    # ── Fit (launch) ──────────────────────────
    log.info(f"\nLaunching training job: {job_name}")
    estimator.fit(
        inputs=data_channels,
        job_name=job_name,
        wait=args.wait,      # set to False to return immediately (async)
        logs="All",
    )

    if args.wait:
        log.info(f"\nJob complete. Model artifacts uploaded to: {args.s3_output}")
        log.info(f"To evaluate: download checkpoints/best/ and run evaluate.py")

    return job_name


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")

    parser.add_argument("--s3_data",   required=True,
                        help="S3 URI for manifest CSV  e.g. s3://my-bucket/data/manifests/")
    parser.add_argument("--s3_output", required=True,
                        help="S3 URI for model output  e.g. s3://my-bucket/checkpoints/")
    parser.add_argument("--role",      default=None,
                        help="IAM role ARN (auto-detected inside SageMaker Studio)")
    parser.add_argument("--debug",     action="store_true",
                        help="Use CPU instance + 2 epochs for a quick smoke test")
    parser.add_argument("--wait",      action="store_true",
                        help="Block until job completes (otherwise async)")

    args = parser.parse_args()
    launch_job(args)
