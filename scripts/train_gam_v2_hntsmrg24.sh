#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 DATA_ROOT [GPU_ID] [RUN_DIR] [BASELINE_CKPT]" >&2
  exit 2
fi

DATA_ROOT=$1
GPU_ID=${2:-0}
RUN_DIR=${3:-runs/gam_sacb_minimal_v2_hntsmrg24_seed2026}
BASELINE_CKPT=${4:-}
TRAIN_MANIFEST="${DATA_ROOT}/manifests/train.csv"
VALIDATION_MANIFEST="${DATA_ROOT}/manifests/validation.csv"

for required_path in \
  "${DATA_ROOT}" \
  "${TRAIN_MANIFEST}" \
  "${VALIDATION_MANIFEST}"
do
  if [[ ! -e "${required_path}" ]]; then
    echo "missing required path: ${required_path}" >&2
    exit 1
  fi
done

arguments=(
  python train_gam.py
  --config configs/gam_sacb_hntsmrg24.json
  --data-root "${DATA_ROOT}"
  --train-manifest "${TRAIN_MANIFEST}"
  --validation-manifest "${VALIDATION_MANIFEST}"
  --output-dir "${RUN_DIR}"
  --device cuda:0
)

if [[ -n "${BASELINE_CKPT}" ]]; then
  if [[ ! -f "${BASELINE_CKPT}" ]]; then
    echo "missing baseline checkpoint: ${BASELINE_CKPT}" >&2
    exit 1
  fi
  arguments+=(--baseline-checkpoint "${BASELINE_CKPT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${arguments[@]}"
