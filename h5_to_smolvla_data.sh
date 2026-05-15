#!/usr/bin/env bash
set -euo pipefail

CENTER_CAMERA="${CENTER_CAMERA:-front}"

python h5_to_lerobot_smolvla.py \
  --input /mnt/som/passover-0509 \
  --pattern "align_v0_*.h5" \
  --output ./lerobot_smolvla_dataset \
  --repo-id Dinzhen123/my_smolvla_dataset \
  --task-ranges ./task_ranges_smolvla.json \
  --cameras "${CENTER_CAMERA},left,right" \
  --action-mode next \
  --fps 30 \
  --overwrite
