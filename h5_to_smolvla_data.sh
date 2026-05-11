python h5_to_lerobot_smolvla.py \
  --input passover-0426 \
  --pattern "align_v*_*.h5" \
  --output ./lerobot_smolvla_dataset \
  --repo-id your_hf_name/my_smolvla_dataset \
  --task "your task instruction here" \
  --fps 30 \
  --overwrite

