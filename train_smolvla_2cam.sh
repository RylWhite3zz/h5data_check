CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/test_data_smolvla \
  --dataset.root=/media/raid/workspace/zhaoyanpeng/data/transfer_center_wyf/test_data_smolvla \
  --batch_size=25 \
  --steps=20000 \
  --output_dir=../../model/outputs/train/smolvla_test_2cam \
  --job_name=smolvla_my_data \
  --policy.device=cuda \
  --wandb.enable=false \
  --peft.method_type=LORA \
  --peft.r=64 \
  --save_freq=1000 \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.left": "observation.images.camera1","observation.images.right": "observation.images.camera2"}' \
  --policy.empty_cameras=1
