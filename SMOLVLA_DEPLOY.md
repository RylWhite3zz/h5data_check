# SmolVLA remote deployment

`deploy_smolvla.py` is a two-role script:

- `--role server`: run on the GPU desktop. It loads the SmolVLA/LoRA checkpoint and serves action chunks over TCP.
- `--role robot`: run on the ROS machine attached to the arm. It subscribes to camera and joint topics, sends observations to the server, and publishes 14-D actions as left/right `JointState` commands.

The action/state order matches the converted dataset and `infer.py`:

```text
[left_joint_0..6, right_joint_0..6]
```

## 1. Start the server on the GPU desktop

Use either the training output directory or the `pretrained_model` directory. The script auto-resolves `checkpoints/last/pretrained_model` when present.

3-camera model:

```bash
python deploy_smolvla.py \
  --role server \
  --model-path ../../model/outputs/train/smolvla_test_3cam \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

2-camera model:

```bash
python deploy_smolvla.py \
  --role server \
  --model-path ../../model/outputs/train/smolvla_test_2cam \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

If the 2-camera checkpoint config did not save `empty_cameras=1`, add:

```bash
--empty-cameras 1
```

For LoRA checkpoints, the server environment must have `peft` installed and must be able to load the base model referenced by `adapter_config.json` (usually `lerobot/smolvla_base`). If that path is different on the deployment desktop, add `--base-model-path lerobot/smolvla_base` or point it to a local base model directory.

## 2. Start the robot bridge on the ROS machine

Keep the existing robot startup steps unchanged:

```bash
# terminal 1
roscore

# terminal 2
cd /home/agilex/cobot_magic/camera_ws
source devel/setup.bash
roslaunch astra_camera multi_camera_yann.launch

# terminal 3
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
./can_muti_activate.sh
source devel/setup.bash
roslaunch piper start_ms_piper_yann.launch mode:=1 auto_enable:=true
```

Then replace the original step 5 `python yann_rospy_control.py` with this deployment script. It publishes the same `sensor_msgs/JointState` command format to `/master/joint_left` and `/master/joint_right`, while reading feedback from `/puppet/joint_left` and `/puppet/joint_right`.

3-camera deployment:

```bash
cd /home/agilex/cobot_magic/collect_data
python deploy_smolvla.py \
  --role robot \
  --server-host <GPU_DESKTOP_IP> \
  --camera-mode 3cam \
  --task "your task instruction here" \
  --publish-rate 30
```

2-camera deployment:

```bash
cd /home/agilex/cobot_magic/collect_data
python deploy_smolvla.py \
  --role robot \
  --server-host <GPU_DESKTOP_IP> \
  --camera-mode 2cam \
  --task "your task instruction here" \
  --publish-rate 30
```

Default ROS topics:

```text
left image:        /camera_l/color/image_raw
right image:       /camera_r/color/image_raw
front image:       /camera_f/color/image_raw
left state:        /puppet/joint_left
right state:       /puppet/joint_right
left command:      /master/joint_left
right command:     /master/joint_right
```

Override them with `--img-left-topic`, `--img-right-topic`, `--img-front-topic`, `--puppet-arm-left-topic`, `--puppet-arm-right-topic`, `--puppet-arm-left-cmd-topic`, and `--puppet-arm-right-cmd-topic`.

If the robot has more physical cameras than the model uses, keep `--camera-mode` as the model mode and map each model input slot to the desired physical camera topic:

```bash
python deploy_smolvla.py \
  --role robot \
  --server-host <GPU_DESKTOP_IP> \
  --camera-mode 3cam \
  --camera-topic-map 'left=/camera_l/color/image_raw,right=/camera_r/color/image_raw,front=/camera_e/color/image_raw'
```

This example feeds the new `camera_e` stream into the model's `front` input slot. The model still receives `left/right/front`; only their ROS sources change.

If the camera topics follow the usual `/<node>/color/image_raw` convention, you can map by node/prefix instead:

```bash
python deploy_smolvla.py \
  --role robot \
  --server-host <GPU_DESKTOP_IP> \
  --camera-mode 3cam \
  --camera-node-map 'left=camera_l,right=camera_r,front=camera_e'
```

For a 2-camera checkpoint, only map the slots that are actually used, normally `left` and `right`.

## Safety switches

Start with command publishing disabled:

```bash
cd /home/agilex/cobot_magic/collect_data
python deploy_smolvla.py --role robot --server-host <GPU_DESKTOP_IP> --camera-mode 3cam --dry-run --verbose
```

By default, each published action is delta-limited with:

```text
0.01,0.01,0.01,0.01,0.01,0.01,0.2
```

This 7-D limit is applied to both arms. Override with `--max-joint-step`, or disable it with `--disable-step-limit` after verifying behavior.

## Runtime behavior

The robot side is asynchronous. It keeps a local action queue and asks the server for a fresh SmolVLA action chunk when the queue drops below `--queue-low-watermark`. New chunks replace older queued actions by default; use `--no-replace-action-queue` to append instead.

Optional temporal aggregation can be enabled on the robot side:

```bash
python deploy_smolvla.py \
  --role robot \
  --server-host <GPU_DESKTOP_IP> \
  --camera-mode 3cam \
  --temporal-agg \
  --temporal-agg-query-interval 5 \
  --temporal-agg-k 0.01
```

With `--temporal-agg`, the GPU still returns normal action chunks. The robot side aligns overlapping chunks by publish step and computes an exponential weighted average for the current step. `--temporal-agg-k` controls how strongly older predictions are down-weighted; larger values favor newer chunks more. `--temporal-agg-query-interval` controls how often to request a fresh overlapping chunk.

The server sends unnormalized 14-D joint targets. Normalization and camera key renaming are loaded from the saved LeRobot processors in the checkpoint. If a checkpoint has no saved processors, pass:

```bash
--dataset-stats ./lerobot_smolvla_dataset/meta/stats.json
```
