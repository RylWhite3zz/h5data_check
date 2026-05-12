import argparse
import io
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


CAMERAS = ("left", "right", "front")
TASK_RANGES = (
    (
        0,
        19,
        "Pick up the banana with the left hand, hand it to the right, and place it in the purple cup.",
    ),
    (
        20,
        39,
        "Pick up the banana with the left hand, hand it to the right, and place it in the brown cup.",
    ),
    (
        40,
        49,
        "Pick up the banana with the left hand, hand it to the right, and place it in the blue cup.",
    ),
    (
        50,
        59,
        "Pick up the banana with the left hand, hand it to the right, and place it in the green cup.",
    ),
)


def import_lerobot_dataset():
    # Current LeRobot rejects the deprecated LEROBOT_HOME variable during import.
    os.environ.pop("LEROBOT_HOME", None)

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def decode_compressed_image(root, camera_name, frame_index, offsets):
    base = f"obs/image/{camera_name}"
    start = int(offsets[camera_name][frame_index])
    end = int(offsets[camera_name][frame_index + 1])
    encoded = np.asarray(root[f"{base}/raw"][start:end], dtype=np.uint8)

    try:
        image = Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {camera_name} frame {frame_index}") from exc

    # Match visualize.py and h5_to_3camera_mp4.py: this dataset displays
    # correctly when OpenCV-decoded bytes are interpreted as RGB.
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[..., ::-1])


def image_offsets(root, cameras):
    offsets = {}
    for camera_name in cameras:
        lengths = np.asarray(root[f"obs/image/{camera_name}/len"][:], dtype=np.int64)
        offsets[camera_name] = np.concatenate([[0], np.cumsum(lengths)])
    return offsets


def episode_length(root, cameras):
    image_count = min(int(root[f"obs/image/{camera_name}/cnt"][0]) for camera_name in cameras)
    qpos_count = min(
        int(root["act/qpos/left/cnt"][0]),
        int(root["act/qpos/right/cnt"][0]),
    )
    return min(image_count, qpos_count)


def load_qpos(root, length):
    left = np.asarray(root["act/qpos/left/raw"][:length], dtype=np.float32)
    right = np.asarray(root["act/qpos/right/raw"][:length], dtype=np.float32)
    return np.concatenate([left, right], axis=1)


def build_features(first_h5, cameras, use_videos):
    image_dtype = "video" if use_videos else "image"

    with h5py.File(first_h5, "r") as root:
        offsets = image_offsets(root, cameras)
        qpos = load_qpos(root, 1)
        left_dim = int(root["act/qpos/left/raw"].shape[1])
        right_dim = int(root["act/qpos/right/raw"].shape[1])
        joint_names = [f"left_joint_{i}" for i in range(left_dim)] + [
            f"right_joint_{i}" for i in range(right_dim)
        ]

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (int(qpos.shape[1]),),
                "names": joint_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (int(qpos.shape[1]),),
                "names": joint_names,
            },
        }

        for camera_name in cameras:
            image = decode_compressed_image(root, camera_name, 0, offsets)
            features[f"observation.images.{camera_name}"] = {
                "dtype": image_dtype,
                "shape": tuple(int(v) for v in image.shape),
                "names": ["height", "width", "channels"],
            }

    return features


def iter_h5_files(input_path, pattern):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob(pattern))
    return [path for path in files if path.suffix.lower() in {".h5", ".hdf5"}]


def episode_id_from_h5(h5_path):
    matches = re.findall(r"\d+", h5_path.stem)
    if not matches:
        raise ValueError(f"Cannot infer episode id from file name: {h5_path.name}")
    return int(matches[-1])


def task_for_h5(h5_path, default_task=None):
    episode_id = episode_id_from_h5(h5_path)
    for start, end, task in TASK_RANGES:
        if start <= episode_id <= end:
            return task

    if default_task is not None:
        return default_task

    raise ValueError(
        f"No task configured for episode id {episode_id} from {h5_path.name}. "
        "Add it to TASK_RANGES or pass --task as a fallback."
    )


def convert_episode(dataset, h5_path, task, cameras, action_mode, max_frames=None):
    with h5py.File(h5_path, "r") as root:
        offsets = image_offsets(root, cameras)
        length = episode_length(root, cameras)
        if max_frames is not None:
            length = min(length, max_frames)

        qpos = load_qpos(root, length)

        for frame_index in range(length):
            if action_mode == "next":
                action_index = min(frame_index + 1, length - 1)
            else:
                action_index = frame_index

            frame = {
                "observation.state": qpos[frame_index],
                "action": qpos[action_index],
                "task": task,
            }

            for camera_name in cameras:
                frame[f"observation.images.{camera_name}"] = decode_compressed_image(
                    root, camera_name, frame_index, offsets
                )

            dataset.add_frame(frame)

    dataset.save_episode()
    return length


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert compressed robot HDF5 episodes to a LeRobot dataset for SmolVLA."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Input H5 file or directory containing H5/HDF5 files.",
    )
    parser.add_argument(
        "--pattern",
        default="align_v0_*.h5",
        help="Glob pattern used when --input is a directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output LeRobot dataset directory.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Dataset repo id, for example your_hf_name/my_smolvla_dataset.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Fallback natural language task instruction for files outside TASK_RANGES.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="custom_bimanual")
    parser.add_argument(
        "--action-mode",
        choices=("current", "next"),
        default="current",
        help="Use qpos[t] or qpos[t+1] as the action target.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Store images instead of MP4 videos. SmolVLA usually uses videos.",
    )
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    h5_files = iter_h5_files(args.input, args.pattern)
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found from {args.input} with {args.pattern}")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output)

    LeRobotDataset = import_lerobot_dataset()
    features = build_features(h5_files[0], CAMERAS, use_videos=not args.no_videos)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=not args.no_videos,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )

    total_frames = 0
    for episode_index, h5_path in enumerate(h5_files):
        task = task_for_h5(h5_path, args.task)
        length = convert_episode(
            dataset,
            h5_path,
            task=task,
            cameras=CAMERAS,
            action_mode=args.action_mode,
            max_frames=args.max_frames,
        )
        total_frames += length
        print(f"saved episode {episode_index}: {h5_path.name}, frames={length}, task={task!r}")

    dataset.finalize()
    print(f"converted {len(h5_files)} episodes, total_frames={total_frames}")
    print(f"output: {args.output}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["lerobot", "smolvla", "robotics"],
            license="apache-2.0",
            push_videos=not args.no_videos,
        )


if __name__ == "__main__":
    main()
