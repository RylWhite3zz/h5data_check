import argparse
import inspect
import io
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


CAMERA_TO_OPENPI_KEY = {
    "front": "observation.images.top",
    "left": "observation.images.left_wrist",
    "right": "observation.images.right_wrist",
}


def import_lerobot_dataset(output_root):
    # Current LeRobot rejects the deprecated LEROBOT_HOME variable during import.
    # Keep only HF_LEROBOT_HOME for OpenPI/LeRobot dataset discovery.
    os.environ.pop("LEROBOT_HOME", None)
    os.environ["HF_LEROBOT_HOME"] = str(output_root.expanduser().resolve())

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets import LeRobotDataset

    return LeRobotDataset


def call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return fn(**kwargs)

    supported = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return fn(**supported)


def decode_compressed_image(root, camera_name, frame_index, offsets, resize=None):
    base = f"obs/image/{camera_name}"
    start = int(offsets[camera_name][frame_index])
    end = int(offsets[camera_name][frame_index + 1])
    encoded = np.asarray(root[f"{base}/raw"][start:end], dtype=np.uint8)

    try:
        image = Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {camera_name} frame {frame_index}") from exc

    if resize is not None:
        image = image.resize(resize, resample=RESAMPLE_BICUBIC)

    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


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


def qpos_names(root):
    left_dim = int(root["act/qpos/left/raw"].shape[1])
    right_dim = int(root["act/qpos/right/raw"].shape[1])
    return [f"left_joint_{i}" for i in range(left_dim)] + [
        f"right_joint_{i}" for i in range(right_dim)
    ]


def build_features(first_h5, cameras, image_dtype, resize):
    with h5py.File(first_h5, "r") as root:
        offsets = image_offsets(root, cameras)
        qpos = load_qpos(root, 1)
        names = qpos_names(root)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (int(qpos.shape[1]),),
                "names": names,
            },
            "action": {
                "dtype": "float32",
                "shape": (int(qpos.shape[1]),),
                "names": names,
            },
        }

        for camera_name in cameras:
            image = decode_compressed_image(root, camera_name, 0, offsets, resize=resize)
            features[CAMERA_TO_OPENPI_KEY[camera_name]] = {
                "dtype": image_dtype,
                "shape": tuple(int(v) for v in image.shape),
                "names": ["height", "width", "channel"],
            }

    return features


def iter_h5_files(input_path, pattern):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.glob(pattern))
    return [path for path in files if path.suffix.lower() in {".h5", ".hdf5"}]


def make_actions(qpos, action_mode):
    if action_mode == "current":
        return qpos, qpos

    if qpos.shape[0] < 2:
        raise ValueError("Need at least 2 frames for next/delta action modes")

    states = qpos[:-1]
    next_qpos = qpos[1:]
    if action_mode == "next":
        return states, next_qpos
    if action_mode == "delta":
        return states, next_qpos - states

    raise ValueError(f"Unsupported action mode: {action_mode}")


def load_task_map(path):
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--task-json must be a JSON object")
    return {str(key): str(value) for key, value in data.items()}


def task_for_h5(h5_path, default_task, task_map):
    return task_map.get(h5_path.name) or task_map.get(h5_path.stem) or default_task


def convert_episode(dataset, h5_path, task, cameras, action_mode, resize, max_frames=None):
    with h5py.File(h5_path, "r") as root:
        offsets = image_offsets(root, cameras)
        length = episode_length(root, cameras)
        if max_frames is not None:
            length = min(length, max_frames)

        qpos = load_qpos(root, length)
        states, actions = make_actions(qpos, action_mode)

        for frame_index, (state, action) in enumerate(zip(states, actions)):
            frame = {
                "observation.state": state.astype(np.float32, copy=False),
                "action": action.astype(np.float32, copy=False),
                "task": task,
            }

            for camera_name in cameras:
                frame[CAMERA_TO_OPENPI_KEY[camera_name]] = decode_compressed_image(
                    root, camera_name, frame_index, offsets, resize=resize
                )

            dataset.add_frame(frame)

    dataset.save_episode()
    return len(states)


def finish_dataset(dataset):
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert compressed bimanual HDF5 episodes to an OpenPI-compatible LeRobot dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Input H5 file or directory containing H5/HDF5 files.",
    )
    parser.add_argument(
        "--pattern",
        default="align_v*_*.h5",
        help="Glob pattern used when --input is a directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./lerobot_openpi"),
        help="LeRobot root. Dataset is saved under output-root/repo-id.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Dataset repo id, for example your_hf_name/my_openpi_dataset.",
    )
    parser.add_argument("--task", required=True, help="Default language instruction.")
    parser.add_argument(
        "--task-json",
        type=Path,
        default=None,
        help="Optional JSON mapping H5 file name or stem to task instruction.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="custom_bimanual")
    parser.add_argument(
        "--action-mode",
        choices=("current", "next", "delta"),
        default="next",
        help=(
            "current: action=qpos[t]; next: action=qpos[t+1]; "
            "delta: action=qpos[t+1]-qpos[t]."
        ),
    )
    parser.add_argument(
        "--image-dtype",
        choices=("image", "video"),
        default="image",
        help="OpenPI examples use image. Use video only if your LeRobot version expects it.",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional fixed image resize before writing.",
    )
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cameras = tuple(CAMERA_TO_OPENPI_KEY)
    h5_files = iter_h5_files(args.input, args.pattern)
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found from {args.input} with {args.pattern}")

    dataset_path = args.output_root / args.repo_id
    if dataset_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_path} exists. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    resize = tuple(args.resize) if args.resize is not None else None
    task_map = load_task_map(args.task_json)

    LeRobotDataset = import_lerobot_dataset(args.output_root)
    features = build_features(h5_files[0], cameras, args.image_dtype, resize)

    dataset = call_with_supported_kwargs(
        LeRobotDataset.create,
        repo_id=args.repo_id,
        root=dataset_path,
        robot_type=args.robot_type,
        fps=args.fps,
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        use_videos=args.image_dtype == "video",
    )

    total_frames = 0
    for episode_index, h5_path in enumerate(h5_files):
        task = task_for_h5(h5_path, args.task, task_map)
        length = convert_episode(
            dataset,
            h5_path,
            task=task,
            cameras=cameras,
            action_mode=args.action_mode,
            resize=resize,
            max_frames=args.max_frames,
        )
        total_frames += length
        print(f"saved episode {episode_index}: {h5_path.name}, frames={length}, task={task!r}")

    finish_dataset(dataset)
    print(f"converted {len(h5_files)} episodes, total_frames={total_frames}")
    print(f"output: {dataset_path}")

    if args.push_to_hub:
        call_with_supported_kwargs(
            dataset.push_to_hub,
            tags=["openpi", "lerobot", "robotics"],
            private=args.private,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main()
