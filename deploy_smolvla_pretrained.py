#!/usr/bin/env python3
# -- coding: UTF-8
"""
Deploy the official pretrained SmolVLA base checkpoint against the existing ROS
topic bridge.

This is intentionally separate from deploy_smolvla.py. The local fine-tuned
checkpoints in this workspace use a 14-D bimanual state/action layout, while
lerobot/smolvla_base is a generic 6-D base policy. This script adapts the local
14-D observation into the official 6-D policy input, then expands the 6-D output
back into a 14-D command shape so the existing robot bridge can consume it.

The default mode is dry-run. Use --allow-publish only after inspecting the
predicted actions, because the official base model is not calibrated to this
custom bimanual robot.

Example server:
    python deploy_smolvla_pretrained.py --role server --device cuda --camera-mode 3cam

Example robot bridge:
    python deploy_smolvla_pretrained.py \
      --role robot \
      --server-host 192.168.1.20 \
      --camera-mode 3cam \
      --task "Pick up the banana with the left hand, hand it to the right, and place it in the purple cup." \
      --verbose
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

import deploy_smolvla as bridge


OFFICIAL_MODEL_ID = "lerobot/smolvla_base"
OFFICIAL_IMAGE_KEYS = {
    "front": "observation.images.camera1",
    "back": "observation.images.camera1",
    "left": "observation.images.camera2",
    "right": "observation.images.camera3",
}
DEFAULT_STATE_INDICES = "7,8,9,10,11,13"


def parse_indices(value: str, *, source_dim: int, expected_len: int, name: str) -> list[int]:
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    indices = [int(part) for part in parts]
    if len(indices) != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} indices, got {len(indices)}")
    bad = [idx for idx in indices if idx < 0 or idx >= source_dim]
    if bad:
        raise ValueError(f"{name} contains out-of-range indices for {source_dim}-D local state: {bad}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate indices: {indices}")
    return indices


def import_policy_components() -> tuple[Any, Any, Any]:
    try:
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError:
        from lerobot.configs.policies import PreTrainedConfig

    try:
        from lerobot.policies import make_pre_post_processors
    except ImportError:
        from lerobot.policies.factory import make_pre_post_processors

    try:
        from lerobot.policies.smolvla import SmolVLAPolicy
    except ImportError:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return PreTrainedConfig, make_pre_post_processors, SmolVLAPolicy


def set_default_empty_cameras(args: argparse.Namespace) -> None:
    if args.empty_cameras is not None:
        return
    camera_count = len(bridge.camera_names_from_args(args))
    args.empty_cameras = max(0, 3 - camera_count)


class OfficialPretrainedSmolVLAServer:
    def __init__(self, args: argparse.Namespace):
        bridge.maybe_add_lerobot_src(args.lerobot_src)
        self.args = args
        self.device = args.device
        self.pretrained_path = args.model_path or OFFICIAL_MODEL_ID
        self.state_indices = parse_indices(
            args.state_indices,
            source_dim=14,
            expected_len=6,
            name="--state-indices",
        )
        self.action_indices = parse_indices(
            args.action_indices or args.state_indices,
            source_dim=14,
            expected_len=6,
            name="--action-indices",
        )

        import torch

        PreTrainedConfig, make_pre_post_processors, SmolVLAPolicy = import_policy_components()

        config = PreTrainedConfig.from_pretrained(self.pretrained_path)
        config.device = self.device
        set_default_empty_cameras(args)
        config.empty_cameras = args.empty_cameras
        if args.num_steps is not None:
            config.num_steps = args.num_steps
        if args.n_action_steps is not None:
            config.n_action_steps = args.n_action_steps

        policy = SmolVLAPolicy.from_pretrained(self.pretrained_path, config=config, strict=False)
        policy.to(self.device)
        policy.eval()

        self.torch = torch
        self.policy = policy
        self.policy_config = config
        self.action_dim = int(config.output_features["action"].shape[0])
        self.state_dim = int(config.input_features["observation.state"].shape[0])
        if self.state_dim != 6 or self.action_dim != 6:
            raise ValueError(
                f"This adapter expects the official 6-D base model, got state={self.state_dim}, action={self.action_dim}"
            )

        processor_overrides = {"device_processor": {"device": self.device}}
        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy_config,
                pretrained_path=self.pretrained_path,
                preprocessor_overrides=processor_overrides,
            )
        except TypeError:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy_config,
                self.pretrained_path,
                preprocessor_overrides=processor_overrides,
            )

        bridge.call_policy_method(self.policy, "reset")
        print(f"Loaded official SmolVLA base from: {self.pretrained_path}")
        print(f"Device: {self.device}")
        print(f"Using local 14-D state indices for 6-D model input: {self.state_indices}")
        print(f"Writing 6-D model actions back to local 14-D indices: {self.action_indices}")
        print(f"empty_cameras={config.empty_cameras}")

    def build_observation(self, header: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], np.ndarray]:
        import torch

        local_state = np.asarray(header["state"], dtype=np.float32)
        if local_state.shape != (14,):
            raise ValueError(f"Expected local 14-D state, got shape {local_state.shape}")
        model_state = local_state[self.state_indices]

        obs: dict[str, Any] = {
            "observation.state": torch.from_numpy(model_state),
            "task": header.get("task") or self.args.task,
            "robot_type": header.get("robot_type") or self.args.robot_type,
        }

        encoded_images = bridge.unpack_images(header, payload)
        for camera, encoded in encoded_images.items():
            if camera not in OFFICIAL_IMAGE_KEYS:
                raise ValueError(f"Unsupported camera slot for official base model: {camera!r}")
            image = bridge.decode_jpeg_rgb(encoded)
            tensor = torch.from_numpy(image).to(dtype=torch.float32).div(255.0)
            tensor = tensor.permute(2, 0, 1).contiguous()
            obs[OFFICIAL_IMAGE_KEYS[camera]] = tensor
        return obs, local_state

    def expand_to_local_actions(self, model_actions: np.ndarray, local_state: np.ndarray) -> np.ndarray:
        model_actions = np.asarray(model_actions, dtype=np.float32)
        if model_actions.ndim == 1:
            model_actions = model_actions[None, :]
        if model_actions.ndim == 3:
            model_actions = model_actions[0]
        if model_actions.ndim != 2 or model_actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected model action shape [T, {self.action_dim}], got {model_actions.shape}")

        local_actions = np.repeat(local_state[None, :], model_actions.shape[0], axis=0)
        local_actions[:, self.action_indices] = model_actions
        return local_actions

    def infer(self, header: dict[str, Any], payload: bytes) -> dict[str, Any]:
        start = time.perf_counter()
        obs, local_state = self.build_observation(header, payload)
        processed = self.preprocessor(obs)

        with self.torch.inference_mode():
            if self.args.server_action_mode == "single":
                action = bridge.call_policy_method(self.policy, "select_action", processed)
            else:
                action = bridge.call_policy_method(self.policy, "predict_action_chunk", processed)
            action = self.postprocessor(action)

        action_np = action.detach().cpu().numpy()
        local_action_np = self.expand_to_local_actions(action_np, local_state)
        if self.args.max_actions_per_response > 0:
            local_action_np = local_action_np[: self.args.max_actions_per_response]

        elapsed = time.perf_counter() - start
        return {
            "type": "action_chunk",
            "version": bridge.PROTOCOL_VERSION,
            "request_id": header.get("request_id"),
            "actions": local_action_np.astype(float).tolist(),
            "server_latency_s": elapsed,
            "model_path": self.pretrained_path,
            "adapter": {
                "state_indices": self.state_indices,
                "action_indices": self.action_indices,
                "model_action_dim": self.action_dim,
                "local_action_dim": 14,
            },
        }


def run_server(args: argparse.Namespace) -> None:
    server = OfficialPretrainedSmolVLAServer(args)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(args.backlog)
        print(f"Listening on {args.host}:{args.port}")

        while True:
            conn, addr = srv.accept()
            print(f"Client connected: {addr[0]}:{addr[1]}")
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                bridge.call_policy_method(server.policy, "reset")
                while True:
                    try:
                        header, payload = bridge.recv_packet(conn)
                    except EOFError:
                        break
                    except Exception as exc:
                        print(f"Receive error from {addr}: {exc}")
                        break

                    try:
                        msg_type = header.get("type")
                        if msg_type == "reset":
                            bridge.call_policy_method(server.policy, "reset")
                            bridge.send_packet(conn, {"type": "reset_ok", "version": bridge.PROTOCOL_VERSION})
                        elif msg_type == "infer":
                            response = server.infer(header, payload)
                            bridge.send_packet(conn, response)
                        else:
                            bridge.send_packet(conn, {"type": "error", "message": f"Unknown message type: {msg_type}"})
                    except Exception as exc:
                        bridge.send_packet(
                            conn,
                            {
                                "type": "error",
                                "request_id": header.get("request_id"),
                                "message": repr(exc),
                            },
                        )
            print(f"Client disconnected: {addr[0]}:{addr[1]}")


def build_parser() -> argparse.ArgumentParser:
    parser = bridge.build_parser()
    parser.description = "Official pretrained SmolVLA base deployment adapter for the local ROS bridge."
    parser.set_defaults(
        model_path=OFFICIAL_MODEL_ID,
        robot_type="so100_follower",
        dry_run=True,
        server_action_mode="chunk",
    )
    parser.add_argument(
        "--state-indices",
        default=DEFAULT_STATE_INDICES,
        help=(
            "Six indices selected from the local 14-D state and fed to lerobot/smolvla_base. "
            "Default selects the right arm except joint5: 7,8,9,10,11,13."
        ),
    )
    parser.add_argument(
        "--action-indices",
        default=None,
        help=(
            "Six local 14-D command indices overwritten by the model action. "
            "Defaults to --state-indices."
        ),
    )
    parser.add_argument(
        "--allow-publish",
        action="store_true",
        help="Actually publish JointState commands on the robot side. By default this script dry-runs.",
    )
    parser.add_argument(
        "--print-official-config",
        action="store_true",
        help="Print the official base model adapter-relevant settings and exit.",
    )
    return parser


def print_official_config(args: argparse.Namespace) -> None:
    bridge.maybe_add_lerobot_src(args.lerobot_src)
    PreTrainedConfig, _, _ = import_policy_components()
    config = PreTrainedConfig.from_pretrained(args.model_path or OFFICIAL_MODEL_ID)
    data = {
        "model_path": args.model_path or OFFICIAL_MODEL_ID,
        "input_features": {
            key: {"type": str(value.type), "shape": list(value.shape)}
            for key, value in config.input_features.items()
        },
        "output_features": {
            key: {"type": str(value.type), "shape": list(value.shape)}
            for key, value in config.output_features.items()
        },
        "chunk_size": getattr(config, "chunk_size", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "empty_cameras": getattr(config, "empty_cameras", None),
    }
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.allow_publish:
        args.dry_run = False

    if args.print_official_config:
        print_official_config(args)
        return

    if args.model_path != OFFICIAL_MODEL_ID and not Path(args.model_path).expanduser().exists():
        print(f"Using Hugging Face model id/path: {args.model_path}")

    if args.role == "server":
        run_server(args)
    else:
        bridge.run_robot(args)


if __name__ == "__main__":
    main()
