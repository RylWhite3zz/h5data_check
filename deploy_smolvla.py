#!/usr/bin/env python3
# -- coding: UTF-8
"""
Deploy a fine-tuned SmolVLA policy on a remote inference machine and bridge it
to the local ROS-controlled bimanual arm.

Run on the GPU desktop:
    python deploy_smolvla.py --role server --model-path /path/to/pretrained_model

Run on the robot/ROS machine:
    python deploy_smolvla.py --role robot --server-host 192.168.1.20 --camera-mode 3cam
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import socket
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PACKET_LEN = struct.Struct("!I")
MAX_HEADER_BYTES = 16 * 1024 * 1024
PROTOCOL_VERSION = 1
JOINT_NAMES = [f"joint{i}" for i in range(7)]
DATASET_IMAGE_KEYS = {
    "left": "observation.images.left",
    "right": "observation.images.right",
    "front": "observation.images.front",
}
POLICY_IMAGE_KEYS = {
    "left": "observation.images.camera1",
    "right": "observation.images.camera2",
    "front": "observation.images.camera3",
}
DEFAULT_STEP_LIMIT_7D = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]


@dataclass
class ObservationFrame:
    state: np.ndarray
    images: dict[str, np.ndarray]
    timestamp: float
    query_step: int = 0


def camera_names_from_args(args: argparse.Namespace) -> list[str]:
    if args.cameras:
        cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]
    elif args.camera_mode == "2cam":
        cameras = ["left", "right"]
    else:
        cameras = ["left", "right", "front"]

    unknown = sorted(set(cameras) - set(DATASET_IMAGE_KEYS))
    if unknown:
        raise ValueError(f"Unsupported camera name(s): {unknown}")
    return cameras


def parse_float_list(value: str | None, *, valid_lengths: tuple[int, ...]) -> list[float] | None:
    if value is None or value == "":
        return None
    parts = value.replace(",", " ").split()
    numbers = [float(part) for part in parts]
    if len(numbers) not in valid_lengths:
        raise ValueError(f"Expected {valid_lengths} values, got {len(numbers)} from {value!r}")
    return numbers


def parse_assignment_map(value: str | None) -> dict[str, str]:
    if value is None or value.strip() == "":
        return {}

    result: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE item in mapping, got {item!r}")
        key, mapped_value = item.split("=", 1)
        key = key.strip()
        mapped_value = mapped_value.strip()
        if not key or not mapped_value:
            raise ValueError(f"Expected non-empty KEY=VALUE item in mapping, got {item!r}")
        if key not in DATASET_IMAGE_KEYS:
            raise ValueError(f"Unsupported model camera slot {key!r}. Use one of {sorted(DATASET_IMAGE_KEYS)}")
        result[key] = mapped_value
    return result


def build_camera_topic_map(args: argparse.Namespace, camera_names: list[str]) -> dict[str, str]:
    topic_by_camera = {
        "left": args.img_left_topic,
        "right": args.img_right_topic,
        "front": args.img_front_topic,
    }

    node_map = parse_assignment_map(args.camera_node_map)
    for camera, node_name in node_map.items():
        topic_by_camera[camera] = args.camera_topic_template.format(node=node_name, camera=camera)

    topic_by_camera.update(parse_assignment_map(args.camera_topic_map))

    missing = [camera for camera in camera_names if not topic_by_camera.get(camera)]
    if missing:
        raise ValueError(f"Missing ROS image topic for model camera slot(s): {missing}")
    return {camera: topic_by_camera[camera] for camera in camera_names}


def expand_joint_limits(values: list[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) == 1:
        return np.repeat(np.asarray(values, dtype=np.float32), 14)
    if len(values) == 7:
        return np.asarray(values + values, dtype=np.float32)
    if len(values) == 14:
        return np.asarray(values, dtype=np.float32)
    raise ValueError("Joint limits must contain 1, 7, or 14 values")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_len"] = len(payload)
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {len(header_bytes)} bytes")
    sock.sendall(PACKET_LEN.pack(len(header_bytes)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)


def recv_packet(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_len = PACKET_LEN.unpack(recv_exact(sock, PACKET_LEN.size))[0]
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {header_len} bytes")
    header = json.loads(recv_exact(sock, header_len).decode("utf-8"))
    payload_len = int(header.get("payload_len", 0))
    payload = recv_exact(sock, payload_len) if payload_len else b""
    return header, payload


def pack_images(encoded_images: dict[str, bytes], camera_names: list[str]) -> tuple[dict[str, int], bytes]:
    lengths: dict[str, int] = {}
    chunks: list[bytes] = []
    for camera in camera_names:
        data = encoded_images[camera]
        lengths[camera] = len(data)
        chunks.append(data)
    return lengths, b"".join(chunks)


def unpack_images(header: dict[str, Any], payload: bytes) -> dict[str, bytes]:
    camera_names = header["camera_names"]
    lengths = header["image_lengths"]
    offset = 0
    images: dict[str, bytes] = {}
    for camera in camera_names:
        size = int(lengths[camera])
        images[camera] = payload[offset : offset + size]
        offset += size
    if offset != len(payload):
        raise ValueError(f"Image payload length mismatch: consumed={offset}, payload={len(payload)}")
    return images


def encode_jpeg_rgb(image_rgb: np.ndarray, quality: int) -> bytes:
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim == 2:
        image_rgb = np.repeat(image_rgb[:, :, None], 3, axis=2)
    if image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]
    image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8, copy=False))

    try:
        import cv2

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return encoded.tobytes()
    except Exception:
        pass

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def decode_jpeg_rgb(encoded: bytes) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(encoded)) as image:
            return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    except Exception:
        import cv2

        buffer = np.frombuffer(encoded, dtype=np.uint8)
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode JPEG image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(image_rgb)


def maybe_add_lerobot_src(path: str | None) -> None:
    if path:
        sys.path.insert(0, str(Path(path).expanduser().resolve()))


def is_local_path_like(value: str) -> bool:
    expanded = str(value).strip()
    return (
        expanded.startswith((".", "/", "~"))
        or "\\" in expanded
        or expanded.count("/") > 1
    )


def resolve_pretrained_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if not path.exists():
        if is_local_path_like(model_path):
            raise FileNotFoundError(
                f"Local model path does not exist from cwd={Path.cwd()}: {model_path}"
            )
        return model_path

    if (path / "last" / "pretrained_model").exists():
        return str((path / "last" / "pretrained_model").resolve())
    if (path / "checkpoints" / "last" / "pretrained_model").exists():
        return str((path / "checkpoints" / "last" / "pretrained_model").resolve())
    if (path / "pretrained_model").exists():
        return str((path / "pretrained_model").resolve())
    direct_checkpoints = sorted(path.glob("*/pretrained_model"))
    if direct_checkpoints:
        return str(direct_checkpoints[-1].resolve())
    checkpoints = sorted((path / "checkpoints").glob("*/pretrained_model")) if (path / "checkpoints").exists() else []
    if checkpoints:
        return str(checkpoints[-1].resolve())
    return str(path.resolve())


def load_json_stats(path: str | None) -> dict[str, dict[str, Any]] | None:
    if not path:
        return None
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def call_policy_method(policy: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    candidates = [policy]
    for attr_path in (("base_model",), ("base_model", "model"), ("model",)):
        current = policy
        for attr in attr_path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            candidates.append(current)

    for candidate in candidates:
        if hasattr(candidate, method_name):
            return getattr(candidate, method_name)(*args, **kwargs)
    raise AttributeError(f"Policy object does not expose {method_name}()")


class SmolVLAServer:
    def __init__(self, args: argparse.Namespace):
        maybe_add_lerobot_src(args.lerobot_src)
        self.args = args
        self.device = args.device
        self.pretrained_path = resolve_pretrained_path(args.model_path)

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        config = PreTrainedConfig.from_pretrained(self.pretrained_path)
        config.device = self.device
        if args.empty_cameras is not None:
            config.empty_cameras = args.empty_cameras
        if args.num_steps is not None:
            config.num_steps = args.num_steps
        if args.n_action_steps is not None:
            config.n_action_steps = args.n_action_steps

        local_path = Path(self.pretrained_path)
        adapter_config = local_path / "adapter_config.json"
        if local_path.exists() and adapter_config.exists():
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise ImportError("This checkpoint is a LoRA/PEFT adapter, but peft is not installed") from exc

            peft_config = PeftConfig.from_pretrained(self.pretrained_path)
            base_path = args.base_model_path or peft_config.base_model_name_or_path
            if not base_path:
                raise ValueError(f"No base_model_name_or_path in {adapter_config}")
            policy = SmolVLAPolicy.from_pretrained(base_path, config=config, strict=False)
            policy = PeftModel.from_pretrained(policy, self.pretrained_path, config=peft_config)
            policy.to(self.device)
            policy.eval()
        else:
            policy = SmolVLAPolicy.from_pretrained(self.pretrained_path, config=config, strict=False)

        self.policy = policy
        self.policy_config = config

        processor_overrides = {"device_processor": {"device": self.device}}
        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy_config,
                pretrained_path=self.pretrained_path,
                preprocessor_overrides=processor_overrides,
            )
            self.processor_key_mode = "dataset"
        except Exception as exc:
            stats = load_json_stats(args.dataset_stats)
            if stats is None:
                raise RuntimeError(
                    "Could not load saved policy processors from the checkpoint. "
                    "Pass --dataset-stats /path/to/meta/stats.json to build processors from stats."
                ) from exc
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.policy_config,
                dataset_stats=stats,
            )
            self.processor_key_mode = "policy"

        call_policy_method(self.policy, "reset")
        print(f"Loaded policy from: {self.pretrained_path}")
        print(f"Processor input key mode: {self.processor_key_mode}")
        print(f"Device: {self.device}")

    def build_observation(self, header: dict[str, Any], payload: bytes) -> dict[str, Any]:
        import torch

        state = np.asarray(header["state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14-D state, got shape {state.shape}")

        obs: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": header.get("task") or self.args.task,
            "robot_type": header.get("robot_type") or self.args.robot_type,
        }

        image_key_map = DATASET_IMAGE_KEYS if self.processor_key_mode == "dataset" else POLICY_IMAGE_KEYS
        encoded_images = unpack_images(header, payload)
        for camera, encoded in encoded_images.items():
            image = decode_jpeg_rgb(encoded)
            tensor = torch.from_numpy(image).to(dtype=torch.float32).div(255.0)
            tensor = tensor.permute(2, 0, 1).contiguous()
            obs[image_key_map[camera]] = tensor
        return obs

    def infer(self, header: dict[str, Any], payload: bytes) -> dict[str, Any]:
        start = time.perf_counter()
        obs = self.build_observation(header, payload)
        processed = self.preprocessor(obs)

        with self.torch.inference_mode():
            if self.args.server_action_mode == "single":
                action = call_policy_method(self.policy, "select_action", processed)
            else:
                action = call_policy_method(self.policy, "predict_action_chunk", processed)
            action = self.postprocessor(action)

        action_np = action.detach().cpu().numpy()
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        if action_np.ndim == 3:
            action_np = action_np[0]
        if self.args.max_actions_per_response > 0:
            action_np = action_np[: self.args.max_actions_per_response]

        if action_np.shape[-1] != 14:
            raise ValueError(f"Expected 14-D action, got shape {action_np.shape}")

        elapsed = time.perf_counter() - start
        return {
            "type": "action_chunk",
            "version": PROTOCOL_VERSION,
            "request_id": header.get("request_id"),
            "actions": action_np.astype(float).tolist(),
            "server_latency_s": elapsed,
            "model_path": self.pretrained_path,
        }


def run_server(args: argparse.Namespace) -> None:
    server = SmolVLAServer(args)
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
                call_policy_method(server.policy, "reset")
                while True:
                    try:
                        header, payload = recv_packet(conn)
                    except EOFError:
                        break
                    except Exception as exc:
                        print(f"Receive error from {addr}: {exc}")
                        break

                    try:
                        msg_type = header.get("type")
                        if msg_type == "reset":
                            call_policy_method(server.policy, "reset")
                            send_packet(conn, {"type": "reset_ok", "version": PROTOCOL_VERSION})
                        elif msg_type == "infer":
                            response = server.infer(header, payload)
                            send_packet(conn, response)
                        else:
                            send_packet(conn, {"type": "error", "message": f"Unknown message type: {msg_type}"})
                    except Exception as exc:
                        send_packet(
                            conn,
                            {
                                "type": "error",
                                "request_id": header.get("request_id"),
                                "message": repr(exc),
                            },
                        )
            print(f"Client disconnected: {addr[0]}:{addr[1]}")


class RosOperator:
    def __init__(self, args: argparse.Namespace, camera_names: list[str]):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, JointState

        self.args = args
        self.camera_names = camera_names
        self.rospy = rospy
        self.Image = Image
        self.JointState = JointState
        self.bridge = CvBridge()
        self.topic_by_camera = build_camera_topic_map(args, camera_names)
        self.lock = threading.Lock()
        self.image_deques = {camera: deque(maxlen=args.buffer_size) for camera in camera_names}
        self.left_joint_deque: deque[Any] = deque(maxlen=args.buffer_size)
        self.right_joint_deque: deque[Any] = deque(maxlen=args.buffer_size)
        self.left_pub = None
        self.right_pub = None
        self.last_command: np.ndarray | None = None

        self.max_joint_step = None if args.disable_step_limit else expand_joint_limits(args.max_joint_step)
        self.joint_min = expand_joint_limits(args.joint_min)
        self.joint_max = expand_joint_limits(args.joint_max)

        rospy.init_node(args.ros_node_name, anonymous=True)
        self._init_subscribers()
        self.left_pub = rospy.Publisher(args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.right_pub = rospy.Publisher(args.puppet_arm_right_cmd_topic, JointState, queue_size=10)

    @staticmethod
    def _image_to_rgb(image: np.ndarray, encoding: str) -> np.ndarray:
        image = np.asarray(image)
        encoding = (encoding or "").lower()
        if image.ndim == 2:
            if image.dtype != np.uint8:
                max_value = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
                image = np.clip(image.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)
            return np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] == 4:
            if encoding in {"bgra8", "bgra"}:
                return np.ascontiguousarray(image[:, :, [2, 1, 0]])
            return np.ascontiguousarray(image[:, :, :3])
        if encoding in {"bgr8", "bgr"}:
            return np.ascontiguousarray(image[:, :, ::-1])
        return np.ascontiguousarray(image[:, :, :3])

    def image_msg_to_rgb(self, msg: Any) -> np.ndarray:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.args.ros_image_encoding)
            if self.args.ros_image_encoding.lower() == "rgb8":
                return np.ascontiguousarray(image)
            return self._image_to_rgb(image, self.args.ros_image_encoding)
        except Exception:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            return self._image_to_rgb(image, getattr(msg, "encoding", ""))

    def _init_subscribers(self) -> None:
        for camera in self.camera_names:
            topic = self.topic_by_camera[camera]
            self.rospy.loginfo("Subscribing model camera slot '%s' from ROS topic '%s'", camera, topic)
            self.rospy.Subscriber(
                topic,
                self.Image,
                lambda msg, camera=camera: self._image_callback(camera, msg),
                queue_size=1000,
                tcp_nodelay=True,
            )
        self.rospy.Subscriber(
            self.args.puppet_arm_left_topic,
            self.JointState,
            self._left_joint_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            self.args.puppet_arm_right_topic,
            self.JointState,
            self._right_joint_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )

    def _image_callback(self, camera: str, msg: Any) -> None:
        with self.lock:
            self.image_deques[camera].append(msg)

    def _left_joint_callback(self, msg: Any) -> None:
        with self.lock:
            self.left_joint_deque.append(msg)

    def _right_joint_callback(self, msg: Any) -> None:
        with self.lock:
            self.right_joint_deque.append(msg)

    @staticmethod
    def _stamp(msg: Any) -> float:
        return float(msg.header.stamp.to_sec())

    @staticmethod
    def _pop_at_or_after(msg_deque: deque[Any], timestamp: float) -> Any | None:
        while len(msg_deque) > 1 and RosOperator._stamp(msg_deque[0]) < timestamp:
            msg_deque.popleft()
        if not msg_deque or RosOperator._stamp(msg_deque[0]) < timestamp:
            return None
        return msg_deque.popleft()

    def _latest_state_locked(self) -> np.ndarray | None:
        if not self.left_joint_deque or not self.right_joint_deque:
            return None
        left = np.asarray(self.left_joint_deque[-1].position, dtype=np.float32)
        right = np.asarray(self.right_joint_deque[-1].position, dtype=np.float32)
        if left.shape[0] < 7 or right.shape[0] < 7:
            return None
        return np.concatenate([left[:7], right[:7]], axis=0)

    def get_frame(self) -> ObservationFrame | None:
        with self.lock:
            if any(not self.image_deques[camera] for camera in self.camera_names):
                return None
            if not self.left_joint_deque or not self.right_joint_deque:
                return None

            frame_time = min(self._stamp(self.image_deques[camera][-1]) for camera in self.camera_names)
            if self._stamp(self.left_joint_deque[-1]) < frame_time:
                return None
            if self._stamp(self.right_joint_deque[-1]) < frame_time:
                return None

            image_msgs = {
                camera: self._pop_at_or_after(self.image_deques[camera], frame_time)
                for camera in self.camera_names
            }
            if any(msg is None for msg in image_msgs.values()):
                return None
            left_joint = self._pop_at_or_after(self.left_joint_deque, frame_time)
            right_joint = self._pop_at_or_after(self.right_joint_deque, frame_time)
            if left_joint is None or right_joint is None:
                return None

            left = np.asarray(left_joint.position, dtype=np.float32)
            right = np.asarray(right_joint.position, dtype=np.float32)

        images: dict[str, np.ndarray] = {}
        for camera, msg in image_msgs.items():
            images[camera] = self.image_msg_to_rgb(msg)

        state = np.concatenate([left[:7], right[:7]], axis=0)
        if state.shape != (14,):
            return None
        return ObservationFrame(state=state, images=images, timestamp=frame_time)

    def current_state(self) -> np.ndarray | None:
        with self.lock:
            return self._latest_state_locked()

    def prepare_action(self, action: np.ndarray) -> np.ndarray:
        target = np.asarray(action, dtype=np.float32).reshape(14)
        if self.max_joint_step is not None:
            reference = self.last_command if self.last_command is not None else self.current_state()
            if reference is not None and reference.shape == (14,):
                delta = np.clip(target - reference, -self.max_joint_step, self.max_joint_step)
                target = reference + delta
        if self.joint_min is not None:
            target = np.maximum(target, self.joint_min)
        if self.joint_max is not None:
            target = np.minimum(target, self.joint_max)
        return target

    def publish_action(self, action: np.ndarray) -> None:
        target = self.prepare_action(action)
        self.last_command = target.copy()
        if self.args.dry_run:
            self.rospy.loginfo_throttle(1.0, f"dry-run action left={target[:7]} right={target[7:]}")
            return

        left_msg = self.JointState()
        left_msg.header.stamp = self.rospy.Time.now()
        left_msg.name = JOINT_NAMES
        left_msg.position = target[:7].astype(float).tolist()

        right_msg = self.JointState()
        right_msg.header.stamp = left_msg.header.stamp
        right_msg.name = JOINT_NAMES
        right_msg.position = target[7:].astype(float).tolist()

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        args: argparse.Namespace,
        camera_names: list[str],
        request_queue: queue.Queue[ObservationFrame],
        action_queue: deque[np.ndarray],
        action_lock: threading.Lock,
        stop_event: threading.Event,
        in_flight_event: threading.Event,
        temporal_aggregator: "TemporalActionAggregator | None" = None,
    ):
        super().__init__(daemon=True)
        self.args = args
        self.camera_names = camera_names
        self.request_queue = request_queue
        self.action_queue = action_queue
        self.action_lock = action_lock
        self.stop_event = stop_event
        self.in_flight_event = in_flight_event
        self.temporal_aggregator = temporal_aggregator
        self.sock: socket.socket | None = None
        self.request_id = 0

    def close_socket(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def ensure_socket(self) -> socket.socket:
        if self.sock is not None:
            return self.sock
        sock = socket.create_connection(
            (self.args.server_host, self.args.port),
            timeout=self.args.connect_timeout,
        )
        sock.settimeout(self.args.socket_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_packet(sock, {"type": "reset", "version": PROTOCOL_VERSION})
        recv_packet(sock)
        self.sock = sock
        return sock

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                sock = self.ensure_socket()
                encoded = {
                    camera: encode_jpeg_rgb(frame.images[camera], self.args.jpeg_quality)
                    for camera in self.camera_names
                }
                image_lengths, payload = pack_images(encoded, self.camera_names)
                self.request_id += 1
                header = {
                    "type": "infer",
                    "version": PROTOCOL_VERSION,
                    "request_id": self.request_id,
                    "camera_mode": self.args.camera_mode,
                    "camera_names": self.camera_names,
                    "image_lengths": image_lengths,
                    "state": frame.state.astype(float).tolist(),
                    "task": self.args.task,
                    "robot_type": self.args.robot_type,
                    "timestamp": frame.timestamp,
                }
                send_packet(sock, header, payload)
                response, _ = recv_packet(sock)
                if response.get("type") == "error":
                    raise RuntimeError(response.get("message", "unknown server error"))
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.shape[-1] != 14:
                    raise ValueError(f"Expected 14-D action(s), got {actions.shape}")
                skipped = 0
                if self.args.skip_actions > 0 and actions.shape[0] > self.args.skip_actions:
                    actions = actions[self.args.skip_actions :]
                    skipped = self.args.skip_actions

                with self.action_lock:
                    if self.temporal_aggregator is not None:
                        self.temporal_aggregator.add_chunk(frame.query_step + skipped, actions)
                    else:
                        if self.args.replace_action_queue:
                            self.action_queue.clear()
                        for action in actions:
                            self.action_queue.append(action.copy())

                if self.args.verbose:
                    latency = response.get("server_latency_s")
                    print(f"received {len(actions)} actions, server_latency_s={latency}")
            except Exception as exc:
                print(f"inference worker error: {exc}")
                self.close_socket()
                time.sleep(self.args.reconnect_interval)
            finally:
                self.in_flight_event.clear()

        self.close_socket()


class TemporalActionAggregator:
    def __init__(self, k: float, max_chunks: int):
        if k < 0:
            raise ValueError("--temporal-agg-k must be >= 0")
        if max_chunks <= 0:
            raise ValueError("--temporal-agg-max-chunks must be > 0")
        self.k = float(k)
        self.max_chunks = int(max_chunks)
        self.chunks: list[tuple[int, np.ndarray]] = []

    def add_chunk(self, base_step: int, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected action chunk shape [T, 14], got {actions.shape}")
        self.chunks.append((int(base_step), actions.copy()))
        self.chunks.sort(key=lambda item: item[0])
        if len(self.chunks) > self.max_chunks:
            self.chunks = self.chunks[-self.max_chunks :]

    def future_horizon(self, step: int) -> int:
        horizons = [base + len(actions) - step for base, actions in self.chunks if base + len(actions) > step]
        return max(horizons, default=0)

    def get_action(self, step: int) -> np.ndarray | None:
        candidates: list[tuple[int, np.ndarray]] = []
        live_chunks: list[tuple[int, np.ndarray]] = []

        for base, actions in self.chunks:
            end = base + len(actions)
            if end > step:
                live_chunks.append((base, actions))
            if base <= step < end:
                candidates.append((base, actions[step - base]))

        self.chunks = live_chunks[-self.max_chunks :]
        if not candidates:
            return None

        offsets = np.asarray([step - base for base, _ in candidates], dtype=np.float32)
        if self.k == 0:
            weights = np.ones_like(offsets)
        else:
            # Smaller offset means a newer prediction for this execution step.
            weights = np.exp(-self.k * offsets)
        weights = weights / np.sum(weights)
        stacked = np.stack([action for _, action in candidates], axis=0)
        return np.sum(stacked * weights[:, None], axis=0).astype(np.float32)


def put_latest(request_queue: queue.Queue[ObservationFrame], frame: ObservationFrame) -> None:
    try:
        request_queue.put_nowait(frame)
        return
    except queue.Full:
        pass
    try:
        request_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        request_queue.put_nowait(frame)
    except queue.Full:
        pass


def run_robot(args: argparse.Namespace) -> None:
    camera_names = camera_names_from_args(args)
    args.max_joint_step = parse_float_list(args.max_joint_step, valid_lengths=(1, 7, 14))
    args.joint_min = parse_float_list(args.joint_min, valid_lengths=(1, 7, 14))
    args.joint_max = parse_float_list(args.joint_max, valid_lengths=(1, 7, 14))
    if args.temporal_agg and args.server_action_mode == "single":
        raise ValueError("--temporal-agg requires --server-action-mode chunk")
    if args.temporal_agg and args.temporal_agg_query_interval <= 0:
        raise ValueError("--temporal-agg-query-interval must be > 0")

    ros = RosOperator(args, camera_names)
    rospy = ros.rospy
    rate = rospy.Rate(args.publish_rate)

    request_queue: queue.Queue[ObservationFrame] = queue.Queue(maxsize=1)
    action_queue: deque[np.ndarray] = deque(maxlen=args.action_queue_size)
    action_lock = threading.Lock()
    stop_event = threading.Event()
    in_flight_event = threading.Event()
    temporal_aggregator = (
        TemporalActionAggregator(args.temporal_agg_k, args.temporal_agg_max_chunks)
        if args.temporal_agg
        else None
    )
    worker = InferenceWorker(
        args,
        camera_names,
        request_queue,
        action_queue,
        action_lock,
        stop_event,
        in_flight_event,
        temporal_aggregator,
    )
    worker.start()

    last_request_time = 0.0
    last_request_step = -10**9
    publish_count = 0
    try:
        while not rospy.is_shutdown():
            now = time.monotonic()
            with action_lock:
                future_horizon = (
                    temporal_aggregator.future_horizon(publish_count)
                    if temporal_aggregator is not None
                    else len(action_queue)
                )

            if temporal_aggregator is not None:
                step_due = publish_count - last_request_step >= args.temporal_agg_query_interval
                horizon_due = future_horizon <= args.queue_low_watermark
                should_request = step_due or horizon_due
            else:
                should_request = future_horizon <= args.queue_low_watermark

            if (
                should_request
                and not in_flight_event.is_set()
                and now - last_request_time >= args.min_request_interval
            ):
                frame = ros.get_frame()
                if frame is not None:
                    frame.query_step = publish_count + args.temporal_agg_base_offset
                    put_latest(request_queue, frame)
                    in_flight_event.set()
                    last_request_time = now
                    last_request_step = publish_count

            action = None
            with action_lock:
                if temporal_aggregator is not None:
                    action = temporal_aggregator.get_action(publish_count)
                elif action_queue:
                    action = action_queue.popleft()

            if action is not None:
                ros.publish_action(action)
                publish_count += 1
            elif args.hold_position and ros.last_command is not None:
                ros.publish_action(ros.last_command)

            if args.verbose and publish_count and publish_count % args.log_every == 0:
                rospy.loginfo(
                    f"published={publish_count}, future_horizon={future_horizon}, "
                    f"queued_actions={len(action_queue)}"
                )
            rate.sleep()
    finally:
        stop_event.set()
        worker.join(timeout=2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolVLA remote inference deployment for the bimanual ROS arm.")
    parser.add_argument("--role", choices=("server", "robot"), required=True)
    parser.add_argument("--camera-mode", choices=("2cam", "3cam"), default="3cam")
    parser.add_argument("--cameras", default=None, help="Optional comma list, e.g. left,right or left,right,front.")
    parser.add_argument("--task", default="your task instruction here")
    parser.add_argument("--robot-type", default="custom_bimanual")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    parser.add_argument("--server-host", default="127.0.0.1", help="Robot-side server IP/hostname.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--reconnect-interval", type=float, default=1.0)

    parser.add_argument("--model-path", default=None, help="Checkpoint path, train output dir, or HF model id.")
    parser.add_argument("--base-model-path", default=None, help="Override LoRA adapter base_model_name_or_path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lerobot-src", default=None, help="Optional path to lerobot/src if not installed.")
    parser.add_argument("--dataset-stats", default=None, help="Fallback LeRobot meta/stats.json if processors are absent.")
    parser.add_argument("--empty-cameras", type=int, default=None, help="Override policy.config.empty_cameras.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override SmolVLA flow sampling steps.")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Override policy.config.n_action_steps.")
    parser.add_argument("--server-action-mode", choices=("chunk", "single"), default="chunk")
    parser.add_argument("--max-actions-per-response", type=int, default=0)

    parser.add_argument("--ros-node-name", default="smolvla_remote_deploy")
    parser.add_argument("--img-left-topic", default="/camera_l/color/image_raw")
    parser.add_argument("--img-right-topic", default="/camera_r/color/image_raw")
    parser.add_argument("--img-front-topic", default="/camera_f/color/image_raw")
    parser.add_argument(
        "--camera-topic-map",
        default=None,
        help=(
            "Map model camera slots to concrete ROS image topics, e.g. "
            "'left=/camera_l/color/image_raw,right=/camera_r/color/image_raw,front=/camera_e/color/image_raw'."
        ),
    )
    parser.add_argument(
        "--camera-node-map",
        default=None,
        help=(
            "Map model camera slots to camera node/prefix names, e.g. 'left=camera_l,right=camera_r,front=camera_e'. "
            "The topic is built with --camera-topic-template."
        ),
    )
    parser.add_argument(
        "--camera-topic-template",
        default="/{node}/color/image_raw",
        help="Template used with --camera-node-map. Available fields: {node}, {camera}.",
    )
    parser.add_argument("--puppet-arm-left-topic", default="/puppet/joint_left")
    parser.add_argument("--puppet-arm-right-topic", default="/puppet/joint_right")
    parser.add_argument("--puppet-arm-left-cmd-topic", default="/master/joint_left")
    parser.add_argument("--puppet-arm-right-cmd-topic", default="/master/joint_right")
    parser.add_argument("--publish-rate", type=float, default=30.0)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--ros-image-encoding",
        default="rgb8",
        help="cv_bridge desired color encoding. Use passthrough if the camera driver rejects rgb8 conversion.",
    )
    parser.add_argument("--action-queue-size", type=int, default=100)
    parser.add_argument("--queue-low-watermark", type=int, default=10)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    parser.add_argument("--skip-actions", type=int, default=0)
    parser.add_argument(
        "--temporal-agg",
        action="store_true",
        help="Aggregate overlapping action chunks by execution step with exponential weighting.",
    )
    parser.add_argument(
        "--temporal-agg-k",
        type=float,
        default=0.01,
        help="Exponential decay for temporal aggregation. Larger values favor newer chunks more strongly.",
    )
    parser.add_argument(
        "--temporal-agg-query-interval",
        type=int,
        default=5,
        help="When --temporal-agg is enabled, request a fresh action chunk every N published steps.",
    )
    parser.add_argument(
        "--temporal-agg-max-chunks",
        type=int,
        default=16,
        help="Maximum number of overlapping chunks kept for temporal aggregation.",
    )
    parser.add_argument(
        "--temporal-agg-base-offset",
        type=int,
        default=0,
        help="Shift the returned chunk base step by this many publish steps for latency compensation.",
    )
    replace_group = parser.add_mutually_exclusive_group()
    replace_group.add_argument("--replace-action-queue", dest="replace_action_queue", action="store_true", default=True)
    replace_group.add_argument("--no-replace-action-queue", dest="replace_action_queue", action="store_false")
    hold_group = parser.add_mutually_exclusive_group()
    hold_group.add_argument("--hold-position", dest="hold_position", action="store_true", default=True)
    hold_group.add_argument("--no-hold-position", dest="hold_position", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-step-limit", action="store_true")
    parser.add_argument(
        "--max-joint-step",
        default=",".join(str(v) for v in DEFAULT_STEP_LIMIT_7D),
        help="Per publish max joint delta: 1, 7, or 14 comma-separated values.",
    )
    parser.add_argument("--joint-min", default=None, help="Optional 1, 7, or 14 comma-separated joint lower bounds.")
    parser.add_argument("--joint-max", default=None, help="Optional 1, 7, or 14 comma-separated joint upper bounds.")
    parser.add_argument("--log-every", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.role == "server":
        if not args.model_path:
            parser.error("--model-path is required for --role server")
        run_server(args)
    else:
        run_robot(args)


if __name__ == "__main__":
    main()
