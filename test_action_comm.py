#!/usr/bin/env python3
# -- coding: UTF-8
"""
Small TCP communication test for the SmolVLA server/robot packet structure.

Server:
    python test_action_comm.py --role server --host 0.0.0.0 --port 8766 --pattern sine

Robot/client:
    python test_action_comm.py --role robot --server-host <SERVER_IP> --port 8766 --num-requests 5

This does not load a model or cameras. The server returns synthetic 14-D target
actions near the state sent by the robot side. By default the robot side reads
the current joint state and publishes the received actions with a small step
limit, using the same packet framing as deploy_smolvla.py.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import socket
import threading
import time
from typing import Any

import numpy as np

import deploy_smolvla as bridge


def parse_actions(value: str | None, action_dim: int) -> np.ndarray | None:
    if value is None or value.strip() == "":
        return None

    try:
        data = json.loads(value)
        actions = np.asarray(data, dtype=np.float32)
    except json.JSONDecodeError:
        rows = []
        for row in value.split(";"):
            row = row.strip()
            if row:
                rows.append([float(part) for part in row.replace(",", " ").split()])
        actions = np.asarray(rows, dtype=np.float32)

    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        raise ValueError(f"--actions must have shape [T, {action_dim}], got {actions.shape}")
    return actions


def generated_delta(step: int, action_dim: int, pattern: str, scale: float) -> np.ndarray:
    if pattern == "zero":
        return np.zeros(action_dim, dtype=np.float32)

    indices = np.arange(action_dim, dtype=np.float32)
    if pattern == "ramp":
        phase = ((step % 9) - 4) / 4.0
        return (scale * phase * (indices + 1) / action_dim).astype(np.float32)
    if pattern == "sine":
        return (scale * np.sin(0.25 * step + 0.3 * indices)).astype(np.float32)
    if pattern == "alternating":
        signs = np.where((indices.astype(np.int32) + step) % 2 == 0, 1.0, -1.0)
        return (scale * signs).astype(np.float32)

    raise ValueError(f"Unsupported action pattern: {pattern}")


class ManualActionSource:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.actions = parse_actions(args.actions, args.action_dim)
        self.cursor = 0

    def reset(self) -> None:
        self.cursor = 0

    def _target_from_delta(self, base_state: np.ndarray, delta: np.ndarray) -> np.ndarray:
        max_delta = float(self.args.max_target_delta)
        if max_delta > 0:
            delta = np.clip(delta, -max_delta, max_delta)
        return (base_state + delta).astype(np.float32)

    def next_chunk(self, base_state: np.ndarray) -> np.ndarray:
        base_state = np.asarray(base_state, dtype=np.float32).reshape(self.args.action_dim)

        if self.actions is None:
            chunk = np.stack(
                [
                    self._target_from_delta(
                        base_state,
                        generated_delta(
                            self.cursor + offset,
                            self.args.action_dim,
                            self.args.pattern,
                            self.args.scale,
                        ),
                    )
                    for offset in range(self.args.chunk_size)
                ],
                axis=0,
            )
            self.cursor += self.args.chunk_size
            return chunk.astype(np.float32)

        rows = []
        for _ in range(self.args.chunk_size):
            if self.cursor >= len(self.actions):
                if not self.args.cycle:
                    break
                self.cursor = 0
            action = self.actions[self.cursor]
            rows.append(action if self.args.absolute_actions else self._target_from_delta(base_state, action))
            self.cursor += 1

        if not rows:
            raise EOFError("manual action sequence exhausted")
        return np.stack(rows, axis=0).astype(np.float32)


class JointActionExecutor:
    def __init__(self, args: argparse.Namespace):
        import rospy
        from sensor_msgs.msg import JointState

        self.args = args
        self.rospy = rospy
        self.JointState = JointState
        self.lock = threading.Lock()
        self.left_joint_deque: deque[Any] = deque(maxlen=args.buffer_size)
        self.right_joint_deque: deque[Any] = deque(maxlen=args.buffer_size)
        self.last_command: np.ndarray | None = None

        parsed_max_step = bridge.parse_float_list(args.max_joint_step, valid_lengths=(1, 7, 14))
        parsed_joint_min = bridge.parse_float_list(args.joint_min, valid_lengths=(1, 7, 14))
        parsed_joint_max = bridge.parse_float_list(args.joint_max, valid_lengths=(1, 7, 14))
        self.max_joint_step = None if args.disable_step_limit else bridge.expand_joint_limits(parsed_max_step)
        self.joint_min = bridge.expand_joint_limits(parsed_joint_min)
        self.joint_max = bridge.expand_joint_limits(parsed_joint_max)

        rospy.init_node(args.ros_node_name, anonymous=True)
        rospy.Subscriber(
            args.puppet_arm_left_topic,
            JointState,
            self._left_joint_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            args.puppet_arm_right_topic,
            JointState,
            self._right_joint_callback,
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.left_pub = rospy.Publisher(args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.right_pub = rospy.Publisher(args.puppet_arm_right_cmd_topic, JointState, queue_size=10)

    def _left_joint_callback(self, msg: Any) -> None:
        with self.lock:
            self.left_joint_deque.append(msg)

    def _right_joint_callback(self, msg: Any) -> None:
        with self.lock:
            self.right_joint_deque.append(msg)

    def current_state(self) -> np.ndarray | None:
        with self.lock:
            if not self.left_joint_deque or not self.right_joint_deque:
                return None
            left = np.asarray(self.left_joint_deque[-1].position, dtype=np.float32)
            right = np.asarray(self.right_joint_deque[-1].position, dtype=np.float32)
        if left.shape[0] < 7 or right.shape[0] < 7:
            return None
        return np.concatenate([left[:7], right[:7]], axis=0)

    def wait_for_state(self, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while not self.rospy.is_shutdown() and time.monotonic() < deadline:
            state = self.current_state()
            if state is not None:
                return state
            self.rospy.sleep(0.02)
        raise TimeoutError(
            f"Timed out waiting for joint feedback on {self.args.puppet_arm_left_topic} "
            f"and {self.args.puppet_arm_right_topic}"
        )

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
        left_msg.name = bridge.JOINT_NAMES
        left_msg.position = target[:7].astype(float).tolist()

        right_msg = self.JointState()
        right_msg.header.stamp = left_msg.header.stamp
        right_msg.name = bridge.JOINT_NAMES
        right_msg.position = target[7:].astype(float).tolist()

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)


def handle_connection(conn: socket.socket, addr: tuple[str, int], args: argparse.Namespace) -> None:
    source = ManualActionSource(args)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"client connected: {addr[0]}:{addr[1]}", flush=True)

    while True:
        try:
            header, payload = bridge.recv_packet(conn)
        except EOFError:
            break

        start = time.perf_counter()
        msg_type = header.get("type")
        try:
            if msg_type == "reset":
                source.reset()
                bridge.send_packet(conn, {"type": "reset_ok", "version": bridge.PROTOCOL_VERSION})
            elif msg_type == "infer":
                if payload:
                    print(f"ignored payload bytes: {len(payload)}", flush=True)
                base_state = np.asarray(header.get("state", [0.0] * args.action_dim), dtype=np.float32)
                if base_state.shape != (args.action_dim,):
                    raise ValueError(f"Expected state shape ({args.action_dim},), got {base_state.shape}")
                sequence_start = source.cursor
                actions = source.next_chunk(base_state)
                response = {
                    "type": "action_chunk",
                    "version": bridge.PROTOCOL_VERSION,
                    "request_id": header.get("request_id"),
                    "sequence_start": sequence_start,
                    "actions": actions.astype(float).tolist(),
                    "server_latency_s": time.perf_counter() - start,
                    "source": "manual_action_comm_test",
                }
                bridge.send_packet(conn, response)
                if args.verbose:
                    print(
                        f"request_id={header.get('request_id')} sent actions shape={actions.shape}",
                        flush=True,
                    )
            else:
                bridge.send_packet(
                    conn,
                    {"type": "error", "request_id": header.get("request_id"), "message": f"unknown type: {msg_type}"},
                )
        except Exception as exc:
            bridge.send_packet(
                conn,
                {"type": "error", "request_id": header.get("request_id"), "message": repr(exc)},
            )

    print(f"client disconnected: {addr[0]}:{addr[1]}", flush=True)


def run_server(args: argparse.Namespace) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(args.backlog)
        print(f"manual action test server listening on {args.host}:{args.port}", flush=True)

        while True:
            conn, addr = srv.accept()
            with conn:
                handle_connection(conn, addr, args)


def summarize_actions(actions: np.ndarray) -> str:
    first = np.array2string(actions[0], precision=4, suppress_small=True)
    if len(actions) == 1:
        return f"shape={actions.shape}, action={first}"
    last = np.array2string(actions[-1], precision=4, suppress_small=True)
    return f"shape={actions.shape}, first={first}, last={last}"


def run_robot(args: argparse.Namespace) -> None:
    executor: JointActionExecutor | None = None
    rate = None
    if args.execute:
        if args.action_dim != 14:
            raise ValueError("--execute requires --action-dim 14")
        executor = JointActionExecutor(args)
        state = executor.wait_for_state(args.state_timeout)
        rate = executor.rospy.Rate(args.publish_rate)
        print(f"joint feedback ok, initial_state={np.array2string(state, precision=4, suppress_small=True)}")

    with socket.create_connection((args.server_host, args.port), timeout=args.connect_timeout) as sock:
        sock.settimeout(args.socket_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        bridge.send_packet(sock, {"type": "reset", "version": bridge.PROTOCOL_VERSION})
        response, _ = bridge.recv_packet(sock)
        if response.get("type") != "reset_ok":
            raise RuntimeError(f"reset failed: {response}")
        print("reset ok")

        for request_id in range(1, args.num_requests + 1):
            if executor is None:
                state = np.zeros(args.action_dim, dtype=np.float32)
            else:
                state = executor.current_state()
                if state is None:
                    state = executor.wait_for_state(args.state_timeout)

            header: dict[str, Any] = {
                "type": "infer",
                "version": bridge.PROTOCOL_VERSION,
                "request_id": request_id,
                "state": state.astype(float).tolist(),
                "task": args.task,
                "robot_type": args.robot_type,
                "timestamp": time.time(),
            }
            bridge.send_packet(sock, header)
            response, _ = bridge.recv_packet(sock)
            if response.get("type") == "error":
                raise RuntimeError(f"server error on request {request_id}: {response.get('message')}")
            if response.get("type") != "action_chunk":
                raise RuntimeError(f"unexpected response on request {request_id}: {response}")

            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2 or actions.shape[1] != args.action_dim:
                raise ValueError(f"expected actions shape [T, {args.action_dim}], got {actions.shape}")

            latency = response.get("server_latency_s")
            print(
                f"request {request_id}: {summarize_actions(actions)}, server_latency_s={latency}",
                flush=True,
            )

            if executor is not None:
                for action in actions:
                    if executor.rospy.is_shutdown():
                        return
                    executor.publish_action(action)
                    rate.sleep()

            if args.request_interval > 0:
                time.sleep(args.request_interval)

    mode = "executed" if executor is not None and not args.dry_run else "received"
    print(f"communication ok: {mode} {args.num_requests} response chunk(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual action TCP communication test.")
    parser.add_argument("--role", choices=("server", "robot"), required=True)
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    parser.add_argument("--server-host", default="127.0.0.1", help="Robot-side server IP/hostname.")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--socket-timeout", type=float, default=10.0)
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--request-interval", type=float, default=0.2)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.003)
    parser.add_argument("--max-target-delta", type=float, default=0.005)
    parser.add_argument("--pattern", choices=("zero", "ramp", "sine", "alternating"), default="sine")
    parser.add_argument(
        "--actions",
        default=None,
        help=(
            "Optional manual delta sequence as JSON or semicolon-separated rows, "
            "for example '[[0,0,...],[0.01,0,...]]' or '0 0 0; 0.01 0 0'."
        ),
    )
    parser.add_argument(
        "--absolute-actions",
        action="store_true",
        help="Interpret --actions as absolute targets. By default --actions are small deltas from current state.",
    )
    cycle_group = parser.add_mutually_exclusive_group()
    cycle_group.add_argument("--cycle", dest="cycle", action="store_true", default=True)
    cycle_group.add_argument("--no-cycle", dest="cycle", action="store_false")
    execute_group = parser.add_mutually_exclusive_group()
    execute_group.add_argument("--execute", dest="execute", action="store_true", default=True)
    execute_group.add_argument("--no-execute", dest="execute", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Read ROS state but log commands instead of publishing.")
    parser.add_argument("--ros-node-name", default="manual_action_comm_test")
    parser.add_argument("--puppet-arm-left-topic", default="/puppet/joint_left")
    parser.add_argument("--puppet-arm-right-topic", default="/puppet/joint_right")
    parser.add_argument("--puppet-arm-left-cmd-topic", default="/master/joint_left")
    parser.add_argument("--puppet-arm-right-cmd-topic", default="/master/joint_right")
    parser.add_argument("--publish-rate", type=float, default=10.0)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--disable-step-limit", action="store_true")
    parser.add_argument(
        "--max-joint-step",
        default="0.002,0.002,0.002,0.002,0.002,0.002,0.02",
        help="Per publish max joint delta: 1, 7, or 14 comma-separated values.",
    )
    parser.add_argument("--joint-min", default=None, help="Optional 1, 7, or 14 comma-separated joint lower bounds.")
    parser.add_argument("--joint-max", default=None, help="Optional 1, 7, or 14 comma-separated joint upper bounds.")
    parser.add_argument("--task", default="manual action communication test")
    parser.add_argument("--robot-type", default="comm_test")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action_dim <= 0:
        raise ValueError("--action-dim must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be > 0")

    if args.role == "server":
        run_server(args)
    else:
        run_robot(args)


if __name__ == "__main__":
    main()
