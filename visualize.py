import os
import cv2
import glob
import h5py
import math
import time
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import rerun as rr
import argparse


def blueprint_row_images(origins):
    from rerun.blueprint import Horizontal, Spatial2DView

    return Horizontal(
        *(
            Spatial2DView(
                name=org,
                origin=org,
            )
            for org in origins
        ),
    )


class HDF5Dataset:
    def __init__(self, args):
        self.args = args
        path = os.path.join(args.root, args.task)
        ds = glob.glob(os.path.join(path, "*.h5"))
        ds.extend(glob.glob(os.path.join(path, "*.hdf5")))
        self.ds = ds

    def log_images(
        self, root, step, compressed=False, raw=False, start=[0, 0, 0], primary="image"
    ):
        if raw:
            ls = []
            for i, camera_name in enumerate(["left", "right", "front"]):
                if compressed and primary == "image":
                    s = start[i]
                    l = root[f"/obs/image/{camera_name}/len"][step]
                    image = root[f"/obs/image/{camera_name}/raw"][s : s + l]
                    image = cv2.imdecode(image, 1)
                    rr.log(f"/{primary}/{camera_name}", rr.Image(image))
                    ls.append(l)
                else:
                    image = root[f"/obs/{primary}/{camera_name}/raw"][step]
                    if primary == "depth":
                        image = (image / max(image.max(), 1) * 255).astype(np.uint8)
                    rr.log(
                        f"/{primary}/{camera_name}",
                        rr.Image(image),
                    )
                    ls.append(0)
            return ls
        for _, camera_name in enumerate(["left", "right", "front"]):
            if compressed and primary == "image":
                image = root[f"/obs/image/{camera_name}"][step]
                length = root[f"/len/image/{camera_name}"][step]
                image = cv2.imdecode(image[:length], 1)
                rr.log(f"/{primary}/{camera_name}", rr.Image(image))
            else:
                image = root[f"/obs/{primary}/{camera_name}"][step]
                if primary == "depth":
                    image = (image / image.max() * 255).astype(np.uint8)
                rr.log(
                    f"/{primary}/{camera_name}",
                    rr.Image(image),
                )

    def log_qpos_dict(self, step, root, raw=False):
        if raw:
            pose_left = root["/act/qpos/left/raw"][step]
            pose_right = root["/act/qpos/right/raw"][step]
        else:
            pose = root["/act/qpos"][step]
            pose_left, pose_right = pose[:7], pose[7:]
        for i, (left, right) in enumerate(zip(pose_left[:-1], pose_right[:-1])):
            rr.log(f"/action_dict/joint/{i}/left", rr.Scalars(left))
            rr.log(f"/action_dict/joint/{i}/right", rr.Scalars(right))
        rr.log("/action_dict/joint/gripper/left", rr.Scalars(pose_left[-1]))
        rr.log("/action_dict/joint/gripper/right", rr.Scalars(pose_right[-1]))

        if raw:
            pose_left = root["/act/qpos/left/vel"][step]
            pose_right = root["/act/qpos/right/vel"][step]
        else:
            pose = root["/act/qvel"][step]
            pose_left, pose_right = pose[:7], pose[7:]
        for i, (left, right) in enumerate(zip(pose_left[:-1], pose_right[:-1])):
            rr.log(f"/action_dict/velocity/{i}/left", rr.Scalars(left))
            rr.log(f"/action_dict/velocity/{i}/right", rr.Scalars(right))
        rr.log("/action_dict/velocity/gripper/left", rr.Scalars(pose_left[-1]))
        rr.log("/action_dict/velocity/gripper/right", rr.Scalars(pose_right[-1]))

        if raw:
            pose_left = root["/act/qpos/left/eff"][step]
            pose_right = root["/act/qpos/right/eff"][step]
        else:
            pose = root["/act/qeff"][step]
            pose_left, pose_right = pose[:7], pose[7:]
        for i, (left, right) in enumerate(zip(pose_left[:-1], pose_right[:-1])):
            rr.log(f"/action_dict/effort/{i}/left", rr.Scalars(left))
            rr.log(f"/action_dict/effort/{i}/right", rr.Scalars(right))
        rr.log("/action_dict/effort/gripper/left", rr.Scalars(pose_left[-1]))
        rr.log("/action_dict/effort/gripper/right", rr.Scalars(pose_right[-1]))

    def log_epos_dict(self, step, root, raw=False):
        if raw:
            pose_left = root["/act/epos/left/raw"][step]
            pose_right = root["/act/epos/right/raw"][step]
        else:
            pose = root["/act/epos"][step]
            pose_left, pose_right = pose[:7], pose[7:]
        translation, orientation = pose_left[:3], pose_left[3:6]
        rotation_mat = Rotation.from_euler("xyz", orientation).as_matrix()

        rr.log(
            "/action_dict/cartesian_position/cord/left",
            rr.Transform3D(translation=translation, mat3x3=rotation_mat),
        )
        #        rr.log(
        #            "/action_dict/cartesian_position/origin/left", rr.Points3D([translation])
        #        )

        translation, orientation = pose_right[:3], pose_right[3:6]
        rotation_mat = Rotation.from_euler("xyz", orientation).as_matrix()
        rr.log(
            "/action_dict/cartesian_position/cord/right",
            rr.Transform3D(translation=translation, mat3x3=rotation_mat),
        )
        #        rr.log(
        #            "/action_dict/cartesian_position/origin/right", rr.Points3D([translation])
        #        )

        for i, (left, right) in enumerate(zip(pose_left[:-1], pose_right[:-1])):
            rr.log(f"/action_dict/endpose/{i}/left", rr.Scalars(left))
            rr.log(f"/action_dict/endpose/{i}/right", rr.Scalars(right))
        rr.log("/action_dict/endpose/gripper/left", rr.Scalars(pose_left[-1]))
        rr.log("/action_dict/endpose/gripper/right", rr.Scalars(pose_right[-1]))

    def log_robot_dataset(
        self, entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]]
    ):
        for i, episode in enumerate(self.ds):
            path, name = episode.rsplit("/", 1)  # raw_000001.h5
            idx = int(name.rsplit(".", 1)[0].rsplit("_", 1)[-1])
            if name != self.args.episode:  # idx != self.args.episode_idx:
                continue
            print(f"visualize `{episode}`")
            if name.startswith(("raw", "align_v")):
                start = [0, 0, 0]
                with h5py.File(episode, "r") as root:
                    compressed = root.attrs.get("compress", False)
                    depth_max_step = 0
                    if "depth" in root["/obs"]:
                        depth_max_step = min(
                            root["/obs/depth/left/ts"].shape[0],
                            root["/obs/depth/right/ts"].shape[0],
                            root["/obs/depth/front/ts"].shape[0],
                        )
                    image_max_step = min(
                        root["/obs/image/left/ts"].shape[0],
                        root["/obs/image/right/ts"].shape[0],
                        root["/obs/image/front/ts"].shape[0],
                    )
                    qpos_max_step = min(
                        root["/act/qpos/left/ts"].shape[0],
                        root["/act/qpos/right/ts"].shape[0],
                    )
                    epos_max_step = min(
                        root["/act/epos/left/ts"].shape[0],
                        root["/act/epos/right/ts"].shape[0],
                    )
                    max_step = max(
                        [depth_max_step, image_max_step, qpos_max_step, epos_max_step]
                    )
                    for step in range(max_step):
                        if step < image_max_step:
                            ns = root["/obs/image/left/ts"][step]
                            rr.set_time("left_image_time", timestamp=ns / 1e9)
                            lengths = self.log_images(
                                root,
                                step,
                                compressed,
                                raw=True,
                                start=start,
                                primary="image",
                            )
                            start = list(x + y for x, y in zip(start, lengths))
                        if step < depth_max_step:
                            ns = root["/obs/depth/left/ts"][step]
                            rr.set_time("left_depth_time", timestamp=ns / 1e9)
                            lengths = self.log_images(
                                root,
                                step,
                                compressed,
                                raw=True,
                                start=start,
                                primary="depth",
                            )
                        if step < qpos_max_step:
                            ns = root["/act/qpos/left/ts"][i]
                            rr.set_time("left_qpos_time", timestamp=ns / 1e9)
                            self.log_qpos_dict(step, root, raw=True)
                        if step < epos_max_step:
                            ns = root["/act/epos/left/ts"][i]
                            rr.set_time("left_epos_time", timestamp=ns / 1e9)
                            self.log_epos_dict(step, root, raw=True)
            else:  # elif False: # aligned
                cur_time_ns = 0
                with h5py.File(episode, "r") as root:
                    compressed = root.attrs.get("compress", False)
                    max_step = root["/ts/image"].shape[0]
                    for step in range(max_step):
                        ns = root["/ts/image"][:, 0][step]
                        rr.set_time("left_view_time", timestamp=ns / 1e9)
                        self.log_images(root, step, compressed)
                        self.log_qpos_dict(step, root)
                        self.log_epos_dict(step, root)

    def blueprint(self):
        from rerun.blueprint import (
            Blueprint,
            Horizontal,
            Vertical,
            Spatial3DView,
            TimeSeriesView,
            Tabs,
            SelectionPanel,
            TimePanel,
            TextDocumentView,
        )

        return Blueprint(
            Horizontal(
                Vertical(
                    Spatial3DView(name="spatial view", origin="/", contents=["/**"]),
                    blueprint_row_images(
                        [
                            f"/depth/{camera_name}"
                            for camera_name in ["left", "right", "front"]
                        ]
                    ),
                    blueprint_row_images(
                        [
                            f"/image/{camera_name}"
                            for camera_name in ["left", "right", "front"]
                        ]
                    ),
                    row_shares=[2, 1],
                ),
                Vertical(
                    Tabs(  # Tabs for all the different time serieses.
                        Vertical(
                            *(
                                [
                                    TimeSeriesView(origin=f"/action_dict/joint/{i}")
                                    for i in range(6)
                                ]
                                + [
                                    TimeSeriesView(origin="/action_dict/joint/gripper"),
                                ]
                            ),
                            name="joint",
                        ),
                        Vertical(
                            *(
                                [
                                    TimeSeriesView(origin=f"/action_dict/velocity/{i}")
                                    for i in range(6)
                                ]
                                + [
                                    TimeSeriesView(
                                        origin="/action_dict/velocity/gripper"
                                    ),
                                ]
                            ),
                            name="velocity",
                        ),
                        Vertical(
                            *(
                                [
                                    TimeSeriesView(origin=f"/action_dict/effort/{i}")
                                    for i in range(6)
                                ]
                                + [
                                    TimeSeriesView(
                                        origin="/action_dict/effort/gripper"
                                    ),
                                ]
                            ),
                            name="effort",
                        ),
                        Vertical(
                            *(
                                [
                                    TimeSeriesView(origin=f"/action_dict/endpose/{i}")
                                    for i in range(6)
                                ]
                                + [
                                    TimeSeriesView(
                                        origin="/action_dict/endpose/gripper"
                                    ),
                                ]
                            ),
                            name="endpose",
                        ),
                        active_tab=0,
                    ),
                    row_shares=[7, 1],
                ),
                column_shares=[2, 1],
            ),
            SelectionPanel(expanded=False),
            TimePanel(expanded=False),
        )


def truncated_radians(deg: float) -> float:
    return float(int(math.radians(deg) * 1000.0)) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizes the HDF5 dataset using Rerun."
    )

    parser.add_argument(
        "--root", required=False, type=Path, default="/home/agilex/data"
    )
    parser.add_argument("--task", required=False, type=str)
    parser.add_argument(
        "--episode",
        action="store",
        type=str,
        help="episode name.",
        default="raw_000000.h5",
        required=True,
    )
    args = parser.parse_args()

    rlds_scene = HDF5Dataset(args)

    rr.init("hdf5-visualized", spawn=True)

    rr.log(
        "/action_dict/cartesian_position/cord/left",
        rr.Boxes3D(
            half_sizes=np.array([1, 1, 1]) * 0.06,
            fill_mode="solid",
            colors=(255, 0, 0),
            labels="L",
            centers=[0, 0.5, 0],
        ),
        # rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 1, 0], colors=(255, 0, 0), labels="R"),
    )
    rr.log(
        "/action_dict/cartesian_position/cord/right",
        rr.Boxes3D(
            half_sizes=np.array([1, 1, 1]) * 0.06,
            fill_mode="solid",
            colors=(0, 255, 0),
            labels="R",
            centers=[0, -0.5, 0],
        ),
        # rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 1, 0], colors=(0, 255, 0), labels="R"),
    )

    rr.send_blueprint(rlds_scene.blueprint())

    rlds_scene.log_robot_dataset({})


#    rr.set_time("tick", sequence=0)
#    rr.log(
#        "/box",
#        rr.Boxes3D(half_sizes=[3.0, 2.0, 1.0], fill_mode=rr.components.FillMode.Solid, colors=(255, 0, 0), labels="L"),
#        #rr.Arrows3D(origins=[0, 0, 0], vectors=[0, 10, 0], colors=(0, 255, 0), labels="R"),
#    )
#
#    for t in range(100):
#        rr.set_time("tick", sequence=t + 1)
#        rr.log(
#            "/box",
#            rr.Transform3D(
#                translation=[0, 0, t / 10.0],
#                rotation_axis_angle=rr.RotationAxisAngle(axis=[0.0, 1.0, 0.0], radians=truncated_radians(t * 4)),
#            ),
#        )
#
#        time.sleep(0.2)


if __name__ == "__main__":
    main()
    rr.log("piper", rr.TextDocument("episode", media_type="text/markdown"))
