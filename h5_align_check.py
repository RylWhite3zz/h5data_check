import argparse

import h5py
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot H5 timestamp alignment.")
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="align_v0_interp_000000.h5",
        help="Path to the input H5 file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with h5py.File(args.h5_path, "r") as f:
        front_ts = f["obs/image/front/ts"][:]
        left_img_ts = f["obs/image/left/ts"][:]
        right_img_ts = f["obs/image/right/ts"][:]

        left_qpos_ts = f["act/qpos/left/ts"][:]
        right_qpos_ts = f["act/qpos/right/ts"][:]

    plt.figure()
    plt.plot(front_ts, label="front image")
    plt.plot(left_img_ts, label="left image")
    plt.plot(right_img_ts, label="right image")
    plt.plot(left_qpos_ts, label="left qpos")
    plt.plot(right_qpos_ts, label="right qpos")
    plt.title("Timestamps")
    plt.xlabel("index")
    plt.ylabel("timestamp")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
