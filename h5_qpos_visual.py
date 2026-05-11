import argparse

import h5py
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot H5 arm qpos.")
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
        left_qpos = f["act/qpos/left/raw"][:]
        right_qpos = f["act/qpos/right/raw"][:]

    print("left_qpos:", left_qpos.shape)
    print("right_qpos:", right_qpos.shape)

    plt.figure()
    plt.plot(left_qpos)
    plt.title("Left arm qpos")
    plt.xlabel("timestep")
    plt.ylabel("joint position")
    plt.legend([f"joint_{i}" for i in range(left_qpos.shape[1])])
    plt.show()

    plt.figure()
    plt.plot(right_qpos)
    plt.title("Right arm qpos")
    plt.xlabel("timestep")
    plt.ylabel("joint position")
    plt.legend([f"joint_{i}" for i in range(right_qpos.shape[1])])
    plt.show()


if __name__ == "__main__":
    main()
