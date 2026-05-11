import argparse

import h5py
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot H5 end-effector positions.")
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
        left_epos = f["act/epos/left/raw"][:]
        right_epos = f["act/epos/right/raw"][:]

    print("left_epos:", left_epos.shape)
    print("right_epos:", right_epos.shape)

    plt.figure()
    plt.plot(left_epos[:, :3])
    plt.title("Left end-effector position")
    plt.xlabel("timestep")
    plt.ylabel("position")
    plt.legend(["x", "y", "z"])
    plt.show()

    plt.figure()
    plt.plot(right_epos[:, :3])
    plt.title("Right end-effector position")
    plt.xlabel("timestep")
    plt.ylabel("position")
    plt.legend(["x", "y", "z"])
    plt.show()


if __name__ == "__main__":
    main()
