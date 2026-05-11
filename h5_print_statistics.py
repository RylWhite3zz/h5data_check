import argparse

import h5py
import numpy as np


ts_keys = {
    "front image": "obs/image/front/ts",
    "left image": "obs/image/left/ts",
    "right image": "obs/image/right/ts",
    "left qpos": "act/qpos/left/ts",
    "right qpos": "act/qpos/right/ts",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Print H5 timestamp statistics.")
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
        for name, key in ts_keys.items():
            ts = f[key][:]
            diff = np.diff(ts)

            print("=" * 60)
            print(name, key)
            print("dtype:", ts.dtype)
            print("shape:", ts.shape)
            print("first 5:", ts[:5])
            print("last 5:", ts[-5:])
            print("min:", np.min(ts))
            print("max:", np.max(ts))
            print("span:", np.max(ts) - np.min(ts))

            if len(diff) > 0:
                print("diff min:", np.min(diff))
                print("diff max:", np.max(diff))
                print("diff median:", np.median(diff))
                print("non-positive diff count:", np.sum(diff <= 0))


if __name__ == "__main__":
    main()
