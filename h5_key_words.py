import argparse

import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Print H5 keys and selected values.")
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="align_v0_interp_000000.h5",
        help="Path to the input H5 file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        with h5py.File(args.h5_path, "r") as f:
            # visititems 会遍历每一个 key 和对应的对象 (obj)
            f.visititems(
                lambda name, obj: print(
                    f"名称: {name:<20} 形状: {getattr(obj, 'shape', 'N/A')}"
                )
            )

            data_cnt = f['act/epos/left/cnt'][0]
            print(f"cnt 的具体数值是: {data_cnt}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {args.h5_path}")
    except Exception as e:
        print(f"发生错误：{e}")


if __name__ == "__main__":
    main()
