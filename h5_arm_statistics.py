import argparse
from pathlib import Path

import h5py
import numpy as np


ARM_DATASETS = (
    ("qpos", "act/qpos/{arm}/raw"),
    ("qvel", "act/qpos/{arm}/vel"),
    ("effort", "act/qpos/{arm}/eff"),
    ("epos", "act/epos/{arm}/raw"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print left/right arm statistics for all H5 files in a folder."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing .h5/.hdf5 files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.h5",
        help="Glob pattern used to find files. Use '*.hdf5' for HDF5 suffix files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search folders recursively.",
    )
    parser.add_argument(
        "--include",
        default="qpos,epos",
        help="Comma-separated dataset groups to print: qpos,qvel,effort,epos.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimals shown for vectors.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print aggregate statistics across files.",
    )
    return parser.parse_args()


def iter_h5_files(folder, pattern, recursive):
    folder = Path(folder)
    glob_fn = folder.rglob if recursive else folder.glob
    files = sorted(path for path in glob_fn(pattern) if path.is_file())
    return [
        path
        for path in files
        if path.suffix.lower() in {".h5", ".hdf5"}
    ]


def fmt_vector(values, precision):
    values = np.asarray(values)
    return np.array2string(
        values,
        precision=precision,
        suppress_small=True,
        separator=", ",
    )


def stats_for_array(array):
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "shape": array.shape,
        "all_zero": bool(np.allclose(array, 0.0)),
        "nan_count": int(np.isnan(array).sum()),
        "min": np.nanmin(array, axis=0),
        "max": np.nanmax(array, axis=0),
        "mean": np.nanmean(array, axis=0),
        "std": np.nanstd(array, axis=0),
        "first": array[0],
        "last": array[-1],
    }


def print_stats(label, stats, precision):
    if stats is None:
        print(f"  {label}: empty")
        return

    print(f"  {label}: shape={stats['shape']} all_zero={stats['all_zero']} nan_count={stats['nan_count']}")
    print(f"    first: {fmt_vector(stats['first'], precision)}")
    print(f"    last : {fmt_vector(stats['last'], precision)}")
    print(f"    min  : {fmt_vector(stats['min'], precision)}")
    print(f"    max  : {fmt_vector(stats['max'], precision)}")
    print(f"    mean : {fmt_vector(stats['mean'], precision)}")
    print(f"    std  : {fmt_vector(stats['std'], precision)}")


class AggregateStats:
    def __init__(self):
        self.file_count = 0
        self.missing_count = 0
        self.empty_count = 0
        self.all_zero_count = 0
        self.row_count = 0
        self.value_count = None
        self.value_sum = None
        self.value_sumsq = None
        self.value_min = None
        self.value_max = None

    def add_missing(self):
        self.missing_count += 1

    def update(self, array):
        self.file_count += 1
        array = np.asarray(array, dtype=np.float64)
        if array.size == 0:
            self.empty_count += 1
            return

        if array.ndim == 1:
            array = array[:, None]

        if np.allclose(array, 0.0):
            self.all_zero_count += 1

        self.row_count += array.shape[0]
        finite = np.isfinite(array)
        counts = finite.sum(axis=0)
        values = np.where(finite, array, 0.0)
        value_sum = values.sum(axis=0)
        value_sumsq = np.square(values).sum(axis=0)

        with np.errstate(all="ignore"):
            value_min = np.nanmin(array, axis=0)
            value_max = np.nanmax(array, axis=0)

        if self.value_count is None:
            self.value_count = counts
            self.value_sum = value_sum
            self.value_sumsq = value_sumsq
            self.value_min = value_min
            self.value_max = value_max
            return

        self.value_count += counts
        self.value_sum += value_sum
        self.value_sumsq += value_sumsq
        self.value_min = np.fmin(self.value_min, value_min)
        self.value_max = np.fmax(self.value_max, value_max)

    def mean(self):
        return self.value_sum / np.maximum(self.value_count, 1)

    def std(self):
        mean = self.mean()
        variance = self.value_sumsq / np.maximum(self.value_count, 1) - np.square(mean)
        return np.sqrt(np.maximum(variance, 0.0))

    def print(self, label, precision):
        print(f"  {label}:")
        print(
            f"    files={self.file_count} missing={self.missing_count} empty={self.empty_count} "
            f"all_zero_files={self.all_zero_count} rows={self.row_count}"
        )
        if self.value_count is None:
            return
        print(f"    min : {fmt_vector(self.value_min, precision)}")
        print(f"    max : {fmt_vector(self.value_max, precision)}")
        print(f"    mean: {fmt_vector(self.mean(), precision)}")
        print(f"    std : {fmt_vector(self.std(), precision)}")


def load_dataset(file_handle, data_name, arm):
    key = data_name.format(arm=arm)
    if key not in file_handle:
        return key, None
    return key, file_handle[key][:]


def print_file_stats(path, selected_names, precision):
    print("=" * 80)
    print(path)

    try:
        with h5py.File(path, "r") as f:
            for group_name, key_template in ARM_DATASETS:
                if group_name not in selected_names:
                    continue

                for arm in ("left", "right"):
                    key, data = load_dataset(f, key_template, arm)
                    if data is None:
                        print(f"  {group_name} {arm}: missing key {key}")
                        continue
                    print_stats(f"{group_name} {arm} ({key})", stats_for_array(data), precision)
    except OSError as exc:
        print(f"  failed to read: {exc}")


def update_summary(path, selected_names, summary):
    try:
        with h5py.File(path, "r") as f:
            for group_name, key_template in ARM_DATASETS:
                if group_name not in selected_names:
                    continue

                for arm in ("left", "right"):
                    key, data = load_dataset(f, key_template, arm)
                    aggregate = summary[(group_name, arm)]
                    if data is None:
                        aggregate.add_missing()
                    else:
                        aggregate.update(data)
    except OSError:
        for group_name, _ in ARM_DATASETS:
            if group_name not in selected_names:
                continue
            for arm in ("left", "right"):
                summary[(group_name, arm)].add_missing()


def print_summary(summary, selected_names, precision):
    print("=" * 80)
    print("summary")
    for group_name, _ in ARM_DATASETS:
        if group_name not in selected_names:
            continue
        for arm in ("left", "right"):
            summary[(group_name, arm)].print(f"{group_name} {arm}", precision)


def main():
    args = parse_args()
    selected_names = {
        item.strip()
        for item in args.include.split(",")
        if item.strip()
    }
    valid_names = {name for name, _ in ARM_DATASETS}
    unknown = sorted(selected_names - valid_names)
    if unknown:
        raise ValueError(f"Unknown --include group(s): {unknown}. Valid groups: {sorted(valid_names)}")

    files = iter_h5_files(args.folder, args.pattern, args.recursive)
    print(f"found {len(files)} H5 file(s)")
    if not files:
        return

    summary = {
        (group_name, arm): AggregateStats()
        for group_name, _ in ARM_DATASETS
        for arm in ("left", "right")
    }
    for path in files:
        update_summary(path, selected_names, summary)
        if not args.summary_only:
            print_file_stats(path, selected_names, args.precision)

    print_summary(summary, selected_names, args.precision)


if __name__ == "__main__":
    main()
