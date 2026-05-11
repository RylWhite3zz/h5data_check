import argparse

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert three H5 camera streams to MP4.")
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="align_v0_interp_000000.h5",
        help="Path to the input H5 file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="episode_three_cameras.mp4",
        help="Path to the output MP4 file.",
    )
    return parser.parse_args()


def get_compressed_frame(f, camera_name, idx):
    base = f"obs/image/{camera_name}"

    raw = f[f"{base}/raw"]
    lengths = f[f"{base}/len"][:]

    offsets = np.concatenate([[0], np.cumsum(lengths)])

    start = offsets[idx]
    end = offsets[idx + 1]

    encoded = np.asarray(raw[start:end], dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"第 {idx} 帧解码失败，camera={camera_name}")

    return img


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    new_w = int(w * target_h / h)
    return cv2.resize(img, (new_w, target_h))


def main():
    args = parse_args()

    global cv2
    import cv2

    with h5py.File(args.h5_path, "r") as f:
        cameras = ["left", "right", "front"]

        num_frames = min(
            int(f[f"obs/image/{cam}/cnt"][0])
            for cam in cameras
        )

        fps = 30
        frames = []

        for i in range(num_frames):
            imgs = []

            for cam in cameras:
                img = get_compressed_frame(f, cam, i)
                img = resize_to_height(img, 360)

                cv2.putText(
                    img,
                    cam,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )

                imgs.append(img)

            canvas = cv2.hconcat(imgs)
            frames.append(canvas)

        h, w = frames[0].shape[:2]

        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        for frame in frames:
            # visualize.py logs the decoded image directly to rr.Image; keep that
            # display color order while composing, then convert for OpenCV video IO.
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()

    print("saved:", args.output)


if __name__ == "__main__":
    main()
