import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "align.h5"

image_keys = []
array_keys = []

def collect_keys(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")

        if len(obj.shape) == 4 and obj.shape[-1] in [1, 3, 4]:
            image_keys.append(name)
        elif len(obj.shape) == 2:
            array_keys.append(name)

with h5py.File(path, "r") as f:
    f.visititems(collect_keys)

    print("\nPossible image keys:")
    for k in image_keys:
        print(" ", k)

    print("\nPossible low-dim array keys:")
    for k in array_keys:
        print(" ", k)

    # 显示第一个相机
    if image_keys:
        key = image_keys[0]
        print("Showing image key:", key)

        images = f[key]
        for i in range(len(images)):
            img = images[i]

            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow(key, img)
            if cv2.waitKey(30) == 27:
                break

        cv2.destroyAllWindows()

    # 画第一个二维数组
    if array_keys:
        key = array_keys[0]
        data = f[key][:]
        print("Plotting array key:", key, data.shape)

        plt.plot(data)
        plt.title(key)
        plt.xlabel("timestep")
        plt.show()
