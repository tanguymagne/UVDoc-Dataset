import argparse
import os

import cv2
import hdf5storage as h5
import numpy as np
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the final dataset",
    )

    args = parser.parse_args()
    root = args.path

    all_samples = sorted(
        [x[:-4] for x in os.listdir(os.path.join(root, "grid3d"))]
    )
    mins_grid3d = []
    maxs_grid3d = []
    mins_wc = []
    maxs_wc = []
    for sample in tqdm(all_samples):
        # Get stats for Grid3D
        grid3d = h5.loadmat(os.path.join(root, "grid3d", f"{sample}.mat"))[
            "grid3d"
        ]
        mins_grid3d.append(np.min(grid3d, axis=(0, 1)))
        maxs_grid3d.append(np.max(grid3d, axis=(0, 1)))

        # Get stats for world coordinates
        wc = cv2.imread(
            os.path.join(root, "wc", f"{sample}.exr"),
            cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR,
        )
        wc = cv2.cvtColor(wc, cv2.COLOR_BGR2RGB)

        mins_wc.append(np.min(wc, axis=(0, 1)))
        maxs_wc.append(np.max(wc, axis=(0, 1)))

    print(f"Grid3D min : {np.min(np.array(mins_grid3d), axis=0)}")
    print(f"Grid3D Max : {np.max(np.array(maxs_grid3d), axis=0)}")
    print()
    print(f"WC min : {np.min(np.array(mins_wc), axis=0)}")
    print(f"WC Max : {np.max(np.array(maxs_wc), axis=0)}")
