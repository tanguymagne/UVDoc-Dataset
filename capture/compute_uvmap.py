import os

import cv2
import hdf5storage as h5
import numpy as np
from scipy.interpolate import griddata

GRID_SHAPE = (89, 61)


def create_UV_map(grid2d, seg, im_rgb):
    """
    Create the UVmap from the 2D grid
    """
    points = grid2d.reshape(-1, 2)

    xx = np.linspace(0, 1, GRID_SHAPE[1])
    yy = np.linspace(0, 1, GRID_SHAPE[0])

    yy, xx = np.meshgrid(xx, yy)
    values_V = xx.flatten()
    values_U = yy.flatten()

    xs = np.arange(im_rgb.shape[1])
    ys = np.arange(im_rgb.shape[0])
    ys, xs = np.meshgrid(xs, ys)
    pi = np.dstack([ys, xs]).reshape(-1, 2)

    U = griddata(points, values_U, pi).reshape(*im_rgb.shape[:2])
    V = griddata(points, values_V, pi).reshape(*im_rgb.shape[:2])

    seg_nan = seg.astype(float)
    seg_nan[seg_nan == 0] = np.nan

    uv = np.dstack([U * seg_nan, V * seg_nan])
    return uv


def save_uv(root_path, sample_id, cam_number):
    """
    Load the files required, create the UVmap, and save it.
    """
    # Load everything required
    grid2d_path = os.path.join(
        root_path, f"grid2d{cam_number}", f"{sample_id}.mat"
    )
    grid2d = h5.loadmat(grid2d_path)["grid2d"]
    seg_path = os.path.join(root_path, f"seg{cam_number}", f"{sample_id}.mat")
    seg = h5.loadmat(seg_path)["seg"]
    im_path = os.path.join(root_path, f"RGB{cam_number}", f"{sample_id}.png")
    im_rgb = cv2.imread(im_path)
    # Create the UVmap
    uv = create_UV_map(grid2d=grid2d, seg=seg, im_rgb=im_rgb)
    # Save the UVmap
    h5.savemat(
        os.path.join(root_path, f"UVmap{cam_number}", f"{sample_id}.mat"),
        dict(uv=uv),
    )


def process_starter(i, args):
    save_uv(*args)
    return f"Done sample {i}"


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to a session of capture."
    )

    args = parser.parse_args()
    root = args.path

    folders = [x for x in os.listdir(root) if x.startswith("seg")]
    for f in folders:
        cam_number = int(f[-1])
        samples = [
            x[:-4]
            for x in os.listdir(os.path.join(root, f))
            if x.endswith(".mat")
        ]
        with mp.Pool(8) as pool:
            TASKS = [(i, (root, s, cam_number)) for i, s in enumerate(samples)]
            print(f"Creating UV map for {len(TASKS)} samples:")
            results = [pool.apply_async(process_starter, t) for t in TASKS]
            for r in results:
                print("\t", r.get())
