import json
import os
import warnings

import hdf5storage as h5
import numpy as np
from scipy.interpolate import griddata

DEPTH_THRESHOLD = 0.5
GRID_SHAPE = (89, 61)


def _complete_nans(array2d, verbose=True):
    completed = np.copy(array2d)
    if verbose:
        print(f"Number of NaNs : {np.isnan(completed).sum()}")
    i = 1
    while np.where(np.isnan(completed))[0].shape[0] != 0:
        xs, ys = np.where(np.isnan(completed))
        xs_to_correct, ys_to_correct = [], []
        for x, y in zip(xs, ys):
            non_nan_neighbors = 0
            borders = 0
            if x + 1 < completed.shape[0]:
                if not np.isnan(completed[x + 1, y]):
                    non_nan_neighbors += 1
            else:
                borders += 1
            if x - 1 >= 0:
                if not np.isnan(completed[x - 1, y]):
                    non_nan_neighbors += 1
            else:
                borders += 1
            if y + 1 < completed.shape[1]:
                if not np.isnan(completed[x, y + 1]):
                    non_nan_neighbors += 1
            else:
                borders += 1
            if y - 2 >= 0:
                if not np.isnan(completed[x, y - 1]):
                    non_nan_neighbors += 1
            else:
                borders += 1

            if borders > 1 or non_nan_neighbors > 1:
                xs_to_correct.append(x)
                ys_to_correct.append(y)

        if verbose:
            print(f"Step {i} - Correcting {len(xs_to_correct)} nans.")
        xs, ys = np.where(np.isnan(completed))
        for x, y in zip(xs_to_correct, ys_to_correct):
            if x + 2 < completed.shape[0]:
                x_pos = (
                    completed[x + 1, y]
                    + completed[x + 1, y]
                    - completed[x + 2, y]
                )
            else:
                x_pos = np.nan
            if x - 2 >= 0:
                x_neg = (
                    completed[x - 1, y]
                    + completed[x - 1, y]
                    - completed[x - 2, y]
                )
            else:
                x_neg = np.nan
            if y + 2 < completed.shape[1]:
                y_pos = (
                    completed[x, y + 1]
                    + completed[x, y + 1]
                    - completed[x, y + 2]
                )
            else:
                y_pos = np.nan
            if y - 2 >= 0:
                y_neg = (
                    completed[x, y - 1]
                    + completed[x, y - 1]
                    - completed[x, y - 2]
                )
            else:
                y_neg = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                completed[x, y] = np.nanmean([x_pos, x_neg, y_pos, y_neg])
        i += 1
    return completed


def complete_nans_border(depth_grid, __iter=0):
    """
    Functions allowing to extrapolate values to borders for fields
    (e.g. depth) with missing values.
    """
    depth_copy = np.copy(depth_grid)

    left = depth_copy[:, 1] - depth_copy[:, 2] + depth_copy[:, 1]
    right = depth_copy[:, -2] - depth_copy[:, -3] + depth_copy[:, -2]
    bottom = depth_copy[1, :] - depth_copy[2, :] + depth_copy[1, :]
    top = depth_copy[-2, :] - depth_copy[-3, :] + depth_copy[-2, :]

    depth_copy[:, 0] = left
    depth_copy[:, -1] = right
    depth_copy[0, :] = bottom
    depth_copy[-1, :] = top

    if np.sum(np.isnan(depth_copy)) > 0 and __iter == 0:
        # Handling the completion of the corners
        return complete_nans_border(depth_copy, __iter=1)
    elif np.sum(np.isnan(depth_copy)) > 0 and __iter == 1:
        # In case there are nans not at the borders
        return _complete_nans(depth_copy, verbose=False)
    else:
        return depth_copy


def create_3dgrid(grid2d, depth, segmentation, K):
    """
    Create a 3D grid from a 2D grid, a depth image and the intrinsics of
    the camera.
    """

    # Process depth
    depths_processed = np.array(depth) * segmentation.astype(float)
    depths_processed[
        depths_processed > DEPTH_THRESHOLD
    ] = 0  # Remove too high value of depths
    depths_processed[depths_processed == 0] = np.nan

    # Interpolate depths on grid coordinate
    points = np.where(~np.isnan(depths_processed))
    points = np.dstack([points[1], points[0]])[0]

    values = depths_processed[~np.isnan(depths_processed)]

    pi = grid2d.reshape(-1, 2)

    depth_grid = griddata(points, values, pi)
    depth_grid = depth_grid.reshape(grid2d.shape[:2])

    depth_grid_full = complete_nans_border(depth_grid)

    # Create 3D grid from 2D pixel coordinate, depth map and calibration matrix
    pts = np.vstack(
        [
            grid2d[:, :, 0].flatten(),
            grid2d[:, :, 1].flatten(),
            np.ones_like(grid2d[:, :, 0]).flatten(),
        ]
    )
    grid3d = np.expand_dims(depth_grid_full.flatten(), axis=0) * (
        np.linalg.inv(K) @ pts
    )
    grid3d = grid3d.T.reshape((*GRID_SHAPE, -1))
    grid3d = grid3d - np.mean(grid3d, axis=(0, 1))

    x = np.copy(grid3d[:, :, 0])
    y = np.copy(grid3d[:, :, 1])
    z = np.copy(grid3d[:, :, 2])
    grid3d[:, :, 0], grid3d[:, :, 1], grid3d[:, :, 2] = -z, x, -y

    return grid3d


def save_3dgrid(root_path, sample_id, cam_number):
    """
    Load the files required, create the 3D grid, and save it.
    """
    # Load everything required
    grid2d_path = os.path.join(
        root_path, f"grid2d{cam_number}", f"{sample_id}.mat"
    )
    grid2d = h5.loadmat(grid2d_path)["grid2d"]
    seg_path = os.path.join(root_path, f"seg{cam_number}", f"{sample_id}.mat")
    seg = h5.loadmat(seg_path)["seg"]
    depth_path = os.path.join(root_path, f"D{cam_number}", f"{sample_id}.mat")
    depth = h5.loadmat(depth_path)["depth"]
    metadata_path = os.path.join(root_path, "metadata.json")

    with open(metadata_path, "r") as f:
        intrinsics = json.load(f)["cam1"]["intrinsics"]
    K = np.zeros([3, 3])
    K[1, 1] = intrinsics["fx"]
    K[0, 0] = intrinsics["fy"]
    K[1, 2] = intrinsics["ppx"]
    K[0, 2] = intrinsics["ppy"]
    K[2, 2] = 1

    # Create the 3D grid
    grid3d = create_3dgrid(
        grid2d=grid2d, depth=depth[0], segmentation=seg, K=K
    )
    # Saves the 3D grid
    h5.savemat(
        os.path.join(root_path, f"grid3d{cam_number}", f"{sample_id}.mat"),
        dict(grid3d=grid3d),
    )


def process_starter(i, args):
    save_3dgrid(*args)
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
            print(f"Creating grid for {len(TASKS)} samples:")
            results = [pool.apply_async(process_starter, t) for t in TASKS]
            for r in results:
                print("\t", r.get())
