import argparse
import json
import multiprocessing as mp
import os
import time

import hdf5storage as h5
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def get_ind(i, j, shape):
    return int(i * shape[1] + j)


def create_faces(grid):
    shape = grid.shape
    faces = []
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            faces.append(
                [
                    3,
                    get_ind(i, j, shape),
                    get_ind(i + 1, j, shape),
                    get_ind(i, j + 1, shape),
                ]
            )
            faces.append(
                [
                    3,
                    get_ind(i + 1, j + 1, shape),
                    get_ind(i + 1, j, shape),
                    get_ind(i, j + 1, shape),
                ]
            )
    return faces


def visualize_grid3d(root, ind):
    ind = f"{ind:05d}"

    # Load sample metadata
    with open(os.path.join(root, "metadata_sample", f"{ind}.json"), "r") as f:
        geom_name = json.load(f)["geom_name"]

    # Load 3D grid
    grid3d_path = os.path.join(root, "grid3d", f"{geom_name}.mat")
    grid3d = h5.loadmat(grid3d_path)["grid3d"]

    # Convert to pyvista.PolyData object and plot
    faces = np.hstack(create_faces(grid3d))
    vertices = np.copy(grid3d).reshape(-1, 3)
    surf = pv.PolyData(vertices, faces)
    surf.plot(
        background="white",
        show_edges=True,
        line_width=1,
        show_axes=False,
        eye_dome_lighting=True,
    )


def visualize_sample(root, ind):
    ind = f"{ind:05d}"

    # Load sample metadata
    with open(os.path.join(root, "metadata_sample", f"{ind}.json"), "r") as f:
        geom_name = json.load(f)["geom_name"]

    # Load everything
    grid2d_path = os.path.join(root, "grid2d", f"{geom_name}.mat")
    grid2d = h5.loadmat(grid2d_path)["grid2d"]
    grid2d = grid2d.reshape(-1, 2)

    img_blank_path = os.path.join(root, "img_geom", f"{geom_name}.png")
    img_blank = plt.imread(img_blank_path)

    img_final_path = os.path.join(root, "img", f"{ind}.png")
    img_final = plt.imread(img_final_path)

    uvmap_path = os.path.join(root, "uvmap", f"{geom_name}.mat")
    uvmap = h5.loadmat(uvmap_path)["uv"]

    warped_texture_path = os.path.join(root, "warped_textures", f"{ind}.png")
    warped_texture = plt.imread(warped_texture_path)

    # Plot
    fig, axes = plt.subplots(1, 6, figsize=(20, 6.5), num="UVDoc Visualizer")
    fig.suptitle(f"Sample {ind}", fontsize=16, y=0.98)
    axes[0].imshow(img_blank)
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)
    axes[0].set_title("Initial image")

    axes[1].imshow(warped_texture)
    axes[1].xaxis.set_visible(False)
    axes[1].yaxis.set_visible(False)
    axes[1].set_title("Warped Texture")

    axes[2].imshow(uvmap[:, :, 0])
    axes[2].xaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)
    axes[2].set_title("UVmap 0")

    axes[3].imshow(uvmap[:, :, 1])
    axes[3].xaxis.set_visible(False)
    axes[3].yaxis.set_visible(False)
    axes[3].set_title("UVmap 1")

    axes[4].imshow(img_final)
    axes[4].xaxis.set_visible(False)
    axes[4].yaxis.set_visible(False)
    axes[4].set_title("Final image")

    axes[5].imshow(img_final)
    axes[5].scatter(x=grid2d[:, 0], y=grid2d[:, 1], c="r", s=2)
    axes[5].xaxis.set_visible(False)
    axes[5].yaxis.set_visible(False)
    axes[5].set_title("Final image with grid overlay")

    fig = fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, help="Path to the final dataset")
    parser.add_argument(
        "--sample", type=int, help="ID of the sample to visualize"
    )

    args = parser.parse_args()

    p1 = mp.Process(target=visualize_sample, args=(args.path, args.sample))
    p2 = mp.Process(target=visualize_grid3d, args=(args.path, args.sample))
    p1.start()
    time.sleep(0.5)
    p2.start()
