import argparse
import json
import multiprocessing as mp
import os
import random
from os.path import join as pjoin

import cv2
import hdf5storage as h5
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    grey_dilation,
)
from tqdm import tqdm

from augment_with_flip import create_augmented_dataset_with_flips


def create_color_grid():
    """
    Create an image containing 3x3 squares with different colors.
    This image will be used for color transfer.
    """
    # Create an image containing squares with different colors
    color_grid = np.zeros([300, 300, 3], dtype=np.uint8)
    color_grid[:100, :100] = [0, 0, 0]
    color_grid[100:200, :100] = [0, 255, 255]
    color_grid[200:, :100] = [0, 255, 0]
    color_grid[:100, 100:200] = [255, 255, 255]
    color_grid[100:200, 100:200] = [255, 255, 0]
    color_grid[200:, 100:200] = [255, 255, 255]
    color_grid[:100, 200:] = [255, 0, 255]
    color_grid[100:200, 200:] = [0, 0, 255]
    color_grid[200:, 200:] = [255, 0, 0]
    return color_grid


def color_transfer_fn(fg, bg, mask, warped_color_grid):
    """
    Function to transfer the color tone of the foreground (the document)
     to the background.

    """
    # Per channel color correction (allow to shift colors)
    fg = fg[mask[:, :, 0] == 1]

    fg_perchannel_mean = np.mean(fg, axis=0)
    fg_perchannel_mean -= np.median(fg_perchannel_mean)
    fg_perchannel_mean = np.clip(fg_perchannel_mean, -64, 64)

    color_corrected = np.copy(bg).astype(float)
    color_corrected = color_corrected + fg_perchannel_mean
    color_corrected = np.clip(color_corrected, 0, 255).astype(np.uint8)

    # Global color correction (allow to shift brightness)
    fg_colored = warped_color_grid[mask[:, :, 0] == 1]
    fg_global_mean = np.mean(fg_colored)
    color_corrected_mean = np.mean(color_corrected)
    diff = fg_global_mean - color_corrected_mean
    if diff > 0:
        diff *= 0.2
    final = color_corrected + diff

    final = np.clip(final, 0, 255).astype(np.uint8)

    return final


def fill(data, invalid=None):
    """
    Taken from https://stackoverflow.com/a/27745627

    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where
                 data value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """

    if invalid is None:
        invalid = np.isnan(data)

    ind = distance_transform_edt(
        invalid, return_distances=False, return_indices=True
    )
    return data[tuple(ind)]


def apply_texture(geom_path, texture, uv_path):
    """
    Compose a texture with a geometry using the uvmap.
    """

    # Load geometry, texture, background and uvmap
    geom = cv2.cvtColor(cv2.imread(geom_path), cv2.COLOR_BGR2RGB)
    if isinstance(texture, str):
        texture = cv2.cvtColor(cv2.imread(texture), cv2.COLOR_BGR2RGB)
    uvmap = h5.loadmat(uv_path)["uv"]

    # Warp the texture based on the uv
    torch_texture_unwarp = torch.from_numpy(
        np.expand_dims(texture.transpose(2, 0, 1), axis=0)
    ).float()
    uvmap = torch.from_numpy(np.expand_dims(uvmap * 2 - 1, axis=0)).float()
    warped_texture = F.grid_sample(
        torch_texture_unwarp, uvmap, align_corners=False
    )
    warped_texture = (
        np.clip(warped_texture[0].numpy().transpose(1, 2, 0), 0, 255) / 255
    )

    # Create masks (Segmentation of the document)
    grey = np.all(warped_texture == 0.5, axis=-1)
    warped_texture[grey] = np.nan

    mask = 1 - np.all(np.isnan(warped_texture), axis=-1).astype(int)
    mask_small = binary_erosion(mask).astype(int)

    mask = np.expand_dims(mask, axis=-1)
    mask_small = np.expand_dims(mask_small, axis=-1)

    # Set the one pixel at the border of the document to its nearest neighbord
    # value because it might contains wrong value
    warped_texture[np.repeat(~mask_small.astype(bool), 3, axis=-1)] = np.nan
    warped_texture = fill(warped_texture)
    warped_texture[np.repeat(~mask_small.astype(bool), 3, axis=-1)] = 1

    warped_texture_save = np.copy(warped_texture)
    warped_texture_save[np.repeat(~mask_small.astype(bool), 3, axis=-1)] = 0
    warped_texture_save = (warped_texture_save * 255).astype(np.uint8)

    # Combine geometry and texture
    blur_texture = np.copy(warped_texture) * 255
    for i in range(3):
        blur_texture[:, :, i] = gaussian_filter(blur_texture[:, :, i], 0.5)

    geom_textures_mult = blur_texture.astype(float) * geom.astype(float) / 255
    geom_textures = geom_textures_mult * 0.75 * mask + geom * (1 - 0.75 * mask)

    return (
        geom_textures,
        mask_small,
        warped_texture_save,
    )


def apply_background(
    geom_path,
    background_path,
    uv_path,
    geom_textures,
    mask_small,
    color_transfer,
):
    """
    Add a background to the image created from the geometry and the texture.
    """
    # Load geometry, texture, background and uvmap
    geom = cv2.cvtColor(cv2.imread(geom_path), cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB)
    uvmap = h5.loadmat(uv_path)["uv"]
    uvmap = torch.from_numpy(np.expand_dims(uvmap * 2 - 1, axis=0)).float()

    # Create the blending mask between background and document
    white = torch.from_numpy(np.ones([1, 3, 1000, 1000])).float()
    white[:, :, 0] = 0
    white[:, :, -1] = 0
    white[:, :, :, 0] = 0
    white[:, :, :, -1] = 0

    warped_white = F.grid_sample(white, uvmap, align_corners=False)
    warped_white = warped_white[0].numpy().transpose(1, 2, 0)
    warped_white[np.isnan(warped_white)] = 0
    warped_white = gaussian_filter(warped_white, sigma=0.75)
    warped_white = np.clip(warped_white, 0, 1)
    warped_white = grey_dilation(warped_white[:, :, 0], size=(3, 3))
    warped_white = grey_dilation(warped_white, size=(3, 3))
    mask = np.expand_dims(warped_white, axis=-1)

    # Adjust color of background to match the document
    if color_transfer:
        warped_color_grid = apply_texture(
            geom_path, create_color_grid(), uv_path
        )[0]
        background = color_transfer_fn(
            geom, background, mask_small, warped_color_grid
        )

    # Blend background with document
    geom_textures = np.copy(geom_textures)
    geom_textures[np.repeat(~mask_small.astype(bool), 3, axis=-1)] = np.nan
    geom_textures = fill(geom_textures)

    background = cv2.resize(background, (geom.shape[:2][::-1]))
    res = geom_textures * mask + (1 - mask) * background

    return res.astype(np.uint8)


def create_sample(
    i, path, name, sample_name, background_path, texture_path, color_transfer
):
    """
    Create an image and its associated metadata
    """
    # Create image and save it
    img_path = pjoin(path, name, "img_geom", f"{sample_name}.png")
    uv_path = pjoin(path, name, "uvmap", f"{sample_name}.mat")
    img, mask_small, warped_texture = apply_texture(
        img_path, texture_path, uv_path
    )
    img_RGB = apply_background(
        img_path, background_path, uv_path, img, mask_small, color_transfer
    )

    cv2.imwrite(
        pjoin(path, name, "img", f"{i:05d}.png"),
        cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        pjoin(path, name, "warped_textures", f"{i:05d}.png"),
        cv2.cvtColor(warped_texture, cv2.COLOR_RGB2BGR),
    )

    # Create and save metadata
    md = dict(
        geom_name=sample_name,
        texture_name=texture_path.split("textures/")[-1],
        background_name=background_path.split("backgrounds/")[-1],
        sample_id=f"{i:05d}",
    )
    with open(pjoin(path, name, "metadata_sample", f"{i:05d}.json"), "w") as f:
        json.dump(md, f)

    return f"Done sample {i}"


def create_final_dataset(
    path, n_sample, color_transfer, name, num_processes, benchmark_set=False
):
    """
    Create the full final dataset.
    """
    # Load the texture, background and sample path
    with open(pjoin(path, name, "split.json"), "r") as f:
        infos = json.load(f)
    all_textures = [pjoin(path, "textures", x) for x in infos["textures"]]
    all_backgrounds = [
        pjoin(path, "backgrounds", x) for x in infos["backgrounds"]
    ]

    all_samples = [
        x[:-5] for x in sorted(os.listdir(pjoin(path, name, "metadata_geom")))
    ]

    # Create and run in multiple processes all tasks to create final samples
    iterator = iter(all_samples)
    TASKS = []

    if benchmark_set:
        random.shuffle(all_textures)
        random.shuffle(all_backgrounds)

        assert len(all_textures) >= n_sample, "Not enough textures"
        assert len(all_backgrounds) >= n_sample, "Not enough backgrounds"

    for i in tqdm(range(n_sample)):
        try:
            sample_name = next(iterator)
        except:
            iterator = iter(all_samples)
            sample_name = next(iterator)

        if not benchmark_set:
            background_path = np.random.choice(all_backgrounds)
            texture_path = np.random.choice(all_textures)
        else:
            background_path = all_backgrounds[i]
            texture_path = all_textures[i]

        TASKS.append(
            (
                i,
                path,
                name,
                sample_name,
                background_path,
                texture_path,
                color_transfer,
            )
        )
    with mp.Pool(num_processes) as pool:
        print(f"Creating {len(TASKS)} samples.")
        results = [pool.apply_async(create_sample, t) for t in TASKS]
        for r in results:
            print("\t", r.get())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the raw dataset")
    parser.add_argument(
        "--n-sample",
        "-n",
        type=int,
        default=5000,
        help="Number of samples to create (int)",
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[488, 712],
        help="Width and height (2 int)",
    )
    parser.add_argument(
        "--subprocess",
        type=int,
        default=32,
        help="Number of subprocesses to use",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="To create a validation set",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="The ratio to use for dataset splitting. Between 0 and 1",
    )
    parser.add_argument(
        "--no_color_transfer",
        action="store_true",
        default=False,
        help="Whether to use color transfer or not",
    )
    parser.add_argument(
        "--benchmark_set",
        action="store_true",
        default=False,
        help="Whether to create a benchmark set (meaning tight cropping and no flip)",
    )
    args = parser.parse_args()

    split_ratio = args.ratio
    name = "final"
    color_transfer = not args.no_color_transfer
    print(f"Color transfer : {color_transfer}")
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # Make dirs
    for subdir in [
        "",
        "grid2d",
        "grid3d",
        "img",
        "img_geom",
        "metadata_geom",
        "metadata_sample",
        "seg",
        "wc",
        "uvmap",
        "textures",
        "warped_textures",
    ]:
        if args.split:
            os.makedirs(
                os.path.join(args.path, name + "_train", subdir),
                exist_ok=False,
            )
            os.makedirs(
                os.path.join(args.path, name + "_val", subdir), exist_ok=False
            )
        else:
            os.makedirs(os.path.join(args.path, name, subdir), exist_ok=False)

    create_augmented_dataset_with_flips(
        path=args.path,
        name=name,
        img_size=args.img_size,
        n_sample=args.n_sample,
        split=args.split,
        split_ratio=split_ratio,
        num_processes=args.subprocess,
        benchmark_set=args.benchmark_set,
    )
    if args.split:
        create_final_dataset(
            args.path,
            args.n_sample,
            color_transfer=color_transfer,
            name=name + "_train",
            num_processes=args.subprocess,
            benchmark_set=args.benchmark_set,
        )
        create_final_dataset(
            args.path,
            int(args.n_sample * split_ratio),
            color_transfer=color_transfer,
            name=name + "_val",
            num_processes=args.subprocess,
            benchmark_set=args.benchmark_set,
        )
    else:
        create_final_dataset(
            args.path,
            args.n_sample,
            color_transfer=color_transfer,
            name=name,
            num_processes=args.subprocess,
            benchmark_set=args.benchmark_set,
        )
