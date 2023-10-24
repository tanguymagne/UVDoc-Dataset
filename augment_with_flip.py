import json
import multiprocessing as mp
import os
from os.path import join as pjoin

import cv2
import hdf5storage as h5
import imageio
import numpy as np
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
imageio.plugins.freeimage.download()


DEPTH_THRESHOLD = 0.5


def get_all_images(folder_path):
    """
    Find all .jpg and .png images in a specified folder and its sub-folder.
    """

    all_images = []
    for f in os.listdir(folder_path):
        if os.path.isdir(pjoin(folder_path, f)):
            all_images.extend(
                [
                    pjoin(f, x)
                    for x in os.listdir(pjoin(folder_path, f))
                    if x.endswith((".jpg", ".png"))
                ]
            )
        elif f.endswith((".jpg", ".png")):
            all_images.append(f)
    return sorted(all_images)


def copy_and_downsample_texture(
    path, img_size, texture_names, save_name, test_set=False
):
    """
    Copy and downsample the texture images to the final folder.
    """
    print("Copying and downsampling textures")
    for texture_name in tqdm(texture_names):
        texture_path = pjoin(path, "textures", texture_name)
        texture = cv2.imread(texture_path)
        if not test_set:
            texture = cv2.resize(texture, img_size)
        cv2.imwrite(
            pjoin(path, save_name, "textures", texture_path.split("/")[-1]),
            texture,
        )


def crop_tight_fn(img, seg, grid2d, uvmap, wc):
    """
    Function used to crop the image tightly around the document.
    """
    import random

    size = img.shape

    minx = np.floor(np.amin(grid2d[:, :, 0])).astype(int)
    maxx = np.ceil(np.amax(grid2d[:, :, 0])).astype(int)
    miny = np.floor(np.amin(grid2d[:, :, 1])).astype(int)
    maxy = np.ceil(np.amax(grid2d[:, :, 1])).astype(int)
    s = 20
    s = min(
        min(s, minx), miny
    )  # s shouldn't be smaller than actually available natural padding is
    s = min(min(s, size[1] - 1 - maxx), size[0] - 1 - maxy)

    img = img[miny - s : maxy + s, minx - s : maxx + s]
    seg = seg[miny - s : maxy + s, minx - s : maxx + s]
    uvmap = uvmap[miny - s : maxy + s, minx - s : maxx + s]
    wc = wc[miny - s : maxy + s, minx - s : maxx + s]

    cx1 = random.randint(0, max(s - 5, 1))
    cx2 = random.randint(0, max(s - 5, 1)) + 1
    cy1 = random.randint(0, max(s - 5, 1))
    cy2 = random.randint(0, max(s - 5, 1)) + 1

    img = img[cy1:-cy2, cx1:-cx2]
    seg = seg[cy1:-cy2, cx1:-cx2]
    uvmap = uvmap[cy1:-cy2, cx1:-cx2]
    wc = wc[cy1:-cy2, cx1:-cx2]

    t = miny - s + cy1
    l = minx - s + cx1
    grid2d[:, :, 0] -= l
    grid2d[:, :, 1] -= t

    return img, seg, grid2d, uvmap, wc


def create_flipped_copies(
    path, name, sample_id, img_size, benchmark_set=False
):
    """
    Copy a sample and create 3 augmentations of it with horizontal
    and/or vertical flips.
    """
    for flip_horizontal in [True, False] if not benchmark_set else [False]:
        for flip_vertical in [True, False] if not benchmark_set else [False]:
            # Get paths
            img_path = pjoin(path, "samples", "rgb", f"{sample_id}.png")
            seg_path = pjoin(path, "samples", "seg", f"{sample_id}.mat")
            grid2d_path = pjoin(path, "samples", "grid2d", f"{sample_id}.mat")
            grid3d_path = pjoin(path, "samples", "grid3d", f"{sample_id}.mat")
            uv_path = pjoin(path, "samples", "uvmap", f"{sample_id}.mat")
            depth_path = pjoin(path, "samples", "depth", f"{sample_id}.mat")
            metadata_path = os.path.join(
                path, "samples", "sample_metadata", f"{sample_id}.json"
            )  # use to compute world coordinates

            # Load files
            img_geom = cv2.imread(img_path)
            seg = h5.loadmat(seg_path)["seg"]
            grid2d = h5.loadmat(grid2d_path)["grid2d"]
            grid3d = h5.loadmat(grid3d_path)["grid3d"]
            uvmap = h5.loadmat(uv_path)["uv"]
            depth = h5.loadmat(depth_path)["depth"][0]
            with open(metadata_path, "r") as f:
                intrinsics = json.load(f)["cam"]["intrinsics"]
            K = np.zeros([3, 3])
            K[1, 1] = intrinsics["fx"]
            K[0, 0] = intrinsics["fy"]
            K[1, 2] = intrinsics["ppx"]
            K[0, 2] = intrinsics["ppy"]
            K[2, 2] = 1

            depth = np.array(depth) * seg.astype(float)
            depth[
                depth > DEPTH_THRESHOLD
            ] = 0  # Remove too high value of depths
            depth[np.isnan(depth)] = 0  # Remove NaNs

            # Create 3D coordinates from depth
            depth[depth == 0] = np.nan

            u = np.linspace(0, depth.shape[0] - 1, depth.shape[0])
            v = np.linspace(0, depth.shape[1] - 1, depth.shape[1])
            u_coor, v_coor = np.meshgrid(v, u)

            pts = np.vstack(
                [
                    u_coor.flatten(),
                    v_coor.flatten(),
                    np.ones_like(u_coor).flatten(),
                ]
            )

            wc = np.expand_dims(depth.flatten(), axis=0) * (
                np.linalg.inv(K) @ pts
            )

            wc = wc.T.reshape((depth.shape[0], depth.shape[1], -1))
            wc = wc - np.nanmean(wc, axis=(0, 1))

            x = np.copy(wc[:, :, 0])
            y = np.copy(wc[:, :, 1])
            z = np.copy(wc[:, :, 2])
            wc[:, :, 0], wc[:, :, 1], wc[:, :, 2] = -z, -x, -y
            wc[np.isnan(wc)] = 0  # Like DewarpNet, make background 0

            # Flip
            if flip_horizontal:
                img_geom = img_geom[:, ::-1]
                seg = seg[:, ::-1]

                uvmap = uvmap[:, ::-1]
                uvmap[:, :, 0] = 1 - uvmap[:, :, 0]

                grid2d[:, :, 0] = uvmap.shape[1] - grid2d[:, :, 0]
                grid2d = grid2d[:, ::-1]

                grid3d[:, :, 1] *= -1
                grid3d = grid3d[:, ::-1]

                wc[:, :, 1] *= -1
                wc = wc[:, ::-1]

            if flip_vertical:
                img_geom = img_geom[::-1]
                seg = seg[::-1]

                uvmap = uvmap[::-1]
                uvmap[:, :, 1] = 1 - uvmap[:, :, 1]

                grid2d[:, :, 1] = uvmap.shape[0] - grid2d[:, :, 1]
                grid2d = grid2d[::-1]

                grid3d[:, :, 2] *= -1
                grid3d = grid3d[::-1]

                wc[:, :, 2] *= -1
                wc = wc[::-1]

            # Resize
            ratio = img_size[1] / img_size[0]
            excess = int((img_geom.shape[0] - ratio * img_geom.shape[1]) / 2)
            min_pos = np.min(grid2d[:, :, 1])
            max_pos = img_geom.shape[0] - np.max(grid2d[:, :, 1])

            MIN_DISTANCE_FROM_BORDER = 25

            if min_pos - MIN_DISTANCE_FROM_BORDER < excess:
                b = int(max(0, min_pos - MIN_DISTANCE_FROM_BORDER))
                diff = excess - b
                t = excess + diff
            elif max_pos - MIN_DISTANCE_FROM_BORDER < excess:
                t = int(max(0, max_pos - MIN_DISTANCE_FROM_BORDER))
                diff = excess - t
                b = excess + diff
            else:
                t = excess
                b = excess

            seg = seg[b:-t]
            wc = wc[b:-t]
            uvmap = uvmap[b:-t]
            img_geom = img_geom[b:-t]
            grid2d[:, :, 1] -= b
            if (
                np.min(grid2d[:, :, 1]) < MIN_DISTANCE_FROM_BORDER
                or np.max(grid2d[:, :, 1])
                > img_geom.shape[0] - MIN_DISTANCE_FROM_BORDER
            ):
                print(
                    f"Wrong Cropping for sample {sample_id}."
                    "The sample might be not fully entirely in the image!"
                )

            if benchmark_set:
                img_geom, seg, grid2d, uvmap, wc = crop_tight_fn(
                    img_geom, seg, grid2d, uvmap, wc
                )
            else:
                # If no crop tight, images get resized
                zeros = np.expand_dims(np.zeros_like(seg), axis=-1)
                uvmap = np.dstack([uvmap, zeros])
                uvmap = cv2.resize(uvmap, img_size)[:, :, :2]

                grid2d = (
                    grid2d
                    / np.array([[seg.shape[::-1]]])
                    * np.array([[img_size]])
                )

                img_geom = cv2.resize(img_geom, img_size)
                seg = cv2.resize(seg, img_size)
                wc = cv2.resize(
                    wc, img_size, interpolation=cv2.INTER_NEAREST
                )  # Use nearest neighbor interpolation to avoid interpolation of wc values outside of the document

            # Save files
            sample_name = (
                f"{sample_id}_{int(flip_horizontal)}_{int(flip_vertical)}"
            )
            cv2.imwrite(
                pjoin(path, name, "img_geom", f"{sample_name}.png"),
                img_geom,
            )
            h5.savemat(
                pjoin(path, name, "seg", f"{sample_name}.mat"),
                dict(seg=seg),
            )
            h5.savemat(
                pjoin(path, name, "grid2d", f"{sample_name}.mat"),
                dict(grid2d=grid2d),
            )
            h5.savemat(
                pjoin(path, name, "grid3d", f"{sample_name}.mat"),
                dict(grid3d=grid3d),
            )
            h5.savemat(
                pjoin(path, name, "uvmap", f"{sample_name}.mat"),
                dict(uv=uvmap),
            )
            # Save wc as an .exr file using imageio
            wc = wc.astype("float32")
            imageio.imwrite(pjoin(path, name, "wc", f"{sample_name}.exr"), wc)

            # Create and save metadata
            md = dict(
                sample_id=sample_id,
                flip_horizontal=str(flip_horizontal),
                flip_vertical=str(flip_vertical),
            )
            with open(
                pjoin(path, "samples", "sample_metadata", f"{sample_id}.json"),
                "r",
            ) as f:
                md.update(json.load(f))

            with open(
                pjoin(path, name, "metadata_geom", f"{sample_name}.json"),
                "w",
            ) as f:
                json.dump(md, f)
    return f"Done {sample_id}"


def create_augmented_dataset_with_flips(
    path,
    name,
    img_size,
    n_sample,
    split=True,
    split_ratio=0.0,
    num_processes=8,
    benchmark_set=False,
):
    """
    Create an augmented copy of the dataset.
    In the copy, each sample is augmented with horizontal and/or
    vertical flips.
    Also, split the dataset into train and validation if asked.
    """
    # Get all geometries
    all_samples = [
        x[:-4] for x in get_all_images(pjoin(path, "samples", "rgb"))
    ]
    # Avoid copying all the geometries if we don't need to, except for the test set, where we want variety
    if n_sample < 4 * len(all_samples) and not benchmark_set:
        all_samples = all_samples[: int(n_sample / 4 + 1)]

    print(f"Number of different geometries : {len(all_samples)}")
    print(f"    With flips : {len(all_samples) * 4}")

    # Create a split partition if required
    if split:
        total_samples = len(all_samples)
        val_samples = int(total_samples * split_ratio)
        if val_samples == 0:
            raise ValueError(
                "You are using a too-small split ratio.\n"
                "There are no samples in the validation set.\n"
                "Please choose a higher split ratio, or do not use split."
            )

        partition = np.zeros(total_samples)
        partition[:val_samples] = 1
        np.random.shuffle(partition)
    else:
        partition = None

    def get_name(index):
        if partition is None:
            return name
        elif partition[index]:
            return name + "_val"
        else:
            return name + "_train"

    # Create and run in multiple processes all tasks to copy and augment
    # samples
    TASKS = [
        [path, get_name(i), sample_name, img_size, benchmark_set]
        for i, sample_name in enumerate(all_samples)
    ]
    with mp.Pool(num_processes) as pool:
        results = [pool.apply_async(create_flipped_copies, t) for t in TASKS]
        for r in results:
            print("\t", r.get())

    # Split also textures and backgrounds, and save the split in json file
    textures_path = pjoin(path, "textures")
    backgrounds_path = pjoin(path, "backgrounds")
    all_textures = get_all_images(textures_path)
    all_backgrounds = get_all_images(backgrounds_path)
    if split:
        total_textures = len(all_textures)
        val_textures = int(total_textures * split_ratio)
        partition_textures = np.zeros(total_textures)
        partition_textures[:val_textures] = 1
        np.random.shuffle(partition_textures)

        total_backgrounds = len(all_backgrounds)
        val_backgrounds = int(total_backgrounds * split_ratio)
        partition_backgrounds = np.zeros(total_backgrounds)
        partition_backgrounds[:val_backgrounds] = 1
        np.random.shuffle(partition_backgrounds)

        split_dict_train = dict(
            textures=[
                all_textures[i]
                for i in range(total_textures)
                if not partition_textures[i]
            ],
            backgrounds=[
                all_backgrounds[i]
                for i in range(total_backgrounds)
                if not partition_backgrounds[i]
            ],
            geoms=[
                all_samples[i]
                for i in range(total_samples)
                if not partition[i]
            ],
        )

        split_dict_val = dict(
            textures=[
                all_textures[i]
                for i in range(total_textures)
                if partition_textures[i]
            ],
            backgrounds=[
                all_backgrounds[i]
                for i in range(total_backgrounds)
                if partition_backgrounds[i]
            ],
            geoms=[
                all_samples[i] for i in range(total_samples) if partition[i]
            ],
        )
        with open(pjoin(path, name + "_train", "split.json"), "w") as f:
            json.dump(split_dict_train, f)
        with open(pjoin(path, name + "_val", "split.json"), "w") as f:
            json.dump(split_dict_val, f)

        copy_and_downsample_texture(
            path=path,
            img_size=img_size,
            texture_names=split_dict_train["textures"],
            save_name=name + "_train",
            test_set=benchmark_set,
        )
        copy_and_downsample_texture(
            path=path,
            img_size=img_size,
            texture_names=split_dict_val["textures"],
            save_name=name + "_val",
            test_set=benchmark_set,
        )

    else:
        split_dict = dict(
            textures=all_textures,
            backgrounds=all_backgrounds,
            geoms=all_samples,
        )
        with open(pjoin(path, name, "split.json"), "w") as f:
            json.dump(split_dict, f)

        copy_and_downsample_texture(
            path=path,
            img_size=img_size,
            texture_names=all_textures,
            save_name=name,
            test_set=benchmark_set,
        )
