import argparse
import json
import os
import shutil

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="./",
        help="Path to the folder containing all sessions.",
    )
    args = parser.parse_args()
    root = args.path
    sources = ["D", "grid2d", "grid3d", "RGB", "seg", "UV", "UVmap"]
    targets = ["depth", "grid2d", "grid3d", "rgb", "seg", "uv", "uvmap"]
    extensions = [".mat", ".mat", ".mat", ".png", ".mat", ".png", ".mat"]

    os.makedirs(os.path.join(root, "samples"), exist_ok=True)
    for dir in targets + ["sample_metadata"]:
        os.makedirs(os.path.join(root, "samples", dir), exist_ok=True)

    for sess in sorted(os.listdir(root)):
        if sess.isdigit():
            cameras = [
                x[-1]
                for x in os.listdir(os.path.join(root, sess))
                if x.startswith("grid2d")
            ]
            with open(os.path.join(root, sess, "metadata.json"), "r") as f:
                metadata_sess = json.load(f)

            for cam in cameras:
                annotated_samples = [
                    x[:-4]
                    for x in os.listdir(
                        os.path.join(root, sess, f"grid2d{cam}")
                    )
                ]
                print(f"Copying samples from session {sess} and camera {cam}")

                for sample in tqdm(annotated_samples):
                    sample_name = f"{sess}_{sample}_{cam}"

                    for source, target, ext in zip(
                        sources, targets, extensions
                    ):
                        shutil.copy(
                            os.path.join(
                                root, sess, f"{source}{cam}", f"{sample}{ext}"
                            ),
                            os.path.join(
                                root,
                                "samples",
                                target,
                                sample_name + ext,
                            ),
                        )

                    with open(
                        os.path.join(
                            root, sess, "sampleMetadata", f"{sample}.json"
                        ),
                        "r",
                    ) as f:
                        sample_info = json.load(f)

                    sample_metadata = dict(
                        sample_info=sample_info, cam=metadata_sess[f"cam{cam}"]
                    )

                    with open(
                        os.path.join(
                            root,
                            "samples",
                            "sample_metadata",
                            sample_name + ".json",
                        ),
                        "w",
                    ) as f:
                        json.dump(sample_metadata, f)
