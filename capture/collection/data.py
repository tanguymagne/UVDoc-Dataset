import os
import json
import cv2
import hdf5storage as h5

import numpy as np
from datetime import datetime
from typing import List

from realsense_capture import SERIAL_NUMBERS
from image_switcher import AlignementVerifier


def get_cam_number(serial_number):
    return np.where(np.array(SERIAL_NUMBERS) == serial_number)[0][0] + 1


class DataCollector:
    """
    Class that serves as backend for the collection.
    Controls how samples are saved.
    """

    def __init__(self, cameras: List[str] = []) -> None:
        self.root = None
        self.current_session = None
        self.sample_number = None
        self.cameras = cameras
        self.expecting = None  # "RGB" or "UVD"
        self.alignementVisualizers = dict()

    def setRoot(self, root):
        """
        Set the root to where data should be saved, and check if
        sessions/samples already exist.
        """
        self.root = root
        os.makedirs(root, exist_ok=True)

        self.current_session = self.getLastSession()
        self.sample_number = self.getSampleNumber()

    def createSession(self, intrinsics=None, devices_info=None):
        """
        Create a new session including all subfolders for saving the samples.
        """
        if self.current_session is None:
            new_session_id = f"{0:02d}"
        else:
            new_session_id = f"{int(self.current_session) + 1:02d}"
        if not os.path.isdir(os.path.join(self.root, new_session_id)):
            # Create session
            os.makedirs(os.path.join(self.root, new_session_id))
            metadata = dict(session_begin=str(datetime.now()), sample_number=0)

            # For each cameras, create all necessary subfolders, and update
            # metadata.
            for camera in self.cameras:
                cam_number = get_cam_number(camera)
                os.makedirs(
                    os.path.join(self.root, new_session_id, f"D{cam_number}")
                )
                os.makedirs(
                    os.path.join(self.root, new_session_id, f"RGB{cam_number}")
                )
                os.makedirs(
                    os.path.join(self.root, new_session_id, f"UV{cam_number}")
                )
                os.makedirs(
                    os.path.join(
                        self.root, new_session_id, f"UVmap{cam_number}"
                    )
                )
                os.makedirs(
                    os.path.join(
                        self.root, new_session_id, f"grid2d{cam_number}"
                    )
                )
                os.makedirs(
                    os.path.join(
                        self.root, new_session_id, f"grid3d{cam_number}"
                    )
                )
                os.makedirs(
                    os.path.join(self.root, new_session_id, f"seg{cam_number}")
                )

                metadata[f"cam{cam_number}"] = dict()
                if devices_info is not None and camera in devices_info:
                    metadata[f"cam{cam_number}"]["info"] = devices_info[camera]
                else:
                    metadata[f"cam{cam_number}"]["info"] = dict(
                        serial_number=camera
                    )
                if intrinsics is not None and camera in intrinsics:
                    metadata[f"cam{cam_number}"]["intrinsics"] = intrinsics[
                        camera
                    ]

            os.makedirs(
                os.path.join(self.root, new_session_id, "sampleMetadata")
            )

            # Create a annotate.json file that contains all samples that needs to
            # be annotated.
            annotate = []
            with open(
                os.path.join(self.root, new_session_id, "annotate.json"), "w"
            ) as f:
                json.dump(annotate, f)

            # Write metadata
            with open(
                os.path.join(self.root, new_session_id, "metadata.json"), "w"
            ) as f:
                json.dump(metadata, f)

            # Set current session and sample ID
            self.current_session = new_session_id
            self.sample_number = f"{0:05d}"
        else:
            raise ValueError(
                f"Session {new_session_id} already exists, please remove it."
            )

    def getAllSession(self):
        """
        Returns a list of int containing all sessions already created.
        """
        return [
            int(x)
            for x in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, x)) and x.isdigit()
        ]

    def getLastSession(self):
        """
        Returns the index of the last session or None if no session has been
        created.
        """
        all_sessions = self.getAllSession()
        if all_sessions != []:
            return f"{np.max(self.getAllSession()):02d}"
        else:
            return None

    def getSampleNumber(self):
        """
        Return the number of samples in the current session or None if no
        session has been created.
        """
        if self.current_session is not None:
            with open(
                os.path.join(self.root, self.current_session, "metadata.json"),
                "r",
            ) as f:
                md = json.load(f)
            return md["sample_number"]
        else:
            return None

    def addToAnnotate(self, serial_number):
        """
        Add the current sample from camera serial_number to the list of samples
        to annotate.
        """
        cam_number = int(get_cam_number(serial_number))
        with open(
            os.path.join(self.root, self.current_session, "annotate.json"),
            "r",
        ) as f:
            annotate = json.load(f)
        annotate.append(dict(sample=self.sample_number, cam=cam_number))
        with open(
            os.path.join(self.root, self.current_session, "annotate.json"),
            "w",
        ) as f:
            json.dump(annotate, f)

    def updateMetadata(self):
        """
        Update metadata.json with the current number of samples.
        """
        with open(
            os.path.join(self.root, self.current_session, "metadata.json"), "r"
        ) as f:
            metadata = json.load(f)
        metadata["sample_number"] = int(self.sample_number)
        with open(
            os.path.join(self.root, self.current_session, "metadata.json"), "w"
        ) as f:
            json.dump(metadata, f)

    def updateSample(self):
        """
        Update the number of samples.
        Used once a sample has been fully collected.
        """
        self.sample_number = f"{int(self.sample_number) + 1:05d}"
        self.updateMetadata()

    def save(self, image, depth, serial_number, sample_type):
        """
        Saves the input image/depth/sample_types to the correct folders,
        depending on the expected data type.
        """
        if self.expecting == "RGBD":
            self.saveRGBD(image, depth, serial_number, sample_type)
        elif self.expecting == "UV":
            self.saveUV(image, serial_number)
        else:
            raise ValueError("Can only save RGB or UVD.")

    def saveUV(self, UV_image, serial_number):
        """
        Saves the input UV image in the directory corresponding to the camera
        serial_number.
        """
        cam_number = get_cam_number(serial_number)
        cv2.imwrite(
            os.path.join(
                self.root,
                self.current_session,
                f"UV{cam_number}",
                f"{self.sample_number}.png",
            ),
            cv2.cvtColor(UV_image, cv2.COLOR_RGB2BGR),
        )

    def saveRGBD(self, RGB_image, depth_image, serial_number, sample_type):
        """
        Saves the input RGB image, Depth image and sample type in the
        directories corresponding to the camera serial_number.
        """
        cam_number = get_cam_number(serial_number)
        cv2.imwrite(
            os.path.join(
                self.root,
                self.current_session,
                f"RGB{cam_number}",
                f"{self.sample_number}.png",
            ),
            cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR),
        )
        h5.savemat(
            os.path.join(
                self.root,
                self.current_session,
                f"D{cam_number}",
                f"{self.sample_number}.mat",
            ),
            mdict={"depth": depth_image},
            format="7.3",
        )
        metadata_path = os.path.join(
            self.root,
            self.current_session,
            "sampleMetadata",
            f"{self.sample_number}.json",
        )
        if not os.path.isfile(metadata_path):
            with open(metadata_path, "w") as f:
                json.dump(sample_type, f)

    def verifyAlignement(self, serial_number):
        """
        Show a window to verify that the current sample's RGB and UV images
        from serial_number camera are aligned.
        """
        cam_number = get_cam_number(serial_number)
        rgb_path = os.path.join(
            self.root,
            self.current_session,
            f"RGB{cam_number}",
            f"{self.sample_number}.png",
        )
        uv_path = os.path.join(
            self.root,
            self.current_session,
            f"UV{cam_number}",
            f"{self.sample_number}.png",
        )
        self.alignementVisualizers[cam_number] = AlignementVerifier(
            rgb_path, uv_path
        )
        self.alignementVisualizers[cam_number].show()

    def deleteRGBD(self):
        """
        Delete the latest captured RGB-D data.
        """
        for cam in self.cameras:
            cam_number = get_cam_number(cam)
            os.remove(
                os.path.join(
                    self.root,
                    self.current_session,
                    f"RGB{cam_number}",
                    f"{self.sample_number}.png",
                ),
            )
            os.remove(
                os.path.join(
                    self.root,
                    self.current_session,
                    f"D{cam_number}",
                    f"{self.sample_number}.mat",
                ),
            )
        os.remove(
            os.path.join(
                self.root,
                self.current_session,
                "sampleMetadata",
                f"{self.sample_number}.json",
            ),
        )

    def deleteLast(self):
        """
        Delete the latest captured sample data.
        """
        self.sample_number = f"{int(self.sample_number) - 1:05d}"
        for cam in self.cameras:
            cam_number = get_cam_number(cam)
            for t, ext in zip(
                ["UV", "RGB", "D", "grid2d"], ["png", "png", "mat", "mat"]
            ):
                path = os.path.join(
                    self.root,
                    self.current_session,
                    f"{t}{cam_number}",
                    f"{self.sample_number}.{ext}",
                )
                if os.path.exists(path):
                    os.remove(path)

        os.remove(
            os.path.join(
                self.root,
                self.current_session,
                "sampleMetadata",
                f"{self.sample_number}.json",
            ),
        )
        self.updateMetadata()
        with open(
            os.path.join(self.root, self.current_session, "annotate.json"),
            "r",
        ) as f:
            annotate = json.load(f)
        if len(annotate) > 0 and int(annotate[-1]["sample"]) == int(
            self.sample_number
        ):
            annotate.pop(-1)
            with open(
                os.path.join(self.root, self.current_session, "annotate.json"),
                "w",
            ) as f:
                json.dump(annotate, f)
