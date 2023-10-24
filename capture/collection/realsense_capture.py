from PyQt5 import QtCore, QtGui
import pyrealsense2 as rs
import numpy as np
import cv2
import json

SERIAL_NUMBERS = ["636203000257"]  # , "846112071065"]
RGB_SHARPNESS = str(80)


class CameraCapturer(QtCore.QThread):
    """
    Used to capture images from the RealSense SR305 or D435 cameras.
    """

    ImageUpdate = QtCore.pyqtSignal(QtGui.QImage)
    DepthUpdate = QtCore.pyqtSignal(QtGui.QImage)
    CaptureUpdate = QtCore.pyqtSignal(object)

    def __init__(
        self,
        serial_number,
    ):
        super(CameraCapturer, self).__init__()
        self.capture = False
        self.serial_number = serial_number
        self.setMaxResolution()

    def setMaxResolution(self):
        """
        Set the RGB and Depth resolution.
        """
        realsense_ctx = rs.context()

        self.rgb_width = 1920
        self.rgb_height = 1080

        for x in realsense_ctx.devices:
            if x.get_info(rs.camera_info.serial_number) == self.serial_number:
                cam_name = x.get_info(rs.camera_info.name)
                if "D435" in cam_name:
                    self.depth_width = 1280
                    self.depth_height = 720
                elif "SR305" in cam_name:
                    self.depth_width = 640
                    self.depth_height = 480
                else:
                    raise ValueError("Camera not supported yet !")

    def getIntrinsic(self):
        """
        Returns a dict with camera intrinsics.
        Requires the camera to be running.
        """
        intr = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return dict(
            fx=intr.fx,
            fy=intr.fy,
            height=intr.height,
            width=intr.width,
            ppx=intr.ppx,
            ppy=intr.ppy,
            distorsion=dict(model=str(intr.model), coeffs=intr.coeffs),
        )

    def getDeviceInfo(self):
        """
        Returns a dict with camera info.
        Requires the camera to be running.
        """
        return dict(
            name=self.device.get_info(rs.camera_info.name),
            firmware_version=self.device.get_info(
                rs.camera_info.firmware_version
            ),
            product_id=self.device.get_info(rs.camera_info.product_id),
            serial_number=self.device.get_info(rs.camera_info.serial_number),
        )

    def run(self):
        """
        Starts the camera.
        Capture an image (RGB+Depth) every time the capture attribute is set
        to True.
        """
        # Start the camera and configure it
        self.ThreadActive = True
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_device(self.serial_number)
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            30,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.rgb_width,
            self.rgb_height,
            rs.format.rgb8,
            30,
        )

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        # Increase sharpness for D435 camera
        if "D435" in self.device.get_info(rs.camera_info.name):
            self.advnc_mode = rs.rs400_advanced_mode(self.device)
            params = json.loads(self.advnc_mode.serialize_json())
            params["parameters"]["controls-color-sharpness"] = RGB_SHARPNESS
            if type(next(iter(params))) != str:
                params = {
                    k.encode("utf-8"): v.encode("utf-8")
                    for k, v in params.items()
                }
            json_string = str(params).replace("'", '"')
            self.advnc_mode.load_json(json_string)

        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        depths = []

        while self.ThreadActive:
            # Acquire frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # Emit RGB image for display
            FlippedImage = cv2.flip(color_image, 1)
            ConvertToQtFormat = QtGui.QImage(
                FlippedImage.data,
                FlippedImage.shape[1],
                FlippedImage.shape[0],
                QtGui.QImage.Format_RGB888,
            )
            Pic = ConvertToQtFormat.scaled(640, 360, QtCore.Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)

            # Emit Depth image for display
            depth_image_meters = (
                np.asanyarray(aligned_depth_frame.get_data())
                * self.depth_scale
            )
            rgb_depth = np.repeat(
                np.expand_dims(depth_image_meters, axis=-1), repeats=3, axis=2
            )
            rgb_depth = np.log(1 + rgb_depth)
            rgb_depth = rgb_depth / np.max(rgb_depth) * 255
            rgb_depth = rgb_depth.astype(np.uint8)
            rgb_depth = cv2.flip(rgb_depth, 1)
            depth_ConvertToQtFormat = QtGui.QImage(
                rgb_depth.data,
                rgb_depth.shape[1],
                rgb_depth.shape[0],
                QtGui.QImage.Format_RGB888,
            )
            Depth = depth_ConvertToQtFormat.scaled(
                640, 360, QtCore.Qt.KeepAspectRatio
            )
            self.DepthUpdate.emit(Depth)

            # Emit images for capture
            if self.capture:
                if len(depths) < 10:
                    if len(depths) == 0:
                        tokeep_color_image = cv2.rotate(
                            color_image, cv2.ROTATE_90_CLOCKWISE
                        )
                    depths.append(
                        cv2.rotate(depth_image_meters, cv2.ROTATE_90_CLOCKWISE)
                    )
                else:
                    self.CaptureUpdate.emit(
                        [
                            tokeep_color_image,
                            np.array(depths),
                            self.serial_number,
                        ]
                    )
                    depths = []
                    self.capture = False

    def stop(self):
        """
        Stop the thread and the camera capture.
        """
        self.ThreadActive = False
        self.pipeline.stop()
        self.quit()


class Waiter(QtCore.QThread):
    """
    Helper class.
    Used to capture both UV and RGBD images.
    """

    Done = QtCore.pyqtSignal(object)

    def __init__(self, num_cam):
        super(Waiter, self).__init__()
        self.num_cam = num_cam
        self.count = 0

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            if self.num_cam == self.count:
                self.Done.emit(1)
                self.count = 0

    def stop(self):
        self.ThreadActive = False
        self.quit()
