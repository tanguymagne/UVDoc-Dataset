import argparse
import json
import os
import sys

import cv2
import numpy as np
from base_component import MinMaxGroup, ParameterGroup
from kp_annotation import GRID_SIZE, KeypointDetector, KeypointSorter
from kp_visualization import KPVisualizer
from PyQt5 import QtWidgets
from res_visualization import ResVisualizer
from seg_visualization import SegmentationVisualizer

if GRID_SIZE == [45, 31]:
    DEFAULT_PARAMS = dict(
        minDistBetweenBlobs=10,
        minThreshold=30,
        maxThreshold=205,
        thresholdStep=2,
        filterByArea=True,
        minArea=75,
        maxArea=250,
        filterByCircularity=True,
        minCircularity=0.6,
        maxCircularity=1.0,
        filterByConvexity=True,
        minConvexity=0.9,
        maxConvexity=1.0,
        filterByInertia=True,
        minInertiaRatio=0.25,
        maxInertiaRatio=1.0,
    )
elif GRID_SIZE == [89, 61]:
    DEFAULT_PARAMS = dict(
        minDistBetweenBlobs=5,
        minThreshold=30,
        maxThreshold=205,
        thresholdStep=2,
        filterByArea=True,
        minArea=30,
        maxArea=200,
        filterByCircularity=True,
        minCircularity=0.6,
        maxCircularity=1.0,
        filterByConvexity=True,
        minConvexity=0.9,
        maxConvexity=1.0,
        filterByInertia=True,
        minInertiaRatio=0.25,
        maxInertiaRatio=1.0,
    )

DEFAULT_DISTANCE_FROM_BORDER = 1


class KPDetectionOrdering(QtWidgets.QWidget):
    """
    Main class and UI used to annotate the 2D grid.
    """

    def __init__(self, annotated=[]) -> None:
        super(KPDetectionOrdering, self).__init__()
        self.setWindowTitle("Keypoint Detection")
        self.kpDetector = KeypointDetector()
        self.alreadySetup = False
        self.grid = []
        self.grid_index = []
        self.resVisualizer = ResVisualizer(annotated)
        self.kpSorter = KeypointSorter(None)

    def setup(self, rgb_path, uv_path, grid2d_path, segmentation_path):
        """
        Set the various images.
        """

        self.rgb_path = rgb_path
        self.uv_path = uv_path
        self.grid2d_path = grid2d_path
        self.segmentation_path = segmentation_path

        self.rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        self.min_rgb = np.min(self.rgb, axis=-1)

        self.uv_full = cv2.cvtColor(cv2.imread(uv_path), cv2.COLOR_BGR2RGB)
        self.uv = self.uv_full[:, :, 1]

        self.image = np.repeat(
            np.expand_dims(self.uv, axis=-1), 3, axis=-1
        ).astype(np.uint8)

        if not self.alreadySetup:
            self.setupUi()
            self.alreadySetup = True
        else:
            self.imageVisualizer.labelImage.image_raw = self.image
            self.imageVisualizer.labelImage.setKPImage()
            self.updateKPParams()

    def setupUi(self):
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)

        # Segmentation Display
        self.maskVisualizer = SegmentationVisualizer(self.rgb, self.uv)
        self.layout.addWidget(self.maskVisualizer)
        self.line = QtWidgets.QFrame()
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout.addWidget(self.line)

        # Parameter Selection
        verticalParametersLayout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(verticalParametersLayout)

        label = QtWidgets.QLabel()
        label.setText("<b>Parameters choice for KP detection</b>")
        verticalParametersLayout.addWidget(label)
        verticalParametersLayout.addStretch(1)

        self.groupDistanceBlob = MinMaxGroup(
            "", typ="i", value=10, min_value=1, max_value=500, increment=1
        )
        self.groupDistanceBlob.setTitle("Minimum distrance between blobs")
        self.groupDistanceBlob.setCheckable(False)

        self.groupThreshold = ParameterGroup(
            "Threshold", 100, 175, 0, 255, 1, "i"
        )
        self.groupThreshold.setTitle("Threshold Parameters")
        self.groupThreshold.setCheckable(False)
        self.groupThreshold.maxValueSelect.setCheckable(False)
        self.groupThreshold.minValueSelect.setCheckable(False)
        self.thresholdStep = MinMaxGroup(
            "", typ="i", value=5, min_value=1, max_value=255, increment=1
        )
        self.thresholdStep.setTitle("Threshold Step")
        self.thresholdStep.setCheckable(False)
        self.groupThreshold.layout.addWidget(self.thresholdStep)

        self.groupBoxArea = ParameterGroup("Area", 75, 250, 1, 5000, 1, "i")
        self.groupBoxCircularity = ParameterGroup(
            "Circularity", 0.6, 1.0, 0.0, 1.0, 0.01, "f"
        )
        self.groupBoxConvexity = ParameterGroup(
            "Convexity", 0.9, 1.0, 0.0, 1.0, 0.01, "f"
        )
        self.groupBoxInertia = ParameterGroup(
            "Inertia", 0.15, 1.0, 0.0, 1.0, 0.01, "f"
        )
        self.resetKPParams()

        verticalParametersLayout.addWidget(self.groupDistanceBlob)
        verticalParametersLayout.addWidget(self.groupThreshold)
        verticalParametersLayout.addWidget(self.groupBoxArea)
        verticalParametersLayout.addWidget(self.groupBoxCircularity)
        verticalParametersLayout.addWidget(self.groupBoxConvexity)
        verticalParametersLayout.addWidget(self.groupBoxInertia)
        verticalParametersLayout.addStretch(1)

        self.connectAll()

        # Image Display
        self.imageVisualizer = KPVisualizer(self.image)
        self.updateKPParams()

        # Button
        self.doneButton = QtWidgets.QPushButton()
        self.doneButton.setText("Done")
        self.doneButton.clicked.connect(self.isDone)
        self.doneButton.setEnabled(False)

        self.orderGridAction = QtWidgets.QAction()
        self.orderGridAction.setText("Order Grid (Space)")
        self.orderGridAction.setShortcut("Space")
        self.orderGridAction.triggered.connect(self.orderGrid)
        self.orderGridButton = QtWidgets.QToolButton()
        self.orderGridButton.setDefaultAction(self.orderGridAction)

        self.goBackAction = QtWidgets.QAction()
        self.goBackAction.setText("Go Back To (R)")
        self.goBackAction.setShortcut("R")
        self.goBackAction.triggered.connect(self.goBackTo)
        self.goBackButton = QtWidgets.QToolButton()
        self.goBackButton.setDefaultAction(self.goBackAction)

        self.resetParamButton = QtWidgets.QPushButton()
        self.resetParamButton.setText("Reset Parameters")
        self.resetParamButton.clicked.connect(self.resetKPParams)

        self.runDetectionButton = QtWidgets.QPushButton()
        self.runDetectionButton.setText("Run Detection")
        self.runDetectionButton.clicked.connect(self.updateKeypoints)

        layoutButton = QtWidgets.QHBoxLayout()
        layoutButton.addWidget(self.resetParamButton)
        layoutButton.addWidget(self.runDetectionButton)
        layoutButton.addWidget(self.goBackButton)
        self.goBackAction.setEnabled(False)
        layoutButton.addWidget(self.orderGridButton)
        layoutButton.addWidget(self.doneButton)

        layoutRight = QtWidgets.QVBoxLayout()
        layoutRight.addLayout(layoutButton)
        layoutRight.addWidget(self.imageVisualizer)

        self.layout.addLayout(layoutRight)

    def updateKeypoints(self):
        """
        Detects the keypoints based on the parameters provided.
        """
        img_kp = np.copy(self.image)
        keypoints = self.kpDetector.detectPoints(imageUV=img_kp)
        self.imageVisualizer.labelImage.setKPAlgo(
            [[k.pt[0], k.pt[1]] for k in keypoints]
        )
        self.imageVisualizer.labelImage.setKPImage()

    def connectAll(self):
        """
        Connects the various buttons.
        """
        self.groupDistanceBlob.updateParams.connect(self.updateKPParams)
        self.groupThreshold.minValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupThreshold.maxValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.thresholdStep.updateParams.connect(self.updateKPParams)
        self.groupBoxArea.minValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupBoxArea.maxValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupBoxCircularity.minValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupBoxCircularity.maxValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupBoxConvexity.minValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupBoxConvexity.maxValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupBoxInertia.minValueSelect.updateParams.connect(
            self.updateKPParams
        )
        self.groupBoxInertia.maxValueSelect.updateParams.connect(
            self.updateKPParams
        )

    def getParams(self):
        """
        Returns a dict with all kepoints detection parameters.
        """
        return dict(
            minDistBetweenBlobs=self.groupDistanceBlob.value,
            minThreshold=self.groupThreshold.minValueSelect.value,
            maxThreshold=self.groupThreshold.maxValueSelect.value,
            thresholdStep=self.thresholdStep.value,
            filterByArea=self.groupBoxArea.isChecked(),
            minArea=self.groupBoxArea.minValueSelect.value,
            maxArea=self.groupBoxArea.maxValueSelect.value,
            filterByCircularity=self.groupBoxCircularity.isChecked(),
            minCircularity=self.groupBoxCircularity.minValueSelect.value,
            maxCircularity=self.groupBoxCircularity.maxValueSelect.value,
            filterByConvexity=self.groupBoxConvexity.isChecked(),
            minConvexity=self.groupBoxConvexity.minValueSelect.value,
            maxConvexity=self.groupBoxConvexity.maxValueSelect.value,
            filterByInertia=self.groupBoxInertia.isChecked(),
            minInertiaRatio=self.groupBoxInertia.minValueSelect.value,
            maxInertiaRatio=self.groupBoxInertia.maxValueSelect.value,
        )

    def updateKPParams(self):
        """
        Updates keypoints detection parameters.
        """
        self.kpDetector.setDetectionParams(**self.getParams())
        self.updateKeypoints()

    def resetKPParams(self):
        """
        Resets keypoints detection parameters.
        """
        self.kpDetector.setDetectionParams(**DEFAULT_PARAMS)
        self.updateParamsUi(**DEFAULT_PARAMS)

    def updateParamsUi(
        self,
        minDistBetweenBlobs,
        minThreshold,
        maxThreshold,
        thresholdStep,
        filterByArea,
        minArea,
        maxArea,
        filterByCircularity,
        minCircularity,
        maxCircularity,
        filterByConvexity,
        minConvexity,
        maxConvexity,
        filterByInertia,
        minInertiaRatio,
        maxInertiaRatio,
    ):
        """
        Resets keypoints detection parameters in the UI.
        """
        self.groupDistanceBlob.setValue(minDistBetweenBlobs)
        self.groupThreshold.minValueSelect.setValue(minThreshold)
        self.groupThreshold.maxValueSelect.setValue(maxThreshold)
        self.thresholdStep.setValue(thresholdStep)
        self.groupBoxArea.setChecked(filterByArea)
        self.groupBoxArea.minValueSelect.setValue(minArea)
        self.groupBoxArea.maxValueSelect.setValue(maxArea)
        self.groupBoxCircularity.setChecked(filterByCircularity)
        self.groupBoxCircularity.minValueSelect.setValue(minCircularity)
        self.groupBoxCircularity.maxValueSelect.setValue(maxCircularity)
        self.groupBoxConvexity.setChecked(filterByConvexity)
        self.groupBoxConvexity.minValueSelect.setValue(minConvexity)
        self.groupBoxConvexity.maxValueSelect.setValue(maxConvexity)
        self.groupBoxInertia.setChecked(filterByInertia)
        self.groupBoxInertia.minValueSelect.setValue(minInertiaRatio)
        self.groupBoxInertia.maxValueSelect.setValue(maxInertiaRatio)

    def visualizeFinalGrid(self):
        """
        Visualize the final grid with all detected keypoints.
        """
        self.kpSorter.getFinalGrid(
            distance_from_border=self.groupPushKpBorder.value
        )
        self.imageVisualizer.labelImage.setKPAlgo(
            self.kpSorter.final_grid.reshape(-1, 2)
        )
        self.imageVisualizer.labelImage.setKPImage()

    def goBackTo(self):
        if hasattr(self.imageVisualizer.labelImage, "back_to") and (
            self.imageVisualizer.labelImage.back_to is not None
        ):
            back_to = self.imageVisualizer.labelImage.back_to
            index = self.kpSorter.remove_ordered_to(back_to)
            self.imageVisualizer.labelImage.back_to = None
            self.imageVisualizer.labelImage.next_to_order = None

            ordered = self.kpSorter.already_ordered

            self.imageVisualizer.labelImage.setKPOrdered(ordered)
            self.imageVisualizer.labelImage.setKPImage()

    def isDone(self):
        if self.kpSorter.is_filled:
            self.resVisualizer.setup(
                self.kpSorter.final_grid,
                self.kpSorter.final_grid,
                self.maskVisualizer.mask[:, :, 0],
                self.uv_full,
                self.grid2d_path,
                self.segmentation_path,
            )
            self.resVisualizer.show()
            self.close()
        else:
            self.resVisualizer.failed()
            self.resVisualizer.show()
            self.close()

    def orderGrid(self):
        """
        Order the detected grid, and show the result.
        """

        if not hasattr(self, "firstOrdering"):
            self.firstOrdering = True
        else:
            # If this is not the first ordering, we add the manually ordered keypoint
            self.firstOrdering = False
            if hasattr(self.imageVisualizer.labelImage, "back_to"):
                self.imageVisualizer.labelImage.back_to = None
            if (
                hasattr(self.imageVisualizer.labelImage, "next_to_order")
                and self.imageVisualizer.labelImage.next_to_order is not None
            ):
                next_index = self.imageVisualizer.labelImage.next_to_order
                self.kpSorter.set_ij_manually(next_index)
                self.imageVisualizer.labelImage.next_to_order = None

                ordered = self.kpSorter.already_ordered

                self.imageVisualizer.labelImage.setKPOrdered(ordered)
                self.imageVisualizer.labelImage.setKPImage()
            else:
                return None

        try:
            # Automatically order the grid
            self.imageVisualizer.setMode("view")
            self.imageVisualizer.labelNKP.setEnabled(False)
            self.imageVisualizer.labelCurrentMode.setEnabled(False)
            self.imageVisualizer.changeModeButton.setEnabled(False)

            if self.firstOrdering:
                self.groupPushKpBorder = MinMaxGroup(
                    "Distance From Border",
                    value=DEFAULT_DISTANCE_FROM_BORDER,
                    increment=0.5,
                    max_value=10,
                    min_value=-10,
                    typ="f",
                )
                self.groupPushKpBorder.setCheckable(False)
                self.groupPushKpBorder.updateParams.connect(
                    self.visualizeFinalGrid
                )
                self.imageVisualizer.layout.insertWidget(
                    1, self.groupPushKpBorder
                )

            self.maskVisualizer.setEnabled(False)

            self.groupBoxArea.setEnabled(False)
            self.groupBoxCircularity.setEnabled(False)
            self.groupBoxConvexity.setEnabled(False)
            self.groupBoxInertia.setEnabled(False)
            self.groupDistanceBlob.setEnabled(False)
            self.groupThreshold.setEnabled(False)
            self.doneButton.setEnabled(True)
            self.goBackAction.setEnabled(False)

            self.resetParamButton.setEnabled(False)
            self.runDetectionButton.setEnabled(False)
            self.orderGridButton.setEnabled(False)

            if self.firstOrdering:
                all_KP = self.imageVisualizer.labelImage.getAllKP()
                right_number_of_kp = self.kpSorter.setKeypoints(all_KP)
            else:
                right_number_of_kp = True

            if not right_number_of_kp:
                self.resVisualizer.failed()
                self.resVisualizer.show()
                self.close()
            else:
                if self.firstOrdering:
                    self.imageVisualizer.labelImage.setKPAlgo(all_KP)
                    self.imageVisualizer.labelImage.kp = []
                    self.imageVisualizer.labelImage.setKPImage()

                    self.kpSorter.findTopLeftCorner()
                    self.kpSorter.findBorders(
                        mask=self.maskVisualizer.mask[:, :, 0] * 255
                    )
                    self.kpSorter.initilize_filling_inside_grid()

                self.kpSorter.fillInsideGrid()

                self.imageVisualizer.labelImage.setKPOrdered([])

                self.groupPushKpBorder.setEnabled(True)
                self.kpSorter.getFinalGrid()

                self.imageVisualizer.labelImage.setKPAlgo(
                    self.kpSorter.final_grid.reshape(-1, 2)
                )
                self.imageVisualizer.labelImage.image_raw = self.rgb
                self.imageVisualizer.labelImage.setKPImage()
        except:
            # If the ordering failed, we switch to manual ordering
            self.imageVisualizer.switchToOrderMode()
            self.imageVisualizer.setMode("order")
            self.imageVisualizer.labelNKP.setEnabled(True)
            self.imageVisualizer.labelCurrentMode.setEnabled(True)
            self.imageVisualizer.changeModeButton.setEnabled(True)

            self.groupBoxArea.setEnabled(False)
            self.groupBoxCircularity.setEnabled(False)
            self.groupBoxConvexity.setEnabled(False)
            self.groupBoxInertia.setEnabled(False)
            self.groupDistanceBlob.setEnabled(False)
            self.groupThreshold.setEnabled(False)
            self.doneButton.setEnabled(True)
            self.goBackAction.setEnabled(True)

            self.resetParamButton.setEnabled(False)
            self.runDetectionButton.setEnabled(False)
            self.orderGridButton.setEnabled(True)

            self.groupPushKpBorder.setEnabled(False)

            ordered = self.kpSorter.already_ordered

            self.imageVisualizer.labelImage.setKPOrdered(ordered)
            self.imageVisualizer.labelImage.setKPImage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to the session of capture to annotate."
    )

    args = parser.parse_args()
    root = args.path

    with open(os.path.join(root, "annotate.json"), "r") as f:
        annotate = json.load(f)

    app = QtWidgets.QApplication(sys.argv)
    annotated = []
    for i, info in enumerate(annotate):
        sample_number = info["sample"]
        cam_number = info["cam"]
        annotated.append(info)

        win = KPDetectionOrdering(annotated=annotated)
        win.setup(
            os.path.join(root, f"RGB{cam_number}", f"{sample_number}.png"),
            os.path.join(root, f"UV{cam_number}", f"{sample_number}.png"),
            os.path.join(root, f"grid2d{cam_number}", f"{sample_number}.mat"),
            os.path.join(root, f"seg{cam_number}", f"{sample_number}.mat"),
        )
        win.show()
        app.exec_()

        to_reanotate = [x for x in annotate if x not in annotated]
        with open(os.path.join(root, "annotate.json"), "w") as f:
            json.dump(to_reanotate, f)

    sys.exit()
