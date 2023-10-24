import sys

sys.path.append(".")
import numpy as np
from base_component import MinMaxGroup
from PyQt5 import QtWidgets
from QtImageViewer import QtImageViewer

DEFAULT_THRESHOLD_SEG = 90


class SegmentationVisualizer(QtWidgets.QWidget):
    """
    Allows to visualize the segmentation and modify the threshold used.
    """

    def __init__(self, img_rgb, img_uv):
        super(SegmentationVisualizer, self).__init__()
        self.isReversed = False
        self.image = img_rgb
        self.uv_img = img_uv
        self.setupUi()

    def setupUi(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        # Create controls
        self.reverseMaskButton = QtWidgets.QPushButton()
        self.reverseMaskButton.setText("Reverse mask")
        self.layout.addWidget(self.reverseMaskButton)

        self.thresholdGroup = MinMaxGroup(
            "Threshold",
            value=DEFAULT_THRESHOLD_SEG,
            increment=1,
            max_value=255,
            min_value=0,
            typ="i",
        )
        self.thresholdGroup.setCheckable(False)
        self.layout.addWidget(self.thresholdGroup)

        # Create Visualizer
        self.viewer = QtImageViewer()
        self.viewer.setMinimumWidth(675)
        self.viewer.setMaximumWidth(675)
        self.viewer.setMinimumHeight(1200)
        self.viewer.setMaximumHeight(1200)
        self.layout.addWidget(self.viewer)

        # Connect everything
        self.connectUi()

        # Initialize image
        self.updateMask()
        self.viewer.setImage(self.image * self.maskShowed)

    def connectUi(self):
        self.reverseMaskButton.clicked.connect(self.reverseMask)
        self.reverseMaskButton.clicked.connect(
            lambda: self.viewer.setImage(self.image * self.maskShowed)
        )

        self.thresholdGroup.updateParams.connect(self.updateMask)
        self.thresholdGroup.updateParams.connect(
            lambda: self.viewer.setImage(self.image * self.maskShowed)
        )

    def reverseMask(self):
        """
        Visually reverse the segmentation.
        """
        if not self.isReversed:
            self.maskShowed = self.reverseMask
            self.isReversed = True
        else:
            self.maskShowed = self.mask
            self.isReversed = False

    def updateMask(self):
        """
        Update the segmentation using the defined threshold.
        """
        thresh = self.thresholdGroup.value
        binary = (self.uv_img > thresh).astype(np.uint8)
        binary = np.expand_dims(binary, axis=-1)
        self.mask = binary
        self.reverseMask = 1 - binary
        if self.isReversed:
            self.maskShowed = self.reverseMask
        else:
            self.maskShowed = self.mask
