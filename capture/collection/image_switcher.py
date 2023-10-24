import sys

sys.path.append(".")
from PyQt5 import QtWidgets
from QtImageViewer import QtImageViewer
import cv2


class AlignementVerifier(QtWidgets.QWidget):
    """
    Simple class allowing to switch between two images.
    It is used here to verify that the RGB and UV images are aligned.
    """

    def __init__(self, rgb_path, uv_path):
        super(AlignementVerifier, self).__init__()
        self.rgbIm = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        self.uvIm = cv2.cvtColor(cv2.imread(uv_path), cv2.COLOR_BGR2RGB)
        self.currentImage = "UV"
        self.setupUi()
        self.changeImage()

    def setupUi(self):
        self.setWindowTitle("Verify Alignement")

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        # Add info label
        self.message = QtWidgets.QLabel()
        self.message.setText(
            "This window allows checking that RGB and UV images are aligned."
            "\nYou can close it and click on the 'Delete Last' button if "
            "that's not the case. \nNote: The depth image is automatically "
            "aligned with the RGB image."
        )
        self.layout.addWidget(self.message)

        # Add button to switch between images
        self.switchButton = QtWidgets.QPushButton("Switch")
        self.switchButton.clicked.connect(self.changeImage)
        self.layout.addWidget(self.switchButton)

        # Add image visualizer
        self.imageViewer = QtImageViewer()
        self.imageViewer.setMinimumWidth(560)
        self.imageViewer.setMaximumWidth(560)
        self.imageViewer.setMinimumHeight(1000)
        self.imageViewer.setMaximumHeight(1000)
        self.layout.addWidget(self.imageViewer)

    def changeImage(self):
        # Switch between the two images
        if self.currentImage == "RGB":
            self.imageViewer.setImage(self.uvIm)
            self.currentImage = "UV"
        elif self.currentImage == "UV":
            self.imageViewer.setImage(self.rgbIm)
            self.currentImage = "RGB"
