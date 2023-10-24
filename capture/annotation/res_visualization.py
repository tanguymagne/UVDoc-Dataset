import sys

sys.path.append(".")
import cv2
import hdf5storage as h5
import numpy as np
import torch
import torch.nn.functional as F
from kp_annotation import GRID_SIZE
from PyQt5 import QtCore, QtGui, QtWidgets
from QtImageViewer import QtImageViewer

if GRID_SIZE == [45, 31]:
    CIRCLE_RADIUS = 5
elif GRID_SIZE == [89, 61]:
    CIRCLE_RADIUS = 3
else:
    CIRCLE_RADIUS = 5


def rectangular_bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Unwarp an image (warped_img) according to a 2D grid (point_positions) in a
    specific image size.
    Args:
        warped_img : torch tensor of shape [1, 3, w, h], in the range [0,1]
        point_position : torch tensor of shape []
        img_size : list containing 2ints with width and height

    """
    upsampled_grid = F.interpolate(
        point_positions, size=img_size, mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(
        warped_img,
        upsampled_grid.transpose(1, 2).transpose(2, 3),
        align_corners=True,
    )

    return unwarped_img


def get_ind(i, j, shape):
    return int(i * shape[1] + j)


def create_adjacency_matrix(grid):
    """
    Helper function to visualize the connectivity of the grid.
    Returns the adjacency matrix corresponding to the grid.
    """
    shape = grid.shape
    adj = np.zeros([grid.size, 2], dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ind = get_ind(i, j, shape)
            neighbors = []
            if i + 1 < shape[0]:
                neighbors.append(get_ind(i + 1, j, shape))
            if j + 1 < shape[1]:
                neighbors.append(get_ind(i, j + 1, shape))
            while len(neighbors) < 2:
                neighbors.append(ind)
            adj[ind] = np.array(neighbors)
    return adj


class ResVisualizer(QtWidgets.QWidget):
    """
    Allows to visualize the obtained 2D grid.
    """

    saveToUnfinsihed = QtCore.pyqtSignal(object)

    def __init__(self, annotated):
        super(ResVisualizer, self).__init__()
        self.annotated = annotated
        self.setupUi()

    def setup(
        self,
        grid_index,
        grid,
        segmentation_mask,
        image,
        grid2d_path,
        segmentation_path,
    ):
        self.grid_index = grid_index
        self.grid = grid
        self.segmentation_mask = segmentation_mask
        self.grid2d_path = grid2d_path
        self.segmentation_path = segmentation_path

        self.image = self.createMesh(image)
        self.imageUnwarp = self.unwarpImage(image)

        self.qtPixmapGrid = self.createQtImg(self.image)
        self.qtPixemapUnwarp = self.createQtImg(self.imageUnwarp)

        self.imageGridViewer.setImage(self.qtPixmapGrid)
        self.imageUnwarpViewer.setImage(self.qtPixemapUnwarp)

        self.voidLabel.setText("")
        self.quitButton.setEnabled(True)

    def failed(self):
        """
        Setup UI if detection or ordering of the grid failed.
        """
        self.imageGridViewer.setImage(None)
        self.imageUnwarpViewer.setImage(None)
        self.voidLabel.setText(
            "The grid detection or the grid ordering failed :"
        )
        self.quitButton.setEnabled(False)

    def setupUi(self):
        self.setWindowTitle("Results visualization")

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        # SetupButton
        self.voidLabel = QtWidgets.QLabel()
        self.toUnfinsihedButton = QtWidgets.QPushButton()
        self.toUnfinsihedButton.setText("Add to the list of badly annotated")
        self.toUnfinsihedButton.clicked.connect(self.isDoneReanotate)
        self.quitButton = QtWidgets.QPushButton()
        self.quitButton.setText("Done")
        self.quitButton.clicked.connect(self.isDone)
        layoutButton = QtWidgets.QHBoxLayout()
        layoutButton.addWidget(self.voidLabel, 1)
        layoutButton.addWidget(self.toUnfinsihedButton, 0)
        layoutButton.addWidget(self.quitButton, 0)
        self.mainLayout.addLayout(layoutButton)

        # SetupImage
        self.imageGridViewer = QtImageViewer()
        self.imageUnwarpViewer = QtImageViewer()
        self.imageGridViewer.setMinimumWidth(731)
        self.imageGridViewer.setMaximumWidth(731)
        self.imageGridViewer.setMinimumHeight(1300)
        self.imageGridViewer.setMaximumHeight(1300)
        self.imageUnwarpViewer.setMinimumWidth(920)
        self.imageUnwarpViewer.setMaximumWidth(920)
        self.imageUnwarpViewer.setMinimumHeight(1300)
        self.imageUnwarpViewer.setMaximumHeight(1300)
        layoutImage = QtWidgets.QHBoxLayout()
        layoutImage.addWidget(self.imageGridViewer)
        layoutImage.addWidget(self.imageUnwarpViewer)
        self.mainLayout.addLayout(layoutImage)

    def isDone(self):
        """
        Saves the 2D grid in case of success.
        """

        h5.savemat(self.grid2d_path, mdict={"grid2d": self.grid}, format="7.3")
        h5.savemat(
            self.segmentation_path,
            mdict={"seg": self.segmentation_mask},
            format="7.3",
        )
        self.close()

    def isDoneReanotate(self):
        """
        Remove the sample from the list of annotated samples in case of
        failure.
        """
        self.annotated.pop(-1)
        self.close()

    def createMesh(self, image):
        """
        Create an image with the overlayed 2D mesh.
        """
        image = np.copy(image)
        adj = create_adjacency_matrix(self.grid_index[:, :, 0])
        for i in range(len(adj)):
            for j in range(2):
                if adj[i, j] != i:
                    image = cv2.line(
                        image,
                        [
                            int(self.grid.reshape(-1, 2)[i, 0]),
                            int(self.grid.reshape(-1, 2)[i, 1]),
                        ],
                        [
                            int(self.grid.reshape(-1, 2)[adj[i, j], 0]),
                            int(self.grid.reshape(-1, 2)[adj[i, j], 1]),
                        ],
                        color=(0, 0, 0),
                        thickness=1,
                    )
        scale1 = np.linspace(0, 255, self.grid.shape[0], dtype=np.uint8)
        scale2 = np.linspace(0, 255, self.grid.shape[1], dtype=np.uint8)

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                image = cv2.circle(
                    image,
                    [int(self.grid[i, j, 0]), int(self.grid[i, j, 1])],
                    radius=CIRCLE_RADIUS,
                    color=(int(scale1[i]), int(scale2[j]), 0),
                    thickness=-1,
                )
        return image

    def unwarpImage(self, image):
        """
        Create an unwarped image based on the 2D grid obtained.
        """
        image = np.copy(image)
        unwarped_img = rectangular_bilinear_unwarping(
            torch.unsqueeze(
                torch.from_numpy(image.transpose(2, 0, 1) / 255), dim=0
            ).float(),
            point_positions=torch.unsqueeze(
                torch.from_numpy(
                    self.grid.transpose(2, 0, 1)
                    / np.array([[[1080]], [[1920]]])
                    * 2
                    - 1
                ),
                dim=0,
            ).float(),
            img_size=(1920, 1360),
        )
        return unwarped_img[0].numpy().transpose(1, 2, 0)

    def createQtImg(self, image):
        """
        Convert an array to a QPixmap.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        imageZeros = np.zeros([*image.shape], dtype=np.uint8)
        image = imageZeros + image

        QTimg = QtGui.QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            QtGui.QImage.Format_RGB888,
        )

        return QtGui.QPixmap.fromImage(QTimg)
