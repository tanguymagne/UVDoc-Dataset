import sys

sys.path.append(".")

import cv2
import numpy as np
from kp_annotation import GRID_SIZE
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from QtImageViewer import LEFT_BUTTON, MIDDLE_BUTTON, QtImageViewer
from sklearn.neighbors import NearestNeighbors

if GRID_SIZE == [45, 31]:
    CIRCLE_RADIUS = 5
elif GRID_SIZE == [89, 61]:
    CIRCLE_RADIUS = 3
else:
    CIRCLE_RADIUS = 5


class LabelImage(QtImageViewer):
    """
    Allows to visualize an image and its detected keypoints and manually
    annotate new ones.
    """

    updateNKeypoints = QtCore.pyqtSignal(object)

    def __init__(self, kp, image):
        super(LabelImage, self).__init__()
        self.kp = kp
        self.kp_algo = []
        self.kp_ordered_index = []
        self.next_index = None
        self.image_raw = image
        self.setMinimumWidth(750)
        self.setMaximumWidth(750)
        self.setMinimumHeight(1300)
        self.setMaximumHeight(1300)
        self.setKPImage()
        self.mode = "view"

    def getAllKP(self):
        """
        Return all keypoints
        """
        return np.array(self.kp + self.kp_algo)

    def setKPAlgo(self, kp):
        """
        Set the keypoints detected by the algorithm.
        """
        self.kp_algo = kp

    def setKPOrdered(self, kp_ordered_index):
        self.kp_ordered_index = kp_ordered_index

    def setKPImage(self):
        """
        Visualize the keypoints.
        """

        img_kp = np.copy(self.image_raw)
        for k in self.kp:
            img_kp = cv2.circle(
                img_kp,
                (int(k[0]), int(k[1])),
                CIRCLE_RADIUS,
                (255, 255, 0),
                -1,
            )
        for k in self.kp_algo:
            img_kp = cv2.circle(
                img_kp, (int(k[0]), int(k[1])), CIRCLE_RADIUS, (255, 0, 0), -1
            )

        for ind in self.kp_ordered_index:
            img_kp = cv2.circle(
                img_kp,
                (int(self.kp_algo[ind][0]), int(self.kp_algo[ind][1])),
                CIRCLE_RADIUS,
                (0, 255, 0),
                -1,
            )

        if hasattr(self, "next_to_order") and self.next_to_order is not None:
            img_kp = cv2.circle(
                img_kp,
                (
                    int(self.kp_algo[self.next_to_order][0]),
                    int(self.kp_algo[self.next_to_order][1]),
                ),
                CIRCLE_RADIUS,
                (0, 0, 255),
                -1,
            )

        if hasattr(self, "back_to") and self.back_to is not None:
            img_kp = cv2.circle(
                img_kp,
                (
                    int(self.kp_algo[self.back_to][0]),
                    int(self.kp_algo[self.back_to][1]),
                ),
                CIRCLE_RADIUS,
                (255, 0, 255),
                -1,
            )

        QTimg = QtGui.QImage(
            img_kp.data,
            img_kp.shape[1],
            img_kp.shape[0],
            QtGui.QImage.Format_RGB888,
        )

        self.setImage(QtGui.QPixmap.fromImage(QTimg))
        self.updateNKeypoints.emit(len(self.kp) + len(self.kp_algo))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        To manually annotate keypoints.
        """
        if self.mode == "edit":
            # Allow to add and remove keypoints
            scenePos = self.mapToScene(event.pos())
            x_coord, y_coord = scenePos.x(), scenePos.y()
            if event.button() & LEFT_BUTTON:
                if (
                    x_coord > 0
                    and x_coord < self.sceneRect().width()
                    and y_coord > 0
                    and y_coord < self.sceneRect().height()
                ):
                    self.kp.append([x_coord, y_coord])
                    self.setKPImage()
            if event.button() & MIDDLE_BUTTON:
                kp = self.kp + self.kp_algo
                if len(kp) != 0:
                    KNN = NearestNeighbors(n_neighbors=1).fit(np.array(kp))
                    nearest = KNN.kneighbors(
                        np.array([[x_coord, y_coord]]), return_distance=False
                    )[0, 0]
                    if nearest < len(self.kp):
                        self.kp.pop(nearest)
                    else:
                        self.kp_algo.pop(nearest - len(self.kp))
                    self.setKPImage()
        elif self.mode == "order":
            # Allow to order keypoints
            if event.button() & LEFT_BUTTON:
                scenePos = self.mapToScene(event.pos())
                x_coord, y_coord = scenePos.x(), scenePos.y()
                index_not_ordered = [
                    i
                    for i, k in enumerate(self.kp_algo)
                    if i not in self.kp_ordered_index
                ]
                not_ordered = [
                    k
                    for i, k in enumerate(self.kp_algo)
                    if i not in self.kp_ordered_index
                ]
                if len(not_ordered) != 0:
                    KNN = NearestNeighbors(n_neighbors=1).fit(
                        np.array(not_ordered)
                    )
                    nearest = KNN.kneighbors(
                        np.array([[x_coord, y_coord]]), return_distance=False
                    )[0, 0]
                    self.next_to_order = index_not_ordered[nearest]
                self.setKPImage()
            if event.button() & MIDDLE_BUTTON:
                scenePos = self.mapToScene(event.pos())
                x_coord, y_coord = scenePos.x(), scenePos.y()
                index_ordered = [
                    i
                    for i, k in enumerate(self.kp_algo)
                    if i in self.kp_ordered_index
                ]
                ordered = [
                    k
                    for i, k in enumerate(self.kp_algo)
                    if i in self.kp_ordered_index
                ]
                if len(ordered) != 0:
                    KNN = NearestNeighbors(n_neighbors=1).fit(
                        np.array(ordered)
                    )
                    nearest = KNN.kneighbors(
                        np.array([[x_coord, y_coord]]), return_distance=False
                    )[0, 0]
                    self.back_to = index_ordered[nearest]
                self.setKPImage()
        else:
            return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "view":
            return super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self.mode == "view":
            return super().wheelEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == "view":
            return super().mouseMoveEvent(event)


class KPVisualizer(QtWidgets.QWidget):
    """
    Wrapper around LabelImage, allow to change between edit and view mode and
    show the number of keypoints.
    """

    def __init__(self, image):
        super(KPVisualizer, self).__init__()

        self.active_along_view_mode = "edit"
        self.setupUi(image)

    def setupUi(self, image):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.labelImage = LabelImage([], image)
        self.labelImage.updateNKeypoints.connect(self.updateNKP)

        # Showing number of keypoints
        layoutButtonText = QtWidgets.QHBoxLayout()
        self.layout.addLayout(layoutButtonText)

        self.labelNKP = QtWidgets.QLabel()
        self.labelNKP.setText(
            f"Number of keypoints : {len(self.labelImage.kp)} / {np.product(GRID_SIZE)}"
        )
        layoutButtonText.addWidget(self.labelNKP)
        layoutButtonText.addStretch(1)
        self.layout.addWidget(self.labelImage)

        # Switching from edit to view mode
        self.labelCurrentMode = QtWidgets.QLabel()
        self.labelCurrentMode.setText(
            f"Current Mode : {self.labelImage.mode.title()}"
        )
        layoutButtonText.addWidget(self.labelCurrentMode)

        self.actionChangeMode = QtWidgets.QAction(self)
        self.actionChangeMode.setText("Change to Edit (E)")
        self.actionChangeMode.setShortcut("E")
        self.actionChangeMode.triggered.connect(self.changeMode)
        self.changeModeButton = QtWidgets.QToolButton(self)
        self.changeModeButton.setDefaultAction(self.actionChangeMode)
        layoutButtonText.addWidget(self.changeModeButton)

    def switchToOrderMode(self):
        """
        Switch mode to order keypoints.
        """
        self.active_along_view_mode = "order"
        self.setMode("order")

    def changeMode(self):
        """
        Change between edit and view mode.
        """
        self.actionChangeMode.setText(
            f"Change to {self.labelImage.mode.title()} (E)"
        )

        if self.labelImage.mode == self.active_along_view_mode:
            self.labelImage.mode = "view"
        elif self.labelImage.mode == "view":
            self.labelImage.mode = self.active_along_view_mode

        self.labelCurrentMode.setText(
            f"Current Mode : {self.labelImage.mode.title()}"
        )

    def setMode(self, mode):
        """
        Defines the current mode (edit or view).
        """
        if mode != self.labelImage.mode:
            self.actionChangeMode.setText(
                f"Change to {self.labelImage.mode.title()} (E)"
            )
            self.labelImage.mode = mode
            self.labelCurrentMode.setText(
                f"Current Mode : {self.labelImage.mode.title()}"
            )

    def updateNKP(self, nkp):
        """
        Update the displayed number of keypoints.
        """
        self.labelNKP.setText(
            f"Number of keypoints : {nkp} / {np.product(GRID_SIZE)}"
        )
