from data import DataCollector
from PyQt5 import QtGui, QtWidgets
from realsense_capture import SERIAL_NUMBERS, CameraCapturer, Waiter

DEFORMATION_LIST = ["Perspective", "Curved", "Folded", "Crumpled"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()

        self.resize(800, 600)
        self.dataCollector = DataCollector(SERIAL_NUMBERS)
        self.numCameras = len(SERIAL_NUMBERS)
        self.setupUi()

    def setupUi(self):
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)

        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.layout = QtWidgets.QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        # Create main actions and connect them
        self.actionSetRoot = QtWidgets.QAction(self)
        self.actionSetRoot.setText("Set Root")
        self.actionCreateSession = QtWidgets.QAction(self)
        self.actionCreateSession.setText("Create New Session")
        self.actionCaptureRGB = QtWidgets.QAction(self)
        self.actionCaptureRGB.setText("Capture RGB-D")
        self.actionCaptureUVD = QtWidgets.QAction(self)
        self.actionCaptureUVD.setText("Capture UV-D")
        self.actionDeleteRGB = QtWidgets.QAction(self)
        self.actionDeleteRGB.setText("Delete RGB-D")
        self.actionDeleteLastSample = QtWidgets.QAction(self)
        self.actionDeleteLastSample.setText("Delete Last")
        self.actionSave = QtWidgets.QAction(self)
        self.actionSave.setText("Save")
        self.actionNew = QtWidgets.QAction(self)
        self.actionNew.setText("New")

        self.actionSetRoot.triggered.connect(self.setRoot)
        self.actionCreateSession.triggered.connect(self.newSession)
        self.actionCaptureRGB.triggered.connect(
            lambda: self.captureSample("RGBD")
        )
        self.actionCaptureUVD.triggered.connect(
            lambda: self.captureSample("UV")
        )
        self.actionCaptureUVD.setEnabled(False)
        self.actionDeleteRGB.triggered.connect(self.deleteRGB)
        self.actionDeleteRGB.setEnabled(False)
        self.actionDeleteLastSample.triggered.connect(self.deleteLastSample)
        self.actionDeleteLastSample.setEnabled(True)

        # Create all labels, used to display text information.
        layout_top = QtWidgets.QHBoxLayout()
        self.layout.addLayout(layout_top)

        self.labelRoot = QtWidgets.QLabel()
        self.labelSession = QtWidgets.QLabel()
        self.labelSample = QtWidgets.QLabel()
        self.updateInfo()
        layout_label = QtWidgets.QVBoxLayout()
        layout_label.addWidget(self.labelRoot, 0)
        layout_label.addWidget(self.labelSession, 0)
        layout_label.addWidget(self.labelSample, 0)

        layout_top.addLayout(layout_label)

        # Create all buttons
        layoutButton = QtWidgets.QGridLayout()

        self.changeRootButton = QtWidgets.QToolButton(self.centralWidget)
        self.changeRootButton.setDefaultAction(self.actionSetRoot)
        layoutButton.addWidget(self.changeRootButton, 0, 0)

        self.createSessionButton = QtWidgets.QToolButton(self.centralWidget)
        self.createSessionButton.setDefaultAction(self.actionCreateSession)
        layoutButton.addWidget(self.createSessionButton, 1, 0)

        self.captureRGBButton = QtWidgets.QToolButton(self.centralWidget)
        self.captureRGBButton.setDefaultAction(self.actionCaptureRGB)
        layoutButton.addWidget(self.captureRGBButton, 2, 0)

        self.captureUVDButton = QtWidgets.QToolButton(self.centralWidget)
        self.captureUVDButton.setDefaultAction(self.actionCaptureUVD)
        layoutButton.addWidget(self.captureUVDButton, 2, 1)

        self.DeleteRGBButton = QtWidgets.QToolButton(self.centralWidget)
        self.DeleteRGBButton.setDefaultAction(self.actionDeleteRGB)
        layoutButton.addWidget(self.DeleteRGBButton, 2, 2)

        self.DeleteLastSampleButton = QtWidgets.QToolButton(self.centralWidget)
        self.DeleteLastSampleButton.setDefaultAction(
            self.actionDeleteLastSample
        )
        layoutButton.addWidget(self.DeleteLastSampleButton, 2, 3)

        layout_top.addLayout(layoutButton)

        # Create the sample type selection
        layout_type = QtWidgets.QGridLayout()
        self.TypeSelectionGroupBox = QtWidgets.QGroupBox()
        self.TypeSelectionGroupBox.setTitle("Choose the type of next sample.")
        self.TypeSelectionGroupBox.setLayout(layout_type)

        self.checkboxes = {}
        for i, deformation in enumerate(DEFORMATION_LIST):
            self.checkboxes[deformation] = QtWidgets.QCheckBox(
                deformation, self.TypeSelectionGroupBox
            )
            layout_type.addWidget(self.checkboxes[deformation], i % 2, i // 2)

        layout_top.addWidget(self.TypeSelectionGroupBox)

        # Create the capture tools and the visualization of the cameras.
        self.waiterCapture = Waiter(2 * self.numCameras)  # 2xNumCam
        self.waiterCapture.start()
        self.w = []
        self.CamLabel = []
        self.DepthLabel = []
        gridLayout = QtWidgets.QGridLayout()
        for i, serial_number in enumerate(SERIAL_NUMBERS):
            self.w.append(CameraCapturer(serial_number))
            self.w[-1].start()
            self.w[i].ImageUpdate.connect(
                self.__getattribute__(f"cam{i}UpdateSlot")
            )
            self.w[i].DepthUpdate.connect(
                self.__getattribute__(f"depth{i}UpdateSlot")
            )
            self.w[i].CaptureUpdate.connect(self.captureUpdateSlot)
            self.CamLabel.append(QtWidgets.QLabel())
            self.DepthLabel.append(QtWidgets.QLabel())
            gridLayout.addWidget(self.CamLabel[-1], 0, i)
            gridLayout.addWidget(self.DepthLabel[-1], 1, i)

        self.waiterCapture.Done.connect(self.updateCurrentSample)
        self.layout.addLayout(gridLayout)

    def captureUpdateSlot(self, d):
        """
        Collect the data acquired by the CameraCatpturer and save them.
        """
        image, depth_image, serial_number = d
        self.dataCollector.save(
            image, depth_image, serial_number, self.getSampleType()
        )
        self.waiterCapture.count += 1
        if self.waiterCapture.count == self.numCameras:
            self.actionCaptureUVD.setEnabled(True)
            self.actionDeleteRGB.setEnabled(True)

    def getSampleType(self):
        """
        Returns a dict containing the categories checked for this sample.
        """
        return {
            deformation: self.checkboxes[deformation].isChecked()
            for deformation in DEFORMATION_LIST
        }

    def updateInfo(self):
        """
        Update the labels with the current root/session/sample number.
        """
        self.labelRoot.setText(f"Root : {str(self.dataCollector.root)}")
        if self.dataCollector.sample_number is None:
            sample = str(self.dataCollector.sample_number)
        else:
            sample = int(self.dataCollector.sample_number)
        self.labelSample.setText(f"Current Sample : {sample}")
        if self.dataCollector.sample_number is None:
            session = str(self.dataCollector.current_session)
        else:
            session = int(self.dataCollector.current_session)

        self.labelSession.setText(f"Current Session : {session}")

    def setRoot(self):
        """
        Select the folder to which set the root.
        """
        self.dataCollector.setRoot(
            QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        )
        self.updateInfo()
        self.statusBar().showMessage(
            f"Setting root to {self.dataCollector.root}", 2000
        )

    def newSession(self):
        """
        Create a new session.
        """
        intrinsics = {}
        devices_info = {}
        for i, worker in enumerate(self.w):
            intrinsics[SERIAL_NUMBERS[i]] = worker.getIntrinsic()
            devices_info[SERIAL_NUMBERS[i]] = worker.getDeviceInfo()
        self.dataCollector.createSession(intrinsics, devices_info)
        self.updateInfo()
        self.statusBar().showMessage(
            f"Created Session {int(self.dataCollector.current_session)}", 2000
        )

    def captureSample(self, expecting):
        """
        Set CameraCaturer `capture` attribute to True to capture a sample.
        """
        self.dataCollector.expecting = expecting
        for worker in self.w:
            worker.capture = True
        self.statusBar().showMessage(
            f"Capturing {expecting} for sample {int(self.dataCollector.sample_number)}",
            2000,
        )
        if expecting == "RGBD":
            self.actionCaptureRGB.setEnabled(False)
            self.actionSetRoot.setEnabled(False)
            self.actionCreateSession.setEnabled(False)
            self.actionDeleteLastSample.setEnabled(False)
            self.TypeSelectionGroupBox.setEnabled(False)
        elif expecting == "UV":
            self.actionCaptureUVD.setEnabled(False)
            self.actionDeleteRGB.setEnabled(False)

    def deleteRGB(self):
        """
        Delete the last RGBD captured and allow for new capture.
        """
        self.dataCollector.deleteRGBD()
        self.actionDeleteRGB.setEnabled(False)
        self.actionCaptureUVD.setEnabled(False)
        self.actionCaptureRGB.setEnabled(True)
        self.actionSetRoot.setEnabled(True)
        self.actionDeleteLastSample.setEnabled(True)
        self.actionCreateSession.setEnabled(True)
        self.TypeSelectionGroupBox.setEnabled(True)
        self.waiterCapture.count -= self.numCameras

    def _deleteLastMsgButtonClicked(self, i):
        if "OK" in i.text():
            self.dataCollector.deleteLast()
            self.updateInfo()

    def deleteLastSample(self):
        """
        Delete the last sample if it exists, first displaying a confirmation
        message.
        """
        if (
            self.dataCollector.sample_number is not None
            and int(self.dataCollector.sample_number) > 0
        ):
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Delete Last Sample")
            msg.setText(
                "You are trying to delete last sample!"
                "\n To continue, press 'Ok'."
            )
            msg.setStandardButtons(
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
            )
            msg.setIcon(QtWidgets.QMessageBox.Information)

            msg.buttonClicked.connect(self._deleteLastMsgButtonClicked)
            msg.exec_()

        else:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("No sample")
            msg.setText(
                "There are no sample in this session!"
                "\n Please capture a full sample before trying to delete it!"
            )
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()

    def updateCurrentSample(self, b):
        """
        Update the sample number, add the samples to the list of samples to
        annotate, display the alignment checker, and allow for a new capture.
        Triggered once a full sample has been captured.
        """

        for serial_number in SERIAL_NUMBERS:
            self.dataCollector.addToAnnotate(serial_number)
            self.dataCollector.verifyAlignement(serial_number)
        self.dataCollector.updateSample()
        self.updateInfo()
        self.statusBar().showMessage(
            f"Done ! Captured Sample {int(self.dataCollector.sample_number) - 1}",
            2000,
        )
        self.actionCaptureRGB.setEnabled(True)
        self.actionSetRoot.setEnabled(True)
        self.actionCreateSession.setEnabled(True)
        self.actionDeleteLastSample.setEnabled(True)
        self.TypeSelectionGroupBox.setEnabled(True)
        self.actionDeleteLastSample.setEnabled(True)


def addUpdateSlot(cls, index):
    """
    Add Update slots for each of the cameras.
    """

    def camUpdateSlot(self, Image):
        self.CamLabel[index].setPixmap(QtGui.QPixmap.fromImage(Image))

    def depthUpdateSlot(self, Depth):
        self.DepthLabel[index].setPixmap(QtGui.QPixmap.fromImage(Depth))

    camUpdateSlot.__name__ = f"cam{index}UpdateSlot"
    setattr(cls, camUpdateSlot.__name__, camUpdateSlot)
    depthUpdateSlot.__name__ = f"depth{index}UpdateSlot"
    setattr(cls, depthUpdateSlot.__name__, depthUpdateSlot)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    for i in range(len(SERIAL_NUMBERS)):
        addUpdateSlot(MainWindow, i)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
