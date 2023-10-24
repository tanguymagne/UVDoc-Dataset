from PyQt5 import QtCore, QtWidgets


class MinMaxGroup(QtWidgets.QGroupBox):

    updateParams = QtCore.pyqtSignal(object)

    def __init__(
        self,
        name,
        value=0.5,
        increment=0.01,
        max_value=1,
        min_value=0,
        typ="f",
    ):
        super(MinMaxGroup, self).__init__()
        self.name = name
        self.value = value
        self.increment = increment
        self.max_value = max_value
        self.min_value = min_value
        self.typ = typ
        self.setupUi()

    def setupUi(self):
        self.setTitle(f"{self.name} value")
        self.setCheckable(True)

        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)

        self.label = QtWidgets.QLabel()
        self.updateValue()
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.buttonPlus = QtWidgets.QPushButton()
        self.buttonPlus.setText("+")
        self.buttonPlus.clicked.connect(self.increase)
        self.buttonPlusPlus = QtWidgets.QPushButton()
        self.buttonPlusPlus.setText("++")
        self.buttonPlusPlus.clicked.connect(self.increase_lot)
        self.buttonMax = QtWidgets.QPushButton()
        self.buttonMax.setText("Max")
        self.buttonMax.clicked.connect(self.increase_max)
        self.buttonLess = QtWidgets.QPushButton()
        self.buttonLess.setText("-")
        self.buttonLess.clicked.connect(self.decrease)
        self.buttonLessLess = QtWidgets.QPushButton()
        self.buttonLessLess.setText("--")
        self.buttonLessLess.clicked.connect(self.decrease_lot)
        self.buttonMin = QtWidgets.QPushButton()
        self.buttonMin.setText("Min")
        self.buttonMin.clicked.connect(self.decrease_min)

        self.layout.addWidget(self.buttonMin)
        self.layout.addWidget(self.buttonLessLess)
        self.layout.addWidget(self.buttonLess)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttonPlus)
        self.layout.addWidget(self.buttonPlusPlus)
        self.layout.addWidget(self.buttonMax)

    def increase(self):
        self.value = min(self.value + self.increment, self.max_value)
        self.updateValue()

    def increase_lot(self):
        self.value = min(self.value + 10 * self.increment, self.max_value)
        self.updateValue()

    def increase_max(self):
        self.value = self.max_value
        self.updateValue()

    def decrease(self):
        self.value = max(self.value - self.increment, self.min_value)
        self.updateValue()

    def decrease_lot(self):
        self.value = max(self.value - 10 * self.increment, self.min_value)
        self.updateValue()

    def decrease_min(self):
        self.value = self.min_value
        self.updateValue()

    def setValue(self, value):
        self.value = value
        if self.typ == "i":
            self.label.setText(f"{self.value}")
        elif self.typ == "f":
            self.label.setText(f"{self.value:.2f}")

    def updateValue(self):
        self.setValue(self.value)
        self.updateParams.emit(True)


class ParameterGroup(QtWidgets.QGroupBox):
    def __init__(
        self,
        name,
        default_min,
        default_max,
        min_value,
        max_value,
        increment,
        typ,
    ):
        super(ParameterGroup, self).__init__()
        self.name = name
        self.value_min = default_min
        self.value_max = default_max
        self.min_value = min_value
        self.max_value = max_value
        self.increment = increment
        self.typ = typ
        self.setupUi()

    def setupUi(self):
        self.setTitle(f"Filter by {self.name}")
        self.setCheckable(True)

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.minValueSelect = MinMaxGroup(
            "Min",
            self.value_min,
            self.increment,
            self.max_value,
            self.min_value,
            self.typ,
        )
        self.maxValueSelect = MinMaxGroup(
            "Max",
            self.value_max,
            self.increment,
            self.max_value,
            self.min_value,
            self.typ,
        )
        self.layout.addWidget(self.minValueSelect)
        self.layout.addWidget(self.maxValueSelect)
