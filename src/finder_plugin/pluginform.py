import ida_kernwin
import idautils
import ida_bytes
import ida_nalt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, \
    QTableWidgetItem, QHeaderView, QCheckBox, QFileDialog


class FunctionStartFinderForm(QDialog):
    def __init__(self, parent=None):
        super(FunctionStartFinderForm, self).__init__(parent)
        self.setWindowTitle("Function Start Finder Plugin")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(600, 400)

        self.file_name = None
        self.closed_by_find_button = False

        self.init_ui()

    def exec_(self):
        """Переопределяем exec_ для возврата результатов"""
        super(FunctionStartFinderForm, self).exec_()

        if not self.closed_by_find_button:
            return None

        start_addr = int(self.start_addr_edit.text(), 16)
        end_addr = int(self.end_addr_edit.text(), 16)

        soft_find = self.soft_prediction_check.isChecked()

        return start_addr, end_addr, self.file_name, soft_find

    def init_ui(self):
        # Main layout
        layout = QVBoxLayout()

        # Start layout
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start address : 0x"))

        self.start_addr_edit = QLineEdit()
        self.start_addr_edit.setPlaceholderText("Enter start address ...")
        start_layout.addWidget(self.start_addr_edit)

        layout.addLayout(start_layout)

        # End layout
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End address   : 0x"))

        self.end_addr_edit = QLineEdit()
        self.end_addr_edit.setPlaceholderText("Enter end address ...")
        end_layout.addWidget(self.end_addr_edit)

        layout.addLayout(end_layout)

        # Settings layout
        settings_layout = QHBoxLayout()
        # Prediction checkbox
        self.soft_prediction_check = QCheckBox("Soft prediction")
        self.soft_prediction_check.setCheckState(False)
        self.soft_prediction_check.setToolTip("Doesn't create functions when enabled, but sets comments instead")
        settings_layout.addWidget(self.soft_prediction_check)

        # Import button
        self.import_button = QPushButton("Import model")
        self.import_button.setToolTip("Import RNN or your custom model to find function start")
        self.import_button.clicked.connect(self.on_import_file)
        settings_layout.addWidget(self.import_button)

        layout.addLayout(settings_layout)

        # Find button
        self.find_button = QPushButton("Find")
        self.find_button.clicked.connect(self.on_find)
        layout.addWidget(self.find_button)
        self.setLayout(layout)

    def on_import_file(self):
        self.file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select file with model weights",
            "",
            "Weight Files (*.h5);;All Files (*)"
        )


    def on_find(self):
        self.closed_by_find_button = True
        self.close()
