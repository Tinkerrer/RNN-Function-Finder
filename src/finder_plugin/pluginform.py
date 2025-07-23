import ida_kernwin
import idautils
import ida_bytes
import ida_nalt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView

class FunctionStartFinderForm(QDialog):
    def __init__(self, parent=None):
        super(FunctionStartFinderForm, self).__init__(parent)
        self.setWindowTitle("Function Start Finder Plugin")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(600, 400)
        
        self.init_ui()
    
    def init_ui(self):
        # Main layout
        layout = QVBoxLayout()
        
        # Start layout
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start address :"))

        self.start_addr_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Enter start address ...")
        start_layout.addWidget(self.start_addr_edit)

        layout.addLayout(start_layout)

        # End layout
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End address   :"))

        self.end_addr_edit = QLineEdit()
        self.end_addr_edit.setPlaceholderText("Enter end address ...")
        start_layout.addWidget(self.start_addr_edit)

        layout.addLayout(end_layout)
        
        self.find_button = QPushButton("Find")
        self.find_button.clicked.connect(self.on_find)
        layout.addWidget(self.find_button)
        
        # Results table
        # self.results_table = QTableWidget()
        # self.results_table.setColumnCount(4)
        # self.results_table.setHorizontalHeaderLabels(["Address", "String", "Function", "Segment"])
        # self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # self.results_table.doubleClicked.connect(self.on_double_click)
        #
        # layout.addWidget(self.results_table)
        #
        # self.setLayout(layout)
    
    def on_search(self):
        print("")
        # search_str = self.search_edit.text().strip()
        # if not search_str:
        #     return
        #
        # self.results_table.setRowCount(0)
        #
        # # Search through all strings in the binary
        # for s in idautils.Strings():
        #     if search_str.lower() in str(s).lower():
        #         row = self.results_table.rowCount()
        #         self.results_table.insertRow(row)
        #
        #         # Address
        #         addr_item = QTableWidgetItem(hex(s.ea))
        #         addr_item.setData(QtCore.Qt.UserRole, s.ea)
        #         self.results_table.setItem(row, 0, addr_item)
        #
        #         # String content
        #         str_content = ida_bytes.get_strlit_contents(s.ea, -1, s.strtype)
        #         try:
        #             str_content = str_content.decode('utf-8', errors='replace')
        #         except:
        #             str_content = str(str_content)
        #         self.results_table.setItem(row, 1, QTableWidgetItem(str_content))
        #
        #         # Function name
        #         func = idaapi.get_func(s.ea)
        #         func_name = idaapi.get_func_name(func.start_ea) if func else "N/A"
        #         self.results_table.setItem(row, 2, QTableWidgetItem(func_name))
        #
        #         # Segment name
        #         seg = idaapi.getseg(s.ea)
        #         seg_name = idaapi.get_segm_name(seg) if seg else "N/A"
        #         self.results_table.setItem(row, 3, QTableWidgetItem(seg_name))
    
    def on_double_click(self, index):
        # # Jump to address when row is double-clicked
        # item = self.results_table.item(index.row(), 0)
        # addr = item.data(QtCore.Qt.UserRole)
        # ida_kernwin.jumpto(addr)
        print("")