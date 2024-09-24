import PyQt5.QtWidgets as QtWidgets
from gui.main_window import MainWindow
import sys


# =============================================================================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(app)
    window.showMaximized()
    sys.exit(app.exec_())
