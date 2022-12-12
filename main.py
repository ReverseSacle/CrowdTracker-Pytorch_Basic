import sys
from interface import VideoQt,QtCore,QApplication,QWidget

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    widget = QWidget()
    widget.setStyleSheet('background-color:black;')
    widget.setWindowOpacity(0.8)
    video = VideoQt()
    video.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())
