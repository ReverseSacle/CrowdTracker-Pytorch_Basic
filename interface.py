from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
### lib
from lib.opts import opts
from lib.tracking_utils.log import logger
from lib.tracker.multitracker import JDETracker
### PyQt5
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QFileDialog, QStyle, QMessageBox, QApplication, QProgressBar
from PyQt5 import QtCore, QtWidgets
### src
from src.VideoTimer import VideoTimer
from src.video_track import video_tracker
from src.built_in_camera_track import built_in_camera_tracker
from src.external_camera_track import external_camera_tracker


class VideoQt(QWidget):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1
    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    def __init__(self, video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        QWidget.__init__(self)
        self.input_video_dir = ""
        self.output_video_dir = ""
        self.video_for_browse = ""
        self.root_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

        self.flag_to_video = 0
        self.flag_for_video_to_display = None

        self.video_type = video_type  # 0: offline  1: realTime
        self.auto_play = auto_play
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause

        # Set timer.
        # 设置计时器
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)

        self.threshold = 0.4
        # Choose the GPU that you want to do this with(CPU: -1,GPU_1: 0,GPU_2: 1).
        # 追踪用的GPU(CPU: -1,GPU_1: 0,GPU_2: 1)
        self.use_gpu = '0'


        # Init parameters.
        # 初始化参数
        self.playCapture = cv2.VideoCapture()
        if self.output_video_dir != "":
            self.playCapture.open(self.output_video_dir)
            fps = self.playCapture.get(cv2.CAP_PROP_FPS)
            self.timer.set_fps(fps)
            self.playCapture.release()
            if self.auto_play:
                self.switch_video()

        # set the video coding.
        # 设置编码
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Choose the video for video track.
    # 选择需要预测的视频文件
    def choose_file(self):
        self.input_video_dir, filetype = QFileDialog.getOpenFileName(self, "请选择所需要的文件", os.getcwd(),
                                                                   "All Files(*);;Text Files(*.txt)")
        self.flag_for_video_to_display = 0  # Open
        print(f'输入路径：{self.input_video_dir}')

    # Choose the video that you want to browse.
    # 选择需要打开的视频文件
    def choose_file_to_view(self):
        self.video_for_browse, filetype = QFileDialog.getOpenFileName(self, "请选择所需要的文件", os.getcwd(),
                                                                           "All Files(*);;Text Files(*.txt)")
        self.flag_for_video_to_display = 1
        print(f'选择查看的视频路径：{self.video_for_browse}')

    # The function of the video for video track.
    # 选择需要预测的视频文件的方法函数
    def open_video_file(self):
        self.choose_file()

    # The function of the video that you want to browse.
    # 选择需要打开的视频文件的方法函数
    def open_video_for_view(self):
        self.choose_file_to_view()

    # Refresh thread.
    # 更新进程
    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoQt.STATUS_INIT
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))


    # Display image in the field of the zone of display.
    # 在界面上展示图像
    def show_video_images(self):
        if self.playCapture.isOpened():
            ok, frame = self.playCapture.read()
            if ok:
                frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_AREA)
                height, width = frame.shape[:2]

                if frame.ndim == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.videolabel.setPixmap(temp_pixmap)
            else:
                print("read failed, no frame data")
                ok, frame = self.playCapture.read()
                if not ok and self.video_type is VideoQt.VIDEO_TYPE_OFFLINE:
                    print("play finished")
                    self.reset()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    # The video function of stop and play.
    # 暂停与播放开关
    def switch_video(self):
        flag = 0
        if self.flag_for_video_to_display == 1:
            change_input_video_dir = self.video_for_browse
            flag = 1

        if self.flag_for_video_to_display == 0:
            change_input_video_dir = self.input_video_dir
            flag = 1
        try:
            print(f'当前选择的视频文件：{change_input_video_dir}')
        except:
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning！', 'There is no exist the video file')
            msg_box.exec_()

        if self.flag_for_video_to_display == 0 and flag == 1:
            if self.flag_to_video == 1:
                change_input_video_dir = self.output_video_dir
                if change_input_video_dir == "" or change_input_video_dir is None:
                    return
                if self.status is VideoQt.STATUS_INIT:
                    self.playCapture.open(change_input_video_dir)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                elif self.status is VideoQt.STATUS_PLAYING:
                    self.timer.stop()
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.release()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                elif self.status is VideoQt.STATUS_PAUSE:
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.open(change_input_video_dir)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

                self.status = (VideoQt.STATUS_PLAYING,
                               VideoQt.STATUS_PAUSE,
                               VideoQt.STATUS_PLAYING)[self.status]


        else:
            if self.flag_for_video_to_display == 1 and flag == 1:
                if change_input_video_dir == "" or change_input_video_dir is None:
                    return
                if self.status is VideoQt.STATUS_INIT:
                    self.playCapture.open(change_input_video_dir)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                elif self.status is VideoQt.STATUS_PLAYING:
                    self.timer.stop()
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.release()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                elif self.status is VideoQt.STATUS_PAUSE:
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.open(change_input_video_dir)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

                self.status = (VideoQt.STATUS_PLAYING,
                               VideoQt.STATUS_PAUSE,
                               VideoQt.STATUS_PLAYING)[self.status]
            else:
                if flag == 1:
                    msg_box = QMessageBox(QMessageBox.Warning, 'Warning！',
                                          "There should be ensure that you have done the process of choosing file")
                    msg_box.exec_()
                else:
                    pass

    # Get the selected threshold.
    # 获取选择的阈值
    def onActivated_Threshold(self, text):
        self.threshold = text
        print(f'threshold: {self.threshold}')

    # Get the selected GPU.
    # 获取选择的GPU
    def onActivated_GPU(self, text):
        self.use_gpu = text
        print(f'GPU: {self.use_gpu}')

    # Camera track with built-in camera.
    # 使用当前设备摄像头追踪
    def open_current_device_video(self):
        built_in_camera_tracker(threshold_value=self.threshold, root_dir=self.root_dir, use_gpu=self.use_gpu,
                                opts=opts, JDETracker=JDETracker, fourcc=self.fourcc,
                                QPixmap=QPixmap, QImage=QImage, videolabel=self.videolabel,
                                QMessageBox=QMessageBox, QApplication=QApplication, logger=logger)

    # Camera track with external camera.
    # 使用外置摄像头追踪
    def open_other_device_video(self):
        external_camera_tracker(threshold_value=self.threshold, root_dir=self.root_dir, use_gpu=self.use_gpu,
                                opts=opts, JDETracker=JDETracker, fourcc=self.fourcc,
                                QPixmap=QPixmap, QImage=QImage, videolabel=self.videolabel,
                                QMessageBox=QMessageBox, QApplication=QApplication, logger=logger)

    # Close camera
    # 关闭摄像头
    def close_video(self):
        try:
            self.writer.release()
        except:
            pass
        try:
            self.cap.release()
        except:
            pass
        cv2.destroyAllWindows()

    # The function of video track
    # 视频追踪
    def predict_video(self):
        self.flag_to_video = video_tracker(
            threshold_value=self.threshold,root_dir=self.root_dir,input_video_dir=self.input_video_dir,
            use_gpus=self.use_gpu,opts=opts,JDETracker=JDETracker,fourcc=self.fourcc,
            QMessageBox=QMessageBox,progressBar=self.progressBar,QApplication=QApplication,logger=logger)

    # The function of choosing the seleted video and open it to play.
    # 选择并打开所选的视频文件
    def open_video_file_to_view(self):
        self.open_video_for_view()
        self.switch_video()

    # The fuction of exit.
    # 退出
    def closeEvent(self, event):

        reply = QtWidgets.QMessageBox.question(self, 'Warining！',"Is it already to exit?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            os._exit(0)

    # ---------------Qt Page-------------------------#

    def setupUi(self, Form):
        Form.setObjectName("Form")
        # Windows size.
        # 窗口大小
        Form.resize(960, 610)

        # The field of camera.
        # 摄像头使用的区域
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setStyleSheet('font-size:16px;color:white;')
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 5, 800, 600))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)

        # The field of camera to pad.
        # 摄像头播放区区域
        self.horizontalLayout.setObjectName("horizontalLayout")
        # self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.videolabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.videolabel.setAutoFillBackground(True)
        self.videolabel.setGeometry(0, 5, 800, 600)
        self.videolabel.setScaledContents(True)
        self.videolabel.setObjectName("videolabel")
        self.horizontalLayout.addWidget(self.videolabel)

        # The field of video to open.
        # 设置视频播放区
        # self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.videolabel2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.videolabel2.setAutoFillBackground(True)
        self.videolabel2.setGeometry(0, 5, 800, 600)
        self.videolabel2.setScaledContents(True)
        self.videolabel2.setObjectName("videolabel2")
        self.horizontalLayout.addWidget(self.videolabel2)

        # The field of buttons.
        # 按钮组区域
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(800, 200, 150, 300))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayoutWidget.setStyleSheet('font-size:16px;color:white;')
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        # The field of threshold choosing.
        # 阈值选择区
        self.Thresholdlable = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.Thresholdlable.setObjectName("Thresholdlable")
        self.Thresholdlable.setStyleSheet('font-size:16px;color:white;')
        self.verticalLayout.addWidget(self.Thresholdlable)
        self.Threshold = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.Threshold.setObjectName("Threshold")
        self.Threshold.setStyleSheet('background-color:slategray;font-size:16px;color:white;'
                                     'selection-background-color:gainsboro;'
                                     'selection-color:red;')
        self.Threshold.activated[str].connect(self.onActivated_Threshold)
        self.verticalLayout.addWidget(self.Threshold)

        # The field of threshold choosing.
        # GPU选择区
        self.GPUlable = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.GPUlable.setObjectName("GPUlable")
        self.GPUlable.setStyleSheet('font-size:16px;color:white;')
        self.verticalLayout.addWidget(self.GPUlable)
        self.GPU = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.GPU.setObjectName("GPU")
        self.GPU.setStyleSheet('background-color:slategray;font-size:16px;color:white;'
                                     'selection-background-color:gainsboro;'
                                     'selection-color:red;')
        self.GPU.activated[str].connect(self.onActivated_GPU)
        self.verticalLayout.addWidget(self.GPU)

        # The button of selecting video file.
        # 选择视频文件按钮
        self.selectButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.selectButton.setObjectName("selectButton")
        self.selectButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.selectButton.clicked.connect(self.open_video_file)
        self.verticalLayout.addWidget(self.selectButton)

        # The button of video tracking.
        # 视频检测按钮
        self.videoButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.videoButton.setEnabled(True)
        self.videoButton.setObjectName("videoButton")
        self.videoButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.videoButton.clicked.connect(self.predict_video)
        self.verticalLayout.addWidget(self.videoButton)

        # The button of stop and play.
        # 暂停与播放键按钮
        self.stopButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.stopButton.setObjectName("startButto")
        self.stopButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.stopButton.clicked.connect(self.switch_video)
        self.verticalLayout.addWidget(self.stopButton)

        # The button of opening built-in camera.
        # 打开当前摄像头按钮
        self.openButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.openButton.setObjectName("openButton")
        self.openButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.openButton.setGeometry(0, 0, 32, 32)
        self.openButton.clicked.connect(self.open_current_device_video)
        self.verticalLayout.addWidget(self.openButton)

        # The button of opening external camera.
        # 打开外置摄像头按钮
        self.openButton2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.openButton2.setObjectName("openButton")
        self.openButton2.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.openButton2.setGeometry(0, 0, 32, 32)
        self.openButton2.clicked.connect(self.open_other_device_video)
        self.verticalLayout.addWidget(self.openButton2)

        # The button of closing camera.
        # 关闭摄像头按钮
        self.closeButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.closeButton.setObjectName("closeButton")
        self.closeButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.closeButton.clicked.connect(self.close_video)
        self.verticalLayout.addWidget(self.closeButton)

        # The button of opening video to browsing。
        # 打开视频文件并播放按钮
        self.selectButton2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.selectButton2.setObjectName("selectButton2")
        self.selectButton2.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.selectButton2.clicked.connect(self.open_video_file_to_view)
        self.verticalLayout.addWidget(self.selectButton2)

        # The button of closing windows.
        # 关闭按钮
        self.quitButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.quitButton.setObjectName("quitButton")
        self.quitButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.quitButton.clicked.connect(self.closeEvent)
        self.verticalLayout.addWidget(self.quitButton)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        # The field of progress bar.
        # 进度条区域
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0,100)
        self.progressBar.setValue(0)
        self.verticalLayout.addWidget(self.progressBar)

    def retranslateUi(self, Form):
        self._translate = QtCore.QCoreApplication.translate
        # The title of interface.
        # 界面标题(行人追踪)
        Form.setWindowTitle(self._translate("Form", "CrowdTrack"))
        self.videolabel.setText(self._translate("Form", " "))
        self.videolabel2.setText(self._translate("Form", "Display field"))
        # The choosing bar of threshold value.
        # 阈值选择栏
        self.Thresholdlable.setText(self._translate("Form", "Threshold value"))
        self.Threshold.addItem(self._translate("Form", "0.4"))
        self.Threshold.addItem(self._translate("Form", "0.5"))
        self.Threshold.addItem(self._translate("Form", "0.6"))
        self.Threshold.addItem(self._translate("Form", "0.7"))
        self.Threshold.addItem(self._translate("Form", "0.8"))
        self.Threshold.addItem(self._translate("Form", "0.9"))
        # The choosing bar of GPU.
        # GPU选择栏
        self.GPUlable.setText(self._translate("Form", "GPU"))
        self.GPU.addItem(self._translate("Form", '-1'))
        self.GPU.addItem(self._translate("Form", '0'))
        self.GPU.addItem(self._translate("Form", '1'))
        self.GPU.addItem(self._translate("Form", '2'))
        # The choosing button of the selected video for track.
        # 选择用于追踪的视频文件按钮
        self.selectButton.setText(self._translate("Form", "Video for track"))
        # The choosing button of video track.
        # 视频追踪按钮
        self.videoButton.setText(self._translate("Form", "Video track"))
        # The choosing button of controling the state of the video.
        # 视频当前状态控制按钮
        self.stopButton.setText(self._translate("Form", "Video state"))
        # The choosing button of built-in camera for track.
        # 内置摄像头追踪按钮
        self.openButton.setText(self._translate("Form", "Built-in camera"))
        # The choosing button of external camera for track.
        # 外置摄像头追踪按钮
        self.openButton2.setText(self._translate("Form", "External camera"))
        # The choosing button of closing the camera.
        # 关闭摄像头选择按钮
        self.closeButton.setText(self._translate("Form", "Close camera"))
        # The choosing button of the selected video for browsing.
        # 选择用于浏览的视频按钮
        self.selectButton2.setText(self._translate("Form", "Video for browse"))
        # The choosing button of exit.
        # 退出按钮
        self.quitButton.setText(self._translate("Form", "Exit"))
