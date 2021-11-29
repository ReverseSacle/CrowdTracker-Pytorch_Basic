from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src._init_paths
import cv2
import torch
import os
import sys
import time
import logging
import datetime
import numpy as np

from src.lib.opts import opts
from src.lib.tracking_utils.utils import mkdir_if_missing
from src.lib.tracking_utils.log import logger
from src.lib.tracker.multitracker import JDETracker
from src.lib.tracking_utils.timer import Timer
from src.lib.tracking_utils import visualization as vis
from src.lib.datasets.dataset.jde import letterbox

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QFileDialog, QStyle, QMessageBox, QApplication, QProgressBar
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker


class Communicate(QObject):
    signal = pyqtSignal(str)


class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


class VideoQt(QWidget):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1
    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2


    def __init__(self, video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        QWidget.__init__(self)
        self.video_address = ""
        self.video_output_address = ""
        self.choose_to_view_folder = ""
        self.couting_time = 0
        self.flag_to_video = 0
        self.flag_for_video_to_display = None
        self.video_type = video_type  # 0: offline  1: realTime
        self.auto_play = auto_play
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause

        # Set timer（设置timer)
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)
        #*******************************************************************
        self.set_root_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
        self.set_video_file = self.set_root_dir + '/test.avi'
        self.set_threshold = 0.3
        self.set_match_threshold = 0.8
        #Depending on how man gpus for you to use and '-1' to cpu(取决于你要使用多少个gpu,'-1'为cpu)
        self.set_use_gpus = '1'

        #******************************************************************

        # Init params for video(初始化参数)
        self.playCapture = cv2.VideoCapture()
        if self.video_output_address != "":
            self.playCapture.open(self.video_output_address)
            fps = self.playCapture.get(cv2.CAP_PROP_FPS)
            self.timer.set_fps(fps)
            self.playCapture.release()
            if self.auto_play:
                self.switch_video()

        self.threshold_value = 0.1
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置编码

    # To choose the needed video file to predict(选择需要预测的视频文件)
    def choose_file(self):
        self.video_address, filetype = QFileDialog.getOpenFileName(self, "请选择所需要的文件", os.getcwd(),
                                                                   "All Files(*);;Text Files(*.txt)")
        self.flag_for_video_to_display = 0  # 打开
        print('输入路径：' + self.video_address)

    # To choose the needed video file to view(选择需要打开的视频文件)
    def choose_file_to_view(self):
        self.choose_to_view_folder, filetype = QFileDialog.getOpenFileName(self, "请选择所需要的文件", os.getcwd(),
                                                                           "All Files(*);;Text Files(*.txt)")
        self.flag_for_video_to_display = 1
        print('选择查看的视频路径：' + self.choose_to_view_folder)

    # The function to open the needed video file to predict(选择需要预测的视频文件的方法函数)
    def open_video_file(self):
        self.choose_file()

    # The function to open the needed video file to view(选择需要打开的视频文件的方法函数)
    def open_video_for_view(self):
        self.choose_file_to_view()

    # Reflash thread(更新进程)
    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoQt.STATUS_INIT
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))


    # Show image in the filed of display(在界面上展示图像)
    def show_video_images(self):
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
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
                success, frame = self.playCapture.read()
                if not success and self.video_type is VideoQt.VIDEO_TYPE_OFFLINE:
                    print("play finished")
                    self.reset()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    # The video function of stop and play(暂停与播放开关)
    def switch_video(self):
        flag = 0
        if self.flag_for_video_to_display == 1:
            change_video_address = self.choose_to_view_folder
            flag = 1

        if self.flag_for_video_to_display == 0:
            change_video_address = self.video_address
            flag = 1
        try:
            print('当前选择的视频文件：' + change_video_address)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '提示！', '请确保有选择视频文件')
            msg_box.exec_()

        if self.flag_for_video_to_display == 0 and flag == 1:
            if self.flag_to_video == 1:
                change_video_address = self.video_output_address
                if change_video_address == "" or change_video_address is None:
                    return
                if self.status is VideoQt.STATUS_INIT:
                    self.playCapture.open(change_video_address)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                elif self.status is VideoQt.STATUS_PLAYING:
                    self.timer.stop()
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.release()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                elif self.status is VideoQt.STATUS_PAUSE:
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.open(change_video_address)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

                self.status = (VideoQt.STATUS_PLAYING,
                               VideoQt.STATUS_PAUSE,
                               VideoQt.STATUS_PLAYING)[self.status]


        else:
            if self.flag_for_video_to_display == 1 and flag == 1:
                if change_video_address == "" or change_video_address is None:
                    return
                if self.status is VideoQt.STATUS_INIT:
                    self.playCapture.open(change_video_address)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                elif self.status is VideoQt.STATUS_PLAYING:
                    self.timer.stop()
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.release()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                elif self.status is VideoQt.STATUS_PAUSE:
                    if self.video_type is VideoQt.VIDEO_TYPE_REAL_TIME:
                        self.playCapture.open(change_video_address)
                    self.timer.start()
                    self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

                self.status = (VideoQt.STATUS_PLAYING,
                               VideoQt.STATUS_PAUSE,
                               VideoQt.STATUS_PLAYING)[self.status]
            else:
                if flag == 1:
                    msg_box = QMessageBox(QMessageBox.Warning, '提示！',  "请确保已经进行了视频预测")
                    msg_box.exec_()
                else:
                    pass

    #Get Threshold(获取阈值)
    def onActivated(self, text):
        self.threshold_value = text
        print(self.threshold_value)

    #Use current device's camera to predict(使用当前设备摄像头)
    def open_current_device_video(self):
        flag = 0
        try:
            try:
                cap_test = cv2.VideoCapture(0)
                if cap_test is None or not cap_test.isOpened():
                    print()
                else:
                    flag = 1
                cap_test.release()
            except:
                pass

            if flag == 0:
                msg_box = QMessageBox(QMessageBox.Warning, '提示！', '请确保当前设备有摄像头')
                msg_box.exec_()
            # Set params
            logger.setLevel(logging.INFO)
            # set params
            print('camera_id: ' + str(0))

            set_model_dir = self.set_root_dir + '/models'
            set_model_path = None

            for pth in os.listdir(set_model_dir):
                if pth.split('.')[-1] == 'pth':
                    set_model_path = set_model_dir + '/' + pth
                    break
            print('model_path: ' + set_model_path)

            set_output_path = self.set_root_dir + '/output_result'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            opt = opts(current_dir=self.set_root_dir,
                       model_path=set_model_path,
                       input_path=None,
                       threshold=self.set_threshold,
                       match_threshold=self.set_match_threshold,
                       use_gpu=self.set_use_gpus).init()

            print('current_use_gpus: ')
            print(opt.gpus)
            print('output_path: ' + set_output_path)
            mkdir_if_missing(set_output_path)

            # start to pre_track
            self.capture = cv2.VideoCapture(0)
            frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print('frame_count', frame_count)

            self.progressBar.setMaximum(frame_count)
            frame_rate = 30
            tracker = JDETracker(opt, frame_rate=frame_rate)
            current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
            video_name = current_time + '.mp4'
            print('video_name: ' + video_name)

            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = self.fourcc
            self.writer = cv2.VideoWriter((set_output_path + '/' + video_name), fourcc, frame_rate, (width, height))
            results = []
            frame_id = 0
            timer = Timer()
            use_cuda = True
            if self.set_use_gpus == '-1':
                use_cuda = False

            while (True):
                try:
                    # run tracking
                    ret, img0 = self.capture.read()
                    if not ret:
                        break
                    # img0 = cv2.resize(img0, (1920, 1080))
                    img, _, _, _ = letterbox(img0, height=1088, width=608)
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img, dtype=np.float32)
                    img /= 255.0
                    timer.tic()
                    if use_cuda:
                        blob = torch.from_numpy(img).cuda().unsqueeze(0)
                    else:
                        blob = torch.from_numpy(img).unsqueeze(0)
                    online_targets = tracker.update(blob, img0)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                    timer.toc()
                    # save results
                    # results.append((frame_id + 1, online_tlwhs, online_ids))
                    results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                    fps = 1. / timer.average_time
                    online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                                  fps=fps)
                    frame_id += 1
                    print('detect frame:%d' % (frame_id))

                    height, width = online_im.shape[:2]
                    if online_im.ndim == 3:
                        rgb = cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)
                    elif online_im.ndim == 2:
                        rgb = cv2.cvtColor(online_im, cv2.COLOR_GRAY2BGR)
                    temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                    temp_pixmap = QPixmap.fromImage(temp_image)
                    self.writer.write(online_im)
                    self.videolabel.setPixmap(temp_pixmap)
                    QApplication.processEvents()
                except:
                    self.writer.release()
            self.writer.release()
        except:
            pass

    #Use the other camera of current device to predict(使用外置摄像头)
    def open_other_device_video(self):
        flag = 0
        try:
            try:
                cap_test = cv2.VideoCapture(1)
                if cap_test is None or not cap_test.isOpened():
                        print()
                else:
                        flag = 1
                cap_test.release()
            except:
                pass

            if flag == 0:
                msg_box = QMessageBox(QMessageBox.Warning, '提示！', '请确保有外置摄像头')
                msg_box.exec_()

            if flag == 1:
                # Set params
                logger.setLevel(logging.INFO)
                print('camera_id: ' + str(1))

                set_model_dir = self.set_root_dir + '/models'
                set_model_path = None

                for pth in os.listdir(set_model_dir):
                    if pth.split('.')[-1] == 'pth':
                        set_model_path = set_model_dir + '/' + pth
                        break
                print('model_path: ' + set_model_path)

                set_output_path = self.set_root_dir + '/output_result'
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                opt = opts(current_dir=self.set_root_dir,
                           model_path=set_model_path,
                           input_path=None,
                           threshold=self.set_threshold,
                           match_threshold=self.set_match_threshold,
                           use_gpu=self.set_use_gpus).init()

                print('current_use_gpus: ')
                print(opt.gpus)
                print('output_path: ' + set_output_path)
                mkdir_if_missing(set_output_path)

                # start to pre_track
                self.capture = cv2.VideoCapture(1)
                frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                print('frame_count', frame_count)

                self.progressBar.setMaximum(frame_count)
                frame_rate = 30
                tracker = JDETracker(opt, frame_rate=frame_rate)
                current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
                video_name = current_time + '.mp4'
                print('video_name: ' + video_name)

                width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = self.fourcc
                self.writer = cv2.VideoWriter((set_output_path + '/' + video_name), fourcc, frame_rate, (width, height))
                results = []
                frame_id = 0
                timer = Timer()
                use_cuda = True
                if self.set_use_gpus == '-1':
                    use_cuda = False

                while (True):
                    try:
                        # run tracking
                        ret, img0 = self.capture.read()
                        if not ret:
                            break
                        # img0 = cv2.resize(img0, (1920, 1080))
                        img, _, _, _ = letterbox(img0, height=1088, width=608)
                        img = img[:, :, ::-1].transpose(2, 0, 1)
                        img = np.ascontiguousarray(img, dtype=np.float32)
                        img /= 255.0
                        timer.tic()
                        if use_cuda:
                            blob = torch.from_numpy(img).cuda().unsqueeze(0)
                        else:
                            blob = torch.from_numpy(img).unsqueeze(0)
                        online_targets = tracker.update(blob, img0)
                        online_tlwhs = []
                        online_ids = []
                        online_scores = []
                        for t in online_targets:
                            tlwh = t.tlwh
                            tid = t.track_id
                            vertical = tlwh[2] / tlwh[3] > 1.6
                            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)
                        timer.toc()
                        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                        fps = 1. / timer.average_time
                        # save results
                        # results.append((frame_id + 1, online_tlwhs, online_ids))
                        # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                                      fps=fps)
                        frame_id += 1
                        print('detect frame:%d' % (frame_id))

                        height, width = online_im.shape[:2]
                        if online_im.ndim == 3:
                            rgb = cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)
                        elif online_im.ndim == 2:
                            rgb = cv2.cvtColor(online_im, cv2.COLOR_GRAY2BGR)
                        temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                        temp_pixmap = QPixmap.fromImage(temp_image)
                        self.writer.write(online_im)
                        self.videolabel.setPixmap(temp_pixmap)
                        QApplication.processEvents()
                    except:
                        self.writer.release()
                self.writer.release()
        except:
            pass

    #Close camera(关闭摄像头)
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


    #The function of predict video(视频追踪)
    def predict_video(self):
        flag_predict = 0

        try:
            # Set params
            #self.set_threshold = self.threshold_value  # 阈值
            if os.path.exists(self.video_address):
                set_video_file = self.video_address
                flag_predict = 1
            else:
                msg_box = QMessageBox(QMessageBox.Warning, '提示！', "请确保有选择视频文件")
                msg_box.exec_()
            if flag_predict == 1:
                file_name = (set_video_file.split('.')[0]).split('/')[-1] + '_output'
                self.video_output_address = self.set_root_dir + '/output_result'
                print('输出路径：' + self.video_output_address)

            self.progressBar.setValue(0)
            logger.setLevel(logging.INFO)
            set_model_dir = self.set_root_dir + '/models'
            set_model_path = None

            for pth in os.listdir(set_model_dir):
                if pth.split('.')[-1] == 'pth':
                    set_model_path = set_model_dir + '/' + pth
                    break
            print('model_path: ' + set_model_path)

            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            opt = opts(current_dir=self.set_root_dir,
                       model_path=set_model_path,
                       input_path=set_video_file,
                       threshold=self.set_threshold,
                       match_threshold=self.set_match_threshold,
                       use_gpu=self.set_use_gpus).init()
            opt.output_path = self.video_output_address
            print('current_use_gpus: ')
            print(opt.gpus)

            if opt.output_path:
                mkdir_if_missing(opt.output_path)

            # start to pre_track
            capture = cv2.VideoCapture(set_video_file)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print('frame_count', frame_count)

            self.progressBar.setMaximum(frame_count)

            #start to run track
            frame_rate = 30
            tracker = JDETracker(opt, frame_rate=frame_rate)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = self.fourcc
            writer = cv2.VideoWriter(self.video_output_address + '/' + file_name + '.mp4', fourcc, frame_rate, (width, height))
            results = []
            frame_id = 0
            timer = Timer()
            use_cuda = True
            if self.set_use_gpus == '-1':
                use_cuda = False
            step = 0
            while (True):
                # run tracking
                step += 1
                ret, img0 = capture.read()
                if not ret:
                    break
                img0 = cv2.resize(img0, (1920, 1080))
                img, _, _, _ = letterbox(img0, height=1088, width=608)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                timer.tic()
                if use_cuda:
                    blob = torch.from_numpy(img).cuda().unsqueeze(0)
                else:
                    blob = torch.from_numpy(img).unsqueeze(0)
                online_targets = tracker.update(blob, img0)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                timer.toc()
                # save results
                # results.append((frame_id + 1, online_tlwhs, online_ids))
                results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
                fps = 1. / timer.average_time
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=fps)
                frame_id += 1
                print('detect frame:%d' % (frame_id))

                writer.write(online_im)
                self.progressBar.setValue(step)
                QApplication.processEvents()
            self.flag_to_video = 1
            msg_box = QMessageBox(QMessageBox.Warning, '提示！',  "视频预测完成")
            msg_box.exec_()
            writer.release()
            capture.release()
        except:
            pass



    # The function to choose the seleted video to open(选择并打开所选的视频文件)
    def open_video_file_to_view(self):
        self.open_video_for_view()

    #The fuction to quit(退出)
    def closeEvent(self, event):

        reply = QtWidgets.QMessageBox.question(self, '提示！',
                                               "确定要退出程序吗?",
                                               QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            os._exit(0)

        else:
            pass

            # ---------------qt页面-------------------------#

    def setupUi(self, Form):
        Form.setObjectName("Form")
        # Windows size(窗口大小)
        Form.resize(960, 610)

        # The filed of camera(摄像头使用的区域)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setStyleSheet('font-size:16px;color:white;')
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 5, 800, 600))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)

        # The filed of camera to pad(摄像头播放区区域)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.videolabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.videolabel.setAutoFillBackground(True)
        self.videolabel.setGeometry(0, 5, 800, 600)
        self.videolabel.setScaledContents(True)
        self.videolabel.setObjectName("videolabel")
        self.horizontalLayout.addWidget(self.videolabel)

        # The filed of video to open(设置视频播放区)
        # self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.videolabel2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.videolabel2.setAutoFillBackground(True)
        self.videolabel2.setGeometry(0, 5, 800, 600)
        self.videolabel2.setScaledContents(True)
        self.videolabel2.setObjectName("videolabel2")
        self.horizontalLayout.addWidget(self.videolabel2)

        # The filed of button(按钮组区域)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(800, 200, 150, 300))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayoutWidget.setStyleSheet('font-size:16px;color:white;')
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        # The filed of threshold(阈值选择区)
        self.Thresholdlable = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.Thresholdlable.setObjectName("Thresholdlable")
        self.Thresholdlable.setStyleSheet('font-size:16px;color:white;')
        self.verticalLayout.addWidget(self.Thresholdlable)
        self.Threshold = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.Threshold.setObjectName("Threshold")
        self.Threshold.setStyleSheet('background-color:slategray;font-size:16px;color:white;'
                                     'selection-background-color:gainsboro;'
                                     'selection-color:red;')
        self.Threshold.activated[str].connect(self.onActivated)
        self.verticalLayout.addWidget(self.Threshold)

        # The button of select video file(选择视频文件按钮)
        self.selectButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.selectButton.setObjectName("selectButton")
        self.selectButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.selectButton.clicked.connect(self.open_video_file)
        self.verticalLayout.addWidget(self.selectButton)

        # The button of predict video(视频检测按钮)
        self.videoButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.videoButton.setEnabled(True)
        self.videoButton.setObjectName("videoButton")
        self.videoButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.videoButton.clicked.connect(self.predict_video)
        self.verticalLayout.addWidget(self.videoButton)

        # The button of stop and play(暂停与播放键按钮)
        self.stopButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.stopButton.setObjectName("startButto")
        self.stopButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.stopButton.clicked.connect(self.switch_video)
        self.verticalLayout.addWidget(self.stopButton)

        # The button of use current camera(打开当前摄像头按钮)
        self.openButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.openButton.setObjectName("openButton")
        self.openButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.openButton.setGeometry(0, 0, 32, 32)
        self.openButton.clicked.connect(self.open_current_device_video)
        self.verticalLayout.addWidget(self.openButton)

        # The button of use the other of current camera(打开外置摄像头按钮)
        self.openButton2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.openButton2.setObjectName("openButton")
        self.openButton2.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.openButton2.setGeometry(0, 0, 32, 32)
        self.openButton2.clicked.connect(self.open_other_device_video)
        self.verticalLayout.addWidget(self.openButton2)

        # The button of open video to view(打开视频文件按钮)
        self.selectButton2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.selectButton2.setObjectName("selectButton2")
        self.selectButton2.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.selectButton2.clicked.connect(self.open_video_file_to_view)
        self.verticalLayout.addWidget(self.selectButton2)

        # The button of close camera(关闭摄像头按钮)
        self.closeButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.closeButton.setObjectName("closeButton")
        self.closeButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.closeButton.clicked.connect(self.close_video)
        self.verticalLayout.addWidget(self.closeButton)

        # The button of close windows(关闭按钮)
        self.quitButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.quitButton.setObjectName("quitButton")
        self.quitButton.setStyleSheet('background-color:slategray;font-size:16px;color:white;')
        self.quitButton.clicked.connect(self.closeEvent)
        self.verticalLayout.addWidget(self.quitButton)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        #The filed of progress bar(进度条区域)
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0,100)
        self.progressBar.setValue(0)
        self.verticalLayout.addWidget(self.progressBar)


    def retranslateUi(self, Form):
        self._translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(self._translate("Form", "行人检测与追踪"))
        self.videolabel.setText(self._translate("Form", " "))
        self.videolabel2.setText(self._translate("Form", "展示区"))
        self.Thresholdlable.setText(self._translate("Form", "阈值选择"))
        self.openButton.setText(self._translate("Form", "使用内置摄像头"))
        self.openButton2.setText(self._translate("Form", "使用外置摄像头"))
        self.closeButton.setText(self._translate("Form", "关闭摄像头"))
        self.Threshold.addItem(self._translate("Form", "0.1"))
        self.Threshold.addItem(self._translate("Form", "0.2"))
        self.Threshold.addItem(self._translate("Form", "0.3"))
        self.Threshold.addItem(self._translate("Form", "0.4"))
        self.Threshold.addItem(self._translate("Form", "0.5"))
        self.Threshold.addItem(self._translate("Form", "0.6"))
        self.Threshold.addItem(self._translate("Form", "0.7"))
        self.Threshold.addItem(self._translate("Form", "0.8"))
        self.Threshold.addItem(self._translate("Form", "0.9"))
        self.selectButton.setText(self._translate("Form", "选择需要追踪的视频"))
        self.selectButton2.setText(self._translate("Form", "选择需要打开的视频"))
        self.videoButton.setText(self._translate("Form", "视频追踪"))
        self.stopButton.setText(self._translate("Form", "视频当前状态"))
        self.quitButton.setText(self._translate("Form", "退出"))


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    widget.setStyleSheet('background-color:black;')
    widget.setWindowOpacity(0.8)
    video = VideoQt()
    video.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())
