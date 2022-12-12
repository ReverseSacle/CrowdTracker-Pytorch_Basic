import os
import cv2
import logging
import torch
import time
import numpy as np
from lib.tracking_utils.timer import Timer
from lib.datasets.dataset.jde import letterbox
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils import visualization as vis


def built_in_camera_tracker(threshold_value, root_dir, use_gpu,
                            opts, JDETracker, fourcc, QPixmap, QImage, videolabel,
                            QMessageBox, QApplication, logger):
    try:
        cap_test = cv2.VideoCapture(0)
        if cap_test is None or not cap_test.isOpened():
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning！', 'The build-in camera is not exit')
            msg_box.exec_()
        else:
            cap_test.release()
    
        # if flag == 0:
        #     msg_box = QMessageBox(QMessageBox.Warning, '提示！', '请确保当前设备有摄像头')
        #     msg_box.exec_()

        # Set params
        logger.setLevel(logging.INFO)
        print(f'camera_id: {0}')
    
        model_dir = root_dir + '/models'

        for pth in os.listdir(model_dir):
            if pth.split('.')[-1] == 'pth':
                model_dir += ('/' + pth)
                break

        print(f'model_dir: {model_dir}')
    
        output_video_dir = root_dir + '/output_built_in_camera'
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        opt = opts(current_dir=root_dir,model_path=model_dir,
                   input_path=None,threshold=threshold_value,
                   match_threshold=0.8,use_gpu=use_gpu).init()

        opt.output_root = output_video_dir
        print(f'current_use_gpus: {opt.gpus}')
        print(f'output_video_dir: {output_video_dir}')
        mkdir_if_missing(output_video_dir)
    
        # start to pre_track
        capture = cv2.VideoCapture(0)
        # frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f'frame_count: {frame_count}', frame_count)
        #
        # progressBar.setMaximum(frame_count)
        frame_rate = 30
        tracker = JDETracker(opt, frame_rate=frame_rate)
        video_name = time.strftime('%Y_%m_%d_%H_%M',time.localtime()) + '_.mp4'
        print(f'video_name: {video_name}')
    
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = fourcc
        writer = cv2.VideoWriter((output_video_dir + '/' + video_name), fourcc, frame_rate, (width, height))
        results = []
        frame_id = 0
        timer = Timer()
        use_cuda = True
        if use_gpu == '-1':
            use_cuda = False
    
        while (True):
            try:
                # run tracking
                ok, frame = capture.read()
                if not ok:
                    break
                # frame = cv2.resize(frame, (1920, 1080))
                img, _, _, _ = letterbox(frame, height=1088, width=608)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                timer.tic()

                if use_cuda:
                    blob = torch.from_numpy(img).cuda().unsqueeze(0)
                else:
                    blob = torch.from_numpy(img).unsqueeze(0)

                online_targets = tracker.update(blob, frame)
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
                online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                              fps=fps)
                frame_id += 1
                print(f'detect frame: {frame_id}')
    
                height, width = online_im.shape[:2]
                if online_im.ndim == 3:
                    rgb = cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)
                elif online_im.ndim == 2:
                    rgb = cv2.cvtColor(online_im, cv2.COLOR_GRAY2BGR)

                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                writer.write(online_im)
                videolabel.setPixmap(temp_pixmap)
                QApplication.processEvents()
            except:
                writer.release()
        writer.release()
    except:
        pass