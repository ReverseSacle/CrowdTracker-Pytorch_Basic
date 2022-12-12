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

def video_tracker(threshold_value,root_dir,input_video_dir,use_gpus,
                opts,JDETracker,fourcc,
                QMessageBox,progressBar,QApplication,logger):

    try:
        threshold = threshold_value

        if os.path.exists(input_video_dir):
            file_name = (input_video_dir.split('.')[0]).split('/')[-1] + '_' + time.strftime('%Y_%m_%d_%H_%M',time.localtime())
            output_video_dir = root_dir + '/output_video'
            print(f'output video dir：{output_video_dir}')
        else:
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning！', "There is no exist the video file")
            msg_box.exec_()

        progressBar.setValue(0)
        logger.setLevel(logging.INFO)
        model_dir = root_dir + '/models'

        for pth in os.listdir(model_dir):
            if pth.split('.')[-1] == 'pth':
                model_dir += ('/' + pth)
                break
        print(f'model_dir: {model_dir}')

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        opt = opts(current_dir=root_dir,model_path=model_dir,
                   input_path=input_video_dir,threshold=threshold,
                   match_threshold=0.8,use_gpu=use_gpus).init()
        
        opt.output_root = output_video_dir
        print(f'current_use_gpus: {opt.gpus}')

        if opt.output_root:
            mkdir_if_missing(opt.output_root)

        # start to pre_track
        capture = cv2.VideoCapture(input_video_dir)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'frame_count: {frame_count}')

        progressBar.setMaximum(frame_count)

        # start to run track
        frame_rate = 30
        tracker = JDETracker(opt, frame_rate=frame_rate)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = fourcc
        writer = cv2.VideoWriter(output_video_dir + '/' + file_name + '.mp4', 
                                 fourcc, frame_rate, (width, height))
        results = []
        frame_id = 0
        timer = Timer()
        use_cuda = True
        
        if '-1' == use_gpus:
            use_cuda = False
            
        step = 0
        while (True):
            # run tracking
            step += 1
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (1920, 1080))
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
            online_im = vis.plot_tracking(frame, online_tlwhs, online_ids,
                                          frame_id=frame_id,fps=fps)
            frame_id += 1
            print(f'detect frame: {frame_id}')

            writer.write(online_im)
            progressBar.setValue(step)
            QApplication.processEvents()

        msg_box = QMessageBox(QMessageBox.Warning, '提示！', "视频预测完成")
        msg_box.exec_()
        writer.release()
        capture.release()

        return 1
    except:
        return 0