from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import torch
import numpy as np
import os
import logging
import time

from lib.opts import opts
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
from lib.tracker.multitracker import JDETracker
from lib.tracking_utils.timer import Timer
from lib.tracking_utils import visualization as vis
from lib.datasets.dataset.jde import letterbox

logger.setLevel(logging.INFO)

# set parameters.
# 设置参数
current_dir = os.path.dirname(os.path.realpath(__file__)).\
                    replace('\\','/').replace('/src/tracker','')
input_path =  current_dir + '/videos/MOT16-03.mp4'
input_file_name = (input_path.split('/')[-1]).split('.')[0]
output_path = current_dir + '/output_video/'
model_dir = current_dir + '/models'
threshold = 0.4
match_threshold = 0.8

# Choose the GPU that you want to do this with(CPU: -1,GPU_1: 0,GPU_2: 1).
# 追踪用的GPU(CPU: -1,GPU_1: 0,GPU_2: 1)
set_use_gpu = '-1'
print(f'input_path: {input_path}')

for pth in os.listdir(model_dir):
    if pth.split('.')[-1] == 'pth':
        model_dir += ('/' + pth)
        break

print(f'model_path: {model_dir}')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
opt = opts(current_dir=current_dir, model_path=model_dir,
           input_path=input_path, threshold=threshold,
           match_threshold=match_threshold, use_gpu=set_use_gpu).init()
opt.output_root = output_path
print(f'current_use_gpus: {opt.gpus}')
if opt.output_root:
    mkdir_if_missing(opt.output_root)
# frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

#start to pre_track
capture = cv2.VideoCapture(input_path)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'frame_count: {frame_count}')
frame_rate = 30
tracker = JDETracker(opt, frame_rate=frame_rate)
video_name = input_file_name + '_' + time.strftime('%Y_%m_%d_%H_%M',time.localtime()) + '_.mp4'
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = frame_rate
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(opt.output_root + video_name, fourcc, fps, (width, height))
results = []
frame_id = 0
timer = Timer()
use_cuda = True
if set_use_gpu == '-1':
    use_cuda = False

while(True):
    # run tracking
    ok,frame = capture.read()
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
    results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
    fps = 1. / timer.average_time
    # save results
    #results.append((frame_id + 1, online_tlwhs, online_ids))
    #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
    online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                      fps=fps)
    frame_id += 1
    print(f'detect frame: {frame_id}')
    im = np.array(online_im)
    writer.write(im)
writer.release()
capture.release()
# save results
# write_results(result_filename, results, data_type)
# write_results_score(result_filename, results, data_type)
