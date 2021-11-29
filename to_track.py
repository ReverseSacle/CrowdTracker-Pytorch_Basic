from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import torch
import numpy as np
import os
import logging
import src._init_paths

from src.lib.opts import opts
from src.lib.tracking_utils.utils import mkdir_if_missing
from src.lib.tracking_utils.log import logger
from src.lib.tracker.multitracker import JDETracker
from src.lib.tracking_utils.timer import Timer
from src.lib.tracking_utils import visualization as vis
from src.lib.datasets.dataset.jde import letterbox

logger.setLevel(logging.INFO)

#set params
set_current_dir = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')
set_input_path =  set_current_dir + '/video/MOT16-03.mp4'
set_input_file_name = (set_input_path.split('/')[-1]).split('.')[0]
set_output_path = set_current_dir + './video_result/'
set_model_dir = set_current_dir + '/models'
set_model_path = None
set_threshold = 0.4

#set '-1' to use CPU,set from '0','1' to use the first or second GPU
#CPU设为'-1',GPU设置例子('0'为第一个GPU,'1'为第二个GPU,以此类推)
set_use_gpu = '0'
print('input_path: ' + set_input_path)

for pth in os.listdir(set_model_dir):
    if pth.split('.')[-1] == 'pth':
        set_model_path = set_model_dir + '/' + pth
        break

print('model_path: ' + set_model_path)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
opt = opts(current_dir=set_current_dir,
           model_path=set_model_path,
           input_path=set_input_path,
           threshold=set_threshold,
           use_gpu=set_use_gpu).init()
opt.output_path = set_output_path
print('current_use_gpus: ')
print(opt.gpus)
if opt.output_path:
    mkdir_if_missing(opt.output_path)
# frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

#start to pre_track
capture = cv2.VideoCapture(set_input_path)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print('frame_count', frame_count)
frame_rate = 30
tracker = JDETracker(opt, frame_rate=frame_rate)
video_name = set_input_file_name + '.mp4'
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = frame_rate
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(opt.output_path + video_name, fourcc, fps, (width, height))
results = []
frame_id = 0
timer = Timer()
use_cuda = True
if set_use_gpu == '-1':
    use_cuda = False

while(True):
    # run tracking
    ret,img0 = capture.read()
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
    results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
    fps = 1. / timer.average_time
    # save results
    #results.append((frame_id + 1, online_tlwhs, online_ids))
    #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
    online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                      fps=fps)
    frame_id += 1
    print('detect frame:%d' % (frame_id))
    im = np.array(online_im)
    writer.write(im)
writer.release()
capture.release()
# save results
# write_results(result_filename, results, data_type)
# write_results_score(result_filename, results, data_type)
