#!/usr/bin/env python3

"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import json
import argparse
import time
import queue
from threading import Thread
import json
import logging as log
import os
import random
import sys
import pandas as pd
import cv2 as cv

from utils.network_wrappers import VectorCNN, DetectionsFromFileReader
from mc_tracker.mct import MultiCameraTracker
from utils.analyzer import save_embeddings
from utils.misc import read_py_config, check_pressed_keys, AverageEstimator, set_log_config
from utils.video import MulticamCapture, NormalizerCLAHE
from utils.visualization import visualize_multicam_detections, get_target_size
from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'common/python'))
import monitors


set_log_config()

def save_json_file(save_path, data, description=''):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)
    if description:
        log.info('{} saved to {}'.format(description, save_path))


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
                continue
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


def run(params, config, capture,  reid):
    win_name = 'Multi camera tracking'
    '''
    Here we read the JSON file stored in the real-time mode
    Load the JSON file into the data object
    '''
    f = open(params.detections)
    data = json.load(f)
    
    frame_number = 0
    avg_latency = AverageEstimator()
    output_detections = [[] for _ in range(capture.get_num_sources())]
    
    key = -1

    if config['normalizer_config']['enabled']:
        capture.add_transform(
            NormalizerCLAHE(
                config['normalizer_config']['clip_limit'],
                config['normalizer_config']['tile_size'],
            )
        )

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, config['sct_config'], **config['mct_config'],
                                 visual_analyze=config['analyzer'])
    
    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    frames_read = False
    set_output_params = False

    prev_frames = thread_body.frames_queue.get()
    '''
    In the off-line mode we are not using detection model
    we comment all the detection model inferences in the off-line mode
    and use the the detections from a json file
    '''
    
    presenter = monitors.Presenter(params.utilization_monitors, 0)
    
    while thread_body.process:
        all_detections = [[] for _ in range(capture.get_num_sources())]
        if not params.no_show:
            key = check_pressed_keys(key)
            if key == 27:
                break
            presenter.handleKey(key)
        start = time.perf_counter()
        try:
            frames = thread_body.frames_queue.get_nowait()
            frames_read = True
        except queue.Empty:
            frames = None

        if frames is None:
            continue
        
        '''
        Here, we get the bounding boxes of the objects starting from frame 0.
        We save these boxes to 'all_detections' list to create masks.
        We pass this list to the ReID model to track and re-identify.
        '''
        for i in range(capture.get_num_sources()):
            for j in data[i][frame_number].get('boxes'):
                all_detections[i].append(j)
               
       
        frame_number += 1
        
        
        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(list(all_detections)):
            '''
            Here, we have ommitted the code which separated detections and confidence scores
            since we already separated values in a dictionary.
            '''
           
            all_masks[i] = [det[2] for det in detections if len(det) == 3]
        
        tracker.process(prev_frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()
        print("here")
        if frame_number == (int(len(data[0])-1)):
            break
        latency = max(time.perf_counter() - start, sys.float_info.epsilon)
        avg_latency.update(latency)
        fps = round(1. / latency, 1)
        '''
        We pass a string 'offline' as a flag to visualize the re-identified objects in offline mode.
        '''
        vis = visualize_multicam_detections(prev_frames, 'off-line', tracked_objects, fps, **config['visualization_config'])
        presenter.drawGraphs(vis)
        if not params.no_show:
            cv.imshow(win_name, vis)

        if frames_read and not set_output_params:
            set_output_params = True
            if len(params.output_video):
                frame_size = [frame.shape[::-1] for frame in frames]
                fps = capture.get_fps()
                target_width, target_height = get_target_size(frame_size, None, **config['visualization_config'])
                video_output_size = (target_width, target_height)
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                output_video = cv.VideoWriter(params.output_video, fourcc, min(fps), video_output_size)
            else:
                output_video = None
        if set_output_params and output_video:
            output_video.write(cv.resize(vis, video_output_size))

        print('\rProcessing frame: {}, fps = {} (avg_fps = {:.3})'.format(
                            frame_number, fps, 1. / avg_latency.get()), end="")
        prev_frames, frames = frames, prev_frames
    print(presenter.reportMeans())
    print('')

    thread_body.process = False
    frames_thread.join()

    if len(params.history_file):
        save_json_file(params.history_file, tracker.get_all_tracks_history(), description='History file')
   

    if len(config['embeddings']['save_path']):
        save_embeddings(tracker.scts, **config['embeddings'])


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    """Prepares data for the object tracking demo"""
    parser = argparse.ArgumentParser(description='Multi camera multi object \
                                                  tracking live demo script')
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input sources (indexes of cameras or paths to video files)')
    parser.add_argument('--loop', default=False, action='store_true',
                        help='Optional. Enable reading the input in a loop')
    parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'configs/person.py'), required=False,
                        help='Configuration file')

    parser.add_argument('--detections', type=str, default='Detections_record.json',
                        help='JSON file with bounding boxes')
   
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the object detection model')

    parser.add_argument('--m_reid', type=str, required=True,
                        help='Path to the object re-identification model')

    parser.add_argument('--output_video', type=str, default='', required=False,
                        help='Optional. Path to output video')
    parser.add_argument('--history_file', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save results of the demo')
   
    parser.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                              path to a shared library with the kernels impl.',
                             type=str, default=None)
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially.')

    args = parser.parse_args()
    ''' 
    Removed detection related checkers.
    '''
    if len(args.config):
        log.info('Reading configuration file {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        log.error('No configuration file specified. Please specify parameter \'--config\'')
        sys.exit(1)

    random.seed(config['random_seed'])
    capture = MulticamCapture(args.input, args.loop)
    
    log.info("Creating Inference Engine")
    ie = IECore()


    if args.m_reid:
        object_recognizer = VectorCNN(ie, args.m_reid, args.device, args.cpu_extension)
    else:
        object_recognizer = None
        log.error('ReID model is expected but got {}, please specify ReID model'
                  .format(object_recognizer))

    run(args, config, capture,  object_recognizer)
    log.info('Demo finished successfully')


if __name__ == '__main__':
    main()
