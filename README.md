# Re-Identification with RaspberryPi and Neural Compute Stick 2\* 

This readme is for the 'decoupled'/'after hours'/'offline' ReID version of Multi Camera Multi Target demo of OpenVINO.

## How it works
The project workflow is the following:

1. The application reads tuples of frames from cameras/videos one by one. For each frame in tuple it runs object detector model listed in the models_list and save the detections into a file. The file formate is given below 

```json
[
    [  # Source#0
        {
            "frame_id": 0,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],  # N bounding boxes
            "scores": [score0, score1, ...],  # N scores
        },
        {
            "frame_id": 1,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],
            "scores": [score0, score1, ...],
        },
        ...
    ],
    [  # Source#1
        {
            "frame_id": 0,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],  # N bounding boxes
            "scores": [score0, score1, ...],  # N scores
        },
        {
            "frame_id": 1,
            "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], ...],
            "scores": [score0, score1, ...],
        },
        ...
    ],
    ...
]
```
The detection_only model will also save the recordings of the camera videos for later use in the ReID model.
2. Then in the second part ReID model reads the saved detections and videos from the detections model and run the ReID model to get the cosine descriptor for re-identification of person.
2. All embeddings are passed to tracker which assigns an ID to each object.
3. The project visualizes the resulting bounding boxes and unique object IDs assigned during tracking and save the tracks history into a JSON file.

## Running

### Installation of dependencies

To install required dependencies run

```bash
pip3 install -r requirements.txt
```

### Command line arguments

#### For Detection model
Run the application with the `-h` option to see the following usage message:

```
usage: Detection_only_model.py [-h] -i INPUT [INPUT ...]
                                                  [--loop] [--config CONFIG]
                                                  [--detections DETECTIONS]
                                                  [--m_detector M_DETECTOR]
                                                  [--t_detector T_DETECTOR] 
                                                  [--output_video OUTPUT_VIDEO]
                                                  [--save_detections SAVE_DETECTIONS]
                                                  [--no_show] [-d DEVICE]
                                                  [-l CPU_EXTENSION]
                                                  [-u UTILIZATION_MONITORS]


```

```
# videos
python Detections_only_model.py \
    -i <path_to_video>/video_1.avi <path_to_video>/video_2.avi \
    --m_detector <path_to_model>/person-detection-retail-0013.xml \
    --config configs/person.py

# web-cameras
python multi_camera_multi_person_tracking.py \
    -i 0 1 \
    --m_detector <path_to_model>/person-detection-retail-0013.xml \
    --config configs/person.py
```

# Such file with detections can be saved from the code. Specify the argument `--save_detections` with path to an output file or by default saved in detections_record.json.

#### For Re-identification model

```
usage: offline_ReID_only.py [-h] -i INPUT [INPUT ...]
                                                  [--loop] [--config CONFIG]
                                                  [--detections DETECTIONS]
                                                  [--t_detector T_DETECTOR] 
                                                  [--m_reid M_REID]
                                                  [--output_video OUTPUT_VIDEO]
                                                  [--history_file HISTORY_FILE]
                                                  [--no_show] [-d DEVICE]
                                                  [-l CPU_EXTENSION]
                                                  [-u UTILIZATION_MONITORS]


```

```
# videos
python offline_ReID_only.py \
    -i <path_to_video>/video_1.avi <path_to_video>/video_2.avi \
    --m_reid <path_to_model>/person-reidentification-retail-0103.xml \
    --config configs/person.py

# web-cameras
python multi_camera_multi_person_tracking.py \
    -i 0 1 \
    --m_reid <path_to_model>/person-reidentification-retail-0103.xml \
    --config configs/person.py
```

The saved detections can be passed from the code as specify the argument `--detections` with path to detections file or by default read from detections_record.json.
