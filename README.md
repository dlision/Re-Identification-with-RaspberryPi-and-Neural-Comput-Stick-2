# Re-Identification with RaspberryPi and Neural Compute Stick 2\* 

In this project, we will use two models person detection for retial and re-identification for retail. These models are availble in the Intel OpenVINO model zoo.
The person detection model for the Retail scenario is based on MobileNetV2-like backbone that includes depth-wise convolutions to reduce the amount of computation for the 3x3 convolution block. The single SSD head from 1/16 scale feature map has 12 clustered prior boxes.
This is us person reidentification model uses a whole body image as an input and outputs an embedding vector to match a pair of images by the Cosine distance. The model is based on RMNet backbone that was developed for fast inference. A single reidentification head from the 1/16 scale feature map outputs the embedding vector of 256 floats.

The models are listed in the file [model.lst](https://github.com/dlision/Re-Identification-with-RaspberryPi-and-Neural-Comput-Stick-2/blob/master/models.lst)
## Re-Identification model output
![output_gif](https://github.com/dlision/Re-Identification-with-RaspberryPi-and-Neural-Comput-Stick-2/blob/master/out.gif)

## Benchmarks
Include the benchmark results of running multiple model precisions. 
 The CPU used was **Intel® Core™ i3-8350K CPU @ 4.00GHz × 4** and **16 GB Ram**
 
### For Detection model

| Properties  | CPU         | NCS with CPU |NCS2 with Raspberry |
| ------------| ----------- | ------------ | -----------------  |
|Model Loading| 0.180       | 2.149        | -                  |
|Infer Time   | 0.021       | 0.033        | -                  |
|FPS          | 26.466      | 8.449        | -                  |

### For Re-Idendification model

| Properties  | CPU         | NCS with CPU |NCS2 with Raspberry |
| ------------| ----------- | ------------ | -----------------  |
|Model Loading| 0.277       | 2.203        | -                  |
|Infer Time   | 0.019       | 0.115        | -                  |
|FPS          | 32.54       | 5.971        | -                  |

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

Such file with detections can be saved from the code. Specify the argument `--save_detections` with path to an output file or by default saved in detections_record.json.

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
