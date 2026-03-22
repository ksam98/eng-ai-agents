# Assignment 3: UAV Drone Detection and Tracking

This assignment implements a two-stage pipeline:
1. detect drones in video frames using RT-DETR (Task 1), and
2. track drone motion over time with a Kalman filter (Task 2).

## Quick Links

- Hugging Face dataset (detections parquet): `https://huggingface.co/datasets/ksam98nyu/assignment-3-drone-detections`
- Output video 1 (YouTube): `https://youtu.be/KNFeG1ls3ek`
- Output video 2 (YouTube): `https://youtu.be/w6WIqEVSoAA`

## Test Inputs

Videos used in this run (for your run create this directory and save the
appropriate videos there):
- `videos/drone_video_1.mp4`
- `videos/drone_video_2.mp4`
Then run 
```bash
mkdir -p frames-drone_video_1
ffmpeg -i videos/drone_video_1.mp4 -vf "fps=5" frames-drone_video_1/frame_%04d.jpg
```
and 
```bash
mkdir -p frames-drone_video_2
ffmpeg -i videos/drone_video_2.mp4 -vf "fps=5" frames-drone_video_2/frame_%04d.jpg
```
`frames-drone_video_1/` will have 828 sampled frames (at 5 fps) and
`frames-drone_video_2/` will have 2581 sampled frames (at 5 fps)
## Task 1: Drone Object Detection

### Detector choice and configuration

I used **Ultralytics RT-DETR** (`rtdetr-l.pt`) without fine-tuning for this submission.

Important note: pretrained COCO RT-DETR does not contain a literal `drone` class. For this assignment run, I mapped `airplane` -> drone as a practical proxy.

Current Task 1 script settings:
- model: `rtdetr-l.pt`
- confidence threshold: `0.25`
- sampling rate: `5 FPS`
- accepted video extension: `.mp4`

### Script

- `task_1_drone_object_detection_script.py`
```bash
python task_1_drone_object_detection_script.py
```

### Task 1 outputs

- `detections/`: 1064 frames
- `no-detections/`: 2345 frames

The `no-detections/` folder is intentional. Its purpose is to expose where the detector fails, so errors are visible and diagnosable instead of silently ignored.
In this dataset, many no-detection cases occur when the drone is far from the camera and appears very small in the frame. In those moments, the target is minute and low-detail, and the pretrained detector often misses it.

## Task 2: Kalman Filter Tracking

### Tracker design

- Library: `filterpy`
- State vector: `[x, y, vx, vy]`
  - `x, y`: drone center in image pixels
  - `vx, vy`: velocity in pixels/sec
- Measurement vector: `[x, y]` from detector bounding-box center
- Motion model: constant velocity

Kalman settings used in this run:
- process noise (`Q` scale): `1.0`
- measurement noise (`R` scale): `10.0`
- max missed sampled frames before reset: `8`

### Predict / update / missed detections

For each sampled frame:
1. Predict next state using the Kalman motion model.
2. If a detection exists, update the filter with measured center.
3. If no detection exists, keep predicting for short gaps (`MAX_MISSED_FRAMES`).
4. If misses exceed threshold, reset tracker and wait for re-initialization from a new detection.

This gives smoother trajectories and keeps track continuity during brief detector dropouts.

### Script

- `task_2_kalman_filter_tracking_script.py`

From `assignments/assignment-3`:

```bash
python task_2_kalman_filter_tracking_script.py
```

### Task 2 outputs

- `tracking_outputs/drone_video_1_tracked.mp4`
- `tracking_outputs/drone_video_2_tracked.mp4`

Output videos include:
- detector bounding box overlay
- tracker center and 2D trajectory polyline
- only track-active frames (detection or short prediction window)

## Failure Cases and Observations

Main failure modes observed:
- Drone appears too small/far away -> more misses in `no-detections/`.
- Visual clutter and fast motion can produce unstable boxes.
- Because this run uses pretrained RT-DETR without drone-specific fine-tuning, small-object recall is limited.

How tracking helps:
- Kalman prediction bridges short detection gaps.
- Trajectory remains visually continuous when misses are brief.

## Notes on AI Usage

I used AI support (ChatGPT) for implementation guidance around:
- OpenCV frame sampling/writing patterns,
- RT-DETR API usage,
- Kalman filter setup in `filterpy`, and
- trajectory overlay logic.

I integrated, adapted, and validated the final code and outputs in this repository.
