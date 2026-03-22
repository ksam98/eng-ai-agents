#!/usr/bin/env python3
"""Task 2: Kalman Filter Tracking.

This script:
1. Scans an input directory for `.mp4` video files (configured via `VIDEOS_DIR`).
2. Samples each video at a fixed FPS (`SAMPLE_FPS`).
3. Runs RT-DETR inference on sampled frames to obtain drone detections.
4. Initializes a Kalman filter from the first valid detection and tracks the drone center over time.
5. For each sampled frame, performs Kalman predict/update; if detections are temporarily missing,
   it continues with prediction for up to `MAX_MISSED_FRAMES`.
6. Produces one output video per input video in `OUTPUT_DIR`, containing only track-active frames with:
   - detector bounding box overlay
   - 2D trajectory polyline overlay

NOTES:
- Uses `filterpy` with state vector `[x, y, vx, vy]` and measurement `[x, y]`.
- Uses a pretrained RT-DETR model (`MODEL_WEIGHTS`) with no fine-tuning.
- Pretrained COCO RT-DETR does not include a literal "drone" class; by default,
  this script treats class `"airplane"` as drone via `DRONE_LABELS`.
- Main runtime settings are static constants at the top of the file to reduce complexity.
- ffmpeg re-encoding is attempted when enabled (`USE_FFMPEG`).

NOTES ON THE USAGE OF AI:
- AI (ChatGPT) was used for guidance on Kalman filter setup in `filterpy`,
  RT-DETR/OpenCV integration details, trajectory overlay implementation patterns,
  and FFMPEG integration.
- I adapted, validated, and integrated those ideas into this script’s final structure.
"""


from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
from ultralytics import RTDETR


# Static configuration
VIDEOS_DIR = Path("videos")
OUTPUT_DIR = Path("tracking_outputs")
SAMPLE_FPS = 5.0
MODEL_WEIGHTS = "rtdetr-l.pt"
CONFIDENCE_THRESHOLD = 0.25
DRONE_LABELS = {"airplane"}
MAX_MISSED_FRAMES = 8
PROCESS_NOISE = 1.0
MEASUREMENT_NOISE = 10.0

# Optional runtime toggles
ANY_DETECTION_IS_DRONE = False
DETECTOR_DEVICE = None
USE_FFMPEG = True
FFMPEG_PATH = "ffmpeg"

VIDEO_EXTENSION = ".mp4"


def list_videos(videos_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() == VIDEO_EXTENSION
    )


def build_kalman_filter(dt: float, x: float, y: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State: [x, y, vx, vy]
    kf.F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Measurement: [x, y]
    kf.H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    # Constant-velocity process noise model.
    kf.Q = PROCESS_NOISE * np.array(
        [
            [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
            [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
            [0.5 * dt3, 0.0, dt2, 0.0],
            [0.0, 0.5 * dt3, 0.0, dt2],
        ],
        dtype=float,
    )

    kf.R = np.eye(2, dtype=float) * MEASUREMENT_NOISE
    kf.P = np.diag([100.0, 100.0, 1000.0, 1000.0])
    kf.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)

    return kf


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    xi1 = int(max(0, min(width - 1, x1)))
    yi1 = int(max(0, min(height - 1, y1)))
    xi2 = int(max(0, min(width - 1, x2)))
    yi2 = int(max(0, min(height - 1, y2)))

    if xi2 < xi1:
        xi1, xi2 = xi2, xi1
    if yi2 < yi1:
        yi1, yi2 = yi2, yi1

    return xi1, yi1, xi2, yi2


def bbox_from_center(center_x: float, center_y: float, box_w: float, box_h: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = center_x - box_w / 2.0
    y1 = center_y - box_h / 2.0
    x2 = center_x + box_w / 2.0
    y2 = center_y + box_h / 2.0
    return clamp_bbox(x1, y1, x2, y2, width, height)


def select_drone_detection(result) -> tuple[tuple[int, int, int, int], float] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best: tuple[tuple[int, int, int, int], float] | None = None
    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = str(names[cls_id]).strip().lower()
        if not ANY_DETECTION_IS_DRONE and label not in DRONE_LABELS:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        det = ((int(x1), int(y1), int(x2), int(y2)), conf)

        if best is None or conf > best[1]:
            best = det

    return best


def maybe_reencode_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
    if shutil.which(FFMPEG_PATH) is None:
        return False

    command = [
        FFMPEG_PATH,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def track_single_video(video_path: Path, model: RTDETR, output_dir: Path) -> tuple[int, int, int, Path | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    interval = 1.0 / SAMPLE_FPS
    next_sample_time = 0.0
    frame_idx = 0

    temp_output = output_dir / f"{video_path.stem}_tracked_tmp.mp4"
    final_output = output_dir / f"{video_path.stem}_tracked.mp4"

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        SAMPLE_FPS,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {temp_output}")

    kf: KalmanFilter | None = None
    missed = 0
    last_box_w = 80.0
    last_box_h = 80.0
    trajectory: list[tuple[int, int]] = []

    sampled_frames = 0
    written_frames = 0
    detection_updates = 0

    progress = tqdm(desc=f"Tracking {video_path.name}", unit="sampled_frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms > 0:
            timestamp_seconds = timestamp_ms / 1000.0
        else:
            timestamp_seconds = frame_idx / source_fps

        frame_idx += 1

        if timestamp_seconds + 1e-9 < next_sample_time:
            continue

        next_sample_time += interval
        sampled_frames += 1
        progress.update(1)

        result = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device=DETECTOR_DEVICE,
        )[0]

        detection = select_drone_detection(result)

        draw_bbox: tuple[int, int, int, int] | None = None
        draw_center: tuple[int, int] | None = None
        status_text = ""


        if kf is None:
            # Kalman Filter initialization on first detection.
            if detection is None:
                continue

            (x1, y1, x2, y2), det_conf = detection
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            last_box_w = max(2.0, float(x2 - x1))
            last_box_h = max(2.0, float(y2 - y1))

            kf = build_kalman_filter(dt=interval, x=cx, y=cy)

            missed = 0
            draw_bbox = clamp_bbox(x1, y1, x2, y2, width, height)
            draw_center = (int(cx), int(cy))
            trajectory.append(draw_center)
            detection_updates += 1
            status_text = f"DETECT conf={det_conf:.2f}"
        else:
            # Kalman Filter prediction and update.
            kf.predict()
            pred_x = float(kf.x[0, 0])
            pred_y = float(kf.x[1, 0])

            if detection is not None:
                (x1, y1, x2, y2), det_conf = detection
                meas_x = (x1 + x2) / 2.0
                meas_y = (y1 + y2) / 2.0
                z = np.array([meas_x, meas_y], dtype=float)
                kf.update(z)

                upd_x = float(kf.x[0, 0])
                upd_y = float(kf.x[1, 0])

                last_box_w = max(2.0, float(x2 - x1))
                last_box_h = max(2.0, float(y2 - y1))

                draw_bbox = clamp_bbox(x1, y1, x2, y2, width, height)
                draw_center = (int(upd_x), int(upd_y))
                trajectory.append(draw_center)
                missed = 0
                detection_updates += 1
                status_text = f"DETECT conf={det_conf:.2f}"
            else:
                # Kalman Filter missed detection handling.
                missed += 1
                if missed > MAX_MISSED_FRAMES:
                    kf = None
                    continue

                draw_center = (int(pred_x), int(pred_y))
                draw_bbox = bbox_from_center(pred_x, pred_y, last_box_w, last_box_h, width, height)
                trajectory.append(draw_center)
                status_text = f"PREDICT miss={missed}"

        if draw_bbox is None or draw_center is None:
            continue

        out_frame = frame.copy()
        x1, y1, x2, y2 = draw_bbox
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(out_frame, draw_center, 4, (0, 255, 255), -1)

        if len(trajectory) >= 2:
            traj_points = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out_frame, [traj_points], isClosed=False, color=(0, 0, 255), thickness=2)

        cv2.putText(
            out_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        writer.write(out_frame)
        written_frames += 1

    progress.close()
    cap.release()
    writer.release()

    if written_frames == 0:
        if temp_output.exists():
            temp_output.unlink()
        return sampled_frames, written_frames, detection_updates, None

    if USE_FFMPEG and maybe_reencode_with_ffmpeg(temp_output, final_output):
        temp_output.unlink(missing_ok=True)
        return sampled_frames, written_frames, detection_updates, final_output

    temp_output.replace(final_output)
    return sampled_frames, written_frames, detection_updates, final_output


def main() -> None:
    if SAMPLE_FPS <= 0:
        raise ValueError("SAMPLE_FPS must be > 0")
    if MAX_MISSED_FRAMES < 0:
        raise ValueError("MAX_MISSED_FRAMES must be >= 0")

    videos_dir = VIDEOS_DIR.expanduser().resolve()
    if not videos_dir.exists() or not videos_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {videos_dir}")

    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_dir)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in: {videos_dir}")

    print(f"Input videos directory: {videos_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Videos found: {len(videos)}")
    print(f"Sampling FPS: {SAMPLE_FPS}")
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Drone labels: {sorted(DRONE_LABELS)}")
    print(f"Max missed frames: {MAX_MISSED_FRAMES}")
    print(f"Process noise: {PROCESS_NOISE}")
    print(f"Measurement noise: {MEASUREMENT_NOISE}")

    model = RTDETR(MODEL_WEIGHTS)

    for video_path in videos:
        print(f"\nProcessing {video_path.name}...")
        sampled, written, updates, output_video = track_single_video(
            video_path=video_path,
            model=model,
            output_dir=output_dir,
        )

        print(f"Sampled frames: {sampled}")
        print(f"Frames written (track-active): {written}")
        print(f"Detection updates: {updates}")
        if output_video is None:
            print("No track-active frames were produced for this video.")
        else:
            print(f"Output video: {output_video}")


if __name__ == "__main__":
    main()
