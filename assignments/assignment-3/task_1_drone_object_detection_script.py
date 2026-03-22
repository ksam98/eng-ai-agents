#!/usr/bin/env python3
"""Task 1: Drone Object Detection.

This script:
1. Scans an input directory for .mp4 video files (defaults to "videos" directory).
2. Samples each video at 5 FPS (configurable) and stores them in `frames-<video_name>` folders.
3. Runs RT-DETR inference on all sampled frames to detect drones.
4. Copies each frame into `detections/` or `no-detections/` based on class match.

NOTES:
- No fine-tuning is performed.
- Pretrained COCO RT-DETR does not include a literal "drone" class. By default,
  this script treats detections of class "airplane" as drone detections.
- This script is intentionally fresh-run only. If output directories already exist,
  it exits and asks you to delete them first.

NOTES ON THE USAGE OF AI:
- AI (ChatGPT) was used in generating code snippets for how to use OpenCV to sample frames, 
  certain Path lib functionalities, and the RT-DETR API for inference. 
- I adapted and integrated these snippets into the overall script.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import RTDETR


VIDEO_EXTENSION = ".mp4"
IMAGE_EXTENSION = ".jpg"
MODEL_WEIGHTS = "rtdetr-l.pt"
DETECTION_CONFIDENCE = 0.25
DRONE_LABELS = {"airplane"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "videos_dir",
        nargs="?",
        type=Path,
        default=Path("videos"),
        help="Path to a directory containing input .mp4 video files (default: videos).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Sampling rate for frame extraction (default: 5).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device override (e.g., cpu, cuda:0, mps).",
    )
    return parser.parse_args()


def list_videos(videos_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() == VIDEO_EXTENSION
    )


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() == IMAGE_EXTENSION
    )


def ensure_fresh_output_directories(
    output_dir: Path,
    videos: list[Path],
) -> tuple[list[Path], Path, Path]:
    frames_dirs = [output_dir / f"frames-{video_path.stem}" for video_path in videos]
    detections_dir = output_dir / "detections"
    no_detections_dir = output_dir / "no-detections"

    existing_dirs = [path for path in frames_dirs if path.exists()]
    if detections_dir.exists():
        existing_dirs.append(detections_dir)
    if no_detections_dir.exists():
        existing_dirs.append(no_detections_dir)

    if existing_dirs:
        existing_display = "\n".join(f"- {path}" for path in sorted(existing_dirs))
        raise SystemExit(
            "Issue: output directories already exist. Delete these directories and rerun the script:\n"
            f"{existing_display}"
        )

    return frames_dirs, detections_dir, no_detections_dir


def sample_frames_from_video(video_path: Path, frames_dir: Path, target_fps: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    if target_fps <= 0:
        cap.release()
        raise ValueError("--fps must be > 0.")

    frame_interval_seconds = 1.0 / target_fps
    next_sample_time = 0.0
    frame_idx = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms > 0:
            timestamp_seconds = timestamp_ms / 1000.0
        else:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if source_fps > 0:
                timestamp_seconds = frame_idx / source_fps
            else:
                timestamp_seconds = frame_idx * frame_interval_seconds

        if timestamp_seconds + 1e-9 >= next_sample_time:
            saved_count += 1
            out_path = frames_dir / f"frame_{saved_count:06d}{IMAGE_EXTENSION}"
            cv2.imwrite(str(out_path), frame)
            next_sample_time += frame_interval_seconds

        frame_idx += 1

    cap.release()
    return saved_count


def frame_has_drone_detection(result, drone_labels: set[str]) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False

    names = result.names
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_label = str(names[class_id]).strip().lower()
        if class_label in drone_labels:
            return True
    return False


def process_frames_with_detector(
    model: RTDETR,
    frames_dirs: list[Path],
    detections_dir: Path,
    no_detections_dir: Path,
    conf: float,
    drone_labels: set[str],
    device: str | None,
) -> tuple[int, int, int]:
    all_frames: list[Path] = []
    for frames_dir in frames_dirs:
        all_frames.extend(list_images(frames_dir))

    processed = 0
    detection_count = 0
    no_detection_count = 0

    for frame_path in tqdm(all_frames, desc="Running RT-DETR", unit="frame"):
        results = model.predict(
            source=str(frame_path),
            conf=conf,
            verbose=False,
            device=device,
        )

        has_drone = any(
            frame_has_drone_detection(result, drone_labels)
            for result in results
        )

        target_dir = detections_dir if has_drone else no_detections_dir
        target_name = f"{frame_path.parent.name}__{frame_path.name}"
        shutil.copy2(frame_path, target_dir / target_name)

        processed += 1
        if has_drone:
            detection_count += 1
        else:
            no_detection_count += 1

    return processed, detection_count, no_detection_count


def main() -> None:
    args = parse_args()

    videos_dir = args.videos_dir.expanduser().resolve()
    if not videos_dir.exists() or not videos_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {videos_dir}")

    output_dir = Path.cwd().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_dir)
    if not videos:
        raise FileNotFoundError(f"No {VIDEO_EXTENSION} video files found in: {videos_dir}")

    frames_dirs, detections_dir, no_detections_dir = ensure_fresh_output_directories(output_dir, videos)

    print(f"Input videos directory: {videos_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Videos found: {len(videos)}")
    print(f"Sampling rate: {args.fps} FPS")

    for frames_dir in frames_dirs:
        frames_dir.mkdir(parents=True, exist_ok=False)
    detections_dir.mkdir(parents=True, exist_ok=False)
    no_detections_dir.mkdir(parents=True, exist_ok=False)

    for video_path, frames_dir in zip(videos, frames_dirs):
        saved = sample_frames_from_video(video_path, frames_dir, args.fps)
        print(f"Extracted {saved} frames from {video_path.name} -> {frames_dir.name}")

    print(f"Loading RT-DETR model: {MODEL_WEIGHTS}")
    model = RTDETR(MODEL_WEIGHTS)

    drone_labels = DRONE_LABELS
    print(f"Drone label mapping: {sorted(drone_labels)}")
    print(f"Frame directories to process: {len(frames_dirs)}")

    processed, detection_count, no_detection_count = process_frames_with_detector(
        model=model,
        frames_dirs=frames_dirs,
        detections_dir=detections_dir,
        no_detections_dir=no_detections_dir,
        conf=DETECTION_CONFIDENCE,
        drone_labels=drone_labels,
        device=args.device,
    )

    print("\nDone.")
    print(f"Total frames processed: {processed}")
    print(f"Frames with drone detections: {detection_count}")
    print(f"Frames with no drone detections: {no_detection_count}")
    print(f"Detections directory: {detections_dir}")
    print(f"No-detections directory: {no_detections_dir}")


if __name__ == "__main__":
    main()
