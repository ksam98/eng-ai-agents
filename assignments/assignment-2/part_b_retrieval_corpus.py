#!/usr/bin/env python3
"""Build a video detection corpus (Parquet) from sampled frames."""

from pathlib import Path

import pandas as pd
from ultralytics import YOLO


ASSIGNMENT_DIR = Path(__file__).resolve().parent
FRAMES_DIR = ASSIGNMENT_DIR / "frames"
MODEL_WEIGHTS = ASSIGNMENT_DIR / "models" / "part_a_best_50_epochs.pt"
OUTPUT_PARQUET = ASSIGNMENT_DIR / "outputs" / "video_detections_50_epochs.parquet"

VIDEO_ID = "YcvECxtXoxQ" # From https://www.youtube.com/watch?v=YcvECxtXoxQ&t=1s
FRAME_STRIDE_SECONDS = 5
CONFIDENCE_THRESHOLD = 0.25

def detect_frame(model: YOLO, frame_path: Path) -> list[dict]:
    # Expected format: frame_0001.jpg
    frame_index = int(frame_path.stem.split("_")[1])

    # frame_0001 corresponds to 0 seconds for 5-second sampling.
    timestamp_sec = (frame_index - 1) * FRAME_STRIDE_SECONDS

    rows: list[dict] = []
    results = model.predict(
        source=str(frame_path),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_label = str(result.names[class_id])
            confidence_score = float(box.conf.item())
            x_min, y_min, x_max, y_max = [float(v) for v in box.xyxy[0].tolist()]

            rows.append(
                {
                    "video_id": VIDEO_ID,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "class_id": class_id,
                    "class_label": class_label,
                    "confidence_score": confidence_score,
                    "bounding_box": [x_min, y_min, x_max, y_max],
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "frame_file": frame_path.name,
                    "detector_name": MODEL_WEIGHTS.name,
                }
            )

    return rows


def main() -> None:
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_WEIGHTS}\n"
            "Run part_a_finetune_seg_model.py first."
        )

    frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in: {FRAMES_DIR}")

    print(f"Using model: {MODEL_WEIGHTS}")
    print(f"Frames to process: {len(frame_files)}")

    model = YOLO(str(MODEL_WEIGHTS))
    all_rows: list[dict] = []

    for i, frame_path in enumerate(frame_files, start=1):
        all_rows.extend(detect_frame(model, frame_path))
        if i % 50 == 0 or i == len(frame_files):
            print(f"Processed {i}/{len(frame_files)} frames")

    output_dir = OUTPUT_PARQUET.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Detections saved: {len(df)}")
    print(f"Output file: {OUTPUT_PARQUET}")

    if len(df) > 0:
        print("\nTop detected classes:")
        print(df["class_label"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()


# NOTE: on the use of AI
# I wrote the initial code for processing one frame based on https://docs.ultralytics.com/datasets/segment/carparts-seg/#what-is-the-carparts-segmentation-dataset
# and used an AI assistant to help with writing the loop structure for processing results from model.predict
# (I do understand all code and underlying logic intimately). Additionally I asked the AI assistant to help
# adapt my code for processing one frame to code that can process all frames (again I do understand all code
# written and then underlying logic intimately).