#!/usr/bin/env python3
"""Run image-to-video semantic retrieval using detector labels."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from ultralytics import YOLO


ASSIGNMENT_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS = ASSIGNMENT_DIR / "models" / "part_a_best_50_epochs.pt"
DETECTIONS_PARQUET = ASSIGNMENT_DIR / "outputs" / "video_detections_50_epochs.parquet"
OUTPUT_PARQUET = ASSIGNMENT_DIR / "outputs" / "query_retrieval_results.parquet"

FRAME_STRIDE_SECONDS = 5
QUERY_CONFIDENCE_THRESHOLD = 0.25
QUERY_DATASET_ID = "aegean-ai/rav4-exterior-images"


def merge_timestamps_into_intervals(timestamps: list[int], max_gap_seconds: int) -> list[tuple[int, int]]:
    if not timestamps:
        return []

    intervals: list[tuple[int, int]] = []
    start = timestamps[0]
    end = timestamps[0]

    for ts in timestamps[1:]:
        if ts - end <= max_gap_seconds:
            end = ts
        else:
            intervals.append((start, end))
            start = ts
            end = ts

    intervals.append((start, end))
    return intervals


def build_class_interval_index(detections_df: pd.DataFrame) -> dict[str, list[dict]]:
    class_interval_index: dict[str, list[dict]] = {}

    for class_label, class_df in detections_df.groupby("class_label"):
        timestamps = sorted(class_df["timestamp_sec"].astype(int).unique().tolist())
        intervals = merge_timestamps_into_intervals(
            timestamps=timestamps,
            max_gap_seconds=FRAME_STRIDE_SECONDS,
        )

        interval_rows: list[dict] = []
        for start_ts, end_ts in intervals:
            support_count = int(
                (
                    (class_df["timestamp_sec"] >= start_ts)
                    & (class_df["timestamp_sec"] <= end_ts)
                ).sum()
            )
            interval_rows.append(
                {
                    "start_timestamp": int(start_ts),
                    "end_timestamp": int(end_ts),
                    "number_of_supporting_detections": support_count,
                }
            )

        class_interval_index[str(class_label)] = interval_rows
        #TODO: remove later
        print(f"Class '{class_label}': {len(timestamps)} timestamps merged into {len(intervals)} intervals")

    return class_interval_index


def detect_query_labels(model: YOLO, image) -> set[str]:
    labels: set[str] = set()
    results = model.predict(source=image, conf=QUERY_CONFIDENCE_THRESHOLD, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            labels.add(str(result.names[class_id]))

    return labels


def main() -> None:
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_WEIGHTS}\n"
            "Run part_a_finetune_seg_model.py first."
        )
    if not DETECTIONS_PARQUET.exists():
        raise FileNotFoundError(
            f"Detections parquet not found: {DETECTIONS_PARQUET}\n"
            "Run part_b_retrieval_corpus.py first."
        )

    print(f"Loading model: {MODEL_WEIGHTS}")
    model = YOLO(str(MODEL_WEIGHTS))

    print(f"Loading detections: {DETECTIONS_PARQUET}")
    detections_df = pd.read_parquet(DETECTIONS_PARQUET)
    class_interval_index = build_class_interval_index(detections_df)

    print(f"Loading query dataset: {QUERY_DATASET_ID}")
    query_ds = load_dataset(QUERY_DATASET_ID, split="train")
    print(f"Queries to process: {len(query_ds)}")

    retrieval_rows: list[dict] = []

    for query_index, item in enumerate(query_ds):
        query_timestamp_sec = int(item["timestamp_sec"])
        query_labels = detect_query_labels(model=model, image=item["image"])

        if not query_labels:
            retrieval_rows.append(
                {
                    "query_index": query_index,
                    "query_timestamp_sec": query_timestamp_sec,
                    "class_label_used": None,
                    "start_timestamp": None,
                    "end_timestamp": None,
                    "number_of_supporting_detections": 0,
                }
            )
            continue

        for class_label in sorted(query_labels):
            intervals = class_interval_index.get(class_label, [])
            if not intervals:
                retrieval_rows.append(
                    {
                        "query_index": query_index,
                        "query_timestamp_sec": query_timestamp_sec,
                        "class_label_used": class_label,
                        "start_timestamp": None,
                        "end_timestamp": None,
                        "number_of_supporting_detections": 0,
                    }
                )
                continue

            for interval in intervals:
                retrieval_rows.append(
                    {
                        "query_index": query_index,
                        "query_timestamp_sec": query_timestamp_sec,
                        "class_label_used": class_label,
                        "start_timestamp": interval["start_timestamp"],
                        "end_timestamp": interval["end_timestamp"],
                        "number_of_supporting_detections": interval[
                            "number_of_supporting_detections"
                        ],
                    }
                )

        if (query_index + 1) % 20 == 0 or (query_index + 1) == len(query_ds):
            print(f"Processed {query_index + 1}/{len(query_ds)} queries")

    output_dir = OUTPUT_PARQUET.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Saved retrieval output: {OUTPUT_PARQUET}")
    print(f"Rows written: {len(retrieval_df)}")
    print(f"Unique queries covered: {retrieval_df['query_index'].nunique()}")


if __name__ == "__main__":
    main()


# NOTE: on the use of AI
# I wrote the initial code for processing one query image. I then used an AI assistant to 
# help with writing the loop structure for processing results from detect_query_labels and 
# then writing results to retrieval_rows (I do understand all code and underlying logic intimately)