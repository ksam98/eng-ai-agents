# Assignment 2: Image-to-Video Semantic Retrieval

This assignment builds a semantic retrieval pipeline for car exterior components:

1. fine-tune a part detector,
2. run it over sampled video frames,
3. answer query-image searches by returning matching time intervals.

## 1) Data Preparation

The video corpus is the YouTube video `YcvECxtXoxQ`, downloaded and then sampled every 5 seconds.
[commands are as given in the assignment]
```bash
# download video
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
  -o "input_video.mp4" \
  "https://www.youtube.com/watch?v=YcvECxtXoxQ"

# sample one frame every 5 seconds
mkdir -p frames
ffmpeg -i input_video.mp4 -vf "fps=1/5" frames/frame_%04d.jpg
```

This creates the `frames/` folder used by Part B.

## 2) Script Overview (A -> B -> C)

### `part_a_finetune_seg_model.py`
- Fine-tunes the segmentation model and saves `models/part_a_best.pt`.
- For the final 50-epoch run, training was done on a Lightning AI `A100` instance.
- The script is commented and includes the training configuration used.

### `part_b_retrieval_corpus.py`
- Loads `models/part_a_best.pt`.
- Runs detection on all `frames/frame_*.jpg`.
- Writes one detection per row to parquet (video corpus index):
  - `outputs/video_detections.parquet` (20-epoch model output)
  - `outputs/video_detections_50_epochs.parquet` (50-epoch model output)

### `part_c_semantic_search.py`
- Loads query images from `aegean-ai/rav4-exterior-images`.
- Detects query labels using the same model.
- Matches query labels to corpus detections by `class_label`.
- Merges nearby timestamps into contiguous intervals (5-second gap rule).
- Writes retrieval results to `outputs/query_retrieval_results.parquet`.

## 3) Auto-Generated Folders

- `runs/`: training artifacts and checkpoints from Ultralytics (`results`, `weights`, logs).
- `models/`: stable copied checkpoints (e.g., `part_a_best.pt`).
  - `part_a_best.pt` is the model weights from a 20 epoch run
  - `part_a_best_50_epochs.pt` is the model weights from a 50 epoch run
- `outputs/`: generated parquet outputs for corpus detections and retrieval results.

## 4) Output Files and Retrieval Logic

### Detection corpus
- `video_detections_50_epochs.parquet` is the main corpus artifact.
- Each row is a single detection with:
  - temporal index (`frame_index`, `timestamp_sec`),
  - semantic label (`class_label`),
  - confidence and bounding box (`x_min, y_min, x_max, y_max` and `bounding_box`).

### Semantic retrieval output
- `query_retrieval_results.parquet` contains query-to-video matches.
- For each query, output rows include:
  - `class_label_used`
  - `start_timestamp`
  - `end_timestamp`
  - `number_of_supporting_detections`

High-level logic:
- detect labels in query image,
- find matching class detections in corpus parquet,
- merge consecutive timestamps into intervals,
- return interval(s) supported by detections.
