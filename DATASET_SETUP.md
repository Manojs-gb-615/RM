# How2Sign Dataset Setup Guide

> **Status**: CSV annotation files are available locally. Pose `.npy` files are **NOT** available and must be extracted from raw video files using MediaPipe.

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Obtaining the Dataset](#obtaining-the-dataset)
3. [Files Available Locally](#files-available-locally)
4. [CSV Column Descriptions](#csv-column-descriptions)
5. [Required Directory Structure](#required-directory-structure)
6. [Running prepare_manifests.py](#running-prepare_manifestspy)
7. [Pose File Format](#pose-file-format)
8. [Extracting Poses from Video Using MediaPipe](#extracting-poses-from-video-using-mediapipe)
9. [Known Limitations](#known-limitations)

---

## Dataset Overview

**How2Sign** is a large-scale multimodal American Sign Language production dataset based on instructional "How-To" videos.

| Property | Value |
|----------|-------|
| Full name | How2Sign |
| Authors | Duarte et al. (2021) |
| Domain | Instructional / "How-To" videos |
| Language | American Sign Language (ASL) |
| Total sentences | ~35,191 |
| Total video clips | ~2,631 |
| Total duration | ~89 hours |
| Modalities | RGB front video, RGB side video, depth, audio, English captions |
| License | Requires registration and acceptance of terms |

**Dataset split:**

| Split | Sentences | Duration (approx.) |
|-------|-----------|-------------------|
| Train | 31,128 | ~79 hours |
| Val | 1,741 | ~4 hours |
| Test | 2,322 | ~6 hours |

---

## Obtaining the Dataset

### Step 1: Register

Go to the official dataset page and complete the registration form:

> **https://how2sign.github.io/**

You must agree to the data usage terms. The dataset is provided for non-commercial research purposes only.

### Step 2: Download CSV Annotation Files

The realigned CSV files should be downloaded from the dataset's official release. They contain start/end timestamps and English captions for each sentence.

**Files needed:**
```
how2sign_realigned_train.csv
how2sign_realigned_val.csv
how2sign_realigned_test.csv
```

### Step 3: Download Video Files

After registration, you will receive instructions to download the RGB front-view video files (`sign_rgb_front`). These are large files (hundreds of GB).

> ⚠️ **Storage requirement**: The full dataset requires significant storage. Plan accordingly before downloading.

---

## Files Available Locally

The following CSV annotation files are already on this machine:

```
c:\Users\manoj\Downloads\how2sign_realigned_train.csv
c:\Users\manoj\Downloads\how2sign_realigned_val.csv
c:\Users\manoj\Downloads\how2sign_realigned_test.csv
```

A 30-sample subset directory is also available:
```
c:\Users\manoj\Downloads\context_aware_howsign_30samples_with_csvs\ASL\data\annotations\
```

> ⚠️ **What is NOT available**: Pose `.npy` files. These are required for model training and evaluation.

---

## CSV Column Descriptions

Each realigned CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `VIDEO_ID` | string | Unique identifier for the source video clip |
| `SENTENCE_ID` | string | Unique identifier for this specific sentence |
| `START_REALISED` | float | Start time of the sentence in the video (seconds) |
| `END_REALISED` | float | End time of the sentence in the video (seconds) |
| `SENTENCE` | string | English caption / transcription of the signed sentence |

**Example row:**
```
VIDEO_ID,SENTENCE_ID,START_REALISED,END_REALISED,SENTENCE
_B4aNNi86Vk,_B4aNNi86Vk_7-8-rgb_front,7.2,8.9,"so today I am going to show you"
```

**Notes:**
- `START_REALISED` and `END_REALISED` are the realigned timestamps (more accurate than original automated timestamps).
- The `SENTENCE_ID` typically encodes the `VIDEO_ID` plus a timestamp range.
- Some sentences may appear across multiple rows if they were re-segmented.

---

## Required Directory Structure

The model training code (`src/data/how2sign_dataset.py`) expects the following directory layout:

```
<data_dir>/
├── annotations/
│   ├── how2sign_realigned_train.csv
│   ├── how2sign_realigned_val.csv
│   └── how2sign_realigned_test.csv
│
├── poses/
│   ├── train/
│   │   ├── <SENTENCE_ID>.npy      ← [T, 274] float32 pose arrays
│   │   ├── <SENTENCE_ID>.npy
│   │   └── ...
│   ├── val/
│   │   ├── <SENTENCE_ID>.npy
│   │   └── ...
│   └── test/
│       ├── <SENTENCE_ID>.npy
│       └── ...
│
└── videos/                         ← (optional, only needed for pose extraction)
    ├── train/
    │   ├── <VIDEO_ID>.mp4
    │   └── ...
    ├── val/
    └── test/
```

---

## Running prepare_manifests.py

If a `prepare_manifests.py` script is present in the project, it cross-references the CSV files with the available pose `.npy` files and generates a filtered manifest (a cleaned CSV listing only sentences for which pose data exists).

```powershell
$PYTHON = "c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\python.exe"

& $PYTHON scripts/prepare_manifests.py `
    --csv_dir c:\Users\manoj\Downloads\ `
    --pose_dir <path_to_pose_npy_dir> `
    --output_dir data/manifests/
```

The resulting `manifests/train.csv`, `manifests/val.csv`, and `manifests/test.csv` are then used by `How2SignPoseDataset`.

> ⚠️ **This step cannot be run until pose `.npy` files are available.** Without pose files, the manifest will be empty.

---

## Pose File Format

Each pose file is a NumPy `.npy` array with the following properties:

| Property | Value |
|----------|-------|
| Format | NumPy `.npy` file |
| dtype | `float32` |
| Shape | `[T, 274]` |
| `T` | Number of video frames for this sentence (variable) |
| `274` | 137 keypoints × 2 coordinates (x, y) |
| Coordinate system | Normalised [0, 1] relative to frame dimensions |
| Keypoint set | MediaPipe Holistic (face, pose, left hand, right hand) |

**MediaPipe Holistic landmark breakdown:**

| Region | Keypoints | Index range (approx.) | Pairs (×2 coords) |
|--------|-----------|-----------------------|--------------------|
| Face mesh | 468 | 0–467 | 0–935 |
| Pose body | 33 | 468–500 | 936–1001 |
| Left hand | 21 | 501–521 | 1002–1043 |
| Right hand | 21 | 522–542 | 1044–1085 |

> **Note**: ContextSign uses a **reduced set of 137 keypoints** (selected face landmarks + all pose + both hands) rather than all 543 MediaPipe Holistic landmarks. This selection must be consistent between pose extraction and model training.

**Loading a pose file:**
```python
import numpy as np
poses = np.load("_B4aNNi86Vk_7-8-rgb_front.npy")  # shape: [T, 274]
print(f"Frames: {poses.shape[0]}, Dims: {poses.shape[1]}")
```

---

## Extracting Poses from Video Using MediaPipe

> ⚠️ **This step requires raw RGB video files AND the MediaPipe library.**
> MediaPipe is NOT listed in the main project dependencies — it must be installed separately.

### Install MediaPipe

```powershell
$PIP = "c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\pip.exe"
& $PIP install mediapipe
```

### Extraction Script (Conceptual)

The following outlines the extraction process. A full extraction script would:

1. Load each video clip identified by the CSV timestamps
2. Trim to `[START_REALISED, END_REALISED]` using OpenCV or FFmpeg
3. Run MediaPipe Holistic on each frame
4. Select the 137 keypoints (consistent with training configuration)
5. Stack into `[T, 274]` array
6. Save as `.npy` with filename `<SENTENCE_ID>.npy`

```python
# Conceptual outline — not a production-ready script
import cv2
import mediapipe as mp
import numpy as np

def extract_poses(video_path, start_sec, end_sec, sentence_id, output_dir):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    frame_idx = 0
    
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_idx / fps
            if start_sec <= t <= end_sec:
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Extract and flatten selected 137 keypoints (x, y) → 274 dims
                pose_vec = extract_keypoints(results)  # [274,]
                frames.append(pose_vec)
            frame_idx += 1
    
    cap.release()
    if frames:
        pose_array = np.stack(frames, axis=0).astype(np.float32)  # [T, 274]
        np.save(f"{output_dir}/{sentence_id}.npy", pose_array)
```

### Estimated Extraction Time

On a machine with RTX 2050 (MediaPipe can use GPU):
- Approximately 1–2 seconds per second of video at 30fps
- For the full training set (~79 hours): **estimated 79–158 hours** processing time

> ⚠️ **This is an estimate only.** Actual time depends on hardware, MediaPipe configuration, and video resolution.

---

## Known Limitations

1. **Pose files not included**: The How2Sign dataset release does NOT include pre-extracted pose files. Each researcher must run MediaPipe extraction themselves.

2. **No ground-truth keypoints**: Since pose extraction is done by a separate tool (MediaPipe), the keypoints are **approximations**, not manually annotated ground truth.

3. **Missing keypoints**: MediaPipe may fail to detect hands or face landmarks in some frames (e.g., due to occlusion). These should be handled by zeroing or interpolating missing values.

4. **Temporal alignment**: The realigned CSV timestamps may not be frame-perfectly aligned with the video. Small boundary offsets (±1–2 frames) are expected.

5. **Keypoint consistency**: The 137-keypoint selection used in this project must exactly match between:
   - The extraction script
   - The model's `pose_dim = 274` configuration
   - Normalization statistics computed on the training set

6. **Video download**: Raw video files require significant bandwidth and storage. Some researchers use only a subset of clips.
