import csv
import json

import numpy as np

from ml.convert_openpose import build_real_from_openpose


def _person(x_shift=0.0):
    def flat(n, base_x=0.0, conf=0.9):
        arr = []
        for i in range(n):
            arr.extend([base_x + i, 10.0 + i, conf])
        return arr

    return {
        "pose_keypoints_2d": flat(25, x_shift),
        "hand_left_keypoints_2d": flat(21, 30.0),
        "hand_right_keypoints_2d": flat(21, 40.0),
        "face_keypoints_2d": flat(70, 50.0),
    }


def test_convert_openpose_writes_pose_and_metadata(tmp_path):
    json_dir = tmp_path / "openpose_output" / "json"
    clip = json_dir / "VID_1-1-rgb_front"
    clip.mkdir(parents=True)
    for i in range(3):
        data = {"people": [_person(x_shift=float(i))]}
        (clip / f"VID_1-1-rgb_front_{i:012d}_keypoints.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    csv_path = tmp_path / "val.csv"
    csv_path.write_text(
        "VIDEO_ID,VIDEO_NAME,SENTENCE_ID,SENTENCE_NAME,START_REALIZED,END_REALIZED,SENTENCE\n"
        "VID_1,VID_1-1-rgb_front,VID_1_1,VID_1-1-rgb_front,0.0,2.5,Hello world\n",
        encoding="utf-8",
    )

    out = tmp_path / "data"
    n = build_real_from_openpose(json_dir, csv_path, out, width=100.0, height=50.0, conf=0.05)
    assert n == 1

    pose = np.load(out / "poses" / "VID_1-1-rgb_front.npy")
    assert pose.shape == (3, 274)
    assert pose.dtype == np.float32
    assert abs(pose[1, 0] - (1.0 / 100.0)) < 1e-5
    assert abs(pose[1, 25 * 2] - (30.0 / 100.0)) < 1e-5

    rows = list(csv.DictReader(open(out / "metadata.csv", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["sentence"] == "Hello world"
    assert rows[0]["frames"] == "3"
    assert (out / "DATASET_SOURCE.txt").exists()


def test_low_confidence_points_are_zeroed(tmp_path):
    json_dir = tmp_path / "json"
    clip = json_dir / "A_1-1-rgb_front"
    clip.mkdir(parents=True)
    person = _person()
    person["pose_keypoints_2d"][2] = 0.0  # confidence below threshold
    for i in range(2):
        (clip / f"A_1-1-rgb_front_{i:012d}_keypoints.json").write_text(
            json.dumps({"people": [person]}), encoding="utf-8"
        )
    csv_path = tmp_path / "v.csv"
    csv_path.write_text(
        "VIDEO_ID,SENTENCE_NAME,SENTENCE,START_REALIZED,END_REALIZED\n"
        "A,A_1-1-rgb_front,Test,0,1\n",
        encoding="utf-8",
    )
    out = tmp_path / "o"
    build_real_from_openpose(json_dir, csv_path, out, width=100, height=50)
    pose = np.load(out / "poses" / "A_1-1-rgb_front.npy")
    assert pose[0, 0] == 0.0 and pose[0, 1] == 0.0
