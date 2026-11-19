# Copyright (c) 2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare ignored local inputs from the existing OopsieData episode; no downloads."""
import argparse
import h5py
import json
import numpy as np
from pathlib import Path

import cv2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-root", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source = args.dataset_root / "CMU_Auton/20261001_180831.h5"
with h5py.File(source, "r") as episode:
    robot = json.loads(episode.attrs["robot_profile"])
    states = {
        key: value[()] for key, value in episode["observations/robot_states"].items() if isinstance(value, h5py.Dataset)
    }
    raw = episode["observations/video_paths/exterior_image_1"][()]
    video = source.parent / (raw.decode() if isinstance(raw, bytes) else str(raw))
    instruction = str(episode.attrs["language_instruction"])
assert states["joint_position"].shape == (70, 7), "Only the documented 70-frame episode is supported."
cap = cv2.VideoCapture(str(video))
assert cap.isOpened(), f"Cannot read {video}"
count, fps, width, height = int(cap.get(7)), cap.get(5), int(cap.get(3)), int(cap.get(4))
assert count == 70 and abs(fps - 15) < 1e-6
ok, frame = cap.read()
cap.release()
assert ok
refs = root / "references/drawer"
refs.mkdir(parents=True, exist_ok=True)
assert cv2.imwrite(str(refs / "exterior_image_1-0000.png"), frame)
metadata = {
    "source": "CMU_Auton/20261001_180831.h5",
    "instruction": instruction,
    "robot": robot,
    "states": {key: {"first": value[0].tolist()} for key, value in states.items()},
    "cameras": {
        "exterior_image_1": {
            "path": str(video.resolve()),
            "frames": count,
            "fps": fps,
            "width": width,
            "height": height,
        }
    },
}
(refs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(root / "motion-drawer").mkdir(exist_ok=True)
np.savez(root / "motion-drawer/states.npz", **states)
print(f"Prepared {count} frames at {fps} fps in {root}; dataset files remain untracked.")
