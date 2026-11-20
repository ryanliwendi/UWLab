"""Prepare local ignored reference frames and textures from the existing dataset."""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--episode", type=Path, required=True)
a = p.parse_args()
r = Path(__file__).resolve().parent
source = a.episode
with h5py.File(source, "r") as episode:
    robot = json.loads(episode.attrs["robot_profile"])
    joints = episode["observations/robot_states/joint_position"][()]
    raw = episode["observations/video_paths/front"][()]
    video = source.parent / (raw.decode() if isinstance(raw, bytes) else str(raw))
assert (
    joints.ndim == 2 and joints.shape[1] == 6
), f"Unexpected joint shape: {joints.shape}"
c = cv2.VideoCapture(str(video))
assert c.isOpened(), f"Cannot open {video}"
fps, count = c.get(5), int(c.get(7))
ok, frame = c.read()
c.release()
assert ok and count > 0 and fps > 0
refs = r / "references/plate"
refs.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(refs / "front-0000.png"), frame)
meta = {
    "source": str(source.name),
    "robot": robot,
    "states": {"joint_position": {"first": joints[0].tolist()}},
    "cameras": {
        "front": {
            "path": str(video.resolve()),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "frames": count,
            "fps": fps,
        }
    },
}
(refs / "metadata.json").write_text(json.dumps(meta, indent=2))
patch = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[161:239, 485:675] / 255.0
lin = np.where(patch <= 0.04045, patch / 12.92, ((patch + 0.055) / 1.055) ** 2.4)
mean = lin.mean((0, 1))
new = np.clip(mean * 1.10 + 0.35 * (lin - mean), 0, 1)
srgb = np.where(new <= 0.0031308, new * 12.92, 1.055 * new ** (1 / 2.4) - 0.055)
textures = r / "assets/textures"
textures.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.uint8(srgb * 255)).resize((760, 312), Image.Resampling.LANCZOS).save(
    textures / "subtle-wood.png"
)
print(f"Prepared local reference; real video: {video}")
