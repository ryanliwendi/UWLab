# Copyright (c) 2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export real, simulated, and synchronized videos without modifying source data."""
import argparse
import json
import numpy as np
import subprocess
from pathlib import Path

import cv2

p = argparse.ArgumentParser()
p.add_argument("run")
a = p.parse_args()
root = Path(__file__).resolve().parent
run = root / "renders" / a.run
source = Path(json.loads((root / "references/drawer/metadata.json").read_text())["cameras"]["exterior_image_1"]["path"])
diagnostics = json.loads((run / "contact-motion.json").read_text())
c = cv2.VideoCapture(str(source))
fps = c.get(cv2.CAP_PROP_FPS)
n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
assert n == 70 and fps == 15
out = run / "videos"
out.mkdir(exist_ok=True)
seq = run / "comparison-frames"
seq.mkdir(exist_ok=True)
for i in range(n):
    ok, real = c.read()
    assert ok
    sim = cv2.imread(str(run / "frames" / f"{i:04d}.png"))
    assert sim is not None and sim.shape == real.shape
    canvas = np.full((460, 1344, 3), 245, np.uint8)
    canvas[60:444, :672] = real
    canvas[60:444, 672:] = sim
    cv2.putText(canvas, "REAL - CMU Auton", (12, 23), 0, 0.65, (25, 25, 25), 2)
    cv2.putText(
        canvas,
        (
            "SIM - contact push + hold (no retreat)"
            if diagnostics.get("controller")
            else "SIM - actuator-driven arm / passive drawer"
        ),
        (684, 23),
        0,
        0.58,
        (25, 25, 25),
        2,
    )
    label = f"Frame {i:02d}/69 | {i / 15:.2f} s"
    cv2.putText(canvas, label, (12, 49), 0, 0.48, (40, 40, 40), 1)
    row = diagnostics["frames"][i]
    detail = f"{i / 15:.2f}s | drawer {row['drawer_displacement_m'] * 1000:.1f} mm | arm contact {row['robot_contact_force_N']:.1f} N"
    cv2.putText(canvas, detail, (684, 49), 0, 0.43, (40, 40, 40), 1)
    cv2.putText(
        canvas,
        (
            "Contact physics; passive drawer. Arm holds pushing pose after closure. Real/sim compared at original speed."
            if diagnostics.get("controller")
            else "Contact physics; no drawer motion commands. Recorded arm targets. Geometry estimated; frame-index comparison."
        ),
        (12, 456),
        0,
        0.38,
        (30, 30, 30),
        1,
    )
    cv2.imwrite(str(seq / f"{i:04d}.png"), canvas)
c.release()


def enc(src, dst):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            "15",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
    )


enc(seq / "%04d.png", out / "drawer-real-vs-sim.mp4")
enc(run / "frames/%04d.png", out / "drawer-sim.mp4")
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out / "drawer-real.mp4"),
    ],
    check=True,
)
cv2.imwrite(
    str(out / "comparison-start-middle-end.jpg"),
    np.vstack([cv2.imread(str(seq / f"{i:04d}.png")) for i in [0, 35, 69]]),
)
checks = {}
for p in out.glob("*.mp4"):
    c = cv2.VideoCapture(str(p))
    checks[p.name] = {
        "frames": int(c.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": c.get(cv2.CAP_PROP_FPS),
        "width": int(c.get(3)),
        "height": int(c.get(4)),
    }
    c.release()
    assert checks[p.name]["frames"] == 70
(out / "video-checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
