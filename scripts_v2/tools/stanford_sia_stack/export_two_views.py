"""Export separate simulation and two-view real/sim videos without time warping."""

from pathlib import Path
import argparse, subprocess, json
import cv2, numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
p.add_argument("--real-video", type=Path, required=True)
a = p.parse_args()
out = a.run / "videos"
out.mkdir(exist_ok=True)
files = sorted((a.run / "frames").glob("*.png"))
assert files, "No rendered frames"
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        "15",
        "-i",
        str(a.run / "frames/%04d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out / "simulation.mp4"),
    ],
    check=True,
)
v = cv2.VideoCapture(str(a.real_video))
assert v.isOpened()
fps = v.get(5)
assert fps > 0
proc = subprocess.Popen(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        "1280x520",
        "-framerate",
        "15",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out / "real-vs-sim.mp4"),
    ],
    stdin=subprocess.PIPE,
)
last = None
held = False
for i, f in enumerate(files):
    v.set(cv2.CAP_PROP_POS_FRAMES, int(i / 15 * fps))
    ok, real = v.read()
    if ok:
        last = real
    else:
        held = True
    assert last is not None
    canvas = np.full((520, 1280, 3), 245, np.uint8)
    canvas[40:, :640] = cv2.resize(last, (640, 480))
    canvas[40:, 640:] = cv2.imread(str(f))
    cv2.putText(
        canvas,
        "REAL" + (" (last frame held)" if not ok else ""),
        (12, 27),
        0,
        0.65,
        (25, 25, 25),
        2,
    )
    cv2.putText(
        canvas,
        "SIM - scripted arm / contact physics",
        (650, 27),
        0,
        0.6,
        (25, 25, 25),
        1,
    )
    proc.stdin.write(canvas.tobytes())
proc.stdin.close()
assert proc.wait() == 0
v.release()
(out / "export.json").write_text(
    json.dumps(
        {
            "frames": len(files),
            "fps": 15,
            "real_fps": fps,
            "held_last_real_frame": held,
            "note": "Shared playback clock, not phase-aligned events.",
        },
        indent=2,
    )
)
