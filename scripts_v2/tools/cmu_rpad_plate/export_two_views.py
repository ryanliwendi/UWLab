from pathlib import Path
import sys, json, subprocess
import cv2, numpy as np
import argparse

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--real-video", type=Path, required=True)
args = p.parse_args()
run = args.run
out = run / "videos"
out.mkdir(exist_ok=True)
realpath = args.real_video
v = cv2.VideoCapture(str(realpath))
real = []
while True:
    ok, im = v.read()
    if not ok:
        break
    real.append(im)
v.release()
frames = sorted((run / "frames").glob("*.png"))
assert frames, "No rendered frames found"
output = out / "real-vs-sim-two-views.mp4"
p = subprocess.Popen(
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
        "1920x580",
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
        str(output),
    ],
    stdin=subprocess.PIPE,
)
for i, f in enumerate(frames):
    c = np.full((580, 1920, 3), 245, np.uint8)
    c[40:, :960] = cv2.resize(real[min(i, len(real) - 1)], (960, 540))
    c[40:, 960:] = cv2.resize(cv2.imread(str(f)), (960, 540))
    cv2.putText(c, "REAL - CMU RPAD", (15, 28), 0, 0.7, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(
        c, "SIM - clean tabletop", (975, 28), 0, 0.7, (25, 25, 25), 2, cv2.LINE_AA
    )
    if i >= len(real):
        cv2.putText(
            c,
            "REAL ENDED - last frame held",
            (15, 565),
            0,
            0.6,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    p.stdin.write(c.tobytes())
p.stdin.close()
assert p.wait() == 0
v = cv2.VideoCapture(str(output))
check = {
    "frames": int(v.get(7)),
    "fps": v.get(5),
    "width": int(v.get(3)),
    "height": int(v.get(4)),
}
v.release()
assert check == {"frames": len(frames), "fps": 15.0, "width": 1920, "height": 580}
(out / "video-checks.json").write_text(json.dumps(check, indent=2))
print("VIDEOS_COMPLETE", out, flush=True)
