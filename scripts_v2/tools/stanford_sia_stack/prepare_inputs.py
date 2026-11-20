"""Prepare ignored local references and marker textures from a user-supplied episode."""

from pathlib import Path
import argparse, json
import cv2, h5py, numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--episode", type=Path, required=True)
a = p.parse_args()
r = Path(__file__).resolve().parent
refs = r / "references/stack"
refs.mkdir(parents=True, exist_ok=True)
cameras = {}
with h5py.File(a.episode, "r") as ep:
    robot = json.loads(ep.attrs["robot_profile"])
    q = ep["observations/robot_states/joint_position"][()]
    assert q.ndim == 2 and q.shape[1] == 7
    for name in ["side_cam", "top_cam"]:
        raw = ep["observations/video_paths/" + name][()]
        video = a.episode.parent / (
            raw.decode() if isinstance(raw, bytes) else str(raw)
        )
        v = cv2.VideoCapture(str(video))
        ok, im = v.read()
        assert ok, f"Cannot read {video}"
        assert im.shape[:2] == (480, 640), "Scene fitting expects 640x480 references"
        cameras[name] = {
            "path": str(video.resolve()),
            "width": im.shape[1],
            "height": im.shape[0],
            "fps": v.get(5),
            "frames": int(v.get(7)),
        }
        v.release()
        cv2.imwrite(str(refs / (name + "-0000.png")), im)
(refs / "metadata.json").write_text(
    json.dumps(
        {
            "robot": robot,
            "states": {"joint_position": {"first": q[0].tolist()}},
            "cameras": cameras,
        },
        indent=2,
    )
)
t = r / "assets/textures"
t.mkdir(parents=True, exist_ok=True)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
for b in json.loads((r / "config/grid-layout.json").read_text())["boards"]:
    sheet = np.full((840, 640), 235, np.uint8)
    for row in range(8):
        for col in range(6):
            x, y = 20 + col * 100, 20 + row * 100
            if (row + col) % 2:
                sheet[y : y + 100, x : x + 100] = 10
            else:
                ident = b["id_offset"] + 4 * col + 3 - row // 2
                marker = cv2.aruco.generateImageMarker(dictionary, ident, 72)
                sheet[y + 14 : y + 86, x + 14 : x + 86] = cv2.rotate(
                    marker, cv2.ROTATE_90_COUNTERCLOCKWISE
                )
    cv2.imwrite(str(t / f"board-{b['index']}-6x8.png"), sheet)
print("Prepared local reference frames and five 6x8 marker sheets.")
