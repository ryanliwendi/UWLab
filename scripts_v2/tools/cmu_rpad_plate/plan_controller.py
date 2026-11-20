"""Joint-target controller plan; plate poses are NEVER commanded by this plan."""

from pathlib import Path
import json, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import least_squares

root = Path(__file__).resolve().parent
fit = json.loads((root / "config/fit.json").read_text())
tree = ET.parse(root / "assets/urdf/vx300s_aloha.urdf")
joints = tree.findall("joint")
names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def transform(j):
    o = j.find("origin")
    t = np.eye(4)
    if o is not None:
        t[:3, 3] = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = R.from_euler(
            "xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return t


chain = []
name = "vx300s/gripper_base"
while name != "vx300s/base_link":
    j = next(j for j in joints if j.find("child").get("link") == name)
    chain.insert(0, j)
    name = j.find("parent").get("link")
base = np.array(fit["base"])
baseR = np.diag([-1, -1, 1])


def fk(q):
    t = np.eye(4)
    t[:3, :3] = baseR
    t[:3, 3] = base
    for j in chain:
        m = np.eye(4)
        if j.get("type") == "revolute":
            m[:3, :3] = R.from_rotvec(
                np.fromstring(j.find("axis").get("xyz"), sep=" ")
                * q[names.index(j.get("name"))]
            ).as_matrix()
        t = t @ transform(j) @ m
    return t


limits = {
    j.get("name"): (
        float(j.find("limit").get("lower")) + 0.002,
        float(j.find("limit").get("upper")) - 0.002,
    )
    for j in joints
    if j.get("type") == "revolute"
}
lo = np.array([limits[n][0] for n in names])
hi = np.array([limits[n][1] for n in names])
cam = np.array(fit["position"])
rot = np.array(fit["rotation"])
K = np.array(fit["K"])


def plane(u, v, z):
    ray = rot @ np.linalg.inv(K) @ np.array([u, v, 1])
    return cam + (z - cam[2]) / ray[2] * ray


pc = plane(725, 600, 0)
radius = np.linalg.norm(plane(846, 600, 0.012)[:2] - plane(604, 600, 0.012)[:2]) / 2
# Pinch the elevated rim between upper and lower fingers, approaching from behind.
grasp = pc + np.array([radius * 0.91, 0, 0.012])
local = np.array([0, 0.0035, 0.105])
orient = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
q0 = np.array([fit["q"][n] for n in names])
t0 = fk(q0)
g0 = t0[:3, :3] @ local + t0[:3, 3]
target = plane(565, 215, 0.10)
target[2] = 0.10
# Keyframes: open approach, rim insertion, close, lift, carry, lower, release, retreat.
keys = [
    (0, g0, t0[:3, :3], 0.034, "approach"),
    (2, grasp + [0, 0, 0.12], orient, 0.025, "approach"),
    (3.4, grasp + [0, 0, 0.04], orient, 0.025, "rim_approach"),
    (4.4, grasp, orient, 0.025, "rim_insert"),
    (5.2, grasp, orient, 0.002, "close"),
    (5.8, grasp, orient, 0.002, "hold"),
    (8.0, grasp + [0, 0, 0.40], orient, 0.002, "lift"),
    (10.0, grasp + (target - pc) + [0, 0, 0.33], orient, 0.002, "transport"),
    (12.0, grasp + (target - pc) + [0, 0, 0.003], orient, 0.002, "lower"),
    (12.8, grasp + (target - pc) + [0, 0, 0.003], orient, 0.022, "release"),
    (13.8, grasp + (target - pc) + [0.08, 0, 0.09], orient, 0.03, "retreat"),
    (15.0, grasp + (target - pc) + [0.08, 0, 0.09], orient, 0.03, "settle"),
]
prev = q0.copy()
rows = []
for i, t in enumerate(np.arange(0, 15 + 1 / 30, 1 / 30)):
    k = min(
        max(np.searchsorted([x[0] for x in keys], t, side="right") - 1, 0),
        len(keys) - 2,
    )
    a, b = keys[k : k + 2]
    u = float(np.clip((t - a[0]) / (b[0] - a[0]), 0, 1))
    u = u * u * (3 - 2 * u)
    goal = a[1] * (1 - u) + b[1] * u
    rr = Slerp([0, 1], R.from_matrix([a[2], b[2]]))([u]).as_matrix()[0]
    opening = a[3] * (1 - u) + b[3] * u

    def residual(q):
        tt = fk(q)
        g = tt[:3, :3] @ local + tt[:3, 3]
        return np.r_[
            (g - goal) * 100,
            R.from_matrix(rr.T @ tt[:3, :3]).as_rotvec() * 5,
            (q - prev) * 0.02,
        ]

    sol = least_squares(residual, np.clip(prev, lo, hi), bounds=(lo, hi), max_nfev=100)
    prev = sol.x
    tt = fk(prev)
    rows.append(
        {
            "time_s": float(t),
            "q": dict(zip(names, prev.tolist()))
            | {"left_finger": float(opening), "right_finger": float(opening)},
            "phase": b[4],
            "grasp_target": goal.tolist(),
            "position_error_m": float(
                np.linalg.norm(tt[:3, :3] @ local + tt[:3, 3] - goal)
            ),
            "orientation_error_rad": float(
                np.linalg.norm(R.from_matrix(rr.T @ tt[:3, :3]).as_rotvec())
            ),
        }
    )
report = {
    "plate_start": pc.tolist(),
    "plate_radius": float(radius),
    "target_center": target.tolist(),
    "rows": rows,
    "status": "ARM_JOINT_TARGETS_ONLY_PLATE_NOT_ACTUATED",
}
(root / "config/controller-plan.json").write_text(json.dumps(report, indent=2))
print(
    "position max",
    max(r["position_error_m"] for r in rows),
    "orientation max",
    max(r["orientation_error_rad"] for r in rows),
    "radius",
    radius,
)
