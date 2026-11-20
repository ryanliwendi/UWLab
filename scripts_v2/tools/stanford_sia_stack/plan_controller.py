"""IK arm/finger targets only. Cube poses are never controlled."""

from pathlib import Path
import json, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import least_squares

root = Path(__file__).resolve().parent
joints = ET.parse(root / "assets/urdf/fr3.urdf").findall("joint")
names = [f"fr3_joint{i}" for i in range(1, 8)]
chain = []
link = "fr3_hand_tcp"
while link != "base":
    j = next(j for j in joints if j.find("child").get("link") == link)
    chain.insert(0, j)
    link = j.find("parent").get("link")
origins = []
for j in chain:
    o = j.find("origin")
    t = np.eye(4)
    if o is not None:
        t[:3, 3] = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = R.from_euler(
            "xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    origins.append(t)


def fk(q):
    t = np.eye(4)
    for j, o in zip(chain, origins):
        m = np.eye(4)
        if j.get("name") in names:
            m[:3, :3] = R.from_rotvec(
                np.fromstring(j.find("axis").get("xyz"), sep=" ")
                * q[names.index(j.get("name"))]
            ).as_matrix()
        t = t @ o @ m
    return t


if __name__ == "__main__":
    meta = json.loads((root / "references/stack/metadata.json").read_text())
    q0 = np.array(meta["states"]["joint_position"]["first"])
    t0 = fk(q0)
    lim = [next(j for j in joints if j.get("name") == n).find("limit") for n in names]
    lo = np.array([float(j.get("lower")) + 0.002 for j in lim])
    hi = np.array([float(j.get("upper")) - 0.002 for j in lim])
    placements = json.loads((root / "config/cube-placement.json").read_text())
    red = np.array(placements["RedCube"]["center_world"])
    green = np.array(placements["GreenCube"]["center_world"])
    grasp = red.copy()
    grasp[2] = 0.015
    rr = (
        R.from_euler(
            "z", placements["RedCube"]["yaw_degrees"] - 90, degrees=True
        ).as_matrix()
        @ R.from_euler("x", 180, degrees=True).as_matrix()
    )
    above = green.copy()
    above[2] = 0.16
    place = above.copy()
    place[2] = 0.041
    keys = [
        (0, t0[:3, 3], t0[:3, :3], 0.08, "approach"),
        (3, grasp + [0, 0, 0.09], rr, 0.07, "approach"),
        (5, grasp, rr, 0.07, "descend"),
        (6, grasp, rr, 0.018, "close"),
        (7, grasp, rr, 0.018, "hold"),
        (9, grasp + [0, 0, 0.145], rr, 0.018, "lift"),
        (12, above, rr, 0.018, "transport"),
        (15, place, rr, 0.018, "lower"),
        (16, place, rr, 0.07, "release"),
        (18, above, rr, 0.07, "retreat"),
        (325 / 15, above, rr, 0.07, "settle"),
    ]
    prev = q0.copy()
    rows = []
    for i in range(325):
        t = i / 15
        k = min(
            max(np.searchsorted([x[0] for x in keys], t, side="right") - 1, 0),
            len(keys) - 2,
        )
        a, b = keys[k : k + 2]
        u = np.clip((t - a[0]) / (b[0] - a[0]), 0, 1)
        u = u * u * (3 - 2 * u)
        goal = a[1] * (1 - u) + b[1] * u
        rot = Slerp([0, 1], R.from_matrix([a[2], b[2]]))([u]).as_matrix()[0]

        def residual(q):
            f = fk(q)
            return np.r_[
                100 * (f[:3, 3] - goal),
                5 * R.from_matrix(rot.T @ f[:3, :3]).as_rotvec(),
                0.02 * (q - prev),
            ]

        sol = least_squares(
            residual, np.clip(prev, lo, hi), bounds=(lo, hi), max_nfev=100
        )
        prev = sol.x
        f = fk(prev)
        rows.append(
            dict(
                time_s=t,
                q=prev.tolist(),
                width=float(a[3] * (1 - u) + b[3] * u),
                phase=b[4],
                tcp_target=goal.tolist(),
                ik_error_m=float(np.linalg.norm(f[:3, 3] - goal)),
            )
        )
    report = dict(
        mode="SCRIPTED_IK_JOINT_TARGETS_DYNAMIC_CUBES",
        initial_tcp=t0[:3, 3].tolist(),
        rows=rows,
        max_ik_error_m=max(x["ik_error_m"] for x in rows),
        max_joint_speed_rad_s=np.abs(np.diff([x["q"] for x in rows], axis=0) * 15)
        .max(axis=0)
        .tolist(),
    )
    (root / "config/controller-plan.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "rows"})
