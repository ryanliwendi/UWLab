from pathlib import Path
import json, sys, numpy as np

p = Path(sys.argv[1])
m = json.loads((p / "motion.json").read_text())
f = m["frames"]
red = np.array([r["red_position"] for r in f])
green = np.array([r["green_position"] for r in f])
contact = np.array([r["finger_forces_N"] for r in f])
speed = np.linalg.norm([r["red_linear_velocity"] for r in f], axis=1)
xy = np.linalg.norm((red - green)[:, :2], axis=1)
dz = red[:, 2] - green[:, 2]
settled = (
    (xy < 0.004)
    & (np.abs(dz - 0.025) < 0.003)
    & (speed < 0.01)
    & (np.max(contact, axis=1) < 0.05)
)
r = {
    "verified_lift": m["verified_lift"],
    "bilateral_contact_frames": int(np.all(contact > 0.05, axis=1).sum()),
    "peak_red_center_height_m": float(red[:, 2].max()),
    "final_center_xy_error_m": float(xy[-1]),
    "final_red_above_green_m": float(dz[-1]),
    "final_speed_m_s": float(speed[-1]),
    "settled_stack_final_second": bool(np.all(settled[-15:])),
    "green_displacement_m": float(np.linalg.norm(green[-1] - green[0])),
    "max_tcp_tracking_error_m": float(
        max(
            np.linalg.norm(np.array(row["tcp_actual"]) - np.array(goal["tcp_target"]))
            for row, goal in zip(
                f,
                json.loads(
                    (Path(__file__).parent / "config/controller-plan.json").read_text()
                )["rows"],
            )
        )
    ),
}
(p / "validation.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
