# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a grasp dataset sampled with UWLab's Robotiq 2F-85 for the DROID robot's Robotiq 2F-85.

Both gripper models share the same base frame (fingers point along +x and close along y), so the grasp poses
carry over unchanged. Their passive finger joints differ: in the DROID model the five passive joints mimic
``finger_joint``, so the gripper joint positions are rewritten from ``finger_joint`` alone.

Usage:
    python scripts_v2/tools/convert_grasps_to_droid.py \\
        --input ./Datasets/OmniReset/Grasps/InsertiveCube/grasps.pt \\
        --output ./Datasets/OmniReset_DroidFranka/Grasps/InsertiveCube/grasps.pt
"""

import argparse
import os
import torch

# DROID Robotiq 2F-85 joint position = sign * finger_joint (PhysX mimic joints in the DROID USD)
DROID_GRIPPER_JOINT_SIGNS = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_finger_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_knuckle_joint": -1.0,
    "left_inner_finger_knuckle_joint": -1.0,
}


def main():
    parser = argparse.ArgumentParser(description="Convert a UWLab Robotiq 2F-85 grasp dataset for the DROID robot.")
    parser.add_argument("--input", type=str, required=True, help="grasps.pt sampled with UWLab's Robotiq 2F-85.")
    parser.add_argument("--output", type=str, required=True, help="Where to write the DROID grasps.pt.")
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    grasps = data["grasp_relative_pose"]
    finger_joint = grasps["gripper_joint_positions"]["finger_joint"]
    grasps["gripper_joint_positions"] = {
        name: [sign * q for q in finger_joint] for name, sign in DROID_GRIPPER_JOINT_SIGNS.items()
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(data, args.output)
    print(f"Converted {len(finger_joint)} grasps: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
