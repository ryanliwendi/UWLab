# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DROID Franka robot and names that replace the UR5e ones in the shared OmniReset configs."""

from uwlab_assets.robots.droid_franka_robotiq import (
    IMPLICIT_DROID_FRANKA_ROBOTIQ_2F85,
    IMPLICIT_DROID_FRANKA_UWLAB_ROBOTIQ_2F85,
)

GRIPPER = "uwlab"
"""Which Robotiq 2F-85 model is mounted on the Franka arm.

``"uwlab"``: UWLab's calibrated model, the same gripper as the UR5e. Grasps sampled with
``OmniReset-Robotiq2f85-GraspSampling-v0`` fit it as is.
``"droid"``: the model from NVIDIA's DROID USD. Grasps must first be converted with
``scripts_v2/tools/convert_grasps_to_droid.py``.
"""

ROBOT_CFG = {"uwlab": IMPLICIT_DROID_FRANKA_UWLAB_ROBOTIQ_2F85, "droid": IMPLICIT_DROID_FRANKA_ROBOTIQ_2F85}[GRIPPER]
"""Robot articulation for the chosen gripper."""

ARM_JOINT_NAMES = ["panda_joint[1-7]"]
"""Arm joints (UR5e: ``shoulder.*``, ``elbow.*``, ``wrist.*``)."""

EE_BODY_NAME = "panda_link8"
"""Arm flange, used by the OSC action and observations (UR5e: ``wrist_3_link``)."""

GRIPPER_BODY_NAME = {"uwlab": "robotiq_base_link", "droid": "base_link"}[GRIPPER]
"""Robotiq 2F-85 base, used for grasps and gripper rewards."""

DATASET_DIR = {"uwlab": "./Datasets/OmniReset_FrankaUWLabGripper", "droid": "./Datasets/OmniReset_DroidFranka"}[GRIPPER]
"""Local dataset directory (grasps, partial assemblies, reset states), relative to the repo root.

Reset states store robot joint positions, so each robot needs its own datasets.
"""

EXPERIMENT_NAME = {
    "uwlab": "droid_franka_uwlab_robotiq_2f85_omnireset_agent",
    "droid": "droid_franka_robotiq_2f85_omnireset_agent",
}[GRIPPER]
"""Log folder name under logs/rsl_rl/."""
