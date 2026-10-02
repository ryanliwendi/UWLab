# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_robotiq_gripper.actions import ROBOTIQ_GRIPPER_BINARY_ACTIONS

from ...mdp.actions.actions_cfg import RelCartesianOSCActionCfg
from .constants import ARM_JOINT_NAMES, EE_BODY_NAME

# Same pre-train gains and action scales as the UR5e; PhysX Jacobian and null-space posture control
# because the Franka has 7 joints.
DROID_FRANKA_ROBOTIQ_2F85_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=ARM_JOINT_NAMES,
    body_name=EE_BODY_NAME,
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    jacobian_source="physx",
    nullspace_stiffness=10.0,
    nullspace_damping_ratio=1.0,
)


@configclass
class DroidFrankaRobotiq2f85RelativeOSCAction:
    """Action config using the OSC + binary gripper."""

    arm = DROID_FRANKA_ROBOTIQ_2F85_RELATIVE_OSC
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
