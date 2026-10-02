# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from . import task_space_actions


@configclass
class RelCartesianOSCActionCfg(ActionTermCfg):
    """Configuration for Relative Cartesian OSC action term.

    Uses the analytical Jacobian from calibrated UR5e kinematics and a simple
    task-space PD controller matching the real robot's OSC implementation:
        tau = J^T @ (Kp * pose_error + Kd * vel_error)

    No inertial dynamics decoupling, no mass matrix. Designed to work with
    the DelayedDCMotor actuator for sim2real alignment.

    For other arms, set ``jacobian_source="physx"``. Arms with more than 6 joints (e.g. Franka)
    should also set ``nullspace_stiffness`` so the redundant joints do not drift.
    """

    class_type: type[ActionTerm] = task_space_actions.RelCartesianOSCAction

    @configclass
    class OffsetCfg:
        """Offset configuration for body or frame offsets."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation offset."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Rotation offset as quaternion (w, x, y, z)."""

    joint_names: list[str] = MISSING
    """Joint names for the arm (regex supported)."""

    body_name: str = MISSING
    """End-effector body name (e.g., 'wrist_3_link')."""

    scale_xyz_axisangle: tuple[float, float, float, float, float, float] = MISSING
    """Per-DOF scaling for [x, y, z, rx, ry, rz] action deltas."""

    input_clip: tuple[float, float] | None = None
    """Optional symmetric clip range for scaled actions."""

    motion_stiffness: tuple[float, float, float, float, float, float] = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
    """Task-space stiffness Kp for [x, y, z, rx, ry, rz]."""

    motion_damping_ratio: tuple[float, float, float, float, float, float] = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
    """Task-space damping ratio. Kd = 2 * sqrt(Kp) * damping_ratio."""

    torque_limit: tuple[float, ...] = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    """Per-joint torque limits (clamped after J^T multiplication). One value per arm joint."""

    jacobian_source: Literal["ur5e_analytical", "physx"] = "ur5e_analytical"
    """Where the Jacobian comes from.

    ``"ur5e_analytical"`` uses the calibrated UR5e kinematics and only works for the UR5e.
    ``"physx"`` uses the physics engine's Jacobian and works for any fixed-base arm.
    """

    nullspace_stiffness: float = 0.0
    """Joint-space stiffness pulling the arm toward its default joint positions, applied only in the
    null space of the Jacobian so it does not disturb the end-effector. 0 disables it."""

    nullspace_damping_ratio: float = 1.0
    """Null-space damping ratio. Kd_null = 2 * sqrt(Kp_null) * damping_ratio."""
