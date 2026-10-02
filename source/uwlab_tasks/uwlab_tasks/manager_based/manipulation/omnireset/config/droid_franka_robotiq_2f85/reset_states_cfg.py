# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset state generation environments for the DROID Franka + Robotiq 2F-85.

Same reset distributions as the UR5e configs. This file only swaps the robot, its action, the joint / body
names used for IK and success checks, and the dataset directory.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..ur5e_robotiq_2f85 import reset_states_cfg as ur5e_reset_states_cfg
from .actions import DroidFrankaRobotiq2f85RelativeOSCAction
from .constants import ARM_JOINT_NAMES, DATASET_DIR, GRIPPER_BODY_NAME, ROBOT_CFG


@configclass
class DroidFrankaResetStatesSceneCfg(ur5e_reset_states_cfg.ResetStatesSceneCfg):
    """Same scene as the UR5e (table, mount plate, objects), with the DROID Franka (gripper: ``constants.GRIPPER``)."""

    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def use_droid_franka_names(env_cfg: ur5e_reset_states_cfg.UR5eRobotiq2f85ResetStatesCfg) -> None:
    """Replace the UR5e joint / body names in reset events and success checks, and use the Franka datasets."""
    for term in env_cfg.events.__dict__.values():
        if not isinstance(term, EventTermCfg):
            continue
        if "robot_ik_cfg" in term.params:
            term.params["robot_ik_cfg"] = SceneEntityCfg(
                "robot", joint_names=ARM_JOINT_NAMES, body_names=GRIPPER_BODY_NAME
            )
        if "dataset_dir" in term.params:
            term.params["dataset_dir"] = DATASET_DIR

    env_cfg.terminations.success.params["ee_body_name"] = GRIPPER_BODY_NAME


@configclass
class DroidFrankaObjectAnywhereEEAnywhereResetStatesCfg(ur5e_reset_states_cfg.ObjectAnywhereEEAnywhereResetStatesCfg):
    scene: DroidFrankaResetStatesSceneCfg = DroidFrankaResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    actions: DroidFrankaRobotiq2f85RelativeOSCAction = DroidFrankaRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        use_droid_franka_names(self)


@configclass
class DroidFrankaObjectRestingEEGraspedResetStatesCfg(ur5e_reset_states_cfg.ObjectRestingEEGraspedResetStatesCfg):
    scene: DroidFrankaResetStatesSceneCfg = DroidFrankaResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    actions: DroidFrankaRobotiq2f85RelativeOSCAction = DroidFrankaRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        use_droid_franka_names(self)


@configclass
class DroidFrankaObjectAnywhereEEGraspedResetStatesCfg(ur5e_reset_states_cfg.ObjectAnywhereEEGraspedResetStatesCfg):
    scene: DroidFrankaResetStatesSceneCfg = DroidFrankaResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    actions: DroidFrankaRobotiq2f85RelativeOSCAction = DroidFrankaRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        use_droid_franka_names(self)


@configclass
class DroidFrankaObjectPartiallyAssembledEEAnywhereResetStatesCfg(
    ur5e_reset_states_cfg.ObjectPartiallyAssembledEEAnywhereResetStatesCfg
):
    scene: DroidFrankaResetStatesSceneCfg = DroidFrankaResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    actions: DroidFrankaRobotiq2f85RelativeOSCAction = DroidFrankaRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        use_droid_franka_names(self)


@configclass
class DroidFrankaObjectPartiallyAssembledEEGraspedResetStatesCfg(
    ur5e_reset_states_cfg.ObjectPartiallyAssembledEEGraspedResetStatesCfg
):
    scene: DroidFrankaResetStatesSceneCfg = DroidFrankaResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    actions: DroidFrankaRobotiq2f85RelativeOSCAction = DroidFrankaRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        use_droid_franka_names(self)
