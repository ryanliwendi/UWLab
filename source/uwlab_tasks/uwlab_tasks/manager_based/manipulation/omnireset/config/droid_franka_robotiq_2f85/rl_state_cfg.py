# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL state environments for the DROID Franka + Robotiq 2F-85.

The task (objects, rewards, success, observations) is shared with the UR5e configs. This file only swaps
the robot, its action, the joint / body names that observations and rewards refer to, and the dataset directory.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..ur5e_robotiq_2f85 import rl_state_cfg as ur5e_rl_state_cfg
from .actions import DroidFrankaRobotiq2f85RelativeOSCAction
from .constants import ARM_JOINT_NAMES, DATASET_DIR, EE_BODY_NAME, GRIPPER_BODY_NAME, ROBOT_CFG


@configclass
class DroidFrankaRlStateSceneCfg(ur5e_rl_state_cfg.RlStateSceneCfg):
    """Same scene as the UR5e (table, mount plate, objects), with the DROID Franka (gripper: ``constants.GRIPPER``)."""

    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def use_droid_franka_names(env_cfg: ur5e_rl_state_cfg.Ur5eRobotiq2f85RlStateCfg) -> None:
    """Replace the UR5e joint / body names in observations and rewards, and point resets at the Franka datasets."""
    ee = SceneEntityCfg("robot", body_names=EE_BODY_NAME)
    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        group.end_effector_pose.params["target_asset_cfg"] = ee
        group.insertive_asset_pose.params["root_asset_cfg"] = ee
        group.receptive_asset_pose.params["root_asset_cfg"] = ee
    env_cfg.observations.critic.end_effector_vel_lin_ang_b.params["target_asset_cfg"] = ee

    env_cfg.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    env_cfg.rewards.ee_asset_distance.params["root_asset_cfg"] = SceneEntityCfg("robot", body_names=GRIPPER_BODY_NAME)

    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = DATASET_DIR


# Training configuration (Stage 1: no curriculum, implicit actuator, no sysid DR)
@configclass
class DroidFrankaRobotiq2f85RelCartesianOSCTrainCfg(ur5e_rl_state_cfg.Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    scene: DroidFrankaRlStateSceneCfg = DroidFrankaRlStateSceneCfg(num_envs=32, env_spacing=1.5)
    actions: DroidFrankaRobotiq2f85RelativeOSCAction = DroidFrankaRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        use_droid_franka_names(self)


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class DroidFrankaRobotiq2f85RelCartesianOSCEvalCfg(ur5e_rl_state_cfg.Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    scene: DroidFrankaRlStateSceneCfg = DroidFrankaRlStateSceneCfg(num_envs=32, env_spacing=1.5)
    actions: DroidFrankaRobotiq2f85RelativeOSCAction = DroidFrankaRobotiq2f85RelativeOSCAction()

    def __post_init__(self):
        super().__post_init__()
        use_droid_franka_names(self)
