# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniReset tasks for the DROID robot (Franka Panda + Robotiq 2F-85).

Partial assemblies (``OmniReset-PartialAssemblies-v0``) do not involve the robot and are shared with the UR5e.
"""

import gymnasium as gym

from . import agents

# Register reset states environments
gym.register(
    id="OmniReset-DroidFrankaRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:DroidFrankaObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-DroidFrankaRobotiq2f85-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:DroidFrankaObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-DroidFrankaRobotiq2f85-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:DroidFrankaObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-DroidFrankaRobotiq2f85-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.reset_states_cfg:DroidFrankaObjectPartiallyAssembledEEAnywhereResetStatesCfg"
        )
    },
)

gym.register(
    id="OmniReset-DroidFrankaRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_states_cfg:DroidFrankaObjectPartiallyAssembledEEGraspedResetStatesCfg"
    },
)

# Register RL state environments
gym.register(
    id="OmniReset-DroidFrankaRobotiq2f85-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:DroidFrankaRobotiq2f85RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:DroidFranka_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-DroidFrankaRobotiq2f85-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:DroidFrankaRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:DroidFranka_PPORunnerCfg",
    },
)
