# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...ur5e_robotiq_2f85.agents.rsl_rl_cfg import Base_PPORunnerCfg
from ..constants import EXPERIMENT_NAME


@configclass
class DroidFranka_PPORunnerCfg(Base_PPORunnerCfg):
    """Same PPO settings as the UR5e; separate log directory."""

    experiment_name = EXPERIMENT_NAME
