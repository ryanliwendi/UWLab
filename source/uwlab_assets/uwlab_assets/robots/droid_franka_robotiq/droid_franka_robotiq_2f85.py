# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DROID robot: Franka Panda arm + Robotiq 2F-85 gripper.

The USD is ``franka_robotiq_2f_85_flattened.usd`` from the MIT-licensed Hugging Face dataset
``owhan/DROID-sim-environments`` (the robot model used by the DROID simulation evaluations).
It is not tracked in git; download it into :obj:`DROID_FRANKA_ROBOTIQ_2F85_DIR` with::

    curl -L -o source/uwlab_assets/data/Robots/DroidFrankaRobotiq2f85/franka_robotiq_2f_85_flattened.usd \\
        https://huggingface.co/datasets/owhan/DROID-sim-environments/resolve/main/franka_robotiq_2f_85_flattened.usd

The robot is spawned with :func:`spawn_droid_franka_robotiq_2f85`, which stiffens the Robotiq fingertip joints
(see :obj:`STIFF_FINGERTIP_MIMIC_JOINTS`).

The following configurations are available:

* :obj:`DROID_FRANKA_ARTICULATION`: Base articulation (USD, init state).
* :obj:`IMPLICIT_DROID_FRANKA_ROBOTIQ_2F85`: Full robot with torque-controlled ImplicitActuator arm (for RL training).
* :obj:`IMPLICIT_DROID_FRANKA_UWLAB_ROBOTIQ_2F85`: The same arm with UWLab's calibrated Robotiq 2F-85 (the UR5e's
  gripper) instead of the DROID one. Build its USD once with
  ``scripts_v2/tools/conversions/build_franka_uwlab_gripper_usd.py``.
"""

import os

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.utils import clone

from uwlab_assets import UWLAB_ASSETS_DATA_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import ROBOTIQ_2F85, ROBOTIQ_2F85_DEFAULT_JOINT_POS

DROID_FRANKA_ROBOTIQ_2F85_DIR = os.path.join(UWLAB_ASSETS_DATA_DIR, "Robots", "DroidFrankaRobotiq2f85")
DROID_FRANKA_UWLAB_ROBOTIQ_2F85_DIR = os.path.join(UWLAB_ASSETS_DATA_DIR, "Robots", "DroidFrankaUWLabRobotiq2f85")

# In the USD, the four fingertip joints follow ``finger_joint`` through soft PhysX mimic joints (natural frequency
# 1000, damping ratio 0.05). Under grip force the fingertips flex 5-8 deg, so grasped cubes get squeezed out or
# wobble ~1-3 deg in the hand. We make them as stiff as the USD's own outer knuckle mimic joint so that the pads stay
# parallel like the real linkage.
STIFF_FINGERTIP_MIMIC_JOINTS = {
    "right_inner_finger_joint": "rotX",
    "left_inner_finger_joint": "rotX",
    "right_inner_finger_knuckle_joint": "rotX",
    "left_inner_finger_knuckle_joint": "rotX",
}
FINGERTIP_MIMIC_NATURAL_FREQUENCY = 1.0e6
FINGERTIP_MIMIC_DAMPING_RATIO = 0.0


@clone
def spawn_droid_franka_robotiq_2f85(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the DROID USD, then stiffen the fingertip mimic joints (see :obj:`STIFF_FINGERTIP_MIMIC_JOINTS`)."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for joint_prim in Usd.PrimRange(prim):
        axis = STIFF_FINGERTIP_MIMIC_JOINTS.get(joint_prim.GetName())
        if axis is None:
            continue
        joint_prim.GetAttribute(f"physxMimicJoint:{axis}:naturalFrequency").Set(FINGERTIP_MIMIC_NATURAL_FREQUENCY)
        joint_prim.GetAttribute(f"physxMimicJoint:{axis}:dampingRatio").Set(FINGERTIP_MIMIC_DAMPING_RATIO)
    return prim


# In the DROID USD, the five passive finger joints mimic ``finger_joint`` (PhysX mimic joints).
DROID_ROBOTIQ_2F85_DEFAULT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
}

# DROID home pose (same as the DROID simulation evaluations)
DROID_FRANKA_ARM_DEFAULT_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.6283,  # -pi / 5
    "panda_joint3": 0.0,
    "panda_joint4": -2.5133,  # -4 pi / 5
    "panda_joint5": 0.0,
    "panda_joint6": 1.8850,  # 3 pi / 5
    "panda_joint7": 0.0,
}

DROID_FRANKA_DEFAULT_JOINT_POS = {**DROID_FRANKA_ARM_DEFAULT_JOINT_POS, **DROID_ROBOTIQ_2F85_DEFAULT_JOINT_POS}

# Franka Panda datasheet limits
FRANKA_VELOCITY_LIMITS = {
    "panda_joint[1-4]": 2.175,
    "panda_joint[5-7]": 2.61,
}

FRANKA_EFFORT_LIMITS = {
    "panda_joint[1-4]": 87.0,
    "panda_joint[5-7]": 12.0,
}

DROID_FRANKA_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_droid_franka_robotiq_2f85,
        usd_path=os.path.join(DROID_FRANKA_ROBOTIQ_2F85_DIR, "franka_robotiq_2f_85_flattened.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=DROID_FRANKA_DEFAULT_JOINT_POS
    ),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_DROID_FRANKA_ROBOTIQ_2F85 = DROID_FRANKA_ARTICULATION.copy()  # type: ignore
IMPLICIT_DROID_FRANKA_ROBOTIQ_2F85.actuators = {
    # zero stiffness / damping: the OSC action applies joint torques directly.
    # The USD has no armature; 0.1 (reflected motor inertia, as in common Franka models) keeps the light
    # wrist joints stable under the OSC's explicit damping.
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        stiffness=0.0,
        damping=0.0,
        armature=0.1,
        effort_limit_sim=FRANKA_EFFORT_LIMITS,
        velocity_limit_sim=FRANKA_VELOCITY_LIMITS,
    ),
    # UWLab's Robotiq 2F-85 gains. The USD drive gains (stiffness ~5700, almost no damping) make the
    # fingers chatter and close very slowly.
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        stiffness=17,
        damping=5,
        effort_limit_sim=60,
    ),
}

# The same arm with UWLab's calibrated Robotiq 2F-85 mounted where the DROID gripper was (the UR5e's gripper, so
# grasps sampled with it fit as is). Its USD is built by scripts_v2/tools/conversions/build_franka_uwlab_gripper_usd.py.
DROID_FRANKA_UWLAB_ROBOTIQ_2F85_ARTICULATION = DROID_FRANKA_ARTICULATION.replace(
    spawn=DROID_FRANKA_ARTICULATION.spawn.replace(
        func=sim_utils.spawn_from_usd,
        usd_path=os.path.join(DROID_FRANKA_UWLAB_ROBOTIQ_2F85_DIR, "franka_uwlab_robotiq_2f85.usd"),
    ),
    init_state=DROID_FRANKA_ARTICULATION.init_state.replace(
        joint_pos={**DROID_FRANKA_ARM_DEFAULT_JOINT_POS, **ROBOTIQ_2F85_DEFAULT_JOINT_POS}
    ),
)

IMPLICIT_DROID_FRANKA_UWLAB_ROBOTIQ_2F85 = DROID_FRANKA_UWLAB_ROBOTIQ_2F85_ARTICULATION.copy()  # type: ignore
IMPLICIT_DROID_FRANKA_UWLAB_ROBOTIQ_2F85.actuators = {
    "arm": IMPLICIT_DROID_FRANKA_ROBOTIQ_2F85.actuators["arm"],
    "gripper": ROBOTIQ_2F85.actuators["gripper"],
}
