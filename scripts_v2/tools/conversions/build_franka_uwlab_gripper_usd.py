# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the DROID Franka arm with UWLab's calibrated Robotiq 2F-85 gripper (the gripper the UR5e uses).

The DROID USD's own Robotiq model is switched off and UWLab's gripper is mounted at exactly the same place on
``panda_link8``. Both grippers use the same base-frame convention (fingers along +x, closing along y), so the DROID
mounting joint is reused as is. The result references the two source USDs instead of copying them.

Needs the DROID USD in ``source/uwlab_assets/data/Robots/DroidFrankaRobotiq2f85/`` (see
``uwlab_assets.robots.droid_franka_robotiq``); downloads UWLab's gripper USD next to the output.

Usage:
    python scripts_v2/tools/conversions/build_franka_uwlab_gripper_usd.py
"""

import os
import shutil
import urllib.request

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from uwlab_assets import UWLAB_ASSETS_DATA_DIR, UWLAB_CLOUD_ASSETS_DIR

DROID_DIR = os.path.join(UWLAB_ASSETS_DATA_DIR, "Robots", "DroidFrankaRobotiq2f85")
DROID_USD = os.path.join(DROID_DIR, "franka_robotiq_2f_85_flattened.usd")
OUT_DIR = os.path.join(UWLAB_ASSETS_DATA_DIR, "Robots", "DroidFrankaUWLabRobotiq2f85")
OUT_USD = os.path.join(OUT_DIR, "franka_uwlab_robotiq_2f85.usd")
GRIPPER_FILE = "robotiq_2f85_gripper_calibrated.usd"
GRIPPER_URL = f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/2f85RobotiqGripperCalibrated/{GRIPPER_FILE}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gripper_usd = os.path.join(OUT_DIR, GRIPPER_FILE)
    if not os.path.exists(gripper_usd):
        print(f"Downloading {GRIPPER_URL}")
        urllib.request.urlretrieve(GRIPPER_URL, gripper_usd)
    # same gripper metadata (fingertip offset, approach axis) as the DROID gripper folder
    shutil.copy(os.path.join(DROID_DIR, "metadata.yaml"), os.path.join(OUT_DIR, "metadata.yaml"))

    droid = Usd.Stage.Open(DROID_USD)
    hand_joint = UsdPhysics.Joint(droid.GetPrimAtPath("/panda/panda_link8/panda_hand_joint"))
    droid_base_w = UsdGeom.XformCache().GetLocalToWorldTransform(
        droid.GetPrimAtPath("/panda/Gripper/Robotiq_2F_85/base_link")
    )

    uw = Usd.Stage.Open(gripper_usd)
    uw_root = uw.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    # robotiq_base_link pose relative to the gripper's default prim (USD matrices use row vectors)
    uw_base_in_root = (
        cache.GetLocalToWorldTransform(uw.GetPrimAtPath(f"{uw_root.GetPath()}/robotiq_base_link"))
        * cache.GetLocalToWorldTransform(uw_root).GetInverse()
    )

    if os.path.exists(OUT_USD):
        os.remove(OUT_USD)
    stage = Usd.Stage.CreateNew(OUT_USD)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = stage.DefinePrim("/panda", "Xform")
    robot.GetReferences().AddReference(os.path.relpath(DROID_USD, OUT_DIR))
    stage.SetDefaultPrim(robot)

    # switch off the DROID gripper and its mounting joint
    stage.GetPrimAtPath("/panda/Gripper").SetActive(False)
    stage.GetPrimAtPath("/panda/panda_link8/panda_hand_joint").SetActive(False)

    # add UWLab's gripper so that its robotiq_base_link starts where the DROID base_link was
    gripper = stage.DefinePrim("/panda/UWLabGripper", "Xform")
    gripper.GetReferences().AddReference(f"./{GRIPPER_FILE}")
    xform = UsdGeom.Xformable(gripper)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(uw_base_in_root.GetInverse() * droid_base_w)
    # its fixed joint to the world (and articulation root) is only for the stand-alone gripper
    stage.GetPrimAtPath("/panda/UWLabGripper/root_joint").SetActive(False)

    # mount it on the arm flange with the same joint frames as the DROID gripper
    mount = UsdPhysics.FixedJoint.Define(stage, "/panda/panda_link8/uwlab_gripper_joint")
    mount.CreateBody0Rel().SetTargets(["/panda/panda_link8"])
    mount.CreateBody1Rel().SetTargets(["/panda/UWLabGripper/robotiq_base_link"])
    mount.CreateLocalPos0Attr().Set(hand_joint.GetLocalPos0Attr().Get())
    mount.CreateLocalRot0Attr().Set(hand_joint.GetLocalRot0Attr().Get())
    mount.CreateLocalPos1Attr().Set(hand_joint.GetLocalPos1Attr().Get())
    mount.CreateLocalRot1Attr().Set(hand_joint.GetLocalRot1Attr().Get())

    stage.GetRootLayer().Save()
    print(f"Saved {OUT_USD}")

    # quick check: one articulation root, and the new gripper base sits where the DROID one was
    check = Usd.Stage.Open(OUT_USD)
    roots = [str(p.GetPath()) for p in check.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    base_w = UsdGeom.XformCache().GetLocalToWorldTransform(check.GetPrimAtPath("/panda/UWLabGripper/robotiq_base_link"))
    print(f"Articulation roots: {roots}")
    print(
        "Gripper base position error vs DROID:"
        f" {(base_w.ExtractTranslation() - droid_base_w.ExtractTranslation()).GetLength():.2e} m"
    )
    rot_err = Gf.Rotation(base_w.ExtractRotationQuat() * droid_base_w.ExtractRotationQuat().GetInverse()).GetAngle()
    print(f"Gripper base rotation error vs DROID: {rot_err:.2e} deg")


if __name__ == "__main__":
    main()
