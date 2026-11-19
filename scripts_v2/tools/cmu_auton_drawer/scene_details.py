# Copyright (c) 2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-pose visual details. Cables are not simulated deformable bodies."""
import numpy as np
from scipy.spatial.transform import Rotation

from pxr import UsdGeom, UsdShade, Vt


def add_details(stage, scene, campos, camrot, K, box, mesh, material, plane_point):
    R = Rotation.from_quat(camrot).as_matrix()

    def ray_point(uv, axis, value):
        ray = R @ np.linalg.inv(K) @ np.array([*uv, 1.0])
        return campos + (value - campos[axis]) / ray[axis] * ray

    def cable(name, points, width=0.002, color=(0.008, 0.009, 0.012)):
        c = UsdGeom.BasisCurves.Define(stage, "/World/Detail_" + name)
        c.CreateTypeAttr("cubic")
        c.CreateBasisAttr("bspline")
        c.CreateWrapAttr("nonperiodic")
        pp = [points[0], *points, points[-1]]
        c.CreateCurveVertexCountsAttr([len(pp)])
        c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(pp, dtype="float32")))
        c.CreateWidthsAttr([width])
        c.SetWidthsInterpolation("constant")
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material("/World/Materials/Detail_" + name, color))

    panel = stage.GetPrimAtPath("/World/Panel3")
    UsdShade.MaterialBindingAPI.Apply(panel).Bind(material("/World/Materials/LeftPanel", (0.27, 0.3, 0.35)))
    metal = material("/World/Materials/UFactoryHousing", (0.23, 0.25, 0.27), 0.32)
    for p in stage.Traverse():
        if str(p.GetPath()).startswith("/World/Robot/") and "xarm_gripper_base" in p.GetName():
            UsdShade.MaterialBindingAPI.Apply(p).Bind(metal, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    for i, uv in enumerate([(87, 54), (36, 151), (537, 87), (644, 355)]):
        pos = plane_point(*uv, 0.019)
        box("RailBracket" + str(i), pos, (0.045, 0.028, 0.008), (0.11, 0.12, 0.14))
    cable("RightLead", [plane_point(*p, 0.025) for p in [(651, 378), (644, 354), (638, 330), (631, 313)]], 0.004)
