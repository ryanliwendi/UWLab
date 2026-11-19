# Copyright (c) 2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-only chamfers and handle brackets; original collision geometry retained."""
import itertools
import numpy as np
from scipy.spatial import ConvexHull

from pxr import Gf, UsdGeom, UsdShade, Vt


def refine(stage, material):
    names = [
        "HousingTop",
        "HousingBack",
        "HousingSide0",
        "HousingSide1",
        "DrawerBottom",
        "DrawerFront",
        "DrawerSide0",
        "DrawerSide1",
        "Handle",
    ]
    for name in names:
        parent = "/World" if name.startswith("Housing") else "/World/SlidingDrawer"
        p = stage.GetPrimAtPath(parent + "/" + name)
        cube = UsdGeom.Cube(p)
        ops = UsdGeom.Xformable(p).GetOrderedXformOps()
        scale = np.array(next((o.Get() for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeScale)))
        bevel = np.minimum((0.0025 if name == "Handle" else 0.001) / scale, 0.28)
        pts = []
        for signs in itertools.product([-1, 1], repeat=3):
            for axis in range(3):
                v = 0.5 - bevel
                v[axis] = 0.5
                pts.append(v * np.array(signs))
        pts = np.array(pts)
        hull = ConvexHull(pts)
        faces = []
        for f, eq in zip(hull.simplices, hull.equations):
            f = list(f)
            if np.dot(np.cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]]), eq[:3]) < 0:
                f.reverse()
            faces.extend(f)
        vis = UsdGeom.Mesh.Define(stage, parent + "/Detail_" + name)
        vis.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype("float32")))
        vis.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
        vis.CreateFaceVertexIndicesAttr([int(i) for i in faces])
        vis.CreateSubdivisionSchemeAttr("none")
        vis.CreateDoubleSidedAttr(True)
        vis.AddTransformOp().Set(UsdGeom.Xformable(p).GetLocalTransformation())
        binding = (
            material("/World/Materials/FittedHandle", (0.12, 0.003, 0.015), 0.9)
            if name == "Handle"
            else UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
        )
        UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(binding)
        cube.CreateVisibilityAttr("invisible")
    handle = UsdGeom.Xformable(stage.GetPrimAtPath("/World/SlidingDrawer/Handle"))
    front = UsdGeom.Xformable(stage.GetPrimAtPath("/World/SlidingDrawer/DrawerFront"))
    hp = np.array(handle.GetOrderedXformOps()[0].Get())
    fp = np.array(front.GetOrderedXformOps()[0].Get())
    hs = np.array(handle.GetOrderedXformOps()[2].Get())
    yaw = float(handle.GetOrderedXformOps()[1].Get())
    angle = np.deg2rad(yaw)
    ey = np.array([-np.sin(angle), np.cos(angle), 0])
    ex = np.array([np.cos(angle), np.sin(angle), 0])
    front_anchor = fp + ey * np.dot(hp - fp, ey)
    center = (hp + front_anchor) / 2
    center[2] = hp[2]
    for i, sign in enumerate([-1, 1]):
        b = UsdGeom.Cube.Define(stage, f"/World/SlidingDrawer/HandleBracket{i}")
        b.CreateSizeAttr(1)
        b.AddTranslateOp().Set(Gf.Vec3d(*center + sign * ey * (hs[1] / 2 - (0.022 if sign > 0 else 0.011))))
        b.AddRotateZOp().Set(yaw)
        b.AddScaleOp().Set(Gf.Vec3f(float(abs(np.dot(hp - front_anchor, ex))), 0.0045, 0.005))
        UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(
            material("/World/Materials/HandleBracket" + str(i), (0.12, 0.003, 0.015), 0.85)
        )
