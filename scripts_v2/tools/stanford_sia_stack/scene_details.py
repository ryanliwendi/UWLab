"""Reference-pose visual details. Cables are not simulated deformable bodies."""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt


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
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(
            material("/World/Materials/Detail_" + name, color)
        )

    cable(
        "BasePower",
        [
            plane_point(*p, 0.012)
            for p in [(95, 344), (69, 351), (45, 352), (15, 366), (0, 368)]
        ],
        0.006,
    )
    cable(
        "BaseSignal",
        [
            plane_point(*p, 0.018)
            for p in [(95, 343), (68, 342), (45, 348), (10, 352), (0, 356)]
        ],
        0.003,
    )
    controller = plane_point(78, 343, 0.022)
    box("BaseInterface", controller, (0.075, 0.035, 0.018), (0.035, 0.045, 0.045))
    pcb = box(
        "BaseCircuit",
        controller + np.array([0, 0, 0.011]),
        (0.055, 0.027, 0.002),
        (0.015, 0.09, 0.055),
    )
    led = UsdGeom.Sphere.Define(stage, "/World/Detail_StatusLED")
    led.CreateRadiusAttr(0.009)
    led.AddTranslateOp().Set(Gf.Vec3d(*plane_point(109, 345, 0.03)))
    mat = material("/World/Materials/StatusLED", (0.01, 0.7, 0.4))
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Materials/StatusLED/Shader"))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.05, 3.0, 1.7)
    )
    UsdShade.MaterialBindingAPI.Apply(led.GetPrim()).Bind(mat)
    metal = material("/World/Materials/FingerMetal", (0.32, 0.34, 0.36), 0.32)
    for p in stage.Traverse():
        if str(p.GetPath()).startswith("/World/Robot/") and (
            "leftfinger" in p.GetName() or "rightfinger" in p.GetName()
        ):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                metal, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
