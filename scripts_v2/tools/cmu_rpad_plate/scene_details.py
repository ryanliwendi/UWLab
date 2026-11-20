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

    if scene == "stack":
        cable(
            "HandCable",
            [
                ray_point(p, 1, -0.07)
                for p in [
                    (126, 79),
                    (133, 111),
                    (142, 156),
                    (180, 174),
                    (230, 177),
                    (264, 183),
                    (263, 204),
                    (291, 215),
                    (303, 213),
                ]
            ],
            0.0025,
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
        shader = UsdShade.Shader(
            stage.GetPrimAtPath("/World/Materials/StatusLED/Shader")
        )
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
    elif scene == "drawer":
        panel = stage.GetPrimAtPath("/World/Panel3")
        UsdShade.MaterialBindingAPI.Apply(panel).Bind(
            material("/World/Materials/LeftPanel", (0.27, 0.30, 0.35))
        )
        metal = material("/World/Materials/UFactoryHousing", (0.23, 0.25, 0.27), 0.32)
        for p in stage.Traverse():
            if (
                str(p.GetPath()).startswith("/World/Robot/")
                and "xarm_gripper_base" in p.GetName()
            ):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    metal, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )
        for i, uv in enumerate([(87, 54), (36, 151), (537, 87), (644, 355)]):
            pos = plane_point(*uv, 0.019)
            box("RailBracket" + str(i), pos, (0.045, 0.028, 0.008), (0.11, 0.12, 0.14))
        cable(
            "RightLead",
            [
                plane_point(*p, 0.025)
                for p in [(651, 378), (644, 354), (638, 330), (631, 313)]
            ],
            0.004,
        )
    else:
        # Rear cables and fixture feet are fixed to the scene, not to robot joints.
        backy = plane_point(640, 216, 0)[1]
        for i, pixels in enumerate(
            [
                [(412, 0), (431, 56), (447, 107), (459, 151)],
                [(552, 0), (561, 74), (571, 145), (579, 210)],
            ]
        ):
            cable(
                "RearCable" + str(i),
                [ray_point(p, 1, backy + 0.006) for p in pixels],
                0.003,
            )
        for i, uv in enumerate([(1048, 585), (1126, 681), (930, 672)]):
            box(
                "SideFixture" + str(i),
                plane_point(*uv, 0.03),
                (0.055, 0.065, 0.04),
                (0.018, 0.022, 0.026),
            )
        for i, uv in enumerate([(331, 210), (865, 216), (295, 660), (977, 669)]):
            box(
                "FrameFastener" + str(i),
                plane_point(*uv, 0.022),
                (0.038, 0.025, 0.01),
                (0.17, 0.18, 0.19),
            )
        for i, uv in enumerate([(1091, 470), (105, 440)]):
            c = plane_point(*uv, 0.055)
            cable(
                "ArmLoom" + str(i),
                [
                    c + np.array(v)
                    for v in [
                        (0, 0, 0),
                        (0.01, 0.025, 0.05),
                        (0.03, 0.04, 0.10),
                        (0.07, 0.04, 0.16),
                    ]
                ],
                0.012,
            )
