"""Visual-only molded plate shell. Does not create or modify collision geometry."""

import numpy as np
from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt
from scipy.interpolate import PchipInterpolator


def refine_plate(stage, mesh, material, radius):
    base = np.array(
        [[0, 0.003], [0.075, 0.003], [0.095, 0.005], [0.11, 0.011], [0.12, 0.018]]
    )
    curve = PchipInterpolator(base[:, 0], base[:, 1])
    rr = np.linspace(0, 0.12, 97)
    zs = curve(rr)
    slopes = curve.derivative()(rr) * 0.12 / radius
    n = 192
    nr = len(rr)
    ps = []
    ns = []
    colors = []
    faces = []
    # Subtle pigment variation distinguishes well, shoulder and rolled rim.
    stops = np.array([0, 0.070, 0.077, 0.083, 0.095, 0.110, 0.117, 0.120])
    cols = np.array(
        [
            [0.43, 0.139, 0.139],
            [0.43, 0.139, 0.139],
            [0.40, 0.119, 0.120],
            [0.44, 0.148, 0.145],
            [0.455, 0.151, 0.145],
            [0.43, 0.128, 0.125],
            [0.49, 0.172, 0.163],
            [0.47, 0.154, 0.147],
        ]
    )
    for side in [0, 1]:
        for r, z, slope in zip(rr, zs, slopes):
            col = np.array([np.interp(r, stops, cols[:, j]) for j in range(3)])
            for a in np.arange(n) * 2 * np.pi / n:
                ps.append(
                    [
                        r * radius / 0.12 * np.cos(a),
                        r * radius / 0.12 * np.sin(a),
                        z - side * 0.003,
                    ]
                )
                normal = np.array([-slope * np.cos(a), -slope * np.sin(a), 1])
                normal /= np.linalg.norm(normal)
                ns.append(normal * (1 if side == 0 else -1))
                colors.append(col * (1 if side == 0 else 0.94))
    for side in [0, 1]:
        for j in range(nr - 1):
            for i in range(n):
                a = side * nr * n + j * n + i
                b = side * nr * n + j * n + (i + 1) % n
                c = b + n
                d = a + n
                faces.append([d, c, b, a] if side == 0 else [a, b, c, d])
    for i in range(n):
        a = (nr - 1) * n + i
        b = (nr - 1) * n + (i + 1) % n
        faces.append([a, a + nr * n, b + nr * n, b])
    stage.RemovePrim("/World/Plate/Visual")
    v = mesh("Plate/Visual", ps, faces, (0.43, 0.139, 0.139))
    v.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.array(ns, dtype="float32")))
    v.SetNormalsInterpolation("vertex")
    UsdGeom.PrimvarsAPI(v).CreatePrimvar(
        "plateColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec3fArray.FromNumpy(np.asarray(colors, dtype="float32")))
    mat = material("/World/Materials/PlateDetailFinish", (0.43, 0.139, 0.139), 0.34)
    shader = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Materials/PlateDetailFinish/Shader")
    )
    reader = UsdShade.Shader.Define(stage, "/World/Materials/PlateDetailFinish/Color")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("plateColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
    UsdShade.MaterialBindingAPI.Apply(v.GetPrim()).Bind(mat)
    # Low molded rings and rolled lip are genuine geometry, under 0.6 mm high.
    for k, (r, tube) in enumerate(
        [(0.078, 0.00022), (0.087, 0.00018), (0.112, 0.00024), (0.1195, 0.00045)]
    ):
        pts = []
        fs = []
        for a in np.arange(n) * 2 * np.pi / n:
            for b in np.arange(12) * 2 * np.pi / 12:
                rad = r * radius / 0.12 + tube * np.cos(b)
                pts.append(
                    [
                        rad * np.cos(a),
                        rad * np.sin(a),
                        float(curve(r)) + 0.00008 + tube * np.sin(b),
                    ]
                )
        for i in range(n):
            for j in range(12):
                fs.append(
                    [
                        i * 12 + j,
                        ((i + 1) % n) * 12 + j,
                        ((i + 1) % n) * 12 + (j + 1) % 12,
                        i * 12 + (j + 1) % 12,
                    ]
                )
        bead = mesh("Plate/MoldedRing" + str(k), pts, fs, (0.46, 0.152, 0.147))
        UsdShade.MaterialBindingAPI.Apply(bead.GetPrim()).Bind(
            material("/World/Materials/PlateRing" + str(k), (0.46, 0.152, 0.147), 0.32)
        )
    return float(np.max(np.abs(zs - np.interp(rr, base[:, 0], base[:, 1]))))
