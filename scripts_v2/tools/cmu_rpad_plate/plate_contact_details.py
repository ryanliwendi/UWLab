"""Visual refinements only. Existing table collision box is preserved."""

import numpy as np
from pxr import UsdGeom, UsdShade, Gf, Vt


def refine_table(stage, box, mesh, material, tc, tx, ty):
    # Add the low crossbars of the two U-shaped table supports visible in the real view.
    for i, sign in enumerate([-1, 1]):
        box(
            "TableFootRail" + str(i),
            (float(tc[0] + sign * (tx / 2 - 0.01)), float(tc[1]), 0.005),
            (0.014, float(ty - 0.006), 0.01),
            (0.025, 0.028, 0.026),
        )
    # Thin dark seams/rounded-looking edge strips under the wood top; no enlarged collider.
    for i, sign in enumerate([-1, 1]):
        box(
            "TableEdge" + str(i),
            (float(tc[0]), float(tc[1] + sign * (ty / 2 - 0.001)), 0.0855),
            (float(tx - 0.004), 0.002, 0.002),
            (0.28, 0.17, 0.08),
        )


def refine_plate(stage, mesh, material, radius):
    from scipy.interpolate import PchipInterpolator

    # Smooth closed shell with normals, preserving the collider's measured profile within 1 mm.
    base = np.array(
        [[0, 0.003], [0.075, 0.003], [0.095, 0.005], [0.11, 0.011], [0.12, 0.018]]
    )
    spline = PchipInterpolator(base[:, 0], base[:, 1])
    rs = np.linspace(0, 0.12, 49)
    zs = spline(rs)
    der = spline.derivative()(rs)
    points = []
    normals = []
    faces = []
    n = 128
    nr = len(rs)
    for side in [0, 1]:
        for r, z, d in zip(rs, zs, der):
            for i in range(n):
                a = i * 2 * np.pi / n
                points.append(
                    [
                        r * radius / 0.12 * np.cos(a),
                        r * radius / 0.12 * np.sin(a),
                        z - side * 0.003,
                    ]
                )
                normal = np.array([-d * np.cos(a), -d * np.sin(a), 1.0])
                normal /= np.linalg.norm(normal)
                normals.append(normal * (1 if side == 0 else -1))
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
    v = mesh("Plate/Visual", points, faces, (0.43, 0.135, 0.14))
    v.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.array(normals, dtype="float32")))
    v.SetNormalsInterpolation("vertex")
    UsdShade.MaterialBindingAPI.Apply(v.GetPrim()).Bind(
        material("/World/Materials/PlateFinish", (0.43, 0.135, 0.14), 0.42)
    )
    return float(np.max(np.abs(zs - np.interp(rs, base[:, 0], base[:, 1]))))
