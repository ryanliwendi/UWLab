"""Isolated Isaac Lab visual-fitting scenes. No policies or task training.

Objects/backgrounds are explicit 3-D geometry, not a pasted camera image.
This first pass is visual calibration, NOT validated task physics.
"""

import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--scene", choices=["stack"], default="stack")
p.add_argument("--output", required=True)
p.add_argument("--view", choices=["front", "back", "wrist"], default="front")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation
from PIL import Image
from pxr import UsdGeom, UsdShade, UsdPhysics, Gf, Sdf, Vt
import isaaclab.sim as sim
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg

ROOT = Path(__file__).resolve().parent
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((ROOT / "references" / args.scene / "metadata.json").read_text())
names = {"stack": "fr3"}
model = names[args.scene]
converter = UrdfConverter(
    UrdfConverterCfg(
        asset_path=str(ROOT / "assets/urdf" / f"{model}.urdf"),
        usd_dir=str(ROOT / "assets/usd" / model),
        usd_file_name=f"{model}.usd",
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0, damping=10.0
            )
        ),
    )
)
ctx = sim.SimulationContext(sim.SimulationCfg(dt=1 / 120, device=args.device))
stage = ctx.stage


def material(path, color, rough=0.7, texture=None):
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/Shader")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/ST")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        t = UsdShade.Shader.Define(stage, path + "/Texture")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st.ConnectableAPI(), "result"
        )
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        s.GetInput("diffuseColor").ConnectToSource(t.ConnectableAPI(), "rgb")
    return m


def box(name, pos, size, color, yaw=0):
    c = UsdGeom.Cube.Define(stage, "/World/" + name)
    c.CreateSizeAttr(1)
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.AddRotateZOp().Set(yaw)
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(
        material("/World/Materials/" + name, color)
    )
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def mesh(name, points, faces, color, uv=None, texture=None):
    m = UsdGeom.Mesh.Define(stage, "/World/" + name)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(points, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([x for f in faces for x in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    if uv is not None:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(uv)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
        material("/World/Materials/" + name, color, texture=texture)
    )
    return m


base = (0, 0, 0)
quat = (1, 0, 0, 0)
box(
    "WoodBench",
    (0.43327369, 0.07428705, -0.025),
    (1.22214903, 0.65380458, 0.05),
    (0.6, 0.43, 0.25),
    8.11265369,
)
box("GreenCube", (0.48, 0.01, 0.0125), (0.025, 0.025, 0.025), (0.025, 0.33, 0.08))
box("RedCube", (0.59, -0.045, 0.0125), (0.025, 0.025, 0.025), (0.65, 0.015, 0.025))
rvec = np.array([1.752136704, 0.572361289, -0.287057363])
t = np.array([-0.455136929, 0.263261682, 1.34844733])
R = Rotation.from_rotvec(rvec).as_matrix()
campos = -R.T @ t
camrot = Rotation.from_matrix(R.T).as_quat()
K = np.array(meta["robot"]["intrinsic_calibration_matrix"]["side_cam"])
qdict = dict(
    zip(
        meta["robot"]["robot_state_joint_names"],
        meta["states"]["joint_position"]["first"],
    )
)
qdict.update({"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04})


def plane_point(u, v, z=0):
    Rcw = Rotation.from_quat(camrot).as_matrix()
    ray = Rcw @ np.linalg.inv(K) @ np.array([u, v, 1.0])
    return campos + (z - campos[2]) / ray[2] * ray


def pose_box(name, pos, size=None, yaw=None):
    prim = stage.GetPrimAtPath("/World/" + name)
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*map(float, pos)))
        elif size is not None and op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3f(*map(float, size)))
        elif yaw is not None and op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
            op.Set(float(yaw))


for name, uv in [("GreenCube", (345, 408)), ("RedCube", (405, 432))]:
    pose_box(name, plane_point(*uv, 0.0125))
texdir = ROOT / "assets/textures"
texdir.mkdir(parents=True, exist_ok=True)


def texture_patch(label, ref, rect):
    im = cv2.imread(str(ROOT / "references" / ref))
    x0, y0, x1, y1 = rect
    path = texdir / (label + ".png")
    patch = im[y0:y1, x0:x1]
    patch = (
        np.concatenate([patch, patch[:, ::-1]], axis=1) if "wood" in label else patch
    )
    patch = np.concatenate([patch, patch[::-1]], axis=0) if "wood" in label else patch
    patch = (
        np.clip(patch.astype(float) * 1.16, 0, 255).astype("uint8")
        if label == "stack-cloth"
        else patch
    )
    cv2.imwrite(str(path), patch)
    return path


wood = texture_patch(
    "stack-wood-grain-v2", "stack/top_cam-0000.png", (8, 164, 105, 308)
)
benchR = Rotation.from_euler("z", 8.11265369, degrees=True).as_matrix()
benchP = np.array(
    [
        [-0.611074515, -0.32690229, 0.0001],
        [0.611074515, -0.32690229, 0.0001],
        [0.611074515, 0.32690229, 0.0001],
        [-0.611074515, 0.32690229, 0.0001],
    ]
) @ benchR.T + np.array([0.43327369, 0.07428705, 0])
mesh(
    "BenchGrain",
    benchP,
    [[0, 1, 2, 3]],
    (1, 1, 1),
    [(0, 0), (0, 5), (3, 5), (3, 0)],
    wood,
)
cloth = texture_patch(
    "stack-cloth-continuous", "stack/side_cam-0000.png", (5, 20, 55, 170)
)
tile = cv2.imread(str(cloth))
tile = np.concatenate([tile, tile[:, ::-1]], axis=1)
tile = np.concatenate([tile, tile[::-1]], axis=0)
cv2.imwrite(str(cloth), tile)
path = []
for yy in np.linspace(-0.85, 0.47, 61):
    path.append([-0.3, float(yy)])
for angle in np.linspace(np.pi, np.pi / 2, 21)[1:]:
    path.append([-0.22 + 0.08 * np.cos(angle), 0.47 + 0.08 * np.sin(angle)])
for xx in np.linspace(-0.22, 2.5, 111)[1:]:
    path.append([float(xx), 0.55])
path = np.array(path)
tangent = np.gradient(path, axis=0)
tangent /= np.linalg.norm(tangent, axis=1)[:, None]
normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
nz = 49
nx = len(path)
points = []
uv = []
faces = []
for iz, zz in enumerate(np.linspace(-0.2, 1.3, nz)):
    for ix, xy in enumerate(path):
        dd = distance[ix]
        fold = 0.004 * np.sin(2 * np.pi * dd / 0.31 + 0.4 * zz) + 0.002 * np.sin(
            2 * np.pi * dd / 0.73 - 1.4 * zz
        )
        pos = xy + normal[ix] * fold
        points.append([*pos, zz])
        uv.append([dd / 0.9, (zz + 0.2) / 1.5])
for iz in range(nz - 1):
    for ix in range(nx - 1):
        a = iz * nx + ix
        faces.append([a, a + 1, a + nx + 1, a + nx])
curtain = mesh("GreenCurtain", points, faces, (1, 1, 1), uv, cloth)
pp = np.array(points).reshape(nz, nx, 3)
nn = np.cross(np.gradient(pp, axis=1), np.gradient(pp, axis=0))
nn /= np.linalg.norm(nn, axis=2, keepdims=True)
curtain.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nn.reshape(-1, 3).astype("float32")))
curtain.SetNormalsInterpolation("vertex")
cyl = UsdGeom.Cylinder.Define(stage, "/World/BaseMount")
cyl.CreateRadiusAttr(0.084)
cyl.CreateHeightAttr(0.009)
cyl.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.0045))
UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(
    material("/World/Materials/Mount", (0.008, 0.01, 0.014))
)
robot = Articulation(
    ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim.UsdFileCfg(
            usd_path=converter.usd_path,
            rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=base, rot=quat, joint_pos=qdict),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-7]"], stiffness=2000.0, damping=90.0
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                stiffness=2000.0,
                damping=124.0,
                effort_limit_sim=35.0,
            ),
        },
    )
)
from scene_details import add_details

add_details(stage, args.scene, campos, camrot, K, box, mesh, material, plane_point)
sim.DomeLightCfg(intensity=300, color=(0.92, 0.95, 1.0)).func(
    "/World/Dome", sim.DomeLightCfg(intensity=300, color=(0.92, 0.95, 1.0))
)
light = sim.DistantLightCfg(intensity=650, color=(1.0, 0.94, 0.84), angle=12.0)
light.func("/World/Sun", light, orientation=(0.92388, 0.38268, 0, 0))
layout = json.loads((ROOT / "config/grid-layout.json").read_text())
for board in layout["boards"]:
    mesh(
        "Board" + str(board["index"]),
        board["world_corners"],
        [[0, 1, 2], [0, 2, 3]],
        (1, 1, 1),
        [(0, 1), (1, 1), (1, 0), (0, 0)],
        ROOT / "assets/textures" / ("board-" + str(board["index"]) + "-6x8.png"),
    )
placements = json.loads((ROOT / "config/cube-placement.json").read_text())
for name, placement in placements.items():
    pose_box(name, placement["center_world"], yaw=placement["yaw_degrees"])
if os.environ.get("STACK_VIEW") == "top":
    cf = layout["top_camera_fit"]
    campos = np.array(cf["position"])
    camrot = Rotation.from_matrix(np.array(cf["rotation"])).as_quat()
    K = np.array(cf["K"])
refcam = "side_cam"
width = meta["cameras"][refcam]["width"]
height = meta["cameras"][refcam]["height"]
rw = width + 64
rh = height + 64
favg = float((K[0, 0] + K[1, 1]) / 2)
renderK = np.array([[favg, 0, rw / 2], [0, favg, rh / 2], [0, 0, 1.0]])
camera = Camera(
    CameraCfg(
        prim_path="/World/Camera",
        width=rw,
        height=rh,
        data_types=["rgb"],
        spawn=sim.PinholeCameraCfg.from_intrinsic_matrix(
            renderK.flatten().tolist(),
            width=rw,
            height=rh,
            focal_length=24.0,
            clipping_range=(0.01, 100.0),
        ),
    )
)
ctx.reset()
robot.reset()
q = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(q, torch.zeros_like(q))
robot.set_joint_position_target(q)
camera.set_world_poses(
    torch.tensor(campos[None], dtype=torch.float32, device=args.device),
    torch.tensor([[camrot[3], *camrot[:3]]], dtype=torch.float32, device=args.device),
    convention="ros",
)
for _ in range(30):
    ctx.render()
camera.update(0, force_recompute=True)
im = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
Image.fromarray(im).save(out / "render-overscan.png")
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
mx = ((xx - K[0, 2]) * favg / K[0, 0] + rw / 2).astype("float32")
my = ((yy - K[1, 2]) * favg / K[1, 1] + rh / 2).astype("float32")
im = cv2.remap(im, mx, my, cv2.INTER_LINEAR)
Image.fromarray(im).save(out / "render.png")
stage.GetRootLayer().Export(str(out / "scene.usda"))
report = {
    "status": "VISUAL_PROTOTYPE_NOT_CALIBRATED",
    "joint_names": robot.joint_names,
    "camera_position": campos.tolist(),
    "K": K.tolist(),
}
(out / "report.json").write_text(json.dumps(report, indent=2))
print("RENDER_COMPLETE", out, flush=True)
sys.stdout.flush()
os._exit(0)
