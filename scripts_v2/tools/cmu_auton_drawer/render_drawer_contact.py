# Copyright (c) 2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated Isaac Lab visual-fitting scenes. No policies or task training.

Objects/backgrounds are explicit 3-D geometry, not a pasted camera image.
This first pass is visual calibration, NOT validated task physics.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--scene", choices=["drawer"], default="drawer")
p.add_argument("--output", required=True)
p.add_argument("--preview-only", action="store_true")
p.add_argument("--no-robot-contact", action="store_true")
p.add_argument("--rail-friction", type=float, default=0.6)
p.add_argument("--rail-damping", type=float, default=2.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import cv2
import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from PIL import Image
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parent
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((ROOT / "references" / args.scene / "metadata.json").read_text())
names = {"stack": "fr3", "drawer": "xarm7", "plate": "vx300s_aloha"}
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
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0)
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
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        s.GetInput("diffuseColor").ConnectToSource(t.ConnectableAPI(), "rgb")
    return m


def box(name, pos, size, color, yaw=0):
    c = UsdGeom.Cube.Define(stage, "/World/" + name)
    c.CreateSizeAttr(1)
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.AddRotateZOp().Set(yaw)
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material("/World/Materials/" + name, color))
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
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(uv)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material("/World/Materials/" + name, color, texture=texture))
    return m


base = (0, 0, 0)
quat = (1, 0, 0, 0)
box("WoodWorktop", (0.4, 0, -0.035), (1.0, 0.95, 0.07), (0.55, 0.47, 0.33))
for i, y in enumerate([-0.48, 0.48]):
    box("FrameRail" + str(i), (0.4, y, 0.02), (1.1, 0.025, 0.045), (0.025, 0.025, 0.03))
box("BackWall", (0.94, 0, 0.25), (0.025, 1.0, 0.5), (0.12, 0.13, 0.15))
x, y = (0.48, 0.08)
box("HousingTop", (x, y, 0.125), (0.22, 0.18, 0.012), (0.115, 0.055, 0.36))
for i, yy in enumerate([y - 0.088, y + 0.088]):
    box("HousingSide" + str(i), (x, yy, 0.065), (0.22, 0.012, 0.12), (0.115, 0.055, 0.36))
box("HousingBack", (x + 0.104, y, 0.065), (0.012, 0.18, 0.12), (0.115, 0.055, 0.36))
box("DrawerBottom", (x - 0.09, y, 0.015), (0.2, 0.155, 0.012), (0.27, 0.006, 0.027))
box("DrawerFront", (x - 0.19, y, 0.061), (0.012, 0.17, 0.105), (0.27, 0.006, 0.027))
for i, yy in enumerate([y - 0.074, y + 0.074]):
    box("DrawerSide" + str(i), (x - 0.09, yy, 0.048), (0.2, 0.012, 0.07), (0.27, 0.006, 0.027))
box("Handle", (x - 0.224, y, 0.064), (0.012, 0.115, 0.012), (0.2, 0.007, 0.022))
T = np.array(meta["robot"]["extrinsic_calibration_matrix"])
campos = T[:3, 3]
camrot = Rotation.from_matrix(T[:3, :3]).as_quat()
K = np.array(meta["robot"]["intrinsic_calibration_matrix"])
qdict = dict(zip(meta["robot"]["robot_state_joint_names"], meta["states"]["joint_position"]["first"]))


def plane_point(u, v, z=0):
    z = z - 0.08 if args.scene == "drawer" else z
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


pts = np.array([plane_point(u, v, 0.07) for u, v in [(295, 153), (348, 165), (314, 266), (249, 253)]])
center = pts.mean(0)
ex = (pts[0] + pts[3] - pts[1] - pts[2]) / 2
ey = (pts[2] + pts[3] - pts[0] - pts[1]) / 2
lx = np.linalg.norm(ex[:2])
ly = np.linalg.norm(ey[:2])
ex /= np.linalg.norm(ex)
ey = np.array([-ex[1], ex[0], 0])
yaw = np.degrees(np.arctan2(ex[1], ex[0]))
origin = center.copy()
origin[2] = -0.08


def dbox(n, xyz, sz):
    pose_box(n, origin + ex * xyz[0] + ey * xyz[1] + np.array([0, 0, xyz[2]]), sz, yaw)


dbox("HousingTop", (0, 0, 0.07), (lx, ly, 0.009))
dbox("HousingBack", (lx / 2, 0, 0.035), (0.008, ly, 0.07))
for i, sign in enumerate([-1, 1]):
    dbox("HousingSide" + str(i), (0, sign * ly / 2, 0.035), (lx, 0.008, 0.07))
extension = lx * 0.68
dbox("DrawerBottom", (-extension, 0, 0.009), (lx, ly * 0.9, 0.008))
dbox("DrawerFront", (-lx / 2 - extension, 0, 0.036), (0.008, ly * 0.96, 0.065))
for i, sign in enumerate([-1, 1]):
    dbox("DrawerSide" + str(i), (-extension, sign * ly * 0.44, 0.035), (lx, 0.008, 0.06))
handle_center = plane_point(394, 250.5, 0.034)
handle_a = plane_point(405, 210, 0.034)
handle_b = plane_point(383, 291, 0.034)
handle_yaw = yaw
pose_box("Handle", handle_center, (0.009, float(np.linalg.norm(handle_b - handle_a)), 0.01), handle_yaw)
wpts = np.array([plane_point(u, v, 0) for u, v in [(140, 38), (471, 38), (650, 384), (0, 384)]])
mid = wpts.mean(0)
extent = np.ptp(wpts, axis=0)
pose_box("WoodWorktop", mid + np.array([0, 0, -0.035]), (extent[0] + 0.12, extent[1] + 0.12, 0.07))
for old in ["FrameRail0", "FrameRail1", "BackWall"]:
    stage.RemovePrim("/World/" + old)
edgepix = [(140, 38), (471, 38), (655, 384), (0, 384)]
ep = [plane_point(u, v, 0.015) for u, v in edgepix]
for i, j in [(0, 1), (1, 2), (3, 0)]:
    a, b = (ep[i], ep[j])
    d = b - a
    mid = (a + b) / 2
    box(
        "FrameEdge" + str(i),
        mid,
        (float(np.linalg.norm(d[:2])), 0.026, 0.03),
        (0.008, 0.01, 0.016),
        np.degrees(np.arctan2(d[1], d[0])),
    )
for i, j in [(0, 1), (1, 2), (3, 0)]:
    a, b = (ep[i], ep[j])
    mesh(
        "Panel" + str(i),
        [a, b, b + np.array([0, 0, 0.4]), a + np.array([0, 0, 0.4])],
        [[0, 1, 2, 3]],
        (0.025, 0.03, 0.045),
    )
texdir = ROOT / "assets/textures"
texdir.mkdir(parents=True, exist_ok=True)


def texture_patch(label, ref, rect):
    im = cv2.imread(str(ROOT / "references" / ref))
    x0, y0, x1, y1 = rect
    path = texdir / (label + ".png")
    patch = im[y0:y1, x0:x1]
    patch = np.concatenate([patch, patch[:, ::-1]], axis=1) if "wood" in label else patch
    patch = np.concatenate([patch, patch[::-1]], axis=0) if "wood" in label else patch
    cv2.imwrite(str(path), patch)
    return path


wood = texture_patch("drawer-wood", "drawer/exterior_image_1-0000.png", (145, 80, 245, 220))
wp = np.array(wpts)
wp[:, 2] = -0.0798
mesh("WorktopGrain", wp, [[0, 1, 2, 3]], (1, 1, 1), [(0, 0), (2, 0), (2, 2), (0, 2)], wood)
robot = Articulation(
    ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim.UsdFileCfg(usd_path=converter.usd_path, rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=base, rot=quat, joint_pos=qdict),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=400.0, damping=40.0)},
    )
)
dark = material("/World/Materials/GripperDark", (0.025, 0.028, 0.035))
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith("/World/Robot/") and any((n in prim.GetName() for n in ["finger", "knuckle"])):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(dark, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
for name in [
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]:
    qdict[name] = 0.841
from scene_details import add_details

add_details(stage, args.scene, campos, camrot, K, box, mesh, material, plane_point)
hand = stage.GetPrimAtPath("/World/Robot/link7")
if hand.IsValid():
    UsdShade.MaterialBindingAPI.Apply(hand).Bind(
        material("/World/Materials/HandMetal", (0.28, 0.3, 0.33), 0.3),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
sim.DomeLightCfg(intensity=300, color=(0.92, 0.95, 1.0)).func(
    "/World/Dome", sim.DomeLightCfg(intensity=300, color=(0.92, 0.95, 1.0))
)
light = sim.DistantLightCfg(intensity=650, color=(1.0, 0.94, 0.84), angle=12.0)
light.func("/World/Sun", light, orientation=(0.92388, 0.38268, 0, 0))
refcam = {"stack": "side_cam", "drawer": "exterior_image_1", "plate": "front"}[args.scene]
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
            renderK.flatten().tolist(), width=rw, height=rh, focal_length=24.0, clipping_range=(0.01, 100.0)
        ),
    )
)
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg

parts = ["DrawerBottom", "DrawerFront", "DrawerSide0", "DrawerSide1", "Handle"]
bodypos = origin - ex * extension + np.array([0, 0, 0.035])
body = UsdGeom.Xform.Define(stage, "/World/SlidingDrawer")
body.AddTranslateOp().Set(Gf.Vec3d(*bodypos))
UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
mass.CreateMassAttr(0.15)
phys = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
phys.CreateSolverPositionIterationCountAttr(16)
phys.CreateSolverVelocityIterationCountAttr(4)
phys.CreateMaxDepenetrationVelocityAttr(1.0)
for n in parts:
    old = "/World/" + n
    ops = UsdGeom.Xformable(stage.GetPrimAtPath(old)).GetOrderedXformOps()
    pos = np.array(ops[0].Get())
    ops[0].Set(Gf.Vec3d(*pos - bodypos))
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(old, "/World/SlidingDrawer/" + n)
    assert stage.GetRootLayer().Apply(edit)
    collider = stage.GetPrimAtPath("/World/SlidingDrawer/" + n)
    pc = PhysxSchema.PhysxCollisionAPI.Apply(collider)
    pc.CreateContactOffsetAttr(0.001)
    pc.CreateRestOffsetAttr(0.0)
from drawer_visual_details import refine

refine(stage, material)
joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/DrawerGuide")
joint.CreateBody1Rel().SetTargets(["/World/SlidingDrawer"])
joint.CreateAxisAttr("X")
joint.CreateLocalPos0Attr(Gf.Vec3f(*bodypos))
joint.CreateLocalPos1Attr(Gf.Vec3f(0))
a = np.deg2rad(yaw) / 2
rotation = Gf.Quatf(float(np.cos(a)), Gf.Vec3f(0, 0, float(np.sin(a))))
joint.CreateLocalRot0Attr(rotation)
joint.CreateLocalRot1Attr(rotation)
joint.CreateLowerLimitAttr(0.0)
joint.CreateUpperLimitAttr(float(extension))
joint.CreateCollisionEnabledAttr(False)
PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.0)
robot_links = [
    str(p.GetPath())
    for p in stage.Traverse()
    if str(p.GetPath()).startswith("/World/Robot/") and p.HasAPI(UsdPhysics.RigidBodyAPI)
]
if args.no_robot_contact:
    UsdPhysics.FilteredPairsAPI.Apply(body.GetPrim()).CreateFilteredPairsRel().SetTargets(robot_links)
drawer = RigidObject(
    RigidObjectCfg(
        prim_path="/World/SlidingDrawer", spawn=None, init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(bodypos))
    )
)
contacts = ContactSensor(
    ContactSensorCfg(prim_path="/World/SlidingDrawer", filter_prim_paths_expr=robot_links, update_period=0.0)
)
ctx.reset()
robot.reset()
q = robot.data.default_joint_pos.clone()
for j, n in enumerate(robot.joint_names):
    if n not in meta["robot"]["robot_state_joint_names"]:
        q[0, j] = 0.841
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
assert args.scene == "drawer"
data = np.load(ROOT / "motion-drawer/states.npz")
arm_names = meta["robot"]["robot_state_joint_names"]
joint_ids = [robot.joint_names.index(n) for n in arm_names]
frames = out / "frames"
frames.mkdir(exist_ok=True)
ticks = []
logs = []
assert args.rail_friction >= 0 and args.rail_damping >= 0
rail_logs = []


def apply_rail_friction():
    velocity = float(np.dot(drawer.data.root_lin_vel_w[0].cpu().numpy(), ex))
    magnitude = -args.rail_friction * np.tanh(velocity / 0.01) - args.rail_damping * velocity
    force = torch.tensor((magnitude * ex).reshape(1, 1, 3), device=args.device, dtype=torch.float32)
    drawer.set_external_force_and_torque(force, torch.zeros_like(force), is_global=True)
    drawer.write_data_to_sim()
    rail_logs.append(
        {"slide_velocity_m_s": velocity, "resistance_force_N": float(magnitude), "power_W": float(magnitude * velocity)}
    )


closed_position = extension - 0.004
hold_target = None
hold_start = None
hold_tick = None
for i in range(70):
    for sub in range(8):
        t = max(0.0, i - 1 + (sub + 1) / 8)
        lo = int(t)
        hi = min(69, lo + 1)
        alpha = t - lo
        target = (1 - alpha) * data["joint_position"][lo] + alpha * data["joint_position"][hi]
        now = (i * 8 + sub) / 120
        if hold_target is not None:
            target = hold_target
        q[0, joint_ids] = torch.tensor(target, device=args.device, dtype=q.dtype)
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        apply_rail_friction()
        ctx.step(render=False)
        robot.update(1 / 120)
        drawer.update(1 / 120)
        contacts.update(1 / 120, force_recompute=True)
        pos = drawer.data.root_pos_w[0].cpu().numpy()
        displacement = float((pos - bodypos) @ ex)
        force = contacts.data.force_matrix_w[0].cpu().numpy()
        force_norm = float(np.linalg.norm(force.reshape(-1, 3), axis=1).sum())
        measured = robot.data.joint_pos[0].cpu().numpy()
        assert np.isfinite(measured).all() and np.max(abs(measured)) < 10, "Invalid robot state"
        if hold_target is None and displacement >= closed_position - 0.0001 and (force_norm >= 0.5):
            hold_start = now + 1 / 120
            hold_tick = i * 8 + sub
            hold_target = measured[joint_ids].copy() + 0.1 * (target - measured[joint_ids])
        ticks.append(
            {
                "controller_phase": "hold_push_pose" if hold_target is not None else "recorded_push",
                "frame": i,
                "substep": sub,
                "time_s": (i * 8 + sub + 1) / 120,
                "drawer_displacement_m": displacement,
                "robot_contact_force_N": force_norm,
                "drawer_speed_m_s": float(torch.linalg.vector_norm(drawer.data.root_lin_vel_w[0]).cpu()),
                "joint_positions": measured.tolist(),
                "target_joint_positions": q[0].cpu().tolist(),
            }
        )
    for _ in range(3):
        ctx.render()
    camera.update(0, force_recompute=True)
    im = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
    im = cv2.remap(im, mx, my, cv2.INTER_LINEAR)
    Image.fromarray(im).save(frames / f"{i:04d}.png")
    logs.append(
        {
            "frame": i,
            "time_s": i / 15,
            "drawer_displacement_m": displacement,
            "closed_fraction": displacement / extension,
            "robot_contact_force_N": force_norm,
        }
    )
    print("CONTACT_FRAME", i, displacement, force_norm, flush=True)
settling = []
for j in range(120):
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    apply_rail_friction()
    ctx.step(render=False)
    robot.update(1 / 120)
    drawer.update(1 / 120)
    contacts.update(1 / 120, force_recompute=True)
    pos = drawer.data.root_pos_w[0].cpu().numpy()
    force = contacts.data.force_matrix_w[0].cpu().numpy()
    settling.append(
        {
            "time_s": 70 / 15 + (j + 1) / 120,
            "drawer_displacement_m": float((pos - bodypos) @ ex),
            "drawer_speed_m_s": float(torch.linalg.vector_norm(drawer.data.root_lin_vel_w[0]).cpu()),
            "robot_contact_force_N": float(np.linalg.norm(force.reshape(-1, 3), axis=1).sum()),
        }
    )
summary = {
    "rail_friction_N": args.rail_friction,
    "rail_damping_Ns_m": args.rail_damping,
    "rail_smoothing_velocity_m_s": 0.01,
    "rail_force_log": rail_logs,
    "controller": "push until closed within 0.1 mm and contact >=0.5 N, then hold measured pushing pose plus 10 percent of forward tracking error; no retreat",
    "hold_tick": hold_tick,
    "hold_start_s": hold_start,
    "hold_target": None if hold_target is None else hold_target.tolist(),
    "closed_position_m": float(closed_position),
    "settling": settling,
    "mode": "ACTUATED_ARM_PASSIVE_PRISMATIC_DRAWER",
    "no_robot_contact_control": args.no_robot_contact,
    "table_height_m": -0.08,
    "drawer_mass_kg": 0.15,
    "drawer_travel_m": float(extension),
    "final_displacement_m": displacement,
    "max_contact_force_N": max((x["robot_contact_force_N"] for x in ticks)),
    "first_contact_tick": next((j for j, x in enumerate(ticks) if x["robot_contact_force_N"] > 0.1), None),
    "first_motion_tick": next((j for j, x in enumerate(ticks) if x["drawer_displacement_m"] > 0.001), None),
    "frames": logs,
    "ticks": ticks,
}
(out / "contact-motion.json").write_text(json.dumps(summary, indent=2))
(out / "renderer-source.py").write_text(Path(__file__).read_text())
stage.GetRootLayer().Export(str(out / "scene.usda"))
print("CONTACT_COMPLETE", out, flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
