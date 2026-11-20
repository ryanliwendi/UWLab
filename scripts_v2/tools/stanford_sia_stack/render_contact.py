"""Scripted arm targets; free cubes move through contact physics only."""

from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "scene.py").read_text()
source = source.replace("UsdPhysics, Gf", "UsdPhysics, PhysxSchema, Gf")
prefix = source.split("ctx.reset()")[0]
tail = r"""

from isaaclab.assets import RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
plan=json.loads((ROOT/'config/controller-plan.json').read_text());arm=np.array([r['q'] for r in plan['rows']]);widths=np.array([r['width'] for r in plan['rows']])
from plan_controller import fk
placements=json.loads((ROOT/'config/cube-placement.json').read_text())
objects={}
for name,color in [('RedCube',(.65,.015,.025)),('GreenCube',(.025,.33,.08))]:
 pos=placements[name]['center_world'];angle=np.deg2rad(placements[name]['yaw_degrees']);stage.RemovePrim('/World/'+name)
 cfg=RigidObjectCfg(prim_path='/World/'+name,spawn=sim.CuboidCfg(size=(.025,.025,.025),rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False,solver_position_iteration_count=16,solver_velocity_iteration_count=4),mass_props=sim.MassPropertiesCfg(mass=.03),collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),physics_material=sim.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6,restitution=0.),visual_material=sim.PreviewSurfaceCfg(diffuse_color=color),activate_contact_sensors=True),init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos),rot=(float(np.cos(angle/2)),0,0,float(np.sin(angle/2)))))
 objects[name]=RigidObject(cfg)
fingers=[str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Robot/') and p.HasAPI(UsdPhysics.RigidBodyAPI) and 'finger' in str(p.GetPath())]
assert len(fingers)==2,fingers
contacts=ContactSensor(ContactSensorCfg(prim_path='/World/RedCube',filter_prim_paths_expr=fingers,update_period=0.,history_length=1))
ctx.reset();robot.reset();q=robot.data.default_joint_pos.clone()
def target(i):
 for j,n in enumerate(robot.joint_names):
  if n in meta['robot']['robot_state_joint_names']:q[0,j]=float(arm[i,meta['robot']['robot_state_joint_names'].index(n)])
  elif 'finger_joint' in n:q[0,j]=float(np.clip(widths[i]/2,0,.04))
target(0);robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.set_joint_position_target(q)
camera.set_world_poses(torch.tensor(campos[None],dtype=torch.float32,device=args.device),torch.tensor([[camrot[3],*camrot[:3]]],dtype=torch.float32,device=args.device),convention='ros')
for o in objects.values():o.reset()
contacts.reset()
print('CONTROL_CONFIG',{'stiffness':robot.data.joint_stiffness.cpu().tolist(),'damping':robot.data.joint_damping.cpu().tolist(),'gravity_api':hasattr(robot.root_physx_view,'get_gravity_compensation_forces'),'friction_api':[a for a in dir(robot.root_physx_view) if 'friction' in a]},flush=True)
def step():
 if hasattr(robot.root_physx_view,'get_gravity_compensation_forces'):robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
 robot.write_data_to_sim();ctx.step(render=False);robot.update(1/120)
 for o in objects.values():o.update(1/120)
 contacts.update(1/120,force_recompute=True)
for _ in range(60):step()
initial=objects['RedCube'].data.root_pos_w[0].cpu().numpy().copy()
xx,yy=np.meshgrid(np.arange(width),np.arange(height));mx=((xx-K[0,2])*favg/K[0,0]+rw/2).astype('float32');my=((yy-K[1,2])*favg/K[1,1]+rh/2).astype('float32')
if not os.environ.get('STACK_VALIDATE_ONLY'):
 for _ in range(10):ctx.render()
frames=out/'frames';frames.mkdir(exist_ok=True);rows=[];held=0;lift=False
for i in range(len(arm)):
 target(i);robot.set_joint_position_target(q)
 for _ in range(8):
  step();forces=np.linalg.norm(contacts.data.force_matrix_w[0].cpu().numpy().reshape(-1,3),axis=1);held=held+1 if np.all(forces>.05) else 0
  if held>=24 and objects['RedCube'].data.root_pos_w[0,2].item()-initial[2]>.02:lift=True
 red=objects['RedCube'].data.root_pos_w[0].cpu().numpy();green=objects['GreenCube'].data.root_pos_w[0].cpu().numpy()
 rows.append({'frame':i,'time_s':i/15,'red_position':red.tolist(),'green_position':green.tolist(),'finger_forces_N':forces.tolist(),'verified_lift':lift,'joint_positions':robot.data.joint_pos[0].cpu().tolist(),'target_gripper_width':float(widths[i]),'applied_torque':robot.data.applied_torque[0].cpu().tolist(),'phase':plan['rows'][i]['phase'],'tcp_actual':fk([robot.data.joint_pos[0,robot.joint_names.index('fr3_joint'+str(j))].item() for j in range(1,8)])[:3,3].tolist(),'red_linear_velocity':objects['RedCube'].data.root_lin_vel_w[0].cpu().tolist()})
 if not os.environ.get('STACK_VALIDATE_ONLY'):
  for _ in range(2):ctx.render()
  camera.update(0,force_recompute=True);im=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();Image.fromarray(cv2.remap(im,mx,my,cv2.INTER_LINEAR)).save(frames/f'{i:04d}.png')
 if i%30==0:(out/'partial-frames.json').write_text(json.dumps(rows));print('STACK_FRAME',i,'lift',lift,'red',red.tolist(),'forces',forces.tolist(),flush=True)
result={'mode':'SCRIPTED_IK_JOINT_TARGETS_DYNAMIC_CUBES','physics_hz':120,'render_fps':15,'frames':rows,'verified_lift':lift,'max_red_displacement_m':float(max(np.linalg.norm(np.array(r['red_position'])-initial) for r in rows)),'note':'Scripted IK arm targets fitted to the reconstructed cubes. Dynamic cubes, no attachments or scripted object transforms. Not a trained policy.'}
(out/'motion.json').write_text(json.dumps(result,indent=2));stage.GetRootLayer().Export(str(out/'scene.usda'));print('STACK_COMPLETE',result['verified_lift'],result['max_red_displacement_m'],flush=True);sys.stdout.flush();os._exit(0)
"""
exec(compile(prefix + tail, str(root / "scene.py"), "exec"))
