"""Isolated physical plate experiment; only arm/finger actuator targets change."""
from pathlib import Path
RUN_SOURCE=Path(__file__).read_text()
source=Path(__file__).with_name('scene.py').read_text()
source=source.replace("p.add_argument('--view'", "p.add_argument('--clean-table',action='store_true');p.add_argument('--detailed-plate',action='store_true');p.add_argument('--refine-visuals',action='store_true');p.add_argument('--no-contact',action='store_true');p.add_argument('--quick',action='store_true');p.add_argument('--view'")
source=source.replace('UsdPhysics,Gf','UsdPhysics,PhysxSchema,Gf')
source=source.replace('dt=1/120','dt=1/240')
source=source.replace('disable_gravity=True','disable_gravity=False')
source=source.replace("actuators={'all':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=100.,damping=10.)}","actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['waist','shoulder','elbow','forearm_roll','wrist_angle','wrist_rotate'],stiffness=400.,damping=40.),'fingers':ImplicitActuatorCfg(joint_names_expr=['left_finger','right_finger'],stiffness=2000.,damping=124.,effort_limit_sim=35.,velocity_limit_sim=.2)}")
source=source.replace("refcam={'stack'", "if args.refine_visuals:\n cf=json.loads((ROOT/'config/fit.json').read_text());campos=np.array(cf['position']);camrot=Rotation.from_matrix(np.array(cf['rotation'])).as_quat();K=np.array(cf['K'])\nrefcam={'stack'")
marker='ctx.reset();robot.reset();q='
assert source.count(marker)==1
prefix=source.split(marker)[0]
tail=r'''
from isaaclab.assets import RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
plan=json.loads((ROOT/'config/controller-plan.json').read_text())
start=np.array(plan['plate_start']);radius=plan['plate_radius']
# Preserve fitted backdrop
# Preserve fitted trim
for name in ['Detail_RearCable0','Detail_RearCable1']:
 cable=UsdGeom.BasisCurves(stage.GetPrimAtPath('/World/'+name));pp=np.array(cable.GetPointsAttr().Get());cable.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pp.astype('float32')))
# Closed thin plate surface and compound convex collision geometry match the radial profile.
if args.clean_table:
 for name in ['SideFixture0','SideFixture1','SideFixture2','FrameFastener2','FrameFastener3']:stage.RemovePrim('/World/'+name)
stage.RemovePrim('/World/PinkPlate')
for k in range(3):stage.RemovePrim('/World/RimBead'+str(k))
body=UsdGeom.Xform.Define(stage,'/World/Plate');body.AddTranslateOp().Set(Gf.Vec3d(*start))
UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.06)
rp=PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim());rp.CreateSolverPositionIterationCountAttr(16);rp.CreateSolverVelocityIterationCountAttr(4);rp.CreateMaxDepenetrationVelocityAttr(.5);rp.CreateEnableCCDAttr(True)
physmat=UsdShade.Material.Define(stage,'/World/Materials/PlateFriction');pm=UsdPhysics.MaterialAPI.Apply(physmat.GetPrim());pm.CreateStaticFrictionAttr(.8);pm.CreateDynamicFrictionAttr(.6);pm.CreateRestitutionAttr(0.)
def collider(p):
 UsdPhysics.CollisionAPI.Apply(p);c=PhysxSchema.PhysxCollisionAPI.Apply(p);c.CreateContactOffsetAttr(.0005);c.CreateRestOffsetAttr(0.);UsdShade.MaterialBindingAPI.Apply(p).Bind(physmat,materialPurpose='physics')
profile=np.array([[0,.003],[.075,.003],[.095,.005],[.11,.011],[.12,.018]])
profile[:,0]*=radius/.12
points=[];faces=[];n=96
for side in [0,1]:
 for r,z in profile:
  for i in range(n):a=i*2*np.pi/n;points.append([r*np.cos(a),r*np.sin(a),z-side*.003])
for side in [0,1]:
 for j in range(4):
  for i in range(n):
   a=side*5*n+j*n+i;b=side*5*n+j*n+(i+1)%n;c=side*5*n+(j+1)*n+(i+1)%n;d=side*5*n+(j+1)*n+i;faces.append([a,b,c,d] if side==0 else [d,c,b,a])
for i in range(n):a=4*n+i;b=4*n+(i+1)%n;faces.append([a,a+5*n,b+5*n,b])
v=mesh('Plate/Visual',points,faces,(.48,.16,.15))
# Flat base plus annular wedges retain the raised rim and its underside clearance.
basecol=UsdGeom.Cylinder.Define(stage,'/World/Plate/BaseCollision');basecol.CreateRadiusAttr(float(profile[1,0]));basecol.CreateHeightAttr(.003);basecol.AddTranslateOp().Set(Gf.Vec3d(0,0,.0015));basecol.CreateVisibilityAttr('invisible');collider(basecol.GetPrim())
for j in range(1,4):
 for k in range(32):
  pts=[]
  for side in [0,1]:
   for r,z in profile[j:j+2]:
    for a in [k*2*np.pi/32,(k+1)*2*np.pi/32]:pts.append([r*np.cos(a),r*np.sin(a),z-side*.003])
  m=mesh(f'Plate/Wedge{j}_{k}',pts,[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]],(.48,.16,.15));m.CreateVisibilityAttr('invisible');collider(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
visual_error=0.
if args.refine_visuals:
 from plate_contact_details import refine_plate
 if args.detailed_plate:
  from plate_detail_v2 import refine_plate
 visual_error=refine_plate(stage,mesh,material,radius)
# Uniform contact tolerance and measured force logging on the original finger mesh hulls.
links=[str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Robot/') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
fingers=[p for p in links if 'finger' in p]
assert len(fingers)==2,fingers
for p in stage.Traverse():
 if str(p.GetPath()).startswith('/World/Robot/') and p.HasAPI(UsdPhysics.CollisionAPI):collider(p)
 if str(p.GetPath()).startswith('/World/Robot/') and 'finger' in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI):
  cp=PhysxSchema.PhysxCollisionAPI.Apply(p);cp.CreateTorsionalPatchRadiusAttr(.005);cp.CreateMinTorsionalPatchRadiusAttr(.001)
if args.no_contact:UsdPhysics.FilteredPairsAPI.Apply(body.GetPrim()).CreateFilteredPairsRel().SetTargets(links)
PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
plate=RigidObject(RigidObjectCfg(prim_path='/World/Plate',spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(start))))
contacts=ContactSensor(ContactSensorCfg(prim_path='/World/Plate',filter_prim_paths_expr=fingers,update_period=0.))
ctx.reset();robot.reset();other.reset();plate.reset()
q=robot.data.default_joint_pos.clone()
for j,n in enumerate(robot.joint_names):q[0,j]=plan['rows'][0]['q'].get(n,float(q[0,j]))
robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.set_joint_position_target(q)
oq=other.data.default_joint_pos.clone();other.write_joint_state_to_sim(oq,torch.zeros_like(oq));other.set_joint_position_target(oq)
camera.set_world_poses(torch.tensor(campos[None],dtype=torch.float32,device=args.device),torch.tensor([[camrot[3],*camrot[:3]]],dtype=torch.float32,device=args.device),convention='ros')
xx,yy=np.meshgrid(np.arange(width),np.arange(height));mx=((xx-K[0,2])*favg/K[0,0]+rw/2).astype('float32');my=((yy-K[1,2])*favg/K[1,1]+rh/2).astype('float32')
frames=out/'frames';frames.mkdir(exist_ok=True);ticks=[];flog=[];held_ticks=0;maxheld=0;lift=False
# Physics-only settling at the initial pose. No plate pose writes after creation.
grav_api=hasattr(robot.root_physx_view,'get_gravity_compensation_forces')
def step():
 if grav_api:
  robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
  other.set_joint_effort_target(other.root_physx_view.get_gravity_compensation_forces())
 robot.write_data_to_sim();other.write_data_to_sim();ctx.step(render=False);robot.update(1/240);other.update(1/240);plate.update(1/240);contacts.update(1/240,force_recompute=True)
for _ in range(120):step()
initial=plate.data.root_pos_w[0].cpu().numpy().copy()
for _ in range(10):ctx.render()
for tick in range(round(plan['rows'][-1]['time_s']*240)):
 t=tick/240;k=min(int(t*30),len(plan['rows'])-2);alpha=t*30-k;a,b=plan['rows'][k:k+2]
 for j,n in enumerate(robot.joint_names):q[0,j]=a['q'].get(n,float(q[0,j]))*(1-alpha)+b['q'].get(n,float(q[0,j]))*alpha
 robot.set_joint_position_target(q);step()
 pos=plate.data.root_pos_w[0].cpu().numpy();quatp=plate.data.root_quat_w[0].cpu().numpy();vel=plate.data.root_lin_vel_w[0].cpu().numpy();forces=np.linalg.norm(contacts.data.force_matrix_w[0].cpu().numpy().reshape(-1,3),axis=1)
 bilateral=bool(np.all(forces>.05));held_ticks=held_ticks+1 if bilateral else 0;maxheld=max(maxheld,held_ticks)
 if held_ticks>=48 and pos[2]-initial[2]>.03:lift=True
 rec={'tick':tick,'time_s':t,'phase':a['phase'],'plate_position':pos.tolist(),'plate_quat_wxyz':quatp.tolist(),'plate_velocity':vel.tolist(),'finger_forces_N':forces.tolist(),'bilateral_contact':bilateral,'bilateral_duration_s':held_ticks/240,'verified_lift':lift,'joint_positions':robot.data.joint_pos[0].cpu().tolist(),'joint_velocities':robot.data.joint_vel[0].cpu().tolist(),'joint_targets':q[0].cpu().tolist()};ticks.append(rec)
 if tick%16==0:
  i=tick//16
  if not args.quick or i%15==0:
   for _ in range(2):ctx.render()
   camera.update(0,force_recompute=True);im=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();im=cv2.remap(im,mx,my,cv2.INTER_LINEAR);Image.fromarray(im).save(frames/f'{i:04d}.png')
  flog.append(rec)
  if i%15==0:print('CONTACT',round(t,2),a['phase'],pos.tolist(),forces.tolist(),flush=True)
final=plate.data.root_pos_w[0].cpu().numpy();target=np.array(plan['target_center']);speed=float(np.linalg.norm(plate.data.root_lin_vel_w[0].cpu().numpy()));separated=bool(np.all(forces<.05));inside=bool(abs(final[0]-target[0])<tx*.4 and abs(final[1]-target[1])<ty*.4 and abs(final[2]-.10)<.012)
final_normal=Rotation.from_quat([*quatp[1:],quatp[0]]).apply([0,0,1]);last=ticks[-120:];settled_upright=all(np.linalg.norm(r['plate_velocity'])<.02 and Rotation.from_quat([*r['plate_quat_wxyz'][1:],r['plate_quat_wxyz'][0]]).apply([0,0,1])[2]>.95 and abs(r['plate_position'][2]-.1)<.012 for r in last)
summary={'camera':{'position':campos.tolist(),'rotation':Rotation.from_quat(camrot).as_matrix().tolist(),'K':K.tolist()},'placement_target':target.tolist(),'settled_upright_last_half_second':settled_upright,'final_normal':final_normal.tolist(),'fingertip_patch_radius_m':.005,'finger_kp':2000.,'finger_kd':124.,'finger_force_limit_N':35.,'visual_surface_max_deviation_m':visual_error,'mode':'CONTACT_PHYSICS_NO_PLATE_COMMANDS','no_robot_contact':args.no_contact,'gravity_compensation':grav_api,'finger_body_paths':fingers,'mass_kg':.06,'static_friction':.8,'dynamic_friction':.6,'physics_hz':240,'initial_plate_position':initial.tolist(),'final_plate_position':final.tolist(),'verified_lift':lift,'max_bilateral_duration_s':maxheld/240,'peak_plate_height':max(r['plate_position'][2] for r in ticks),'placement_in_region':inside,'released':separated,'final_speed_m_s':speed,'placement_candidate':bool(inside and separated and settled_upright),'frames':flog,'ticks':ticks,'note':'Estimated physical parameters; placement_candidate is a geometric/velocity check, not benchmark certification.'}
(out/'contact-motion.json').write_text(json.dumps(summary,indent=2));(out/'controller-plan.json').write_text(json.dumps(plan,indent=2));(out/'renderer-source.py').write_text(RUN_SOURCE);stage.GetRootLayer().Export(str(out/'scene.usda'));print('PHYSICS_COMPLETE', {k:v for k,v in summary.items() if k not in ['frames','ticks']},flush=True);sys.stdout.flush();os._exit(0)
'''
exec(compile(prefix+tail,__file__,'exec'))
