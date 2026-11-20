"""Isolated Isaac Lab visual-fitting scenes. No policies or task training.

Objects/backgrounds are explicit 3-D geometry, not a pasted camera image.
This first pass is visual calibration, NOT validated task physics.
"""
import argparse,json,os,sys
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--scene',choices=['plate'],default='plate');p.add_argument('--output',required=True)
p.add_argument('--view',choices=['front','back','wrist'],default='front');AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np,torch,cv2
from scipy.spatial.transform import Rotation
from PIL import Image
from pxr import UsdGeom,UsdShade,UsdPhysics,Gf,Sdf,Vt
import isaaclab.sim as sim
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera,CameraCfg
ROOT=Path(__file__).resolve().parent
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
meta=json.loads((ROOT/'references'/args.scene/'metadata.json').read_text())
names={'stack':'fr3','drawer':'xarm7','plate':'vx300s_aloha'};model=names[args.scene]
converter=UrdfConverter(UrdfConverterCfg(asset_path=str(ROOT/'assets/urdf'/f'{model}.urdf'),usd_dir=str(ROOT/'assets/usd'/model),usd_file_name=f'{model}.usd',fix_base=True,merge_fixed_joints=True,make_instanceable=False,joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.,damping=10.))))
ctx=sim.SimulationContext(sim.SimulationCfg(dt=1/120,device=args.device));stage=ctx.stage

def material(path,color,rough=.7,texture=None):
 m=UsdShade.Material.Define(stage,path);s=UsdShade.Shader.Define(stage,path+'/Shader');s.CreateIdAttr('UsdPreviewSurface');s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough);s.CreateOutput('surface',Sdf.ValueTypeNames.Token);m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
 if texture:
  st=UsdShade.Shader.Define(stage,path+'/ST');st.CreateIdAttr('UsdPrimvarReader_float2');st.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');st.CreateOutput('result',Sdf.ValueTypeNames.Float2)
  t=UsdShade.Shader.Define(stage,path+'/Texture');t.CreateIdAttr('UsdUVTexture');t.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');t.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat');t.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB');t.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture));t.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(),'result');t.CreateOutput('rgb',Sdf.ValueTypeNames.Float3);s.GetInput('diffuseColor').ConnectToSource(t.ConnectableAPI(),'rgb')
 return m
def box(name,pos,size,color,yaw=0):
 c=UsdGeom.Cube.Define(stage,'/World/'+name);c.CreateSizeAttr(1);c.AddTranslateOp().Set(Gf.Vec3d(*pos));c.AddRotateZOp().Set(yaw);c.AddScaleOp().Set(Gf.Vec3f(*size));UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material('/World/Materials/'+name,color));UsdPhysics.CollisionAPI.Apply(c.GetPrim());return c
def mesh(name,points,faces,color,uv=None,texture=None):
 m=UsdGeom.Mesh.Define(stage,'/World/'+name);m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array(points,dtype=np.float32)));m.CreateFaceVertexCountsAttr([len(f) for f in faces]);m.CreateFaceVertexIndicesAttr([x for f in faces for x in f]);m.CreateSubdivisionSchemeAttr('none');m.CreateDoubleSidedAttr(True)
 if uv is not None:UsdGeom.PrimvarsAPI(m).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(uv)
 UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material('/World/Materials/'+name,color,texture=texture));return m
base=(0,0,0);quat=(1,0,0,0)
if args.scene=='stack':
 box('WoodBench',(.38,0,-.025),(1.18,.68,.05),(.60,.43,.25))
 # Green curtain is a corrugated 3-D sheet; dimensions/folds are provisional.
 pts=[];faces=[]
 for iz in range(21):
  for ix in range(81):
   x=-1.5+ix*.05;z=-.2+iz*.07;pts.append([x,.48+.018*np.sin(ix*.7)+.01*np.sin(iz*.8+ix*.2),z])
 for iz in range(20):
  for ix in range(80):a=iz*81+ix;faces.append([a,a+1,a+82,a+81])
 mesh('GreenCurtain',pts,faces,(.004,.12,.072))
 box('GreenCube',(.48,.01,.0125),(.025,.025,.025),(.025,.33,.08));box('RedCube',(.59,-.045,.0125),(.025,.025,.025),(.65,.015,.025))
 rvec=np.array([1.752136704,.572361289,-.287057363]);t=np.array([-.455136929,.263261682,1.348447330]);R=Rotation.from_rotvec(rvec).as_matrix();campos=-R.T@t;camrot=Rotation.from_matrix(R.T).as_quat();K=np.array(meta['robot']['intrinsic_calibration_matrix']['side_cam'])
 # Reference-derived board textures on 3-D table-plane quads, not a background screenshot.
 ref=cv2.imread(str(ROOT/'references/stack/side_cam-0000.png'))
 # Unknown board pixels under real cubes are explicitly neutralized, not retained as fake objects.
 ref[395:417,332:356]=[220,220,210];ref[414:441,392:417]=[220,220,210]
 corners=[[[242,326],[339,351],[282,395],[199,370]],[[156,383],[270,424],[195,479],[88,439]],[[320,394],[438,425],[378,476],[276,442]],[[468,393],[602,433],[561,479],[431,445]]]
 texture_dir=ROOT/'assets/textures';texture_dir.mkdir(parents=True,exist_ok=True)
 for i,c in enumerate(corners):
  src=np.array(c,np.float32);dst=np.array([[0,0],[511,0],[511,399],[0,399]],np.float32)
  texture=texture_dir/f'stack-board-{i}.png';cv2.imwrite(str(texture),cv2.warpPerspective(ref,cv2.getPerspectiveTransform(src,dst),(512,400)))
  points=[]
  for u,v in c:
   ray=R.T@np.linalg.inv(K)@np.array([u,v,1.]);points.append((campos+(.0008-campos[2])/ray[2]*ray).tolist())
  mesh(f'Board{i}',points,[[0,1,2,3]],(1.,1.,1.),[(0.,1.),(1.,1.),(1.,0.),(0.,0.)],texture)
 qdict=dict(zip(meta['robot']['robot_state_joint_names'],meta['states']['joint_position']['first']));qdict.update({'fr3_finger_joint1':.04,'fr3_finger_joint2':.04})
elif args.scene=='drawer':
 box('WoodWorktop',(.40,0,-.035),(1.0,.95,.07),(.55,.47,.33))
 for i,y in enumerate([-.48,.48]):box('FrameRail'+str(i),(.4,y,.02),(1.1,.025,.045),(.025,.025,.03))
 box('BackWall',(.94,0,.25),(.025,1.,.50),(.12,.13,.15))
 # Purple shell, open red tray and handle. Dimensions estimated, not measured.
 x,y=.48,.08
 box('HousingTop',(x,y,.125),(.22,.18,.012),(.12,.037,.426))
 for i,yy in enumerate([y-.088,y+.088]):box('HousingSide'+str(i),(x,yy,.065),(.22,.012,.12),(.12,.037,.426))
 box('HousingBack',(x+.104,y,.065),(.012,.18,.12),(.12,.037,.426))
 box('DrawerBottom',(x-.09,y,.015),(.20,.155,.012),(.55,.015,.06));box('DrawerFront',(x-.19,y,.061),(.012,.17,.105),(.55,.015,.06))
 for i,yy in enumerate([y-.074,y+.074]):box('DrawerSide'+str(i),(x-.09,yy,.048),(.20,.012,.07),(.55,.015,.06))
 box('Handle',(x-.224,y,.064),(.012,.115,.012),(.35,.015,.035))
 # Metadata transform used provisionally as camera-to-base; requires image verification.
 T=np.array(meta['robot']['extrinsic_calibration_matrix']);campos=T[:3,3];camrot=Rotation.from_matrix(T[:3,:3]).as_quat();K=np.array(meta['robot']['intrinsic_calibration_matrix'])
 qdict=dict(zip(meta['robot']['robot_state_joint_names'],meta['states']['joint_position']['first']))
else:
 box('WhiteWorktop',(.30,0,-.025),(1.0,1.0,.05),(.85,.85,.83))
 for i,y in enumerate([-.51,.51]):box('FrameRail'+str(i),(.30,y,.02),(1.05,.025,.04),(.025,.025,.025))
 box('BackWall',(.85,0,.30),(.02,1.1,.60),(.078,.074,.067));box('WoodTrim',(.83,0,.48),(.03,1.1,.04),(.36,.22,.11))
 box('RaisedTable',(.57,.10,.17),(.23,.27,.025),(.50,.29,.13))
 for i,x in enumerate([.48,.66]):
  for j,y in enumerate([0,.20]):box(f'TableLeg{i}{j}',(x,y,.08),(.015,.015,.16),(.06,.06,.055))
 # Concave plate profile: visual surface only until collision decomposition is validated.
 pts=[];faces=[]
 profile=[(float(r),float(z)) for r,z in zip(np.linspace(0,.12,25),np.interp(np.linspace(0,.12,25),[0,.075,.095,.11,.117,.12],[.003,.003,.005,.011,.016,.018]))]
 for rad,z in profile:
  for i in range(96):a=2*np.pi*i/96;pts.append([.32+rad*np.cos(a),-.11+rad*np.sin(a),z])
 for j in range(len(profile)-1):
  for i in range(96):faces.append([j*96+i,j*96+(i+1)%96,(j+1)*96+(i+1)%96,(j+1)*96+i])
 mesh('PinkPlate',pts,[f[::-1] for f in faces],(.48,.16,.15))
 campos=np.array([.15,-.85,.85]);target=np.array([.35,0,.05]);z=(target-campos);z/=np.linalg.norm(z);x=np.cross(z,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(z,x);camrot=Rotation.from_matrix(np.column_stack([x,y,z])).as_quat();K=np.array([[610,0,320],[0,610,240],[0,0,1]])
 qdict={'waist':1.6122,'shoulder':.3068,'elbow':-.1289,'forearm_roll':-.01227,'wrist_angle':.3896,'wrist_rotate':.02454,'left_finger':.025,'right_finger':-.025}

# Image-landmark fitting revision 2. All dimensions remain estimates.
def plane_point(u,v,z=0):
 z=z-.15 if args.scene=='drawer' else z
 Rcw=Rotation.from_quat(camrot).as_matrix();ray=Rcw@np.linalg.inv(K)@np.array([u,v,1.]);return campos+(z-campos[2])/ray[2]*ray

def pose_box(name,pos,size=None,yaw=None):
 prim=stage.GetPrimAtPath('/World/'+name);ops=UsdGeom.Xformable(prim).GetOrderedXformOps()
 for op in ops:
  if op.GetOpType()==UsdGeom.XformOp.TypeTranslate:op.Set(Gf.Vec3d(*map(float,pos)))
  elif size is not None and op.GetOpType()==UsdGeom.XformOp.TypeScale:op.Set(Gf.Vec3f(*map(float,size)))
  elif yaw is not None and op.GetOpType()==UsdGeom.XformOp.TypeRotateZ:op.Set(float(yaw))

if args.scene=='drawer':
 # Fit cabinet top directly to its four measured image corners at estimated 7 cm height.
 pts=np.array([plane_point(u,v,.07) for u,v in [(295,153),(348,165),(314,266),(249,253)]])
 center=pts.mean(0);ex=(pts[0]+pts[3]-pts[1]-pts[2])/2;ey=(pts[2]+pts[3]-pts[0]-pts[1])/2
 lx=np.linalg.norm(ex[:2]);ly=np.linalg.norm(ey[:2]);ex/=np.linalg.norm(ex);ey=np.array([-ex[1],ex[0],0]);yaw=np.degrees(np.arctan2(ex[1],ex[0]));origin=center.copy();origin[2]=-.15
 def dbox(n,xyz,sz):pose_box(n,origin+ex*xyz[0]+ey*xyz[1]+np.array([0,0,xyz[2]]),sz,yaw)
 dbox('HousingTop',(0,0,.07),(lx,ly,.009));dbox('HousingBack',(lx/2,0,.035),(.008,ly,.07))
 for i,sign in enumerate([-1,1]):dbox('HousingSide'+str(i),(0,sign*ly/2,.035),(lx,.008,.07))
 extension=lx*.68
 dbox('DrawerBottom',(-extension/2,0,.009),(lx+extension,ly*.9,.008));dbox('DrawerFront',(-lx/2-extension,0,.036),(.008,ly*.96,.065))
 for i,sign in enumerate([-1,1]):dbox('DrawerSide'+str(i),(-extension/2,sign*ly*.44,.035),(lx+extension,.008,.06))
 dbox('Handle',(-lx/2-extension-.018,0,.034),(.007,ly*.68,.009))
 # Surface and rail footprint from calibrated camera rays.
 wpts=np.array([plane_point(u,v,0) for u,v in [(140,38),(471,38),(650,384),(0,384)]])
 mid=wpts.mean(0);extent=np.ptp(wpts,axis=0);pose_box('WoodWorktop',mid+np.array([0,0,-.035]),(extent[0]+.12,extent[1]+.12,.07))
 # Match the background frame footprint using the same camera and table plane.
 for old in ['FrameRail0','FrameRail1','BackWall']:stage.RemovePrim('/World/'+old)
 edgepix=[(140,38),(471,38),(655,384),(0,384)];ep=[plane_point(u,v,.015) for u,v in edgepix]
 for i,j in [(0,1),(1,2),(3,0)]:
  a,b=ep[i],ep[j];d=b-a;mid=(a+b)/2;box('FrameEdge'+str(i),mid,(float(np.linalg.norm(d[:2])),.026,.03),(.008,.01,.016),np.degrees(np.arctan2(d[1],d[0])))
 for i,j in [(0,1),(1,2),(3,0)]:
  a,b=ep[i],ep[j];mesh('Panel'+str(i),[a,b,b+np.array([0,0,.4]),a+np.array([0,0,.4])],[[0,1,2,3]],(.025,.03,.045))
elif args.scene=='stack':
 for name,uv in [('GreenCube',(345,408)),('RedCube',(405,432))]:pose_box(name,plane_point(*uv,.0125))
else:
 # Correct focal length for the full 1280x720 reference, not the former 640-pixel assumption.
 fit=json.loads((ROOT/'config/fit.json').read_text());K=np.array(fit['K']);campos=np.array(fit['position']);camrot=Rotation.from_matrix(np.array(fit['rotation'])).as_quat()
 top=np.array([plane_point(u,v,.10) for u,v in [(470,151),(690,151),(695,248),(449,248)]])
 tc=top.mean(0);tx=np.linalg.norm(top[1]-top[0]);ty=np.linalg.norm(top[3]-top[0]);pose_box('RaisedTable',tc-np.array([0,0,.009]),(tx,ty,.018))
 for i,xx in enumerate([-1,1]):
  for j,yy in enumerate([-1,1]):pose_box(f'TableLeg{i}{j}',tc+np.array([xx*(tx/2-.01),yy*(ty/2-.01),-.055]),(.014,.014,.09))
 pc=plane_point(725,600,.012);plate_radius=np.linalg.norm(plane_point(846,600,.012)[:2]-plane_point(604,600,.012)[:2])/2;plate=UsdGeom.Mesh(stage.GetPrimAtPath('/World/PinkPlate'));pp=np.array(plate.GetPointsAttr().Get());pp[:,:2]=(pp[:,:2]-np.array([.32,-.11]))*(plate_radius/.12)+pc[:2];plate.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pp.astype('float32')))
 base=tuple(plane_point(1090,475,0));quat=(0.,0.,0.,1.)
 # Bounded visual pose fit, NOT a validated conversion of the source encoder angles.
 raw=np.array(meta['states']['joint_position']['first']);mapped=np.array([raw[0]-np.pi/2,np.pi/2-raw[1],raw[2]-np.pi/2,raw[3],raw[4],raw[5]]);mapped[1]=max(mapped[1],-1.8499);qdict=dict(zip(['waist','shoulder','elbow','forearm_roll','wrist_angle','wrist_rotate'],mapped.tolist()));qdict.update({'left_finger':.022,'right_finger':.022}) # Visual opening estimate from external/wrist references, not calibration.
 inactive_q=fit['inactive_q'];qdict=fit['q'];pleft=plane_point(322,216,0);pright=plane_point(874,216,0);front=plane_point(640,720,0)
 center=(pleft+pright+front)/3;backy=(pleft[1]+pright[1])/2
 pose_box('WhiteWorktop',(center[0],(backy+front[1])/2,-.025),(pright[0]-pleft[0],backy-front[1],.05))
 pose_box('BackWall',(0,backy+.02,.40),(2.6,.025,.80));ray=Rotation.from_quat(camrot).as_matrix()@np.linalg.inv(K)@np.array([640,20,1.]);trimz=(campos+(backy+.24-campos[1])/ray[1]*ray)[2];pose_box('WoodTrim',(0,backy+.24,trimz),(2.6,.04,.035))
 for i,xx in enumerate([pleft[0],pright[0]]):pose_box('FrameRail'+str(i),(xx,(backy+front[1])/2,.015),(.025,backy-front[1],.03))
 for i,xx in enumerate([pleft[0],pright[0]]):box('Upright'+str(i),(xx,backy,.33),(.025,.025,.66),(.008,.009,.01))
 # Mounts and cables are 3-D background geometry; dimensions are provisional.
 for i,uv in enumerate([(105,440),(1090,475)]):
  bp=plane_point(*uv,0);box('Mount'+str(i),bp+np.array([0,0,.015]),(.24,.24,.03),(.008,.009,.01))

# Surface-only reference textures. No robot/object pixels or full background image planes.
texdir=ROOT/'assets/textures';texdir.mkdir(parents=True,exist_ok=True)
def texture_patch(label,ref,rect):
 im=cv2.imread(str(ROOT/'references'/ref));x0,y0,x1,y1=rect;path=texdir/(label+'.png');patch=im[y0:y1,x0:x1];patch=np.concatenate([patch,patch[:,::-1]],axis=1) if 'wood' in label else patch;patch=np.concatenate([patch,patch[::-1]],axis=0) if 'wood' in label else patch;cv2.imwrite(str(path),patch);return path
if args.scene=='stack':
 wood=texture_patch('stack-wood','stack/side_cam-0000.png',(10,385,65,402))
 mesh('BenchGrain',[[-.21,-.34,.0001],[.97,-.34,.0001],[.97,.34,.0001],[-.21,.34,.0001]],[[0,1,2,3]],(1,1,1),[(0,0),(4,0),(4,2),(0,2)],wood)
 cloth=texture_patch('stack-cloth','stack/side_cam-0000.png',(450,20,638,330))
 curtain=UsdGeom.Mesh(stage.GetPrimAtPath('/World/GreenCurtain'));UsdGeom.PrimvarsAPI(curtain).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([(ix/80*3,iz/20) for iz in range(21) for ix in range(81)])
 UsdShade.MaterialBindingAPI.Apply(curtain.GetPrim()).Bind(material('/World/Materials/ClothTexture',(1,1,1),texture=cloth))
 mesh('SideCurtain',[[-.3,-.8,-.1],[-.3,.5,-.1],[-.3,.5,1.2],[-.3,-.8,1.2]],[[0,1,2,3]],(1,1,1),[(0,0),(2,0),(2,1),(0,1)],cloth)
 cyl=UsdGeom.Cylinder.Define(stage,'/World/BaseMount');cyl.CreateRadiusAttr(.084);cyl.CreateHeightAttr(.009);cyl.AddTranslateOp().Set(Gf.Vec3d(0,0,.0045));UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material('/World/Materials/Mount',(.008,.01,.014)))
elif args.scene=='drawer':
 wood=texture_patch('drawer-wood','drawer/exterior_image_1-0000.png',(165,60,275,140))
 wp=np.array(wpts);wp[:,2]=-.1498;mesh('WorktopGrain',wp,[[0,1,2,3]],(1,1,1),[(0,0),(5,0),(5,5),(0,5)],wood)
else:
 wood=texture_patch('plate-wood','plate/front-0000.png',(485,161,675,239))
 tp=top.copy();tp[:,2]=.1001;mesh('RaisedTableGrain',tp,[[0,1,2,3]],(1,1,1),[(0,1),(1,1),(1,0),(0,0)],wood)

robot=Articulation(ArticulationCfg(prim_path='/World/Robot',spawn=sim.UsdFileCfg(usd_path=converter.usd_path,rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=True)),init_state=ArticulationCfg.InitialStateCfg(pos=base,rot=quat,joint_pos=qdict),actuators={'all':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=100.,damping=10.)}))

if args.scene=='plate':
 dark=material('/World/Materials/RobotBlack',(.009,.011,.014),.55)
 UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/Robot')).Bind(dark,bindingStrength=UsdShade.Tokens.weakerThanDescendants)
 # The real scene contains a second inactive arm; preserve it as a separate robot instance.
 b2=tuple(plane_point(105,440,0))
 other=Articulation(ArticulationCfg(prim_path='/World/InactiveRobot',spawn=sim.UsdFileCfg(usd_path=converter.usd_path,rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=True)),init_state=ArticulationCfg.InitialStateCfg(pos=b2,joint_pos=inactive_q),actuators={'all':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=100.,damping=10.)}))
 UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/InactiveRobot')).Bind(dark,bindingStrength=UsdShade.Tokens.weakerThanDescendants)
if args.scene=='drawer':
 dark=material('/World/Materials/GripperDark',(.025,.028,.035))
 for prim in stage.Traverse():
  if str(prim.GetPath()).startswith('/World/Robot/') and any(n in prim.GetName() for n in ['finger','knuckle']):UsdShade.MaterialBindingAPI.Apply(prim).Bind(dark,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
if args.scene=='plate':
  navy=material('/World/Materials/GripperNavy',(.004,.008,.035),.7)
  for prim in stage.Traverse():
   if 'gripper_bar' in prim.GetName():UsdShade.MaterialBindingAPI.Apply(prim).Bind(navy,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
from scene_details import add_details
add_details(stage,args.scene,campos,camrot,K,box,mesh,material,plane_point)

if args.scene=='plate':
 for i,xx in enumerate([-1,1]):
  box('FootBar'+str(i),tc+np.array([xx*(tx/2-.01),0,-.093]),(.014,ty,.014),(.02,.022,.024))
 box('RearFrame',(float((pleft[0]+pright[0])/2),backy,.014),(float(pright[0]-pleft[0]),.025,.028),(.008,.01,.012))
 # Rear wall is a visual-only backdrop in this draft.
 stage.GetPrimAtPath('/World/BackWall').RemoveAPI(UsdPhysics.CollisionAPI)
 for name in ['Detail_ArmLoom0','Detail_ArmLoom1']:stage.RemovePrim('/World/'+name)
 for prim in stage.Traverse():
  if 'gripper_bar' in str(prim.GetPath()):UsdShade.MaterialBindingAPI.Apply(prim).Bind(material('/World/Materials/DarkerBlue',(.003,.006,.025),.8),bindingStrength=UsdShade.Tokens.strongerThanDescendants)
 # Fine circular beads follow the plate rim rather than painting a disk.
 for k,(rad,h) in enumerate([(.091,.0048),(.108,.0105),(.118,.017)]):
  ps=[];fs=[]
  for a in np.linspace(0,2*np.pi,129)[:-1]:
   for b in np.linspace(0,2*np.pi,9)[:-1]:
    rr=rad*plate_radius/.12+.00065*np.cos(b);ps.append([pc[0]+rr*np.cos(a),pc[1]+rr*np.sin(a),h+.00065*np.sin(b)])
  for i in range(128):
   for j in range(8):fs.append([i*8+j,((i+1)%128)*8+j,((i+1)%128)*8+(j+1)%8,i*8+(j+1)%8])
  mesh('RimBead'+str(k),ps,fs,(.47,.155,.145))

sim.DomeLightCfg(intensity=300,color=(.92,.95,1.)).func('/World/Dome',sim.DomeLightCfg(intensity=300,color=(.92,.95,1.)))
light=sim.DistantLightCfg(intensity=650,color=(1.,.94,.84),angle=12.);light.func('/World/Sun',light,orientation=(.92388,.38268,0,0))
if args.scene=='plate' and args.view!='front':
  viewfit=fit[args.view+'_camera'];campos=np.array(viewfit['position']);camrot=Rotation.from_matrix(np.array(viewfit['rotation'])).as_quat();K=np.array(viewfit['K'])
refcam={'stack':'side_cam','drawer':'exterior_image_1','plate':'front'}[args.scene]
width=meta['cameras'][refcam]['width'];height=meta['cameras'][refcam]['height']
# Overscan + analytic lens sampling implements the real principal point without relying on unsupported USD offsets.
rw=width+64;rh=height+64;favg=float((K[0,0]+K[1,1])/2)
renderK=np.array([[favg,0,rw/2],[0,favg,rh/2],[0,0,1.]])
camera=Camera(CameraCfg(prim_path='/World/Camera',width=rw,height=rh,data_types=['rgb'],spawn=sim.PinholeCameraCfg.from_intrinsic_matrix(renderK.flatten().tolist(),width=rw,height=rh,focal_length=24.,clipping_range=(.01,100.))))

# One continuous slab surface material, preserving the original hidden collider.
stage.RemovePrim('/World/RaisedTableGrain')
UsdGeom.Imageable(stage.GetPrimAtPath('/World/RaisedTable')).MakeInvisible()
x0,x1=tc[0]-tx/2,tc[0]+tx/2
y0,y1=tc[1]-ty/2,tc[1]+ty/2
z0,z1=tc[2]-.018,tc[2]
# Top plus sides and bottom. Explicit UVs include the complete top footprint.
quads=[[(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
 [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
 [(x1,y1,z0),(x0,y1,z0),(x0,y1,z1),(x1,y1,z1)],
 [(x0,y1,z0),(x0,y0,z0),(x0,y0,z1),(x0,y1,z1)],
 [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
 [(x0,y1,z0),(x1,y1,z0),(x1,y0,z0),(x0,y0,z0)]]
uvs=[[(0,0),(1,0),(1,1),(0,1)],
 [(0,.08),(1,.08),(1,0),(0,0)],[(1,.92),(0,.92),(0,1),(1,1)],
 [(.08,1),(.08,0),(0,0),(0,1)],[(.92,0),(.92,1),(1,1),(1,0)],
 [(0,1),(1,1),(1,0),(0,0)]]
mesh('ContinuousWoodSlab',[v for face in quads for v in face],[[4*i+j for j in range(4)] for i in range(6)],(1,1,1),[uv for face in uvs for uv in face],ROOT/'assets/textures/subtle-wood.png')

ctx.reset();robot.reset();q=robot.data.default_joint_pos.clone();robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.set_joint_position_target(q)
if args.scene=='plate':
 other.reset();oq=other.data.default_joint_pos.clone();other.write_joint_state_to_sim(oq,torch.zeros_like(oq));other.set_joint_position_target(oq)
camera.set_world_poses(torch.tensor(campos[None],dtype=torch.float32,device=args.device),torch.tensor([[camrot[3],*camrot[:3]]],dtype=torch.float32,device=args.device),convention='ros')
for _ in range(30):ctx.render()
camera.update(0,force_recompute=True);im=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();Image.fromarray(im).save(out/'render-overscan.png');xx,yy=np.meshgrid(np.arange(width),np.arange(height));mx=((xx-K[0,2])*favg/K[0,0]+rw/2).astype('float32');my=((yy-K[1,2])*favg/K[1,1]+rh/2).astype('float32');im=cv2.remap(im,mx,my,cv2.INTER_LINEAR);Image.fromarray(im).save(out/'render.png')
stage.GetRootLayer().Export(str(out/'scene.usda'))
report={'scene':args.scene,'status':'VISUAL_FITTING_DRAFT_NOT_PHYSICS_VALIDATED','robot_usd':converter.usd_path,'joint_names':robot.joint_names,'joint_pos':q[0].cpu().tolist(),'camera_position':campos.tolist(),'camera_quat_xyzw':camrot.tolist(),'K':K.tolist(),'notes':['No policy or training.','Object dimensions and backgrounds are provisional.','Drawer uses the recorded initial zero vector; its validity and timestamp alignment are unverified.','Plate pose uses hypothesized encoder offset/sign conversion with 0.0277-rad shoulder-limit clipping; not a validated calibration. ALOHA-specific meshes replace the generic gripper.','Board textures are rectified real patches with neutral fill under occluded cube pixels; lighting and dimensions need validation.']}
(out/'renderer-source.py').write_text(Path(__file__).read_text());(out/'report.json').write_text(json.dumps(report,indent=2));print('RENDER_COMPLETE',out,flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0) # All artifacts explicitly saved; avoid Kit shutdown deadlock in batch renderer.
