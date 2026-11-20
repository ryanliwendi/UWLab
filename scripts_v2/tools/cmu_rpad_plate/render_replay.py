from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'render_contact.py').read_text()
s=s.replace("p.add_argument('--refine-visuals'","p.add_argument('--recording',required=True);p.add_argument('--refine-visuals'")
a=s.index('for tick in range(round(');b=s.index("(out/'contact-motion.json').write_text",a)
s=s[:a]+r"""
# Render the measured successful contact trajectory with updated visual mesh only.
# No new physics claim: the exported contact/success flags belong to the supplied recording.
summary=json.loads(Path(args.recording).read_text())
for i,rec in enumerate(summary['frames']):
 jq=torch.tensor([rec['joint_positions']],dtype=torch.float32,device=args.device)
 jv=torch.tensor([rec['joint_velocities']],dtype=torch.float32,device=args.device)
 robot.write_joint_state_to_sim(jq,jv)
 pose=torch.tensor([rec['plate_position']+rec['plate_quat_wxyz']],dtype=torch.float32,device=args.device)
 plate.write_root_pose_to_sim(pose)
 for _ in range(2):ctx.render()
 camera.update(0,force_recompute=True);im=camera.data.output['rgb'][0,:,:,:3].cpu().numpy();im=cv2.remap(im,mx,my,cv2.INTER_LINEAR);Image.fromarray(im).save(frames/f'{i:04d}.png')
 if i%30==0:print('REPLAY_FRAME',i,flush=True)
summary['visual_replay_source']='user-supplied recorded states; visual appearance update, no new physics rollout'
"""+s[b:]
exec(compile(s,str(root/'render_contact.py'),'exec'))
