# CMU Auton: close-drawer real-to-sim prototype

A standalone Isaac Lab scene that replays a real xArm7 approach, physically pushes a passive drawer closed, and holds the pushing pose. The handle is parallel to the drawer front. This is **not a registered OmniReset training environment or a learned policy**.

## Run

Tested runtime: Isaac Sim 5.1 / Isaac Lab 2.3.2, with PyTorch, NumPy, SciPy, Pillow, OpenCV, h5py and ffmpeg. Activate your Isaac Lab environment and allocate a GPU. From the repository root:

```bash
python scripts_v2/tools/cmu_auton_drawer/prepare_inputs.py \
  --dataset-root /scr/shared/datasets/oopsie_data/OopsieData-Submissions
python scripts_v2/tools/cmu_auton_drawer/render_drawer_contact.py \
  --output scripts_v2/tools/cmu_auton_drawer/renders/friction \
  --headless --enable_cameras --device cuda:0
python scripts_v2/tools/cmu_auton_drawer/export_drawer_contact_comparison.py friction
```

Inputs come from `CMU_Auton/20261001_180831.h5` and its exterior video (70 frames, 15 fps). Dataset files, generated USDs, textures and videos are ignored and not distributed here. Included xArm meshes/URDF derive from UFACTORY's xarm_ros commit `cf51617d98bc55a88d9aa735acfa43a0bbb8e425`; its BSD license and conversion provenance are preserved in `assets/`.

Optional matched controls:

- `--rail-friction 0 --rail-damping 0`: no explicit rail resistance.
- `--no-robot-contact`: filter robot–drawer collisions; diagnostic only.

## What the simulation does

- Estimated table height: −0.08 m in the robot-base frame; geometry is fitted to image landmarks.
- Seven recorded arm joints are interpolated at 120 Hz and tracked by position actuators. Gripper opening conversion and gains are provisional; arm gravity is disabled.
- A 0.15 kg compound tray moves on a passive prismatic guide. No drawer position commands or closing-force schedule are applied.
- Rail resistance is `F = -0.6*tanh(v/0.01) - 2*v` N along the slide, using measured velocity in m/s. This is a smooth Coulomb-like/viscous approximation, not measured friction or exact static stiction.
- Once closure is within 0.1 mm and robot contact exceeds 0.5 N, hold the measured pushing pose plus 10% of the forward tracking error. No retreat is commanded.
- Log every physics tick, render the original 70-frame duration, then check an additional one-second hold without extending the real-video comparison.

`contact-motion.json` contains actual joint states, targets, drawer travel, robot contact forces, resistance forces and hold checks. The video compares frames at nominal 15 Hz; real sensor timestamp alignment has not been established. The renderer exits after explicitly saving artifacts because Kit shutdown hung in this tested runtime.

## Measured results on this one replay

| Variant | Final travel | Peak robot contact | End-of-hold contact |
|---|---:|---:|---:|
| Parallel handle, continuous push | 61.38 mm | 47.56 N | Not checked |
| Push and hold, no explicit rail resistance | 61.40 mm | 12.76 N | 0.53 N |
| Push and hold, higher resistance (default) | 61.40 mm | 23.98 N | 0.84 N |

The drawer remained closed during the additional one-second hold. An earlier matched contact-disabled control remained open; no contact-disabled rerun of the higher-resistance variant is claimed. These are single-trajectory results, not a robustness benchmark. Force peaks remain a calibration concern.

Latest validated development outputs on Snoopy:
`/scr/burakbar/projects/oopsie-real2sim/renders/drawer-contact-80397/`

Videos are `videos/drawer-real-vs-sim.mp4`, `videos/drawer-sim.mp4`, and `videos/drawer-real.mp4`.

## Remaining work

Validate dimensions, gripper assembly, collision details, gains, friction, reset stability and timing against real measurements. Handle brackets and chamfers are visual-only; simplified collision cubes remain. Then implement task observations/actions, reset distributions, rewards and physical success criteria for OmniReset. Do not use this visual/contact prototype as a ready PPO task or its outputs as ground-truth critic labels. No training, existing policy, or reserved evaluation bank is modified by this contribution.

## Packaging validation

The repository-ready copy was run independently on Snoopy (job 80411), using `prepare_inputs.py` and the commands above. It reproduced the default development result exactly: 61.4033 mm final travel, 23.9792 N peak contact, and 0.8443 N end-of-hold contact. All three exported videos decoded to 70 frames at 15 fps. Python compilation, Black/isort formatting, Ruff undefined/unused checks, mesh-path resolution and Git whitespace checks passed. The full repository pre-commit suite was not run; the runtime emits existing Isaac Sim/force-API deprecation warnings.
