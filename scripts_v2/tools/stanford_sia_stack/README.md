# Stanford SIA: stack red cube on green cube

Standalone Isaac Lab real-to-sim prototype with a Franka FR3/Hand, fitted wooden table, continuous green curtain, and five 6-by-8 marker sheets. The red and green cubes start on the fitted central-sheet cells. A scripted IK controller approaches, closes, lifts, transports, lowers, releases and retreats. **This is not a registered OmniReset environment or a trained policy.**

## Run

Runtime: Isaac Sim 5.1 / Isaac Lab 2.3.2, NumPy, SciPy, Pillow, OpenCV with ArUco, h5py and ffmpeg. Activate Isaac Lab and obtain a GPU allocation. Supply the Stanford SIA stack episode and its local side/top videos.

```bash
P=scripts_v2/tools/stanford_sia_stack
python "$P/fetch_assets.py"
python "$P/prepare_inputs.py" --episode /path/to/stack_episode.h5
python "$P/plan_controller.py"
python "$P/render_contact.py" --output "$P/renders/contact" --headless --enable_cameras --device cuda:0
python "$P/check_result.py" "$P/renders/contact"
python "$P/export_two_views.py" "$P/renders/contact" --real-video /path/to/side_video.mp4
```

Outputs: `motion.json`, `validation.json`, frame PNGs, `videos/simulation.mp4`, and `videos/real-vs-sim.mp4` (real left, sim right). The scripted sequence lasts 325 frames at 15 fps. Comparison uses elapsed time; individual real/sim action events are not phase-aligned. If the real video ends first, its last frame is held and labeled.

Static views:

```bash
python "$P/scene.py" --output "$P/renders/side" --headless --enable_cameras --device cuda:0
STACK_VIEW=top python "$P/scene.py" --output "$P/renders/top" --headless --enable_cameras --device cuda:0
```

`STACK_VALIDATE_ONLY=1` skips frame rendering in the physical rollout; do not run the video exporter on that output.

## Inputs and assets

The HDF5 requires `robot_profile` metadata, seven arm joints at `observations/robot_states/joint_position`, and `side_cam`/`top_cam` video paths under `observations/video_paths`. Fitting assumes 640x480 references. Other episodes need camera/geometry refitting; this is not automatic scene reconstruction.

Mesh downloads are pinned to an upstream revision and SHA-256 checked. Franka license/notice files accompany the derived URDF. Dataset frames, videos, generated textures, mesh downloads, USDs, controller plans and experiment results stay local and are ignored by Git. The five sheets are generated from ArUco marker IDs, not pasted scene images. The wood and curtain textures use small surface-only reference crops.

## Physics and checks

Only arm and finger joint targets are commanded. Cubes are free rigid bodies with no attachments or scripted object transforms. Physics is 120 Hz; gravity and robot gravity compensation remain active. Arm stiffness/damping is 2000/90; hand stiffness/damping is 2000/124 with a 35 N actuator limit. Cubes are 25 mm and 0.03 kg, friction 0.8/0.6, restitution zero. These are provisional simulation parameters, not calibrated hardware gains.

Verified lift requires bilateral finger forces above 0.05 N for 24 consecutive physics ticks and a cube rise above 2 cm. The final-second stack check requires horizontal center offset below 4 mm, vertical center separation 25±3 mm, red-cube speed below 10 mm/s and finger forces below 0.05 N throughout that second. This is a prototype check, not a benchmark success rate.

The local result includes contact forces, cube positions, red-cube linear velocity, actual TCP, joint positions, controller phase, target finger opening and applied torque. Camera fitting, robot geometry, curtain folds and physical parameters remain approximate. Cloth is static visual geometry.

## Packaging validation

Python syntax and undefined-name checks passed. Pinned local mesh hashes, HDF5 input preparation and IK controller generation were checked in a relocated package. The source controller demonstrated contact pickup and stacking before packaging. The relocated GPU physics-only check also completed with verified lift and a released, settled stack; a fresh video render of the relocated package was not repeated. No training or existing policy changes are included.
