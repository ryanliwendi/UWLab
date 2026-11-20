# CMU RPAD: plate-placement real-to-sim prototype

Standalone Isaac Lab scene with a ViperX 300s/ALOHA gripper, estimated 50° camera, wooden placement table and a dynamic plate. A scripted joint-target controller approaches the rim, closes, lifts, carries, lowers and releases. **This is not a registered OmniReset task or a learned policy.**

## Setup and physical trial

Runtime: Isaac Sim 5.1 / Isaac Lab 2.3.2; NumPy, SciPy, Pillow, OpenCV, h5py and ffmpeg. Activate your Isaac Lab environment and obtain a GPU allocation. Supply your own local episode and front video; no dataset media or recorded experiment results are included.

```bash
P=scripts_v2/tools/cmu_rpad_plate
python "$P/fetch_assets.py"
python "$P/prepare_inputs.py" --episode /path/to/plate_episode.h5
python "$P/plan_controller.py"
python "$P/render_contact.py" --refine-visuals --output "$P/renders/contact" --headless --enable_cameras --device cuda:0
python "$P/export_two_views.py" "$P/renders/contact" --real-video /path/to/front_video.mp4
```

The input HDF5 must provide `robot_profile`, six arm joint positions under `observations/robot_states/joint_position`, and the front video path under `observations/video_paths/front`. Image fitting assumes a 1280x720 plate scene. Dataset inputs, generated textures, robot USDs, controller plans, recorded trajectories and output videos are ignored by Git.

Mesh downloads are checked against SHA-256 hashes; upstream licenses are included. The scene configuration contains estimated camera and robot poses, not measured hardware calibration. These estimates are specific to this scene and may need adjustment for another recording.

## Clean tabletop and detailed plate

To render an existing recorded physical trial with the latest appearance:

```bash
python "$P/render_replay.py" --recording "$P/renders/contact/contact-motion.json" --refine-visuals --detailed-plate --clean-table --output "$P/renders/clean-replay" --headless --enable_cameras --device cuda:0
python "$P/export_two_views.py" "$P/renders/clean-replay" --real-video /path/to/front_video.mp4
```

Replay restores saved poses for rendering. It is **not a new physics trial**, and any displayed physical events belong to the supplied recording. Output `videos/real-vs-sim-two-views.mp4` contains only real left and simulation right, with no close-up panels. It runs at 15 fps without phase warping and visibly holds the last real frame if that video ends first. Use a 15 fps real reference for time-matched comparisons.

## Physics and checks

The physical controller commands only robot joint targets; the free plate moves through contact, with no attachment or plate-pose commands during the physical rollout. Physics runs at 240 Hz with gravity compensation. Arm gains are 400/40; finger gains are 2000/124 with a 35 N limit. Plate mass is 0.06 kg, friction 0.8/0.6, and the rim uses compound convex colliders. These are provisional simulation settings.

Verified lift requires both finger forces above 0.05 N for 200 ms and a plate-origin rise above 3 cm. Placement requires the center in the target region, origin near the tabletop, separation from the fingers and stable face-up settling. Results are written locally to `contact-motion.json`.

## Validation and limitations

Python and generated-renderer syntax, asset-hash checks, local input preparation and controller-plan structure were checked during packaging. A GPU rerun of the relocated package has not been completed. No benchmark results or internal experiment records are distributed here.

Camera calibration, geometry, inactive-arm pose and physical parameters remain estimates. The wall is visual-only, and added plate rings do not change the collider. Validate the packaged scene and its reset stability before adding OmniReset observations/actions, resets, rewards or training. No existing policies or evaluation banks are modified.
