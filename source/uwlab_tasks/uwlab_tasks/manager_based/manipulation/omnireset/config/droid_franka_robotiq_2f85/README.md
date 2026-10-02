# OmniReset cube stacking on the DROID Franka

OmniReset's cube-stacking task with a DROID robot including a Franka Panda arm from the DROID simulation
USD and UWLab's calibrated Robotiq 2F-85 gripper (the UR5e's gripper).

## Requirements

UWLab with **Isaac Lab 2.3.2** (commit `b0542fe`) and **Isaac Sim 5.1.0**. Run all commands from the repo root.

## One-time setup

```bash
# 1. Download DROID robot USD
curl -L -o source/uwlab_assets/data/Robots/DroidFrankaRobotiq2f85/franka_robotiq_2f_85_flattened.usd \
  https://huggingface.co/datasets/owhan/DROID-sim-environments/resolve/main/franka_robotiq_2f_85_flattened.usd
# 2. Mount UWLab's gripper on the Franka arm
python scripts_v2/tools/conversions/build_franka_uwlab_gripper_usd.py
```

## Data (OmniReset steps 1-3)

```bash
D=./Datasets/OmniReset_FrankaUWLabGripper   # must match constants.DATASET_DIR
CUBE="env.scene.insertive_object=cube env.scene.receptive_object=cube"

# Step 1: partial assemblies
python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 \
  --num_envs 10 --num_trajectories 10 --headless --dataset_dir $D $CUBE
# Step 2: grasps
python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 \
  --num_envs 8192 --num_grasps 1000 --headless --dataset_dir $D env.scene.object=cube
# Step 3: reset states
for T in ObjectAnywhereEEAnywhere ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped ObjectRestingEEGrasped; do
  python scripts_v2/tools/record_reset_states.py --task OmniReset-DroidFrankaRobotiq2f85-$T-v0 \
    --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir $D $CUBE
done
```

Step 3 takes 1-4 hours for each reset state type.

## Train and play

```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 3 scripts/reinforcement_learning/rsl_rl/train.py \
  --task OmniReset-DroidFrankaRobotiq2f85-RelCartesianOSC-State-v0 --num_envs 16384 --max_iterations 1000 \
  --logger wandb --headless --distributed $CUBE

python scripts/reinforcement_learning/rsl_rl/play.py --task OmniReset-DroidFrankaRobotiq2f85-RelCartesianOSC-State-Play-v0 \
  --num_envs 4 --checkpoint <run folder>/model_999.pt $CUBE
```