# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a success / failure dataset for OmniReset cube stacking from trained RL checkpoints.

Every episode is a full attempt from the task's Play reset (cube on the table, gripper anywhere). Each ``--policy`` is a
checkpoint, optionally with Gaussian action noise (``path:noise_std``). For every episode the script saves:

* ``episode_XXXXX.npz``: per-step robot and cube states, policy / critic observations, actions, rewards, critic values
  (``value``: the acting checkpoint's critic, ``value_ref``: the ``--critic_checkpoint`` critic, the same for all
  episodes) and ``stacked`` (cube stacked after the step)
* ``episode_XXXXX_front.mp4`` and ``episode_XXXXX_wrist.mp4``: 320x240 camera videos at the policy rate
* one row in ``index.csv``. An episode is a success when the cube is stacked at its last step (the training metric).

Usage:
    python scripts_v2/tools/collect_success_failure.py --headless \\
        --task OmniReset-DroidFrankaRobotiq2f85-RelCartesianOSC-State-Play-v0 \\
        --policy <run>/model_1998.pt --policy <run>/model_1998.pt:0.5 --policy <run>/model_800.pt \\
        --critic_checkpoint <run>/model_1998.pt --episodes_per_policy 160 --num_envs 32 \\
        --out_dir ./Datasets/CubeStack_SuccessFailure env.scene.insertive_object=cube env.scene.receptive_object=cube
"""

import argparse
import copy
import csv
import math
import os
import re
import sys

from isaaclab.app import AppLauncher

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../scripts/reinforcement_learning/rsl_rl")
)
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Collect a success / failure dataset from trained RL checkpoints.")
parser.add_argument("--task", type=str, required=True, help="Play task, e.g. OmniReset-...-State-Play-v0.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--policy", action="append", required=True, help="Checkpoint path, optionally ':<action noise>'.")
parser.add_argument("--critic_checkpoint", type=str, required=True, help="Checkpoint whose critic scores all episodes.")
parser.add_argument("--episodes_per_policy", type=int, default=160)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument(
    "--wrist_parent", type=str, default="UWLabGripper/robotiq_base_link", help="Robot prim the wrist camera rides on."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio.v2 as iio
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import TiledCameraCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

# Same cameras as the UR5e RGB data collection (config/ur5e_robotiq_2f85/data_collection_rgb_cfg.py)
FRONT_CAMERA = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
    update_period=0,
    height=240,
    width=320,
    offset=TiledCameraCfg.OffsetCfg(
        pos=(1.0770121, -0.1679045, 0.4486344),
        rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
        convention="opengl",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
)
WRIST_CAMERA_OFFSET = TiledCameraCfg.OffsetCfg(
    pos=(0.0182505, -0.00408447, -0.0689107),
    rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
    convention="opengl",
)

INDEX_FIELDS = [
    "episode_id", "checkpoint", "iteration", "action_noise", "seed", "round", "env", "success", "first_stacked_s",
    "frac_time_stacked", "return", "length_steps", "early_termination", "npz", "front_mp4", "wrist_mp4",
]  # fmt: skip


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.rerender_on_reset = True  # camera images show the new episode right after a reset
    env_cfg.scene.front_camera = FRONT_CAMERA
    env_cfg.scene.wrist_camera = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{args_cli.wrist_parent}/rgb_wrist_camera",
        update_period=0,
        height=240,
        width=320,
        offset=WRIST_CAMERA_OFFSET,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.critic_checkpoint)
    # frozen copy of the reference critic; only the critic part, since the actor's noise distribution can't be copied
    ref_critic = copy.deepcopy(torch.nn.Sequential(runner.alg.policy.critic_obs_normalizer, runner.alg.policy.critic))
    ref_critic.eval()

    robot = unwrapped.scene["robot"]
    cube, base_cube = unwrapped.scene["insertive_object"], unwrapped.scene["receptive_object"]
    cameras = {"front": unwrapped.scene["front_camera"], "wrist": unwrapped.scene["wrist_camera"]}
    progress = unwrapped.reward_manager.get_term_cfg("progress_context").func
    ee_idx = unwrapped.action_manager.get_term("arm")._ee_body_idx
    origins = unwrapped.scene.env_origins
    num_envs, max_len, dt = unwrapped.num_envs, int(unwrapped.max_episode_length), unwrapped.step_dt

    os.makedirs(args_cli.out_dir, exist_ok=True)
    index_path = os.path.join(args_cli.out_dir, "index.csv")
    episode_id = 0
    if os.path.exists(index_path):
        with open(index_path) as f:
            episode_id = sum(1 for _ in csv.DictReader(f))
    with open(index_path, "a", newline="") as index_file:
        index = csv.DictWriter(index_file, fieldnames=INDEX_FIELDS)
        if episode_id == 0:
            index.writeheader()

        for spec in args_cli.policy:
            path, _, noise = spec.partition(":")
            noise = float(noise) if noise else 0.0
            iteration = int(re.search(r"model_(\d+)\.pt", path).group(1))
            runner.load(path)
            policy, net = runner.get_inference_policy(device=unwrapped.device), runner.alg.policy
            num_success = num_episodes = 0
            for round_idx in range(math.ceil(args_cli.episodes_per_policy / num_envs)):
                with torch.inference_mode():  # the env's buffers became inference tensors during the stepping below
                    obs, _ = vec_env.reset()
                ids = list(range(episode_id, episode_id + num_envs))
                episode_id += num_envs
                writers = {
                    (i, cam): iio.get_writer(
                        os.path.join(args_cli.out_dir, f"episode_{ids[i]:05d}_{cam}.mp4"),
                        fps=round(1 / dt),
                        macro_block_size=8,
                    )
                    for i in range(num_envs)
                    for cam in cameras
                }
                steps = {k: [] for k in ["policy_obs", "critic_obs", "joint_pos", "joint_vel", "ee_pos", "ee_quat", "cube_pos",
                                         "cube_quat", "base_cube_pos", "base_cube_quat", "action", "value", "value_ref",
                                         "reward", "stacked"]}  # fmt: skip
                in_episode = torch.ones(num_envs, dtype=torch.bool, device=unwrapped.device)
                length = torch.full((num_envs,), max_len, device=unwrapped.device)
                for t in range(max_len):
                    # state s_t, its camera images and critic values, then the action taken from it
                    with torch.inference_mode():
                        action = policy(obs)
                        if noise > 0:
                            action = action + noise * torch.randn_like(action)
                        steps["value"].append(net.evaluate(obs)[:, 0])
                        steps["value_ref"].append(ref_critic(net.get_critic_obs(obs))[:, 0])
                    ee_pos, ee_quat = math_utils.subtract_frame_transforms(
                        robot.data.root_pos_w, robot.data.root_quat_w,
                        robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx],
                    )  # fmt: skip
                    steps["policy_obs"].append(obs["policy"])
                    steps["critic_obs"].append(obs["critic"])
                    steps["joint_pos"].append(robot.data.joint_pos.clone())
                    steps["joint_vel"].append(robot.data.joint_vel.clone())
                    steps["ee_pos"].append(ee_pos)
                    steps["ee_quat"].append(ee_quat)
                    steps["cube_pos"].append(cube.data.root_pos_w - origins)
                    steps["cube_quat"].append(cube.data.root_quat_w.clone())
                    steps["base_cube_pos"].append(base_cube.data.root_pos_w - origins)
                    steps["base_cube_quat"].append(base_cube.data.root_quat_w.clone())
                    steps["action"].append(action)
                    images = {cam: sensor.data.output["rgb"][..., :3].cpu().numpy() for cam, sensor in cameras.items()}
                    for i in torch.nonzero(in_episode).flatten().tolist():
                        for cam, frames in images.items():
                            writers[(i, cam)].append_data(frames[i])
                    # step; 'stacked' describes the state after the step
                    with torch.inference_mode():
                        obs, reward, done, _ = vec_env.step(action)
                    steps["reward"].append(reward.clone())  # the env reuses its reward buffer every step
                    steps["stacked"].append(progress.success.clone())
                    ended = done.bool() & in_episode
                    length[ended] = t + 1
                    in_episode &= ~done.bool()
                for writer in writers.values():
                    writer.close()

                data = {k: torch.stack(v, dim=1).float().cpu().numpy() for k, v in steps.items()}  # (num_envs, T, ...)
                for i in range(num_envs):
                    n = int(length[i].item())
                    episode = {k: v[i, :n] for k, v in data.items()}
                    stacked = episode["stacked"] > 0.5
                    success = bool(stacked[-1])
                    name = f"episode_{ids[i]:05d}"
                    np.savez_compressed(os.path.join(args_cli.out_dir, name + ".npz"), **episode)
                    first = np.flatnonzero(stacked)
                    index.writerow({
                        "episode_id": ids[i], "checkpoint": path, "iteration": iteration, "action_noise": noise,
                        "seed": args_cli.seed, "round": round_idx, "env": i, "success": int(success),
                        "first_stacked_s": round(float(first[0] + 1) * dt, 2) if len(first) else "",
                        "frac_time_stacked": round(float(stacked.mean()), 3), "return": round(float(episode["reward"].sum()), 3),
                        "length_steps": n, "early_termination": int(n < max_len), "npz": name + ".npz",
                        "front_mp4": name + "_front.mp4", "wrist_mp4": name + "_wrist.mp4",
                    })  # fmt: skip
                    num_success += int(success)
                    num_episodes += 1
                index_file.flush()
                print(f"[{os.path.basename(path)} noise {noise}] round {round_idx}: {num_success}/{num_episodes} successes",
                      flush=True)  # fmt: skip

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
