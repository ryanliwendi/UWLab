# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a critic for a fixed OmniReset policy under a binary success reward (policy evaluation).

The actor of --checkpoint is frozen (weights, exploration noise and observation normalizer) and a freshly initialized
critic is trained with PPO's value loss in the usual OmniReset training environment (same resets). Every reward term is
switched off except success_reward, so the reward is 1 * step_dt = 0.1 at each step the cube is stacked and aligned and
0 otherwise; V(s) is then the discounted future stacked time of the policy (at most 0.1 / (1 - gamma) = 10).

Checkpoints (model_<it>.pt, same format as training) go to logs/rsl_rl/<experiment>/<time>_<run_name>/; their critic
can relabel recorded episodes from the saved critic observations.

Usage:
    python -m torch.distributed.run --nnodes 1 --nproc_per_node 1 scripts_v2/tools/train_binary_critic.py --headless \\
        --task OmniReset-DroidFrankaRobotiq2f85-RelCartesianOSC-State-v0 --policy_checkpoint <run>/model_1998.pt \\
        --num_envs 4096 --max_iterations 500 --run_name binary_critic --logger wandb \\
        env.scene.insertive_object=cube env.scene.receptive_object=cube
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../scripts/reinforcement_learning/rsl_rl")
)
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Train a binary-reward critic for a frozen OmniReset policy.")
parser.add_argument("--task", type=str, required=True, help="Training task, e.g. OmniReset-...-State-v0.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--policy_checkpoint", type=str, required=True, help="Policy checkpoint whose actor is evaluated.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--critic_lr", type=float, default=1e-3, help="Fixed learning rate for the critic.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

# reward terms that stay on: success_reward is the binary reward; progress_context returns 0 but computes the
# alignment success_reward reads, and the reward manager skips zero-weight terms
KEEP_REWARDS = ("success_reward", "progress_context")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations
    # a frozen actor has zero KL, which would make the adaptive schedule raise the learning rate every update; and a
    # fresh critic should regress its returns directly instead of moving at most clip_param per update
    agent_cfg.algorithm.schedule = "fixed"
    agent_cfg.algorithm.learning_rate = args_cli.critic_lr
    agent_cfg.algorithm.use_clipped_value_loss = False
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        env_cfg.seed = agent_cfg.seed = agent_cfg.seed + app_launcher.local_rank

    # binary success reward
    for name, term in vars(env_cfg.rewards).items():
        if hasattr(term, "weight") and name not in KEEP_REWARDS:
            term.weight = 0.0
    print("[INFO] reward weights:", {n: t.weight for n, t in vars(env_cfg.rewards).items() if hasattr(t, "weight")})

    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name,
                                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                           + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else "")))
    env_cfg.log_dir = log_dir
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # load the policy, then: fresh critic, everything else frozen
    print(f"[INFO] Loading policy from: {args_cli.policy_checkpoint}")
    runner.load(args_cli.policy_checkpoint, load_optimizer=False)
    runner.current_learning_iteration = 0
    policy = runner.alg.policy
    for module in policy.critic.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    for name, param in policy.named_parameters():
        param.requires_grad = name.startswith("critic.")
    policy.actor_obs_normalizer.update = lambda *args, **kwargs: None  # keep the actor's input statistics fixed
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    print(f"[INFO] trainable: {trainable}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "evaluated_policy.txt"), "w") as f:
        f.write(f"{os.path.abspath(args_cli.policy_checkpoint)}\n")
    actor_before = {n: p.detach().clone() for n, p in policy.named_parameters() if not p.requires_grad}

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    changed = [n for n, p in policy.named_parameters() if n in actor_before and not torch.equal(p, actor_before[n])]
    print(f"[INFO] frozen parameters changed during training: {changed or 'none'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
