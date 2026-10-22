"""Evaluate an OmniReset critic on recorded episodes and visualize it (CPU only, no simulator).

The critic of --checkpoint (e.g. from train_binary_critic.py) scores every step of every episode of a dataset written by
collect_success_failure.py, from the critic observations saved with each episode. Writes to --out_dir:
  values.npz              V(s) for every episode and step (NaN after an episode's end), with the episode ids
  episodes.png            chosen episodes: this critic and the dataset's reference critic (value_ref) over time
  aggregate.png           successful episodes aligned at the moment the cube is first stacked, and the critic read as
                          time to go, d = log(V / V_max) / log(gamma), against the true time until stacking
  episode_<id>.mp4        camera video next to the two critics drawn in real time

Usage:
  python scripts_v2/tools/critic_values.py --checkpoint <run>/model_499.pt --dataset ./Datasets/CubeStack_SuccessFailure_v2 \\
      --reward_per_step 0.1 --episodes 129 506 518 0 330 640 --out_dir ./outputs/binary_critic
--check_reference first scores the episodes with the critic that produced value_ref, to check the evaluation matches.
"""

import argparse
import csv
import os
from multiprocessing import Pool

import imageio.v2 as iio
import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID, BLUE, ORANGE, GREEN = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
DT = 0.1  # seconds per policy step


def load_critic(checkpoint):
    """Critic MLP plus its observation normalizer, rebuilt from a training checkpoint and its params/agent.yaml."""
    with open(os.path.join(os.path.dirname(checkpoint), "params", "agent.yaml")) as f:
        policy_cfg = yaml.safe_load(f)["policy"]
    act = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[policy_cfg["activation"]]
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    linears = sorted({int(k.split(".")[1]) for k in sd if k.startswith("critic.")})
    layers = []
    for i, idx in enumerate(linears):
        w, b = sd[f"critic.{idx}.weight"], sd[f"critic.{idx}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.load_state_dict({"weight": w, "bias": b})
        layers += [lin] + ([act()] if i < len(linears) - 1 else [])
    mlp = torch.nn.Sequential(*layers).eval()
    mean, std = sd["critic_obs_normalizer._mean"], sd["critic_obs_normalizer._std"]
    eps = 1e-2  # rsl_rl EmpiricalNormalization default

    @torch.no_grad()
    def critic(obs):
        x = (torch.as_tensor(obs, dtype=torch.float32) - mean) / (std + eps)
        return mlp(x)[:, 0].numpy()

    return critic, float(yaml.safe_load(open(os.path.join(os.path.dirname(checkpoint), "params", "agent.yaml")))
                         ["algorithm"]["gamma"])


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)


def episode_title(e, r):
    return f"ep {e} ({'held-out' if e % 10 == 0 else 'training'}), iter {r['iteration']}: " + (
        "success" if r["success"] == "1" else "failure")


def render_video(job):
    dataset, out_dir, r, v_new, v_ref, stacked, label_new, label_ref, camera = job
    e = int(r["episode_id"])
    frames = [f for f in iio.get_reader(os.path.join(dataset, r[f"{camera}_mp4"]))]
    n = len(v_new)
    t = np.arange(n) * DT
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.patch.set_facecolor(SURF)
    writer = iio.get_writer(os.path.join(out_dir, f"episode_{e:05d}.mp4"), fps=10, macro_block_size=8)
    for i, frame in enumerate(frames[:n]):
        ax.clear()
        style(ax)
        ax.fill_between(t[: i + 1], -0.05, 1.08, where=stacked[: i + 1], step="post", color=GREEN, alpha=0.18, lw=0,
                        label="cube stacked")
        ax.plot(t[: i + 1], v_ref[: i + 1], color=INK2, lw=2, ls="--", label=label_ref)
        ax.plot(t[: i + 1], v_new[: i + 1], color=ORANGE, lw=2.4, label=label_new)
        ax.plot(t[i], v_new[i], "o", ms=8, color=ORANGE, mec=SURF, mew=2)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("time (s)", color=INK2)
        ax.set_title(f"{episode_title(e, r)}\n{label_new.split(' (')[0]} {v_new[i]:.2f}   "
                     f"{label_ref.split(' (')[0]} {v_ref[i]:.2f}", loc="left",
                     color=INK if r["success"] == "1" else "#c0392b", fontsize=11)
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
        fig.tight_layout()
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        left = np.asarray(Image.fromarray(frame).resize((640, 480), Image.BILINEAR))
        writer.append_data(np.concatenate([left, right[:480, :640]], axis=1))
    writer.close()
    plt.close(fig)
    return f"episode_{e:05d}.mp4"


def main(args):
    with open(os.path.join(args.dataset, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    critic, gamma = load_critic(args.checkpoint)
    v_max = args.reward_per_step / (1 - gamma)

    if args.check_reference:
        ref_critic, _ = load_critic(args.check_reference)
        err = max(np.abs(ref_critic(d["critic_obs"]) - d["value_ref"]).max()
                  for d in (np.load(os.path.join(args.dataset, r["npz"])) for r in rows[:: max(len(rows) // 20, 1)]))
        print(f"reference check: max |recomputed - value_ref| = {err:.2e} on {len(rows[:: max(len(rows) // 20, 1)])} episodes")

    max_len = max(int(r["length_steps"]) for r in rows)
    values = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    ref = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    stacked = np.zeros((len(rows), max_len), dtype=bool)  # stacked[i, t]: state s_t is stacked
    for i, r in enumerate(rows):
        d = np.load(os.path.join(args.dataset, r["npz"]))
        n = len(d["value_ref"])
        values[i, :n], ref[i, :n] = critic(d["critic_obs"]), d["value_ref"]
        stacked[i, 1:n] = d["stacked"][: n - 1] > 0.5  # d["stacked"][t] flags s_{t+1}
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "values.npz"), episode_id=np.array([int(r["episode_id"]) for r in rows]),
                        value=values, value_ref=ref, checkpoint=args.checkpoint, gamma=gamma, v_max=v_max)
    success = np.array([r["success"] == "1" for r in rows])
    if args.values_only:
        print(f"saved {os.path.join(args.out_dir, 'values.npz')}: V range {np.nanmin(values):.2f}..{np.nanmax(values):.2f}")
        return
    print(f"V range {np.nanmin(values):.2f}..{np.nanmax(values):.2f} (V_max {v_max:.1f}); last step: successes median "
          f"{np.nanmedian(values[success, -1]):.2f}, failures median {np.nanmedian(values[~success, -1]):.2f}")

    ref_scale = args.ref_scale
    label_new, label_ref = f"binary critic V / {v_max:g}", f"original critic V / {ref_scale:g}"
    index = {int(r["episode_id"]): i for i, r in enumerate(rows)}

    # chosen episodes
    eps = args.episodes
    train, held = [e for e in eps if e % 10], [e for e in eps if e % 10 == 0]
    grid = [g for g in (train, held) if g]
    cols = max(len(g) for g in grid)
    fig, axes = plt.subplots(len(grid), cols, figsize=(5 * cols, 3.4 * len(grid)), dpi=110, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURF)
    for ax_row, g in zip(axes, grid):
        for ax, e in zip(ax_row, g):
            i = index[e]
            n = int(rows[i]["length_steps"])
            t = np.arange(n) * DT
            style(ax)
            ax.fill_between(t, -0.05, 1.08, where=stacked[i, :n], step="post", color=GREEN, alpha=0.15, lw=0,
                            label="cube stacked")
            ax.plot(t, ref[i, :n] / ref_scale, color=INK2, lw=2, ls="--", label=label_ref)
            ax.plot(t, values[i, :n] / v_max, color=ORANGE, lw=2.2, label=label_new)
            ax.set_title(episode_title(e, rows[i]), loc="left", color=INK if success[i] else "#c0392b", fontsize=11)
            ax.set_xlim(0, t[-1])
            ax.set_ylim(-0.05, 1.08)
        for ax in ax_row[len(g):]:
            ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Binary-reward critic vs original critic   (top: training episodes, bottom: held-out)", x=0.01,
                 ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "episodes.png"), facecolor=SURF)
    plt.close(fig)

    # successful episodes aligned at the first stacked step, and the critic read as time to go
    lags = np.arange(-100, 31)  # steps relative to the first stacked state
    aligned, ttg_true, ttg_pred = [], [], []
    for i in np.flatnonzero(success):
        n = int(rows[i]["length_steps"])
        k = int(np.argmax(stacked[i, :n]))
        row = np.full(len(lags), np.nan)
        ok = (k + lags >= 0) & (k + lags < n)
        row[ok] = values[i, k + lags[ok]] / v_max
        aligned.append(row)
        steps = np.arange(0, k)
        ttg_true += list((k - steps) * DT)
        ttg_pred += list(np.log(np.clip(values[i, steps] / v_max, 1e-3, None)) / np.log(gamma) * DT)
    aligned = np.array(aligned)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)
    fig.patch.set_facecolor(SURF)
    ax = axes[0]
    style(ax)
    lo, mid, hi = np.nanpercentile(aligned, [25, 50, 75], axis=0)
    ax.fill_between(lags * DT, lo, hi, color=ORANGE, alpha=0.2, lw=0, label="middle 50% of successful episodes")
    ax.plot(lags * DT, mid, color=ORANGE, lw=2.4, label="median, binary critic V / V_max")
    pre = lags <= 0
    ax.plot(lags[pre] * DT, gamma ** (-lags[pre]), color=INK2, lw=1.8, ls="--", label="ideal: gamma^(steps to go)")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("time relative to first stacking (s)", color=INK2)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(f"{success.sum()} successful episodes, aligned at stacking", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax = axes[1]
    style(ax)
    ttg_true, ttg_pred = np.array(ttg_true), np.array(ttg_pred)
    bins = np.arange(0, ttg_true.max() + 0.5, 0.5)
    which = np.digitize(ttg_true, bins)
    centers = [ttg_true[which == b].mean() for b in np.unique(which) if (which == b).sum() >= 20]
    meds = [np.median(ttg_pred[which == b]) for b in np.unique(which) if (which == b).sum() >= 20]
    q = [np.percentile(ttg_pred[which == b], [25, 75]) for b in np.unique(which) if (which == b).sum() >= 20]
    ax.fill_between(centers, [a for a, _ in q], [b for _, b in q], color=BLUE, alpha=0.2, lw=0, label="middle 50%")
    ax.plot(centers, meds, color=BLUE, lw=2.4, marker="o", ms=4, label="median critic time to go")
    lim = max(centers) if centers else 1
    ax.plot([0, lim], [0, lim], color=INK2, lw=1.8, ls="--", label="perfect")
    ax.set_xlabel("true time until stacking (s)", color=INK2)
    ax.set_ylabel("critic time to go: log(V / V_max) / log(gamma) (s)", color=INK2)
    ax.set_title("Binary critic read as time to go (successful episodes, before stacking)", loc="left", color=INK,
                 fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "aggregate.png"), facecolor=SURF)
    plt.close(fig)

    jobs = []
    for e in eps:
        i = index[e]
        n = int(rows[i]["length_steps"])
        jobs.append((args.dataset, args.out_dir, rows[i], values[i, :n] / v_max, ref[i, :n] / ref_scale, stacked[i, :n],
                     label_new, label_ref, args.camera))
    with Pool(args.workers) as pool:
        for done in pool.imap_unordered(render_video, jobs):
            print("saved", done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="training checkpoint (model_<it>.pt) whose critic is used")
    parser.add_argument("--dataset", required=True, help="folder written by collect_success_failure.py")
    parser.add_argument("--reward_per_step", type=float, default=0.1, help="reward per stacked step (V_max = r / (1 - gamma))")
    parser.add_argument("--ref_scale", type=float, default=11.0, help="divide the reference critic's value_ref by this")
    parser.add_argument("--episodes", type=int, nargs="+", default=[129, 506, 518, 0, 330, 640])
    parser.add_argument("--camera", choices=["front", "wrist"], default="front")
    parser.add_argument("--check_reference", default=None, help="checkpoint that produced value_ref, to verify the setup")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--values_only", action="store_true", help="only write values.npz (no plots or videos)")
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
