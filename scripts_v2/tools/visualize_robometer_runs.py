"""Label the same episodes with every fine-tuned Robometer run, and save a chart and two-panel videos per run.

Each run folder (written by finetune_robometer.py) is labeled with its last checkpoint (the highest
adapter_step<N>.pt, else adapter_last.pt; or adapter_step<--step>.pt), on the run's own camera, using every --frame_stride-th frame of the whole
episode (the frame grid of --context training). Zero-shot Robometer labels the same videos for comparison. Episodes
whose id is a multiple of 10 are held out (the validation split of preference_pairs.py); the others are training
episodes.

Output in --out_dir/<run>/:
  progress.png            every episode: zero-shot and fine-tuned progress next to the final critic's V(s)/11
  episode_<id>.mp4        camera video (left) and the same curves drawn in real time (right)
  labels.npz, info.txt    the progress values, and which checkpoint and camera were used

Two steps, the first on a GPU with Robometer's Python environment, the second on any machine:
  /scr/liryan/robometer/.venv/bin/python scripts_v2/tools/visualize_robometer_runs.py label \
      --dataset ./Datasets/CubeStack_SuccessFailure_v2 --episodes 129 506 518 0 330 640 --out_dir ./outputs/robometer_runs
  python scripts_v2/tools/visualize_robometer_runs.py render --dataset ./Datasets/CubeStack_SuccessFailure_v2 \
      --out_dir ./outputs/robometer_runs

compare draws several already-labeled runs (same episodes and frames) in one chart and one video per episode, in
--out_dir/<--name>/:
  python scripts_v2/tools/visualize_robometer_runs.py compare --dataset <dataset> --out_dir <out_dir> \
      --runs <run>_step<N> <run>_step<N> --names "full head" "head LoRA" --name <folder>
"""

import argparse
import csv
import json
import os
import re
from multiprocessing import Pool

import imageio.v2 as iio
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK = "Pick up the green cube and stack it on top of the blue cube."
SURF, INK, INK2, GRID, BLUE, ORANGE, GREEN = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
VMAX = 11.0  # about the critic's ceiling (stacked and staying stacked)


def last_checkpoint(run_dir):
    steps = [int(m.group(1)) for f in os.listdir(run_dir) if (m := re.fullmatch(r"adapter_step(\d+)\.pt", f))]
    if steps:
        return os.path.join(run_dir, f"adapter_step{max(steps)}.pt"), f"step {max(steps)}"
    if os.path.exists(os.path.join(run_dir, "adapter_last.pt")):
        return os.path.join(run_dir, "adapter_last.pt"), "last"
    return None, None


def match_keys(state, model_keys):
    """Rename a checkpoint's progress-head keys to the labeling model's: wrapping a layer with LoRA moves its own
    weights under .base_layer, so a fully trained head's "progress_head.0.weight" becomes
    "progress_head.0.base_layer.weight" here."""
    renamed = {}
    for k, v in state.items():
        if k not in model_keys:
            parent, _, leaf = k.rpartition(".")
            if f"{parent}.base_layer.{leaf}" in model_keys:
                k = f"{parent}.base_layer.{leaf}"
        renamed[k] = v
    unknown = set(renamed) - set(model_keys)
    if unknown:
        raise ValueError(f"checkpoint has {len(unknown)} tensors the labeling model lacks, e.g. {min(unknown)}")
    return renamed


def read_index(dataset):
    with open(os.path.join(dataset, "index.csv")) as f:
        return {int(r["episode_id"]): r for r in csv.DictReader(f)}


def label(args):
    """GPU step: progress of every episode from zero-shot Robometer and from each run's last checkpoint."""
    import torch
    from decord import VideoReader, cpu
    from peft import LoraConfig, inject_adapter_in_model

    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator, setup_peft_model

    device = torch.device("cuda")
    exp_config, tokenizer, processor, model = load_model_from_hf(model_path=args.model_path, device=device)
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    for p in model.parameters():
        p.requires_grad = False
    # every weight any --trainable mode of fine-tuning can change: LoRA on the language model, LoRA on the progress
    # head and the progress head itself. Fresh LoRA adapters are zero, so this state is exactly zero-shot Robometer.
    model = setup_peft_model(model, exp_config.peft)
    linears = [n for n, m in model.progress_head.named_modules() if isinstance(m, torch.nn.Linear)]
    inject_adapter_in_model(LoraConfig(r=exp_config.peft.r, lora_alpha=exp_config.peft.lora_alpha,
                                       lora_dropout=exp_config.peft.lora_dropout, target_modules=linears),
                            model.progress_head)
    for p in model.progress_head.parameters():
        p.requires_grad = True
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device).eval()
    zero_shot = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}

    @torch.no_grad()
    def progress(frames):
        traj = Trajectory(frames=frames, frames_shape=tuple(frames.shape), task=TASK, id="0",
                          metadata={"subsequence_length": int(frames.shape[0])}, video_embeddings=None)
        inputs = collator([ProgressSample(trajectory=traj, sample_type="progress")])["progress_inputs"]
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"), pixel_values_videos=inputs.get("pixel_values_videos"),
                image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=inputs.get("second_per_grid_ts"), sample_type="progress",
            )  # fmt: skip
        logits = out.progress_logits["A"][0].float()  # (F, bins)
        return (torch.softmax(logits, dim=-1) @ torch.linspace(0.0, 1.0, logits.shape[-1], device=device)).cpu().numpy()

    index = read_index(args.dataset)
    frames_cache, zero_shot_cache = {}, {}  # (camera, episode) -> all frames / (camera, episode, k) -> zero-shot progress

    def episode_frames(camera, e):
        if (camera, e) not in frames_cache:
            vr = VideoReader(os.path.join(args.dataset, index[e][f"{camera}_mp4"]), ctx=cpu(0))
            frames_cache[(camera, e)] = vr.get_batch(list(range(len(vr)))).asnumpy()
        return frames_cache[(camera, e)]

    def prefix_indices(t, k):
        """Robometer's linspace_subsample_frames on frames 0..t (as finetune_robometer.py --context)."""
        if t + 1 <= k:
            return list(range(t + 1))
        idx = np.rint(np.linspace(0, t, k)).astype(int).tolist()
        idx[0], idx[-1] = 0, t
        return idx

    def label_episode(camera, e, k):
        """Progress on every --frame_stride-th frame: one video of those frames (k = 0), or, for a run trained with
        --context, each frame scored on its own k-frame video from the episode start."""
        frames = episode_frames(camera, e)
        idx = np.arange(0, int(index[e]["length_steps"]), args.frame_stride)
        if not k:
            return progress(frames[idx])
        return np.array([np.atleast_1d(progress(frames[prefix_indices(int(t), k)]))[-1] for t in idx])  # 1 frame: scalar

    runs = args.runs or sorted(d for d in os.listdir(args.runs_dir) if os.path.exists(os.path.join(args.runs_dir, d, "args.json")))
    for run in runs:
        run_dir = os.path.join(args.runs_dir, run)
        ckpt, which = last_checkpoint(run_dir)
        if args.step is not None:
            ckpt, which = os.path.join(run_dir, f"adapter_step{args.step}.pt"), f"step {args.step}"
            if not os.path.exists(ckpt):
                print(f"skipping {run}: no {os.path.basename(ckpt)}", flush=True)
                continue
        if ckpt is None:
            print(f"skipping {run}: no checkpoint", flush=True)
            continue
        with open(os.path.join(run_dir, "args.json")) as f:
            run_args = json.load(f)
        camera = run_args["camera"]
        # runs trained with the current --context are labeled that way (zero-shot too); older --context runs had no
        # context_frames and are labeled as one video like the others
        k = run_args.get("context_frames", 0) if run_args.get("context") else 0
        state = match_keys(torch.load(ckpt), zero_shot)
        out = {}
        for e in args.episodes:
            if (camera, e, k) not in zero_shot_cache:
                model.load_state_dict(zero_shot, strict=False)
                zero_shot_cache[(camera, e, k)] = label_episode(camera, e, k)
            out[f"zero_shot_{e}"] = zero_shot_cache[(camera, e, k)]
        model.load_state_dict(zero_shot, strict=False)  # weights the checkpoint leaves out stay at zero-shot
        model.load_state_dict(state, strict=False)
        for e in args.episodes:
            out[f"finetuned_{e}"] = label_episode(camera, e, k)
        run_out = os.path.join(args.out_dir, run if args.step is None else f"{run}_step{args.step}")
        os.makedirs(run_out, exist_ok=True)
        np.savez(os.path.join(run_out, "labels.npz"), episodes=np.array(args.episodes), frame_stride=args.frame_stride,
                 **out)
        with open(os.path.join(run_out, "info.txt"), "w") as f:
            f.write(f"run: {run}\ncheckpoint: {ckpt} ({which})\ncamera: {camera}\n"
                    f"frames: every {args.frame_stride}th frame of each episode"
                    + (f", each scored on its own {k}-frame video from the episode start\n" if k else "\n"))
        print(f"labeled {run} with {which} ({camera} camera)", flush=True)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)


def curves(dataset, index, labels, e):
    r = index[e]
    d = np.load(os.path.join(dataset, r["npz"]))
    n = len(d["value_ref"])
    t = np.arange(n) / 10
    stacked = np.r_[False, d["stacked"][:-1] > 0.5]  # frame t shows s_t; stacked[t-1] is its flag
    tr = np.arange(0, n, int(labels["frame_stride"])) / 10
    return r, t, stacked, np.clip(d["value_ref"] / VMAX, -0.05, 1.05), tr, labels[f"zero_shot_{e}"], labels[f"finetuned_{e}"]


def title(e, r):
    return f"ep {e} ({'held-out' if e % 10 == 0 else 'training'}), iter {r['iteration']}: " + (
        "success" if r["success"] == "1" else "failure")


def render_chart(dataset, run_out):
    labels = np.load(os.path.join(run_out, "labels.npz"))
    index = read_index(dataset)
    episodes = list(labels["episodes"])
    train, held = [e for e in episodes if e % 10], [e for e in episodes if e % 10 == 0]
    rows = [row for row in (train, held) if row]
    cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(len(rows), cols, figsize=(5 * cols, 3.4 * len(rows)), dpi=110, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURF)
    for ax_row, row in zip(axes, rows):
        for ax, e in zip(ax_row, row):
            r, t, stacked, v, tr, zs, ft = curves(dataset, index, labels, e)
            style(ax)
            ax.fill_between(t, -0.05, 1.08, where=stacked, step="post", color=GREEN, alpha=0.15, lw=0, label="cube stacked")
            ax.plot(t, v, color=INK2, lw=2, ls="--", label="critic V(s) / 11")
            ax.plot(tr, zs, color=BLUE, lw=2, label="progress, zero-shot")
            ax.plot(tr, ft, color=ORANGE, lw=2, label="progress, fine-tuned")
            ax.set_title(title(e, r), loc="left", color=INK if r["success"] == "1" else "#c0392b", fontsize=11)
            ax.set_xlim(0, t[-1])
            ax.set_ylim(-0.05, 1.08)
        for ax in ax_row[len(row):]:
            ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    with open(os.path.join(run_out, "info.txt")) as f:
        info = dict(line.split(": ", 1) for line in f.read().splitlines())
    fig.suptitle(f"{info['run']}  ({info['checkpoint'].split('(')[-1].rstrip(')')}, {info['camera']} camera)   "
                 "top: training episodes, bottom: held-out", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(run_out, "progress.png"), facecolor=SURF)
    plt.close(fig)


def render_video(job):
    dataset, run_out, e, camera = job
    labels = np.load(os.path.join(run_out, "labels.npz"))
    index = read_index(dataset)
    r, t, stacked, v, tr, zs, ft = curves(dataset, index, labels, e)
    frames = [f for f in iio.get_reader(os.path.join(dataset, r[f"{camera}_mp4"]))]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.patch.set_facecolor(SURF)
    writer = iio.get_writer(os.path.join(run_out, f"episode_{e:05d}.mp4"), fps=10, macro_block_size=8)
    ok = r["success"] == "1"
    for i, frame in enumerate(frames[: len(t)]):
        ax.clear()
        style(ax)
        ax.fill_between(t[: i + 1], -0.05, 1.08, where=stacked[: i + 1], step="post", color=GREEN, alpha=0.18, lw=0,
                        label="cube stacked")
        ax.plot(t[: i + 1], v[: i + 1], color=INK2, lw=2, ls="--", label="critic V(s) / 11")
        k = max(np.searchsorted(tr, t[i], side="right"), 1)  # labeled frames shown so far
        ax.plot(tr[:k], zs[:k], color=BLUE, lw=2.2, label="progress, zero-shot")
        ax.plot(tr[:k], ft[:k], color=ORANGE, lw=2.2, label="progress, fine-tuned")
        for y, col in ((zs[k - 1], BLUE), (ft[k - 1], ORANGE)):
            ax.plot(tr[k - 1], y, "o", ms=8, color=col, mec=SURF, mew=2)
        ax.plot(t[i], v[i], "o", ms=8, color=INK2, mec=SURF, mew=2)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("time (s)", color=INK2)
        ax.set_title(f"{title(e, r)}\nfine-tuned {ft[k - 1]:.2f}   zero-shot {zs[k - 1]:.2f}   critic {v[i]:.2f}",
                     loc="left", color=INK if ok else "#c0392b", fontsize=11)
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
        fig.tight_layout()
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        left = np.asarray(Image.fromarray(frame).resize((640, 480), Image.BILINEAR))
        writer.append_data(np.concatenate([left, right[:480, :640]], axis=1))
    writer.close()
    plt.close(fig)
    return f"{os.path.basename(run_out)}/episode_{e:05d}.mp4"


def render(args):
    """CPU step: charts and videos from the saved labels."""
    run_outs = sorted(os.path.join(args.out_dir, d) for d in os.listdir(args.out_dir)
                      if os.path.exists(os.path.join(args.out_dir, d, "labels.npz")))
    if args.runs:
        run_outs = [p for p in run_outs if os.path.basename(p) in args.runs]
    jobs = []
    for run_out in run_outs:
        render_chart(args.dataset, run_out)
        with open(os.path.join(run_out, "info.txt")) as f:
            camera = dict(line.split(": ", 1) for line in f.read().splitlines())["camera"]
        jobs += [(args.dataset, run_out, int(e), camera) for e in np.load(os.path.join(run_out, "labels.npz"))["episodes"]]
    with Pool(args.workers) as pool:
        for done in pool.imap_unordered(render_video, jobs):
            print("saved", done, flush=True)


COMPARE = ("#eb6834", "#7b52c7", "#c2185b", "#8a6d1f")  # fine-tuned runs, in --runs order


def compare_curves(dataset, index, run_outs, e):
    """curves() for several runs labeled on the same frames; the zero-shot curve is the first run's."""
    labels = [np.load(os.path.join(o, "labels.npz")) for o in run_outs]
    r, t, stacked, v, tr, zs, _ = curves(dataset, index, labels[0], e)
    return r, t, stacked, v, tr, zs, [lab[f"finetuned_{e}"] for lab in labels]


def compare_chart(dataset, run_outs, names, out, episodes):
    index = read_index(dataset)
    cols = min(3, len(episodes))
    rows = -(-len(episodes) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.6 * rows), dpi=110, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURF)
    for ax, e in zip(axes.flat, episodes):
        r, t, stacked, v, tr, zs, fts = compare_curves(dataset, index, run_outs, int(e))
        style(ax)
        ax.fill_between(t, -0.05, 1.08, where=stacked, step="post", color=GREEN, alpha=0.15, lw=0, label="cube stacked")
        ax.plot(t, v, color=INK2, lw=2, ls="--", label="critic V(s) / 11")
        ax.plot(tr, zs, color=BLUE, lw=2, label="zero-shot")
        for ft, name, col in zip(fts, names, COMPARE):
            ax.plot(tr, ft, color=col, lw=2, label=name)
        ax.set_title(title(int(e), r), loc="left", color=INK if r["success"] == "1" else "#c0392b", fontsize=11)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.05, 1.08)
    for ax in axes.flat[len(episodes):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(os.path.basename(out) + ":  " + ",  ".join(os.path.basename(o) for o in run_outs), x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "progress.png"), facecolor=SURF)
    plt.close(fig)


def compare_video(job):
    dataset, run_outs, names, out, e, camera = job
    index = read_index(dataset)
    r, t, stacked, v, tr, zs, fts = compare_curves(dataset, index, run_outs, e)
    frames = [f for f in iio.get_reader(os.path.join(dataset, r[f"{camera}_mp4"]))]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.patch.set_facecolor(SURF)
    writer = iio.get_writer(os.path.join(out, f"episode_{e:05d}.mp4"), fps=10, macro_block_size=8)
    ok = r["success"] == "1"
    for i, frame in enumerate(frames[: len(t)]):
        ax.clear()
        style(ax)
        ax.fill_between(t[: i + 1], -0.05, 1.08, where=stacked[: i + 1], step="post", color=GREEN, alpha=0.18, lw=0,
                        label="cube stacked")
        ax.plot(t[: i + 1], v[: i + 1], color=INK2, lw=2, ls="--", label="critic V(s) / 11")
        k = max(np.searchsorted(tr, t[i], side="right"), 1)  # labeled frames shown so far
        series = [(zs, BLUE, "zero-shot"), *zip(fts, COMPARE, names)]
        for y, col, name in series:
            ax.plot(tr[:k], y[:k], color=col, lw=2.2, label=name)
            ax.plot(tr[k - 1], y[k - 1], "o", ms=8, color=col, mec=SURF, mew=2)
        ax.plot(t[i], v[i], "o", ms=8, color=INK2, mec=SURF, mew=2)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("time (s)", color=INK2)
        ax.set_title(f"{title(e, r)}   critic {v[i]:.2f}\n"
                     + "   ".join(f"{name} {y[k - 1]:.2f}" for y, _, name in series), loc="left",
                     color=INK if ok else "#c0392b", fontsize=9.5)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        left = np.asarray(Image.fromarray(frame).resize((640, 480), Image.BILINEAR))
        writer.append_data(np.concatenate([left, right[:480, :640]], axis=1))
    writer.close()
    plt.close(fig)
    return f"{os.path.basename(out)}/episode_{e:05d}.mp4"


def compare(args):
    """CPU step: one chart and one video per episode with every run's fine-tuned curve."""
    run_outs = [os.path.join(args.out_dir, r) for r in args.runs]
    names = args.names or args.runs
    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    episodes = [int(e) for e in np.load(os.path.join(run_outs[0], "labels.npz"))["episodes"]
                if args.episodes is None or int(e) in args.episodes]
    compare_chart(args.dataset, run_outs, names, out, episodes)
    with open(os.path.join(run_outs[0], "info.txt")) as f:
        camera = dict(line.split(": ", 1) for line in f.read().splitlines())["camera"]
    jobs = [(args.dataset, run_outs, names, out, int(e), camera) for e in episodes]
    with Pool(args.workers) as pool:
        for done in pool.imap_unordered(compare_video, jobs):
            print("saved", done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("label", "render"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, help="folder written by collect_success_failure.py")
        p.add_argument("--out_dir", required=True)
        p.add_argument("--runs", nargs="*", default=None, help="run folder names (default: all)")
    p = sub.choices["label"]
    p.add_argument("--model_path", default="robometer/Robometer-4B")
    p.add_argument("--runs_dir", default="./logs/robometer_pref")
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--frame_stride", type=int, default=3)
    p.add_argument("--step", type=int, default=None,
                   help="use adapter_step<step>.pt instead of the last checkpoint (output folder <run>_step<step>)")
    sub.choices["render"].add_argument("--workers", type=int, default=8)
    p = sub.add_parser("compare")
    p.add_argument("--dataset", required=True, help="folder written by collect_success_failure.py")
    p.add_argument("--out_dir", required=True, help="folder holding the labeled run folders")
    p.add_argument("--runs", nargs="+", required=True, help="labeled run folder names (<run> or <run>_step<N>)")
    p.add_argument("--names", nargs="+", default=None, help="legend name of each run (default: folder names)")
    p.add_argument("--name", required=True, help="output folder name in --out_dir")
    p.add_argument("--episodes", type=int, nargs="+", default=None, help="subset of the labeled episodes (default: all)")
    p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    {"label": label, "render": render, "compare": compare}[args.cmd](args)
