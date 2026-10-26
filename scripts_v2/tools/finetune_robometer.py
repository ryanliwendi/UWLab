"""Fine-tune Robometer with a simulation-trained critic using segment preferences and/or dense value distillation.

Segment preferences: for each segment, Robometer predicts progress p_t on the scored frame.
The whole segment is scored with --progress_reduce (sum, mean, last minus first, or the sum of every
scored frame's gain over the first) on the frames. Its preference is calculated using Bradley-Terry over those scores,
    P(chosen > rejected) = sigmoid((score[chosen] - score[rejected]) / temperature),
trained with cross-entropy against the critic's label.

Dense value distillation: N progress clips per step with the critic's normalized value as the target. 
An episode of --train_dataset is first reduced to at most 64 evenly spaced frames (Robometer's preprocessing cap); 
three distinct frames are drawn and used as start/middle/end by a forward, reverse or rewind strategy (or optionally forward only);
get_segment_indices_with_middle turns them into a clip of at most the model's max_frames frames. Every frame of
the clip (subsampled evenly to max_frames or padded to max_frames by repeating its last frame) is
trained with Robometer's discrete progress loss (cross-entropy against C51-style two-bin targets) toward the critic's
V(s) at that frame. The critic's value is calculated by its unnormalized value divided by the median last-state value 
of the dataset's successful episodes, clipped to [0, 1]. In addition, failures are also included. 
The two losses for preferences and distillation are summed. 

Which components of robometer can be trained:
  lm_lora_head  a LoRA adapter on the language model plus the whole progress head (default)
  head          the whole progress head only
  head_lora     a LoRA adapter on the progress head's linear layers only
  lm_lora_head_lora  LoRA adapters on both the language model and the progress head's linear layers

Frames fed to Robometer for a segment start..end :
  default    every --frame_stride-th (default: 3) frame of the segment are scored. These frames are passed once into
             robometer, and all the progress scores are extracted in that forward pass. 
             These scores are then reduced for the preference loss.
             One potential issue is that Robometer's prompt calls the first frame "the starting state", so each clip is judged as if the task
             began there.
  --context  every --frame_stride-th frame of the segment is scored on its own, using their own subsampled frames (spread evenly from 0 to t)that include 
             the history before the segment. Each frame gets its own forward pass in robometer, where only the last
             score (corresponding to the score of this frame) is extracted.
Note that --context only affects the segment preference training and the evaluation protocol; it does not affect value distillation training.

Run with Robometer's Python environment on a GPU, e.g. preferences with the test-time protocol:
  /scr/liryan/robometer/.venv/bin/python scripts_v2/tools/finetune_robometer.py --context --camera front \
      --trainable head --progress_reduce cum_delta --temperature 0.7 \
      --train_pairs /scr/shared/real_to_sim_rewards/CubeStack_SuccessFailure_v3/preference_pairs/random_L30_all_n100000.jsonl \
      --val_pairs ./Datasets/CubeStack_SuccessFailure_val/preference_pairs/random_L30_n1000.jsonl \
      --train_dataset /scr/shared/real_to_sim_rewards/CubeStack_SuccessFailure_v3 --out_dir ./logs/robometer_pref/run1
or dense value distillation only (evaluated with the test-time protocol):
  /scr/liryan/robometer/.venv/bin/python scripts_v2/tools/finetune_robometer.py --context --camera front \
      --trainable head --pairs_per_step 0 --distill_clips 8 \
      --val_pairs ./Datasets/CubeStack_SuccessFailure_val/preference_pairs/random_L30_n1000.jsonl \
      --train_dataset /scr/shared/real_to_sim_rewards/CubeStack_SuccessFailure_v3 --out_dir ./logs/robometer_pref/run2
Add --eval_only to measure the zero-shot model (or a saved --adapter) without training.

Each evaluation reports seven metrics:
  val_acc  preference accuracy on the first --num_val_pairs validation pairs (logit > 0: agrees with the critic),
           with the same segment frames, --progress_reduce as training
  eval_progress_loss  only with --distill_clips: the distillation loss on --num_val_clips clips from the episodes of
           --val_dataset. The clips are drawn the same way as the training clips, and every evaluation uses the same
           clips.
  pearson  on the successful episodes of --val_dataset: Pearson correlation between progress and time index,
           averaged over episodes
  kendall  on the (successful, suboptimal, failure) triplets of --kendall_val_dataset
           (generate_robometer_finetuning_data.py triplets, episodes of --val_dataset): each episode scored by its
           average progress; Robometer's compute_kendall between the scores and the ranks 3, 2, 1, averaged over
           triplets
  kendall_binary  on the state pairs of --kendall_binary_pairs (generate_robometer_finetuning_data.py kendall_pairs:
           two different episodes of --val_dataset, each at a random frame t up to its first stacked frame for a
           success, its last frame for a failure): each state scored by Robometer's progress at frame t given the
           episode's frames 0..t; Robometer's compute_kendall between the two scores and the critic's V(s_t),
           averaged over pairs
  pearson_critic  on every episode of --val_dataset: Pearson correlation between progress and the ground-truth
           critic's V(s) (value_ref in the episode's .npz) on the same frames, averaged over episodes
  pearson_critic_train  the same on --num_train_eval_episodes episodes of --train_dataset (a fixed random subset; only
           with --train_dataset): how well Robometer fits the critic on episodes it trains on
How pearson, kendall and pearson_critic* score an episode: Robometer predicts progress on every --frame_stride-th frame, 
up to the first frame showing the cube stacked for a success, to the last frame for a failure.
Without --context these frames are fed as one video into robometer; with --context each frame t gets its own video of
--context_frames=8 frames spread from frame 0 to t, as in training.
kendall_binary feeds frames 0..t the same way: every --frame_stride-th frame plus t as one video, keeping t's score;
with --context t's own video.
Outputs in --out_dir:
  args.json           the run's flags (args_resume<N>.json for a run resumed at step N)
  metrics.csv         one row per evaluation: step, training losses and accuracy since the previous evaluation, the
                      seven metrics above, learning rate, wall time (sec) and peak GPU memory (gpu_gb)
  adapter_step<N>.pt  only the trained weights (LoRA and/or progress head) at step N, saved at every evaluation and
                      every --save_every steps; load one with --adapter
  trainer_state.pt    optimizer and learning-rate schedule state at the latest adapter_step<N>.pt, for --resume
"""

import argparse
import csv
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from peft import LoraConfig, inject_adapter_in_model
from torch.utils.data import DataLoader, Dataset

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.data.datasets.helpers import (
    get_segment_indices_with_middle,
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
)
from robometer.evals.eval_metrics_utils import compute_kendall, compute_pearson
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator, setup_peft_model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_path", default="robometer/Robometer-4B")
parser.add_argument("--train_pairs", default=None,
                    help=".jsonl from generate_robometer_finetuning_data.py sample (not needed with --eval_only or "
                         "--pairs_per_step 0)")
parser.add_argument("--val_pairs", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--camera", choices=["front", "wrist"], default="front")
parser.add_argument("--context", action="store_true",
                    help="include the segment's history when scoring the frames")
parser.add_argument("--context_frames", type=int, default=8, help="frames per video with --context (Robometer: 8)")
parser.add_argument("--frame_stride", type=int, default=3, help="frame spacing of the inputs and eval episodes")
parser.add_argument("--progress_reduce", choices=["sum", "mean", "delta", "cum_delta"], default="cum_delta",
                    help="segment score from the scored frames' progress p_0..p_n: sum, mean, delta = p_n - p_0, or "
                         "cum_delta = sum_i (p_i - p_0) (use mean or delta when segment lengths can differ)")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--distill_clips", type=int, default=0, help="dense value distillation clips per step (see above)")
parser.add_argument("--distill_strategies", nargs="+", choices=["forward", "reverse", "rewind"],
                    default=["forward", "reverse", "rewind"], help="clip strategies used for distillation")
parser.add_argument("--pairs_per_step", type=int, default=4, help="preference pairs per step (0: no preference loss)")
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=1000, help="optimizer steps")
parser.add_argument("--eval_every", type=int, default=100)
parser.add_argument("--save_every", type=int, default=100,
                    help="save a checkpoint every this many steps between evals (0: only at evals), so a "
                         "preempted run loses less")
parser.add_argument("--num_val_pairs", type=int, default=500)
parser.add_argument("--num_val_clips", type=int, default=200, help="validation clips for eval_progress_loss")
parser.add_argument("--val_dataset", default="/scr/shared/real_to_sim_rewards/CubeStack_SuccessFailure_val",
                    help="validation episodes (collect_success_failure.py), for pearson and pearson_critic")
parser.add_argument("--kendall_val_dataset", default="/scr/shared/real_to_sim_rewards/CubeStack_KendallTriplets_val",
                    help="folder with triplets.jsonl (generate_robometer_finetuning_data.py triplets) of --val_dataset "
                         "episodes, for kendall")
parser.add_argument("--kendall_binary_pairs", default="/scr/shared/real_to_sim_rewards/CubeStack_SuccessFailure_val/kendall_binary_pairs.jsonl",
                    help="state pairs (generate_robometer_finetuning_data.py kendall_pairs) of --val_dataset "
                         "episodes, for kendall_binary")
parser.add_argument("--train_dataset", default=None,
                    help="training episodes (collect_success_failure.py folder; the training pairs' episodes): the "
                         "--distill_clips clips and their critic targets, and pearson_critic_train")
parser.add_argument("--num_train_eval_episodes", type=int, default=100, help="training episodes evaluated for pearson_critic_train")
parser.add_argument("--trainable", choices=["lm_lora_head", "head", "head_lora", "lm_lora_head_lora"],
                    default="head")
parser.add_argument("--adapter", default=None, help="adapter_*.pt to start from (or to evaluate with --eval_only)")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--resume", action="store_true", help="continue a stopped run in --out_dir from its last adapter")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--wandb", action="store_true", help="also log metrics to Weights & Biases")
parser.add_argument("--wandb_project", default="robometer_pref")
parser.add_argument("--wandb_entity", default=None, help="wandb team or user (default: your default entity)")
parser.add_argument("--wandb_name", default=None, help="run name (default: the --out_dir folder name)")
args = parser.parse_args()


def prefix_indices(t, k):
    """Return k evenly spaced frame indices from t frames, the first and t included."""
    if t + 1 <= k:
        return list(range(t + 1))
    idx = np.rint(np.linspace(0, t, k)).astype(int).tolist()
    idx[0], idx[-1] = 0, t
    return idx


def queries(scored):
    """Frame sequences to feed to Robometer and how many frames are scored in each sequence. 
    With --context one video per scored frame; without, one video."""
    if args.context:
        return [(prefix_indices(int(t), args.context_frames), 1) for t in scored]
    return [(list(map(int, scored)), len(scored))]  # All scored frames scored together


def segment_frames(seg):
    """Queries for a segment: its every --frame_stride-th frame is scored (see queries)."""
    return queries(np.arange(seg["start"], seg["end"] + 1, args.frame_stride))


def load_videos(path, qs):
    """Decodes each frame the queries need once; returns [(frames (F, H, W, 3), n_scored)] in query order."""
    need = sorted({i for idx, _ in qs for i in idx})
    frames = VideoReader(path, ctx=cpu(0)).get_batch(need).asnumpy()
    pos = {i: j for j, i in enumerate(need)}
    return [(frames[[pos[i] for i in idx]], n) for idx, n in qs]


class ShuffledStream(torch.utils.data.Sampler):
    """Yields shuffled pair indices, a new permutation each epoch fixed by the seed. Used for training."""

    def __init__(self, n, seed, skip=0):
        self.n, self.seed, self.skip = n, seed, skip

    def __iter__(self):
        epoch, skip = divmod(self.skip, self.n)  # Skip for resumed runs
        while True:
            yield from np.random.default_rng([self.seed, epoch]).permutation(self.n)[skip:].tolist()
            epoch, skip = epoch + 1, 0


class PairDataset(Dataset):
    """Loads the frames of both segments of each pair from the episode videos."""

    def __init__(self, path, limit=None):
        with open(path) as f:
            self.pairs = [json.loads(line) for line in f]
        if limit is not None:
            self.pairs = self.pairs[:limit]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        item = {"task": p["task"]}
        for side in ("chosen", "rejected"):
            item[side] = load_videos(p[side][f"{args.camera}_mp4"], segment_frames(p[side]))
        return item


def progress_batch(model, collator, videos, task, device):
    """Queries robometer with one or more videos of the same length and returns progress values as 
    well as raw bin logits."""
    samples = [ProgressSample(trajectory=Trajectory(frames=v, frames_shape=tuple(v.shape), task=task, id=str(i),
                                                    metadata={"subsequence_length": int(v.shape[0])},
                                                    video_embeddings=None), sample_type="progress")
               for i, v in enumerate(videos)]
    inputs = collator(samples)["progress_inputs"]
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"), pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"), sample_type="progress",
        )  # fmt: skip
    # (B, F, bins); a one-frame video comes back without its frame axis
    logits = torch.stack([l if l.dim() == 2 else l[None] for l in (out.progress_logits["A"][b] for b in range(len(videos)))])
    logits = logits.float()
    if logits.shape[:2] != (len(videos), len(videos[0])):
        raise RuntimeError(f"expected {len(videos)} x {len(videos[0])} progress values, got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=-1) @ torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device), logits


def scored_progress(model, collator, videos, task, device, batch=1):
    """Runs Robometer on each video from load_videos and keeps its last n_scored (1 for context, all for no-context) 
    progress values: one progress value per scored frame, in order."""
    out = [None] * len(videos)
    by_len = {}
    for i, (frames, _) in enumerate(videos):
        by_len.setdefault(len(frames), []).append(i)
    for ids in by_len.values():
        for s in range(0, len(ids), batch):
            chunk = ids[s : s + batch]
            p, _ = progress_batch(model, collator, [videos[i][0] for i in chunk], task, device)
            for b, i in enumerate(chunk):
                out[i] = p[b, -videos[i][1]:]
    return torch.cat(out)


def load_distill_episodes(path, ref=None):
    """Loads every episode of a dataset folder with its video paths and its distillation targets: the critic's value
    at each frame, divided by the median final value of the successful episodes and clipped to [0, 1].
    If ref is given, the values are divided by ref instead, so validation targets use the training normalizer.
    Returns (episodes, the divisor)."""
    with open(os.path.join(path, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    values = [np.load(os.path.join(path, r["npz"]))["value_ref"] for r in rows]
    if ref is None:
        ref = float(np.median([v[-1] for v, r in zip(values, rows) if r["success"] == "1"]))
    return [{"episode_id": int(r["episode_id"]), "target": np.clip(v / ref, 0.0, 1.0),
             **{k: os.path.join(path, r[k]) for k in ("front_mp4", "wrist_mp4")}} for r, v in zip(rows, values)], ref


def progress_loss(logits, target):
    """Robometer's progress loss: cross-entropy between each frame's 10-bin logits and its target value split over the
    two nearest bins."""
    bins = logits.shape[-1]
    b = torch.as_tensor(np.asarray(target), dtype=torch.float32, device=logits.device) * (bins - 1)
    lo, hi = b.floor().long(), b.ceil().long()
    probs = torch.zeros_like(logits)
    probs.scatter_add_(1, lo[:, None], (hi - b)[:, None])
    probs.scatter_add_(1, hi[:, None], (b - lo)[:, None])
    same = lo == hi  # on a bin: all the probability there
    probs[same] = F.one_hot(lo[same], bins).float()
    return -(probs * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def pair_logit(model, collator, pair, device):
    """Scores both segments of a pair (--progress_reduce of their per-frame progress) and returns
    (score_chosen - score_rejected) / temperature."""
    scores = []
    for side in ("chosen", "rejected"):
        p = scored_progress(model, collator, pair[side], pair["task"], device)
        scores.append({"sum": p.sum(), "mean": p.mean(), "delta": p[-1] - p[0], "cum_delta": (p - p[0]).sum()}[
            args.progress_reduce])
    return (scores[0] - scores[1]) / args.temperature


class DistillClips(Dataset):
    """Given a clip index, picks a random episode, builds a max_frames clip from it with
    the forward, reverse or rewind strategy, and returns the frames with their normalized critic values as targets."""

    def __init__(self, episodes, task, max_frames, ratios, seed, cap=64):
        self.episodes, self.task = episodes, task
        self.max_frames, self.seed, self.cap = max_frames, seed, cap
        self.strategies = [k for k, r in ratios.items() if r > 0]
        self.p = np.array([ratios[k] for k in self.strategies], dtype=float) / sum(ratios.values())

    def __len__(self):
        return 10**12

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        ep = self.episodes[int(rng.integers(len(self.episodes)))]
        T = len(ep["target"])
        stored = np.rint(np.linspace(0, T - 1, min(T, self.cap))).astype(int)  # the episode after preprocessing
        strategy = str(rng.choice(self.strategies, p=self.p))
        f1, f2, f3 = sorted(rng.choice(len(stored), 3, replace=False).tolist())
        start, middle, end = {"forward": (f1, f2, f3), "reverse": (f3, f2, f1), "rewind": (f1, f3, f2)}[strategy]
        idx = get_segment_indices_with_middle(num_frames_total=len(stored), start_idx=start, end_idx=end,
                                              middle_idx=middle if strategy == "rewind" else None,
                                              max_frames=self.max_frames)
        if len(idx) > self.max_frames:  # Robometer then subsamples evenly to max_frames ...
            idx = [idx[k] for k in linspace_subsample_frames(np.array(idx), self.max_frames)[1]]
        frames = stored[idx]
        (video, _), = load_videos(ep[f"{args.camera}_mp4"], [(frames.tolist(), len(frames))])
        # ... and pads a shorter clip to max_frames by repeating its last frame and target
        video, target = pad_trajectory_to_max_frames_np(video, ep["target"][frames].tolist(), self.max_frames)
        return {"frames": video, "target": np.array(target), "task": self.task, "strategy": strategy}


class Counter(torch.utils.data.Sampler):
    def __init__(self, skip=0):
        self.skip = skip

    def __iter__(self):
        i = self.skip
        while True:
            yield i
            i += 1


def load_episodes(path, limit=None):
    """Lists a dataset folder's episodes (skipping early-terminated ones) with their success, first stacked frame and
    file paths; with limit, a fixed random subset of that many.
    Pipeline: evaluation setup; lists the episodes to score and where each success ends."""
    with open(os.path.join(path, "index.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["early_termination"] != "1"]
    if limit is not None and limit < len(rows):
        rows = [rows[i] for i in sorted(np.random.default_rng(0).choice(len(rows), limit, replace=False))]
    episodes = []
    for r in rows:
        stacked = np.load(os.path.join(path, r["npz"]))["stacked"] > 0.5  # stacked[t] flags s_{t+1}
        episodes.append({"episode_id": int(r["episode_id"]), "success": r["success"] == "1",
                         "first_stacked": int(np.argmax(stacked)) + 1 if stacked.any() else None,
                         **{k: os.path.join(path, r[k]) for k in ("npz", "front_mp4", "wrist_mp4")}})
    return episodes


def load_eval_dataset(val_path, triplets_path, binary_pairs_path, train_path=None, num_train_eval_episodes=100):
    """Loads what the episode metrics need: the validation episodes, the Kendall triplets, the kendall_binary state
    pairs and (optionally) the training episodes for pearson_critic_train."""
    with open(os.path.join(triplets_path, "triplets.jsonl")) as f:
        triplets = [json.loads(line) for line in f]
    with open(binary_pairs_path) as f:
        binary_pairs = [json.loads(line) for line in f]
    return (load_episodes(val_path), triplets, binary_pairs,
            load_episodes(train_path, num_train_eval_episodes) if train_path else [])


def label_episodes(model, collator, device, episodes, task):
    """Scores each episode with Robometer (a success up to its first stacked frame, a failure whole) and returns
    {episode_id: progress} and each episode's Pearson correlation with the critic's value on the same frames."""
    p, corr = {}, []
    for ep in episodes:
        idx = np.arange(0, len(VideoReader(ep[f"{args.camera}_mp4"], ctx=cpu(0))), args.frame_stride)
        if ep["success"]:
            idx = idx[: int((idx < ep["first_stacked"]).sum()) + 1]
        videos = load_videos(ep[f"{args.camera}_mp4"], queries(idx))
        p[ep["episode_id"]] = scored_progress(model, collator, videos, task, device).cpu().numpy()
        corr.append(compute_pearson(np.load(ep["npz"])["value_ref"][idx].tolist(), p[ep["episode_id"]].tolist()))
    return p, corr


def state_progress(model, collator, device, state, task):
    """Robometer's progress at a state's frame t given the episode's frames 0..t (every --frame_stride-th frame plus t;
    see queries)."""
    t = state["frame"]
    videos = load_videos(state[f"{args.camera}_mp4"], queries(np.r_[np.arange(0, t, args.frame_stride), t])[-1:])
    return float(scored_progress(model, collator, videos, task, device)[-1])


def episode_metrics(model, collator, device, episodes, triplets, binary_pairs, train_episodes):
    """Computes pearson, kendall, kendall_binary, pearson_critic and (with training episodes) pearson_critic_train.
    Pipeline: evaluation; turns the progress curves into the metrics in metrics.csv."""
    task = triplets[0]["task"]
    p, corr = label_episodes(model, collator, device, episodes, task)
    pearson = [compute_pearson(list(range(len(p[ep["episode_id"]]))), p[ep["episode_id"]].tolist())
               for ep in episodes if ep["success"]]
    kendall = [compute_kendall([1, 2, 3], [float(p[t[k]["episode_id"]].mean()) for k in ("failure", "suboptimal", "successful")])
               for t in triplets]
    kendall_binary = [compute_kendall([s["value"] for s in pair["states"]],
                                      [state_progress(model, collator, device, s, task) for s in pair["states"]])
                      for pair in binary_pairs]
    out = {"pearson": float(np.nanmean(pearson)), "kendall": float(np.mean(kendall)),
           "kendall_binary": float(np.mean(kendall_binary)), "pearson_critic": float(np.nanmean(corr))}
    if train_episodes:
        out["pearson_critic_train"] = float(np.nanmean(label_episodes(model, collator, device, train_episodes, task)[1]))
    return out


@torch.no_grad()
def evaluate(model, collator, loader, device, eval_data, val_clips=None):
    """Runs every eval: val_acc on the validation pairs, eval_progress_loss on the validation clips (if any), then the
    episode metrics."""
    model.eval()
    correct = [pair_logit(model, collator, pair, device).item() > 0 for batch in loader for pair in batch]
    out = {"val_acc": float(np.mean(correct))}
    if val_clips is not None:
        losses = []
        for i in range(args.num_val_clips):
            clip = val_clips[i]
            _, logits = progress_batch(model, collator, [clip["frames"]], clip["task"], device)
            losses.append(progress_loss(logits[0], clip["target"]).item())
        out["eval_progress_loss"] = float(np.mean(losses))
    return {**out, **episode_metrics(model, collator, device, *eval_data)}


def add_head_lora(model, peft_cfg):
    """Adds a LoRA adapter (Robometer's settings) to the progress head's linear layers; only the adapter is trained."""
    linears = [n for n, m in model.progress_head.named_modules() if isinstance(m, torch.nn.Linear)]
    cfg = LoraConfig(r=peft_cfg.r, lora_alpha=peft_cfg.lora_alpha, lora_dropout=peft_cfg.lora_dropout,
                     target_modules=linears)
    inject_adapter_in_model(cfg, model.progress_head)
    for name, p in model.progress_head.named_parameters():
        p.requires_grad = "lora_" in name


def trainable_state(model):
    """The weights being trained (what adapter_step<N>.pt stores). Used for checkpointing."""
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    start_step = 0
    if args.resume:
        saved = [int(m.group(1)) for f in os.listdir(args.out_dir) if (m := re.fullmatch(r"adapter_step(\d+)\.pt", f))]
        if not saved:
            raise ValueError(f"--resume: no adapter_step<N>.pt in {args.out_dir}")
        start_step = max(saved)
        args.adapter = os.path.join(args.out_dir, f"adapter_step{start_step}.pt")
    with open(os.path.join(args.out_dir, f"args_resume{start_step}.json" if args.resume else "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    device = torch.device("cuda")
    if args.wandb:
        import wandb

        name = args.wandb_name or os.path.basename(os.path.normpath(args.out_dir))
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), dir=args.out_dir,
                   name=f"{name}_resume{start_step}" if args.resume else name)

    exp_config, tokenizer, processor, model = load_model_from_hf(model_path=args.model_path, device=device)
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    for p in model.parameters():
        p.requires_grad = False
    if args.trainable.startswith("lm_lora"):
        model = setup_peft_model(model, exp_config.peft)  # LoRA adapter on the language model (r=32, alpha=64)
    if args.trainable.endswith("head_lora"):
        add_head_lora(model, exp_config.peft)
    else:
        for p in model.progress_head.parameters():
            p.requires_grad = True
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # keep trained weights and optimizer state in fp32; the forward runs in bf16
    if args.adapter is not None:
        state = torch.load(args.adapter)
        missing = set(trainable_state(model)) - set(state)
        unknown = set(state) - {k for k, _ in model.named_parameters()}
        if unknown:
            raise ValueError(f"{args.adapter} has {len(unknown)} tensors this model lacks; check --trainable")
        model.load_state_dict(state, strict=False)
        print(f"loaded {args.adapter}" + (f" (missing {len(missing)} trained tensors)" if missing else ""))
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_train:,}", flush=True)

    collate = lambda b: b  # noqa: E731  (frames stay numpy; the Robometer collator runs in the main process)
    val_set = PairDataset(args.val_pairs, args.num_val_pairs)
    val_loader = DataLoader(val_set, batch_size=args.pairs_per_step or 8, num_workers=args.num_workers, collate_fn=collate)
    eval_data = load_eval_dataset(args.val_dataset, args.kendall_val_dataset, args.kendall_binary_pairs, args.train_dataset,
                                  args.num_train_eval_episodes)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    fields = ["step", "train_loss", "train_progress_loss", "eval_progress_loss", "train_acc", "val_acc", "pearson", "kendall", "kendall_binary", "pearson_critic",
              "pearson_critic_train", "lr", "sec", "gpu_gb"]
    sec0 = 0.0  # wall time already spent, so "sec" keeps counting across resumes
    if args.resume:  # keep the rows up to the resumed step, and the columns an earlier version of this script logged
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if int(r["step"]) <= start_step]
        fields += [c for c in reader.fieldnames if c not in fields]
        sec0 = float(rows[-1]["sec"])
    else:
        rows = []
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def log(row):
        row["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(row)
        if args.wandb:
            wandb.log({("eval/" if k in ("val_acc", "pearson", "kendall", "kendall_binary", "pearson_critic", "pearson_critic_train") else "") + k: v for k, v in row.items()
                       if k != "step"}, step=row["step"])

    val_clips = None
    if args.distill_clips > 0:
        episodes, ref = load_distill_episodes(args.train_dataset)
        ratio = list(exp_config.data.progress_strategy_ratio)  # [different_task, forward, reverse, rewind]
        ratios = {k: r for k, r in zip(("forward", "reverse", "rewind"), ratio[1:]) if k in args.distill_strategies}
        clips = DistillClips(episodes, eval_data[1][0]["task"], exp_config.data.max_frames, ratios, args.seed)
        # validation clips: drawn the same way from the validation episodes, with the training normalizer
        val_clips = DistillClips(load_distill_episodes(args.val_dataset, ref)[0], eval_data[1][0]["task"],
                                 exp_config.data.max_frames, ratios, args.seed)
        print(f"distillation: {args.distill_clips} clips per step of at most {exp_config.data.max_frames} frames, "
              f"strategies {dict(zip(clips.strategies, clips.p.round(3)))}; targets critic V(s) / {ref:.3f} "
              "(median last-state value of successes), clipped to [0, 1]", flush=True)

    t0 = time.time() - sec0
    if not args.resume:
        log({"step": 0, **evaluate(model, collator, val_loader, device, eval_data, val_clips), "sec": time.time() - t0})
    if args.eval_only:
        if args.wandb:
            wandb.finish()
        return

    used = start_step * args.pairs_per_step * args.grad_accum  # pairs the stopped run already trained on
    if args.pairs_per_step > 0:
        train_set = PairDataset(args.train_pairs)
        batches = iter(DataLoader(train_set, batch_size=args.pairs_per_step, num_workers=args.num_workers,
                                  sampler=ShuffledStream(len(train_set), args.seed, skip=used), collate_fn=collate))
    if args.distill_clips > 0:
        clip_batches = iter(DataLoader(clips, batch_size=args.distill_clips, num_workers=args.num_workers,
                                       sampler=Counter(start_step * args.distill_clips * args.grad_accum),
                                       collate_fn=collate))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    state_path = os.path.join(args.out_dir, "trainer_state.pt")
    trainer_state = torch.load(state_path, weights_only=False) if args.resume and os.path.exists(state_path) else None
    if trainer_state is not None and trainer_state["step"] != start_step:
        trainer_state = None
    # without a saved schedule a resumed run continues at the post-warmup learning rate
    warm = start_step if args.resume and trainer_state is None else 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1 + warm) / args.warmup_steps))
    if trainer_state is not None:
        opt.load_state_dict(trainer_state["optimizer"])
        sched.load_state_dict(trainer_state["scheduler"])
    if args.resume:
        print(f"resuming at step {start_step}, skipping {used} used pairs; optimizer state "
              + ("restored" if trainer_state is not None else "not saved for this step, AdamW moments restart"))
    step, run_loss, run_acc, run_progress = start_step, [], [], []
    while step < args.max_steps:
        model.train()
        n_pairs = n_clips = 0
        for _ in range(args.grad_accum):
            if args.pairs_per_step > 0:
                batch = next(batches)  # the sampler never runs out, so epochs continue seamlessly
                for pair in batch:  # one pair at a time: segments have different lengths with --context
                    logit = pair_logit(model, collator, pair, device)
                    loss = F.softplus(-logit)
                    (loss / (len(batch) * args.grad_accum)).backward()
                    run_loss.append(loss.item())
                    run_acc.append(float(logit.item() > 0))
                n_pairs += len(batch)
            if args.distill_clips > 0:
                clip_batch = next(clip_batches)
                for clip in clip_batch:
                    _, logits = progress_batch(model, collator, [clip["frames"]], clip["task"], device)
                    loss = progress_loss(logits[0], clip["target"])
                    (loss / (len(clip_batch) * args.grad_accum)).backward()
                    run_progress.append(loss.item())
                n_clips += len(clip_batch)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if args.wandb:  # this step's pairs and clips are the last n_pairs / n_clips entries
            wandb.log({**({"train/loss_step": float(np.mean(run_loss[-n_pairs:])),
                           "train/acc_step": float(np.mean(run_acc[-n_pairs:]))} if n_pairs else {}),
                       **({"train/progress_loss_step": float(np.mean(run_progress[-n_clips:]))} if n_clips else {}),
                       "train/grad_norm": float(grad_norm), "lr": sched.get_last_lr()[0]}, step=step)
        if step % args.eval_every == 0 or step == args.max_steps:
            row = {"step": step, **({"train_loss": float(np.mean(run_loss)), "train_acc": float(np.mean(run_acc))}
                                    if run_loss else {}),
                   **({"train_progress_loss": float(np.mean(run_progress))} if run_progress else {}),
                   **evaluate(model, collator, val_loader, device, eval_data, val_clips), "lr": sched.get_last_lr()[0],
                   "sec": time.time() - t0}
            log(row)
            run_loss, run_acc, run_progress = [], [], []
        if step % args.eval_every == 0 or step == args.max_steps or (args.save_every and step % args.save_every == 0):
            torch.save(trainable_state(model), os.path.join(args.out_dir, f"adapter_step{step}.pt"))
            torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, state_path)
    print(f"done: adapters saved to {args.out_dir}")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
