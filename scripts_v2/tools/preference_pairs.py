"""Preference pairs for fine-tuning Robometer, labeled by the final OmniReset critic.

Two steps (numpy only, no simulator needed):

  build   Gather every episode of a success/failure dataset (see collect_success_failure.py) into one trajectory
          buffer: per-step rewards, final-critic values V(s_t) ("value_ref"), success labels and video paths.
          --values and --success_reward swap in another critic (e.g. the binary-reward critic from
          train_binary_critic.py, scored with critic_values.py) and the binary reward it was trained with.

  sample  Draw K pairs of segments from the buffer (from any two episodes, or both from one episode with
          --strategy within_episode; fixed length, or lengths drawn from a range with --segment_len_max) and label each
          pair with the critic. A segment of L steps starting at state s_0 scores
              -regret(sigma) = sum_t gamma^t A(s_t, a_t) = sum_{t<L} gamma^t r_t + gamma^L V(s_L) - V(s_0)
          (the advantages telescope), and the segment with the higher score is "chosen". Segments where the cube is
          already stacked in every state are dropped unless --keep_done is given.

  triplets  For Kendall tau as in Robometer's policy ranking: label the whole episodes of a separate validation dataset
          successful, suboptimal (failed, critic's normalized last-state value >= --mid_min) or failure (<= --bad_max),
          copy them into a new dataset folder and write triplets of one episode per label.

  kendall_pairs  For Kendall tau against the critic: pairs of states from two different episodes of a validation
          dataset, each at a frame drawn uniformly from 0 to the episode's horizon (its first stacked frame for a
          success, its last frame for a failure), labeled with the critic's V(s) at that frame.

Frame t of an episode video shows state s_t, so a segment covers frames start..end = start + L (inclusive).

Usage:
  python scripts_v2/tools/preference_pairs.py build --dataset ./Datasets/CubeStack_SuccessFailure_v2
  python scripts_v2/tools/preference_pairs.py sample --buffer ./Datasets/CubeStack_SuccessFailure_v2/trajectory_buffer.npz \
      --num_pairs 10000 --segment_len 30 --split train --out ./Datasets/CubeStack_SuccessFailure_v2/preference_pairs/train.jsonl
  python scripts_v2/tools/preference_pairs.py triplets --buffer ./Datasets/CubeStack_SuccessFailure_val/trajectory_buffer.npz \
      --out_dir ./Datasets/CubeStack_KendallTriplets_val --shared_dir /scr/shared/real_to_sim_rewards/CubeStack_KendallTriplets_val
  python scripts_v2/tools/preference_pairs.py kendall_pairs --buffer ./Datasets/CubeStack_SuccessFailure_val/trajectory_buffer.npz \
      --out ./Datasets/CubeStack_SuccessFailure_val/kendall_binary_pairs.jsonl
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np

TASK = "Pick up the green cube and stack it on top of the blue cube."
FINAL_CRITIC = "logs/rsl_rl/droid_franka_uwlab_robotiq_2f85_omnireset_agent/2026-10-03_04-18-29/model_1998.pt"


def build(args):
    dataset = os.path.abspath(args.dataset)
    with open(os.path.join(dataset, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    max_len = max(int(r["length_steps"]) for r in rows)
    reward = np.full((len(rows), max_len), np.nan, dtype=np.float32)  # padded with NaN after each episode's end
    value = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    stacked = np.zeros((len(rows), max_len), dtype=bool)
    for i, r in enumerate(rows):
        d = np.load(os.path.join(dataset, r["npz"]))
        n = len(d["reward"])
        reward[i, :n], value[i, :n], stacked[i, :n] = d["reward"], d["value_ref"], d["stacked"] > 0.5
    if args.values is not None:  # another critic's V(s_t), e.g. from critic_values.py
        v = np.load(args.values)
        row = {int(e): j for j, e in enumerate(v["episode_id"])}
        for i, r in enumerate(rows):
            n = int(r["length_steps"])
            value[i, :n] = v["value"][row[int(r["episode_id"])], :n]
    if args.success_reward is not None:  # binary reward: success_reward * dt for each step that ends stacked
        reward = np.where(np.isfinite(reward), args.success_reward * stacked, np.nan).astype(np.float32)
    out = os.path.join(dataset, args.out)
    np.savez_compressed(
        out,
        episode_id=np.array([int(r["episode_id"]) for r in rows]),
        length=np.array([int(r["length_steps"]) for r in rows]),
        early_termination=np.array([int(r["early_termination"]) for r in rows], dtype=bool),
        success=np.array([int(r["success"]) for r in rows], dtype=bool),
        iteration=np.array([int(r["iteration"]) for r in rows]),
        action_noise=np.array([float(r["action_noise"]) for r in rows], dtype=np.float32),
        front_mp4=np.array([os.path.join(dataset, r["front_mp4"]) for r in rows]),
        wrist_mp4=np.array([os.path.join(dataset, r["wrist_mp4"]) for r in rows]),
        reward=reward,
        value=value,
        stacked=stacked,
        gamma=np.float64(args.gamma),
        task=np.array(TASK),
        critic_checkpoint=np.array(args.critic_checkpoint),
    )
    print(f"saved {out}: {len(rows)} episodes ({int(sum(int(r['success']) for r in rows))} successes), "
          f"{int(np.isfinite(reward).sum())} steps, gamma {args.gamma}")


def segment_score(reward, value, start, length, gamma):
    """-regret of the segment s_start .. s_{start+length}: discounted rewards plus the critic's bootstrap."""
    disc = gamma ** np.arange(length)
    return float(disc @ reward[start : start + length] + gamma**length * value[start + length] - value[start])


def sample(args):
    buf = dict(np.load(args.buffer))  # read every array once; NpzFile decompresses on each access
    gamma = float(buf["gamma"])
    lengths = np.arange(args.segment_len, (args.segment_len_max or args.segment_len) + 1)
    # candidate segments (episode, start, length). A segment needs V at its last state, so it must end by the episode's
    # last stored state; early-terminated episodes are skipped since their terminal penalty falls outside every segment.
    in_val = buf["episode_id"] % 10 == 0  # fixed episode split: every 10th episode is held out for validation
    use = {"all": np.ones_like(in_val), "train": ~in_val, "val": in_val}[args.split]
    candidates = {}  # episode -> (starts, lengths)
    num_total = num_done = 0
    for e in np.flatnonzero(~buf["early_termination"] & use):
        T = buf["length"][e]
        stacked_count = np.concatenate([[0], np.cumsum(buf["stacked"][e][:T])])
        starts, lens = [], []
        for L in lengths:
            s = np.arange(max(T - L, 0))
            num_total += len(s)
            if not args.keep_done:
                # stacked[t] is the flag of state s_{t+1}; s_0 (a reset) is never stacked. Drop a segment when the cube
                # is stacked in every one of its states s_start .. s_{start+L}: the task is already done.
                done = (s > 0) & (stacked_count[s + L] - stacked_count[np.maximum(s - 1, 0)] == L + 1)
                num_done += int(done.sum())
                s = s[~done]
            starts.append(s)
            lens.append(np.full(len(s), L))
        if sum(map(len, starts)) >= 2:
            candidates[e] = (np.concatenate(starts), np.concatenate(lens))
    if not candidates:
        raise ValueError(f"no episode has 2 candidate segments of {lengths[0]}-{lengths[-1]} steps")
    episodes = np.array(list(candidates))
    flat_ep = np.concatenate([np.full(len(candidates[e][0]), e) for e in episodes])
    flat_start = np.concatenate([candidates[e][0] for e in episodes])
    flat_len = np.concatenate([candidates[e][1] for e in episodes])
    print(f"{args.split} split: {len(flat_ep)} candidate segments of {lengths[0]}-{lengths[-1]} steps in "
          f"{len(episodes)} episodes" + ("" if args.keep_done else f" ({num_done} of {num_total} dropped: cube already "
                                         "stacked throughout)"))
    rng = np.random.default_rng(args.seed)

    def draw():
        """Two different segments: uniform over all candidates, or uniform episode then two of its candidates."""
        if args.strategy == "random":
            i, j = rng.choice(len(flat_ep), size=2, replace=False)
            return [(flat_ep[k], flat_start[k], flat_len[k]) for k in (i, j)]
        e = episodes[rng.integers(len(episodes))]
        starts, lens = candidates[e]
        i, j = rng.choice(len(starts), size=2, replace=False)
        return [(e, starts[k], lens[k]) for k in (i, j)]

    def describe(ep, start, L, score):
        return {
            "episode_id": int(buf["episode_id"][ep]), "start": int(start), "end": int(start + L), "length": int(L),
            "score": round(score, 5), "episode_success": bool(buf["success"][ep]),
            "iteration": int(buf["iteration"][ep]), "action_noise": float(buf["action_noise"][ep]),
            "stacked_at_end": bool(buf["stacked"][ep, start + L - 1]),
            "front_mp4": str(buf["front_mp4"][ep]), "wrist_mp4": str(buf["wrist_mp4"][ep]),
        }  # fmt: skip

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    margins, chosen_success, rejected_success, chosen_later, chosen_longer = [], [], [], [], []
    written = 0
    with open(args.out, "w") as f:
        while written < args.num_pairs:
            segs = draw()
            scores = [segment_score(buf["reward"][e], buf["value"][e], s, L, gamma) for e, s, L in segs]
            margin = abs(scores[0] - scores[1])
            if margin < args.min_margin:
                continue
            c, r = (0, 1) if scores[0] >= scores[1] else (1, 0)
            pair = {
                "pair_id": written, "strategy": args.strategy, "split": args.split, "task": str(buf["task"]),
                "segment_len": [int(lengths[0]), int(lengths[-1])], "gamma": gamma, "margin": round(margin, 5),
                "chosen": describe(*segs[c], scores[c]),
                "rejected": describe(*segs[r], scores[r]),
            }  # fmt: skip
            f.write(json.dumps(pair) + "\n")
            margins.append(margin)
            chosen_success.append(buf["success"][segs[c][0]])
            rejected_success.append(buf["success"][segs[r][0]])
            if segs[c][1] != segs[r][1]:
                chosen_later.append(segs[c][1] > segs[r][1])
            if segs[c][2] != segs[r][2]:
                chosen_longer.append(segs[c][2] > segs[r][2])
            written += 1

    meta = {"buffer": os.path.abspath(args.buffer), "critic_checkpoint": str(buf["critic_checkpoint"]), **vars(args)}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    margins = np.array(margins)
    print(f"saved {written} pairs to {args.out}")
    print(f"margin |score difference|: median {np.median(margins):.3f}, under 0.1: {np.mean(margins < 0.1):.0%}, "
          f"under 0.5: {np.mean(margins < 0.5):.0%}")
    print(f"chosen segment from a successful episode: {np.mean(chosen_success):.0%} "
          f"(rejected: {np.mean(rejected_success):.0%})")
    print(f"chosen segment starts later: {np.mean(chosen_later):.0%}"
          + (f", is longer: {np.mean(chosen_longer):.0%}" if chosen_longer else ""))


def triplets(args):
    """Robometer-style ranking dataset: whole episodes labeled successful / suboptimal / failure, and triplets of one
    episode per label. Failed episodes are split by the critic's value at their last state, normalized by the median
    last-state value of the successful episodes (1 = as good as a finished stack) and clipped to [0, 1]."""
    buf = dict(np.load(args.buffer))
    use = np.flatnonzero(~buf["early_termination"])
    final = np.array([buf["value"][e, buf["length"][e] - 1] for e in use])
    success = buf["success"][use].astype(bool)
    ref = float(np.median(final[success]))
    norm = np.clip(final / ref, 0.0, 1.0)
    label = np.where(success, "successful",
                     np.where(norm >= args.mid_min, "suboptimal", np.where(norm <= args.bad_max, "failure", "")))
    os.makedirs(args.out_dir, exist_ok=True)
    rank = {"failure": 1, "suboptimal": 2, "successful": 3}  # Robometer's quality_order
    groups = {k: [] for k in rank}
    with open(os.path.join(args.out_dir, "episodes.jsonl"), "w") as f:
        for e, v, n, lab in zip(use, final, norm, label):
            if not lab:
                continue
            src = os.path.dirname(str(buf["front_mp4"][e]))
            name = f"episode_{int(buf['episode_id'][e]):05d}"
            for suffix in (".npz", "_front.mp4", "_wrist.mp4"):  # the whole episode: states, critic values, videos
                shutil.copy2(os.path.join(src, name + suffix), os.path.join(args.out_dir, name + suffix))
            stacked = buf["stacked"][e, : buf["length"][e]]
            rec = {
                "episode_id": int(buf["episode_id"][e]), "label": str(lab), "rank": rank[lab],
                "success": bool(buf["success"][e]), "length": int(buf["length"][e]),
                # stacked[t] flags s_{t+1}: the first stacked state, as a frame index (frame t shows s_t)
                "first_stacked": int(np.argmax(stacked)) + 1 if stacked.any() else None,
                "critic_final": round(float(v), 4), "critic_norm": round(float(n), 4),
                "iteration": int(buf["iteration"][e]), "action_noise": float(buf["action_noise"][e]),
                "npz": os.path.join(args.shared_dir, name + ".npz"),
                "front_mp4": os.path.join(args.shared_dir, name + "_front.mp4"),
                "wrist_mp4": os.path.join(args.shared_dir, name + "_wrist.mp4"),
            }  # fmt: skip
            f.write(json.dumps(rec) + "\n")
            groups[lab].append(rec)
    combos = len(groups["successful"]) * len(groups["suboptimal"]) * len(groups["failure"])
    if combos == 0:
        raise ValueError(f"a label has no episodes: { {k: len(g) for k, g in groups.items()} }")
    rng = np.random.default_rng(args.seed)
    n = combos if args.num_triplets <= 0 else min(args.num_triplets, combos)
    if n == combos:
        picks = np.stack(np.meshgrid(*[np.arange(len(groups[k])) for k in rank], indexing="ij"), -1).reshape(-1, 3)
    else:  # distinct triplets, uniform over all combinations
        flat = rng.choice(combos, size=n, replace=False)
        picks = np.stack(np.unravel_index(flat, [len(groups[k]) for k in rank]), -1)
    with open(os.path.join(args.out_dir, "triplets.jsonl"), "w") as f:
        for i, idx in enumerate(picks):
            f.write(json.dumps({"triplet_id": i, "task": str(buf["task"]),
                                **{k: groups[k][j] for k, j in zip(rank, idx)}}) + "\n")
    meta = {"buffer": os.path.abspath(args.buffer), "critic_checkpoint": str(buf["critic_checkpoint"]),
            "normalizer_median_success_final_value": round(ref, 4),
            "counts": {k: len(g) for k, g in groups.items()},
            "dropped_between_ranges": int(((~success) & (norm > args.bad_max) & (norm < args.mid_min)).sum()),
            "early_terminated_skipped": int(buf["early_termination"].sum()),
            "num_triplets": int(n), "all_combinations": int(combos), **vars(args)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"labeled {sum(meta['counts'].values())} episodes: {meta['counts']} "
          f"({meta['dropped_between_ranges']} failures between the ranges dropped)")
    print(f"saved {n} of {combos} triplets to {os.path.join(args.out_dir, 'triplets.jsonl')}")


def kendall_pairs(args):
    """Pairs of states for Kendall tau against the critic: two different episodes, each at a frame t drawn uniformly
    from 0 to its horizon (first stacked frame for a success, last frame for a failure), with the critic's V(s_t)."""
    buf = dict(np.load(args.buffer))
    use = np.flatnonzero(~buf["early_termination"])
    horizon = {}
    for e in use:
        T = int(buf["length"][e])
        # stacked[t] flags s_{t+1}: the first stacked state, as a frame index (frame t shows s_t)
        horizon[e] = min(int(np.argmax(buf["stacked"][e, :T])) + 1, T - 1) if buf["success"][e] else T - 1
    rng = np.random.default_rng(args.seed)
    gaps = []
    with open(args.out, "w") as f:
        for i in range(args.num_pairs):
            states = []
            for e in rng.choice(use, size=2, replace=False):
                t = int(rng.integers(horizon[e] + 1))
                states.append({
                    "episode_id": int(buf["episode_id"][e]), "frame": t, "value": round(float(buf["value"][e, t]), 5),
                    "success": bool(buf["success"][e]), "horizon": horizon[e],
                    "front_mp4": str(buf["front_mp4"][e]), "wrist_mp4": str(buf["wrist_mp4"][e]),
                })  # fmt: skip
            f.write(json.dumps({"pair_id": i, "task": str(buf["task"]), "states": states}) + "\n")
            gaps.append(abs(states[0]["value"] - states[1]["value"]))
    meta = {"buffer": os.path.abspath(args.buffer), "critic_checkpoint": str(buf["critic_checkpoint"]), **vars(args)}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {args.num_pairs} pairs of states from {len(use)} episodes to {args.out}; "
          f"|V difference| median {np.median(gaps):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="gather all episodes into one trajectory buffer")
    p.add_argument("--dataset", required=True, help="folder written by collect_success_failure.py")
    p.add_argument("--gamma", type=float, default=0.99, help="discount the critic was trained with (agent.yaml)")
    p.add_argument("--critic_checkpoint", default=FINAL_CRITIC, help="the critic behind the values (recorded only)")
    p.add_argument("--values", default=None,
                   help="values.npz from critic_values.py to use instead of each episode's value_ref")
    p.add_argument("--success_reward", type=float, default=None,
                   help="use a binary reward instead of the recorded one: this much per step that ends stacked "
                        "(0.1 = OmniReset's success_reward * dt)")
    p.add_argument("--out", default="trajectory_buffer.npz", help="buffer file name inside --dataset")
    p = sub.add_parser("sample", help="sample labeled segment pairs from a trajectory buffer")
    p.add_argument("--buffer", required=True)
    p.add_argument("--num_pairs", type=int, required=True)
    p.add_argument("--strategy", choices=["random", "within_episode"], default="random",
                   help="random: two segments from any episodes; within_episode: both from one episode")
    p.add_argument("--segment_len", type=int, default=30, help="steps per segment (10 steps = 1 s)")
    p.add_argument("--segment_len_max", type=int, default=None,
                   help="if set, each segment's length is drawn from segment_len..segment_len_max")
    p.add_argument("--min_margin", type=float, default=0.0, help="drop pairs whose ground truth scores differ by less than this")
    p.add_argument("--keep_done", action="store_true", help="keep segments where the cube is stacked throughout")
    p.add_argument("--split", choices=["all", "train", "val"], default="all",
                   help="episodes to sample from; val = every 10th episode (episode_id % 10 == 0), train = the rest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output .jsonl, one pair per line")
    p = sub.add_parser("triplets", help="Robometer-style ranking triplets (successful, suboptimal, failure) of whole episodes")
    p.add_argument("--buffer", required=True, help="trajectory buffer of a separate validation dataset")
    p.add_argument("--out_dir", required=True, help="new dataset folder (episodes are copied into it)")
    p.add_argument("--shared_dir", required=True, help="where --out_dir is synced; the paths in the .jsonl files")
    p.add_argument("--mid_min", type=float, default=0.5, help="failures with normalized final value >= this: suboptimal")
    p.add_argument("--bad_max", type=float, default=0.4, help="failures with normalized final value <= this: failure")
    p.add_argument("--num_triplets", type=int, default=1000, help="distinct triplets to write (0: every combination)")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("kendall_pairs", help="pairs of states from two episodes, labeled with the critic's V(s)")
    p.add_argument("--buffer", required=True, help="trajectory buffer of a separate validation dataset")
    p.add_argument("--num_pairs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output .jsonl, one pair per line")
    args = parser.parse_args()
    {"build": build, "sample": sample, "triplets": triplets, "kendall_pairs": kendall_pairs}[args.cmd](args)
