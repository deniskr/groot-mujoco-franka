"""Check a dataset written by generate_training_data.py --record against NVIDIA's droid_sample.

Both are loaded with GR00T's own LeRobotEpisodeLoader and the DROID modality config, so a
dataset that loads here loads in training. Decoding AV1 on macOS needs the ffmpeg@7 library
path, which is why this runs as `just check DATASET`.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

OUT = Path(__file__).parent / "outputs"
CONFIG = MODALITY_CONFIGS["oxe_droid_relative_eef_relative_joint"]
CAMS = CONFIG["video"].modality_keys
# (key, index, label): the 7 joints, then eef x, y, z.
PLOTTED = [("joint_position", i, f"j{i + 1}") for i in range(7)] + [("eef_9d", i, "xyz"[i]) for i in range(3)]


def summarize(name, loader):
    lengths = loader.get_episode_lengths()
    print(f"== {name}: {loader.info_meta['codebase_version']}, {len(lengths)} episodes, lengths {lengths}")
    print(f"   video keys in modality.json: {list(loader.modality_meta['video'])}")
    for feature in ("observation.state", "action"):
        s = loader.stats[feature]
        print(f"   {feature:17s} min {np.round(s['min'], 3)}")
        print(f"   {'':17s} max {np.round(s['max'], 3)}")


def describe_episode(name, df):
    """Loading the episode already checked that every video has as many frames as the parquet has rows."""
    grip = np.stack(df["action.gripper_position"])[:, 0]
    print(f"== {name} episode: {len(df)} rows, video frames match; "
          f"{len(np.unique(grip))} distinct gripper actions, e.g. {np.round(np.unique(grip)[:6], 3)}")
    print(f"   instruction: {df['language.annotation.language.language_instruction'].iloc[0]!r}")


def frame_sheet(rows, path):
    """rows: [(label, frames (T, H, W, 3))]; six evenly spaced frames per row."""
    fig, axes = plt.subplots(len(rows), 6, figsize=(18, 2.2 * len(rows)), squeeze=False)
    for r, (label, frames) in enumerate(rows):
        for c, t in enumerate(np.linspace(0, len(frames) - 1, 6).astype(int)):
            ax = axes[r, c]
            ax.imshow(frames[t])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{label}  t={t}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def action_vs_state(columns, path):
    """columns: [(label, episode DataFrame)]; state and action over time, one row per PLOTTED dim.
    In DROID the action runs ahead of the state; ours should look the same."""
    fig, axes = plt.subplots(len(PLOTTED), len(columns), figsize=(7 * len(columns), 1.6 * len(PLOTTED)),
                             sharex="col", squeeze=False)
    for c, (label, df) in enumerate(columns):
        for r, (key, i, name) in enumerate(PLOTTED):
            ax = axes[r, c]
            ax.plot(np.stack(df[f"state.{key}"])[:, i], label="state")
            ax.plot(np.stack(df[f"action.{key}"])[:, i], label="action")
            ax.set_ylabel(name)
        axes[0, c].set_title(label)
        axes[0, c].legend(fontsize=7)
        axes[-1, c].set_xlabel("frame (15 Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).resolve().parent.parent / "Isaac-GR00T/demo_data/droid_sample")
    parser.add_argument("--episode", type=int, default=0, help="our episode to draw")
    parser.add_argument("--reference-episode", type=int, default=1, help="1 is the block-in-bowl episode")
    args = parser.parse_args()

    first = next((args.reference / "data").rglob("*.parquet"))
    if first.read_bytes()[:40].startswith(b"version https://git-lfs"):
        raise SystemExit(f"{args.reference} holds git-lfs pointers, not data: run `git lfs pull` in "
                         f"Isaac-GR00T, or pass --reference /path/to/droid_sample")

    ours = LeRobotEpisodeLoader(args.dataset, CONFIG)
    ref = LeRobotEpisodeLoader(args.reference, CONFIG)
    summarize("ours", ours)
    summarize("droid_sample", ref)
    df, ref_df = ours[args.episode], ref[args.reference_episode]
    describe_episode("ours", df)
    describe_episode("droid_sample", ref_df)

    OUT.mkdir(exist_ok=True)
    frame_sheet([(f"ours {cam}", np.stack(df[f"video.{cam}"])) for cam in CAMS]
                + [(f"droid {cam}", np.stack(ref_df[f"video.{cam}"])) for cam in CAMS],
                OUT / "check_frames.png")
    action_vs_state([(f"ours, episode {args.episode}", df),
                     (f"droid_sample, episode {args.reference_episode}", ref_df)],
                    OUT / "check_action_vs_state.png")
    print(f"wrote {OUT / 'check_frames.png'} and {OUT / 'check_action_vs_state.png'}")


if __name__ == "__main__":
    main()
