"""Closed-loop GR00T inference on the MuJoCo scene.

Each iteration renders the two cameras, reads the joint state, asks the policy for a
40-step joint chunk given the language instruction, and executes the leading part of it
before re-planning.
"""

import os
import socket
from urllib.parse import urlsplit


def hub_reachable(timeout=2.0):
    """Whether the Hugging Face endpoint accepts a connection, i.e. whether this run can
    download anything. DNS failure, no route and a dropped SYN all read as unreachable."""
    endpoint = urlsplit(os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    try:
        socket.create_connection((endpoint.hostname, endpoint.port or 443), timeout).close()
        return True
    except OSError:
        return False


# With no network, every file huggingface_hub revalidates costs five retries on a growing
# backoff before it falls back to the copy it already has -- minutes of DNS failures per
# checkpoint, twice over, before the first inference. HF_HUB_OFFLINE makes the cache
# authoritative and skips the lookups outright. huggingface_hub reads it once, when it is
# imported, which is why this runs above the imports below instead of inside main(); set it
# yourself, to 0 or 1, to override the probe.
if "HF_HUB_OFFLINE" not in os.environ and not hub_reachable():
    print("huggingface.co unreachable: loading the checkpoint from the local cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
# One lookup does not degrade to a cache read: transformers asks the hub whether the
# backbone's tokenizer derives from Mistral, and that call is unguarded, so offline it raises
# instead of being ignored. Isaac-GR00T's gr00t/__init__.py ships the short-circuit -- for a
# repository that is not Mistral's, the answer is known without asking -- behind this flag.
os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
# albumentations, which gr00t imports, asks PyPI for its latest version on import.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse
import json
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from PIL import Image
from scipy.spatial.transform import Rotation

from scene import CAMERAS, data, model, reset

FPS = 15  # DROID's control rate; the policy's video delta indices assume it
HOLD_FRAMES = 10  # the outcome must persist this many control frames, so a transient does not count
# Success: the cube centre this far from where it settled. Half a cube width, chosen for
# legibility rather than as a noise guard -- parked for a whole rollout the cube does not drift
# at all, the measured floor being 0.000 mm across four spawn positions.
MOVE_DISTANCE = 0.02
FRAMES_DIR = Path(__file__).parent / "frames"  # scratch, rewritten every rollout
VIDEO_DIR = Path(__file__).parent / "videos"  # accumulates, never overwritten

# Step 0 of DROID episode 1 (Isaac-GR00T demo_data/droid_sample, observation.state[10:17]):
# end effector high and back, a genuine approach away. The dataset *mean* pose is not usable
# here -- it sits at table height, which is where the cube is, so the gripper would start past
# its target with nothing in the wrist view.
DROID_HOME = np.array(
    [0.0007119614747352898, -0.252167284488678, 0.22461801767349243, -2.060042142868042,
     -0.027827387675642967, 1.5933151245117188, 0.39067181944847107]
)

# The DROID end-effector frame is the flange rotated by Rz(-135 deg), serialized as
# extrinsic-xyz euler, which the GR00T converter re-reads as *intrinsic* XYZ and
# right-multiplies by a fixed axis remap. All of it has to be reproduced.
EE_FROM_FLANGE = Rotation.from_euler("z", -135, degrees=True).as_matrix()
EEF_ROTATION_CORRECT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def eef_9d(position, rotation):
    euler = Rotation.from_matrix(rotation @ EE_FROM_FLANGE).as_euler("xyz")
    corrected = Rotation.from_euler("XYZ", euler).as_matrix() @ EEF_ROTATION_CORRECT
    return np.concatenate([position, corrected[:2].flatten()])


policy = None  # the GR00T policy, loaded once in main()

# Column labels for the DROID head's state and action keys, which share both names and
# layouts, in the order the embodiment config lists them (gr00t/configs/data/
# embodiment_configs.py). eef_9d is XYZ followed by the first two rows of the rotation matrix,
# flattened -- the rot6d convention compute_eef_9d builds. A key without an entry here falls
# back to positional indices, so an unlabelled head still prints.
DIM_LABELS = {
    # "eef_9d": ("x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12"),
    "gp": ("grip",),
    "jp": ("j1", "j2", "j3", "j4", "j5", "j6", "j7"),
}


def log_dims(block, executed=None):
    """One labelled row per timestep of a {key: (1, T, D)} block. Values are printed as they
    are -- no unit conversion, no deltas, nothing derived."""
    keys = [k for k in DIM_LABELS if k in block] + [k for k in block if k not in DIM_LABELS]
    labels = {k: DIM_LABELS.get(k, tuple(str(i) for i in range(block[k].shape[-1]))) for k in keys}
    print("     step " + " ".join(f"{k}[{d}]".rjust(11) for k in keys for d in labels[k]))
    for t in range(block[keys[0]].shape[1]):
        mark = " " if executed is None else ("*" if t < executed else " ")
        print(f"   {mark}{t:4d} " + " ".join(f"{v:11.6f}" for k in keys for v in block[k][0, t]))


def log_observation(obs, index, total, video_delta):
    """Print what is about to be submitted. Pixels are summarised by shape; the rest verbatim."""
    print(f"  observation {index}/{total} into policy.get_action(), verbatim")
    for cam, frames in obs["video"].items():
        print(f"    video[{cam}] {frames.shape} {frames.dtype} = (batch, frame, h, w, rgb),"
              f" frames at delta {list(video_delta)} steps from now")
    for key, text in obs["language"].items():
        print(f"    language[{key}] {text}")
    log_dims(obs["state"])


def log_action(action, index, total, execute_steps):
    """Print the chunk the policy returned, one labelled row per step of the horizon."""
    keys = list(action)
    steps, dims = action[keys[0]].shape[1], sum(action[k].shape[-1] for k in keys)
    print(f"  action {index}/{total} from policy.get_action(), verbatim: {steps} steps x "
          f"{dims} dims, '*' marks the {execute_steps} this loop executes")
    log_dims(action, executed=execute_steps)


def code_version():
    """Short commit of the code that produced a run, marked when the tree is dirty."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def save_video(trial, ok):
    """Stack the two camera views side by side into an mp4 that identifies the run.

    Named <time>_<commit>_trial<n>_<result>, so videos accumulate rather than collide:
    two runs of the same commit and trial still differ by their timestamp.
    """
    if not shutil.which("ffmpeg"):
        print(f"  ffmpeg not found; frames left in {FRAMES_DIR}")
        return None
    VIDEO_DIR.mkdir(exist_ok=True)
    out = (VIDEO_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{code_version()}"
           f"_trial{trial}_{'moved' if ok else 'failed'}.mp4")
    inputs = []
    for cam in CAMERAS:
        inputs += ["-framerate", str(FPS), "-pattern_type", "glob", "-i", f"{FRAMES_DIR / cam}/*.png"]
    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex",
         "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack[out]",
         "-map", "[out]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    print(f"  video: {out.name}")
    return out.name


def completed(path, config):
    """Trials already finished under exactly this configuration, from an earlier invocation.

    The file is append-only and flushed after every trial, so killing a batch loses at most
    the rollout in flight. Its first line pins the configuration: resuming into a batch that
    was produced by different settings would mix two experiments under one denominator.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config) + "\n")
        return {}
    head, *rows = path.read_text().splitlines()
    if json.loads(head) != config:
        raise SystemExit(f"{path} was written by a different configuration.\n"
                         f"  file: {json.loads(head)}\n  now:  {config}\n"
                         f"Pass --results with a new path, or move that file aside.")
    return {row["trial"]: row["success"] for row in map(json.loads, rows)}


def tcp():
    return 0.5 * (data.geom("panda/robotiq/left_pad1").xpos + data.geom("panda/robotiq/right_pad1").xpos)


def rollout(instruction, execute_steps, max_inferences, cube_xy, viewer):
    """One closed loop. True once the cube has been MOVE_DISTANCE from where it settled for
    HOLD_FRAMES consecutive frames, so a bump it rolls back from does not count."""
    renderer = mujoco.Renderer(model, 180, 320)
    # How many frames the head wants, and which: [-15, 0] on the base checkpoint, [0] on the
    # DROID-finetuned one. Read it off the policy rather than assuming, as the reference client
    # does -- a head that wants one frame rejects a stack of two outright.
    video_delta = policy.modality_configs["video"].delta_indices
    history = {cam: deque(maxlen=1 - min(video_delta)) for cam in CAMERAS}
    for cam in CAMERAS:
        (FRAMES_DIR / cam).mkdir(parents=True, exist_ok=True)
        for stale in (FRAMES_DIR / cam).glob("*.png"):
            stale.unlink()
    frame = 0

    def snap(save=True):
        nonlocal frame
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam)
            history[cam].append(renderer.render())
            if save:
                Image.fromarray(history[cam][-1]).save(FRAMES_DIR / cam / f"{frame:05d}.png")
        if save:
            frame += 1

    def observation():
        base, site = data.body("panda/link0"), data.site("panda/attachment_site")
        base_R = base.xmat.reshape(3, 3)
        pose = eef_9d(base_R.T @ (site.xpos - base.xpos), base_R.T @ site.xmat.reshape(3, 3))
        return {
            # delta d is the frame d steps back, so history[-1] is t and history[-16] is t-15
            "video": {cam: np.stack([history[cam][d - 1] for d in video_delta])[None] for cam in CAMERAS},
            "state": {
                "eef_9d": pose[None, None].astype(np.float32),
                "gripper_position": np.float32([[[data.joint("panda/robotiq/right_driver_joint").qpos[0] / 0.8]]]),
                "joint_position": data.qpos[:7][None, None].astype(np.float32),
            },
            "language": {policy.language_key: [[instruction]]},
        }

    steps_per_frame = round(1 / FPS / model.opt.timestep)
    held = 0
    try:
        reset(DROID_HOME, cube_xy)
        cube_home = data.body("cube").xpos.copy()
        for _ in range(history[CAMERAS[0]].maxlen - 1):  # fill the delta window before inferring
            snap(save=False)
        snap()
        for i in range(max_inferences):
            obs = observation()
            # log_observation(obs, i + 1, max_inferences, video_delta)
            action, _ = policy.get_action(obs)
            # log_action(action, i + 1, max_inferences, execute_steps)
            for t in range(execute_steps):
                data.ctrl[:7] = action["joint_position"][0, t]
                # Binarized, as in the reference DROID client: real close commands plateau
                # near 0.7, so a linear mapping never fully shuts a 2F-85 on a small object.
                data.ctrl[-1] = 255 if action["gripper_position"][0, t, 0] > 0.5 else 0
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                snap()
                cube = data.body("cube").xpos
                held = held + 1 if np.linalg.norm(cube - cube_home) > MOVE_DISTANCE else 0
                if held >= HOLD_FRAMES:
                    return True
                if viewer:
                    viewer.sync()
                    if not viewer.is_running():
                        return False
            cube = data.body("cube").xpos
            print(f"  inference {i + 1}/{max_inferences}: TCP-cube {np.linalg.norm(cube - tcp()):.3f} m,"
                  f" cube moved {np.linalg.norm(cube - cube_home):.3f} m")
    finally:
        renderer.close()
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="pick up the red cube")
    # nvidia/GR00T-N1.7-DROID is the same model post-trained on all ~76k DROID episodes. It
    # scored 0/50 against this checkpoint's 2/50 over the same cube positions, so the base one
    # stays the default; see docs/accuracy-analysis.md. Note it is single-frame, which is why
    # the video horizon below is read off the policy rather than assumed.
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.7-3B",
                        help="'nvidia/GR00T-N1.7-DROID' is post-trained on DROID")
    parser.add_argument("--trials", type=int, default=1, help="more than one randomizes the cube position")
    parser.add_argument("--execute-steps", type=int, default=28)
    parser.add_argument("--max-inferences", type=int, default=20)
    parser.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results.jsonl",
                        help="append-only log of finished trials; rerunning the same command resumes from it")
    args = parser.parse_args()

    # Cube positions are drawn up front rather than inside the loop, so trial n gets the same
    # position whether it runs first or after a resume.
    rng = np.random.default_rng(0)  # fixed, so --trials always visits the same cube positions
    cubes = [tuple(rng.uniform([0, -0.12], [0.25, 0.12]).tolist()) if args.trials > 1 else None
             for _ in range(args.trials)]
    config = {k: v for k, v in vars(args).items() if k not in ("viewer", "results")}
    done = completed(args.results, config)
    if done:
        print(f"resuming {args.results.name}: {len(done)}/{args.trials} trials already finished")
    if len(done) == args.trials:  # nothing left to run; do not load 6 GB of checkpoint to say so
        print(f"\n{sum(done.values())}/{len(done)} moved")
        return

    global policy
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag.resolve("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"),
        device=device,
        strict=True,
    )

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    with args.results.open("a") as log:
        for trial, cube_xy in enumerate(cubes, start=1):
            if trial in done:
                continue
            ok = rollout(
                args.instruction,
                execute_steps=args.execute_steps,
                max_inferences=args.max_inferences,
                cube_xy=cube_xy,
                viewer=viewer,
            )
            print(f"trial {trial}/{args.trials}: {'moved' if ok else 'not moved'}")
            video = save_video(trial, ok)
            # Written only once the rollout is over, so an interrupted one is simply re-run.
            log.write(json.dumps({"trial": trial, "success": ok, "cube_xy": cube_xy, "video": video}) + "\n")
            log.flush()
            os.fsync(log.fileno())
            done[trial] = ok
    print(f"\n{sum(done.values())}/{len(done)} moved")


if __name__ == "__main__":
    main()
