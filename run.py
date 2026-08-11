"""Closed-loop GR00T inference on the MuJoCo scene.

Each iteration renders the two cameras, reads the joint state, asks the policy for a
40-step joint chunk given the language instruction, and executes the leading part of it
before re-planning.
"""

import argparse
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


def tcp():
    return 0.5 * (data.geom("panda/robotiq/left_pad1").xpos + data.geom("panda/robotiq/right_pad1").xpos)


def rollout(policy, instruction, execute_steps, max_inferences, cube_xy, frames_dir, viewer):
    """One closed loop. True if the cube was lifted and held, so a nudge does not count."""
    renderer = mujoco.Renderer(model, 180, 320)
    history = {cam: deque(maxlen=16) for cam in CAMERAS}  # the policy's video delta_indices are [-15, 0]
    if frames_dir:
        for cam in CAMERAS:
            (frames_dir / cam).mkdir(parents=True, exist_ok=True)
    frame = 0

    def snap(save=True):
        nonlocal frame
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam)
            history[cam].append(renderer.render())
            if save and frames_dir:
                Image.fromarray(history[cam][-1]).save(frames_dir / cam / f"{frame:05d}.png")
        if save:
            frame += 1

    def observation():
        base, site = data.body("panda/link0"), data.site("panda/attachment_site")
        base_R = base.xmat.reshape(3, 3)
        pose = eef_9d(base_R.T @ (site.xpos - base.xpos), base_R.T @ site.xmat.reshape(3, 3))
        return {
            "video": {cam: np.stack([history[cam][0], history[cam][-1]])[None] for cam in CAMERAS},
            "state": {
                "eef_9d": pose[None, None].astype(np.float32),
                "gripper_position": np.float32([[[data.joint("panda/robotiq/right_driver_joint").qpos[0] / 0.8]]]),
                "joint_position": data.qpos[:7][None, None].astype(np.float32),
            },
            "language": {policy.language_key: [[instruction]]},
        }

    steps_per_frame = round(1 / FPS / model.opt.timestep)
    lifted = 0
    try:
        reset(DROID_HOME, cube_xy)
        cube_z0 = float(data.body("cube").xpos[2])
        for _ in range(history[CAMERAS[0]].maxlen - 1):  # fill the [-15, 0] window before inferring
            snap(save=False)
        snap()
        for i in range(max_inferences):
            action, _ = policy.get_action(observation())
            for t in range(execute_steps):
                data.ctrl[:7] = action["joint_position"][0, t]
                # Binarized, as in the reference DROID client: real close commands plateau
                # near 0.7, so a linear mapping never fully shuts a 2F-85 on a small object.
                data.ctrl[-1] = 255 if action["gripper_position"][0, t, 0] > 0.5 else 0
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                snap()
                lifted = lifted + 1 if data.body("cube").xpos[2] > cube_z0 + 0.05 else 0
                if lifted >= 10:  # 5 cm up for 10 consecutive frames, so a nudge does not count
                    return True
                if viewer:
                    viewer.sync()
                    if not viewer.is_running():
                        return False
            print(f"  inference {i + 1}/{max_inferences}: TCP-cube {np.linalg.norm(data.body('cube').xpos - tcp()):.3f} m")
    finally:
        renderer.close()
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--trials", type=int, default=1, help="more than one randomizes the cube position")
    parser.add_argument("--execute-steps", type=int, default=28)
    parser.add_argument("--max-inferences", type=int, default=8)
    parser.add_argument("--frames", type=Path, help="write per-camera PNGs to this directory")
    parser.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    policy = Gr00tPolicy(
        model_path="nvidia/GR00T-N1.7-3B",
        embodiment_tag=EmbodimentTag.resolve("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"),
        device=device,
        strict=True,
    )

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    rng = np.random.default_rng(0)  # fixed, so --trials always visits the same cube positions
    results = []
    for i in range(args.trials):
        cube_xy = tuple(rng.uniform([0, -0.12], [0.25, 0.12])) if args.trials > 1 else None
        ok = rollout(
            policy,
            args.instruction,
            execute_steps=args.execute_steps,
            max_inferences=args.max_inferences,
            cube_xy=cube_xy,
            frames_dir=args.frames if i == 0 else None,
            viewer=viewer,
        )
        results.append(ok)
        print(f"trial {i + 1}/{args.trials}: {'lifted' if ok else 'not lifted'}")
    print(f"\n{sum(results)}/{len(results)} lifted")


if __name__ == "__main__":
    main()
