"""Scripted reach / close / lift of the red cube with mink IK.

Viewer mode (default; mjpython on macOS) runs --episodes, then h runs the next batch.
--headless runs --episodes without a viewer and exits.
"""

import argparse
import json
import time
from pathlib import Path

import mink
import mujoco
import mujoco.viewer
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats, generate_stats

from run import eef_9d  # pulls in torch and gr00t, so the first import takes a while
from scene import CAMERAS, CUBE_REST_Z, DROID_HOME, data, model, random_layout, reset

start_jtter = 0.05
FRAMES = 45
CLOSE_FRAMES = 2
LIFT_FRAMES = 25
LIFT_HEIGHT = 0.15

ARM_DOF = np.array([model.joint(f"panda/joint{i}").dofadr[0] for i in range(1, 8)])
ARM_QPOS = np.array([model.joint(f"panda/joint{i}").qposadr[0] for i in range(1, 8)])
FPS = 15
STEPS_PER_FRAME = round(1 / FPS / model.opt.timestep)
MAX_DQ = 0.05  # rad per control frame, as in teleop.py

IK_CONFIG = mink.Configuration(model)
# Position only: the pinch point (between the fingertips) goes onto the cube. Orientation is
# left to axis_task below, which fixes where the fingers point but not their roll.
task = mink.FrameTask(
    frame_name="panda/robotiq/pinch",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.0,
)
# Fingers straight down: the pinch site's local +z is the approach axis.
axis_task = mink.AxisAlignTask("panda/robotiq/pinch", "site", axis=(0, 0, -1), cost=1.0)
axis_task.set_target(np.array([0.0, 0.0, 1.0]))
CLOSING_AXIS = np.array([0.0, 1.0, 0.0])
FACE_NORMALS = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
roll_task = mink.AxisAlignTask("panda/robotiq/pinch", "site", axis=CLOSING_AXIS, cost=1.0)

#Regularization: prefer smaller joint velocities
damping_task = mink.DampingTask(model, cost=0.7)
posture_task = mink.PostureTask(model, cost=0.0)

limits = [
    mink.ConfigurationLimit(model),
    mink.VelocityLimit(model, {f"panda/joint{i}": MAX_DQ * FPS for i in range(1, 8)}),
]

def ik_step_mink(target_pos):
    IK_CONFIG.update(data.qpos)
    task.set_target(mink.SE3.from_translation(target_pos))
    vel = mink.solve_ik(
        IK_CONFIG, [task, axis_task, roll_task, damping_task, posture_task], 1 / FPS, "daqp", limits=limits
    )
    step = vel[ARM_DOF] / FPS
    return data.qpos[ARM_QPOS] + step

pending = {"rerun": False}
def on_key(code):
    if chr(code).lower() == "h":
        pending["rerun"] = True


TASK = "pick up the red cube"
MIN_LIFT = 0.10  # an episode is kept when the cube ends at least this far above its resting height
STATE_NAMES = ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12", "grip", *(f"j{i}" for i in range(1, 8))]
EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
# Where each DROID key sits in the 17-d state and action vectors (droid_sample's modality.json).
DROID_SLICES = {"eef_9d": (0, 9), "gripper_position": (9, 10), "joint_position": (10, 17)}


def droid_state(d):
    """[eef_9d(9), gripper(1), joints(7)] exactly as run.py's observation() builds it
    (run.py:215-225): the flange site in the robot base frame, not the world frame."""
    base, site = d.body("panda/link0"), d.site("panda/attachment_site")
    base_R = base.xmat.reshape(3, 3)
    pose = eef_9d(base_R.T @ (site.xpos - base.xpos), base_R.T @ site.xmat.reshape(3, 3))
    grip = d.joint("panda/robotiq/right_driver_joint").qpos[0] / 0.8
    return np.concatenate([pose, [grip], d.qpos[:7]]).astype(np.float32)


# Forward kinematics only, never stepped: where the flange would be if the arm reached the
# command. DROID's action.cartesian_position is the commanded pose too, not the measured one.
fk_data = mujoco.MjData(model)


def commanded_eef(q_arm):
    fk_data.qpos[:] = data.qpos
    fk_data.qpos[ARM_QPOS] = q_arm
    mujoco.mj_kinematics(model, fk_data)
    return droid_state(fk_data)[:9]


def create_dataset(root):
    """An empty LeRobot v2.1 dataset at root, which must not exist yet."""
    def vector():
        return {"dtype": "float32", "shape": (17,), "names": STATE_NAMES}

    def video():
        return {"dtype": "video", "shape": (180, 320, 3), "names": ["height", "width", "channels"]}

    return LeRobotDataset.create(
        repo_id=f"local/{root.name}",
        fps=FPS,
        root=root,
        robot_type="panda_2f85_mujoco",
        features={
            "observation.state": vector(),
            "action": vector(),
            **{f"observation.images.{cam}": video() for cam in CAMERAS},
        },
        image_writer_threads=4,
    )


def render_images(renderer):
    images = {}
    for cam in CAMERAS:
        renderer.update_scene(data, camera=cam)
        images[f"observation.images.{cam}"] = renderer.render()
    return images


def discard_episode(dataset):
    """Drop the frames buffered since the last save_episode(). The image writer runs in
    threads, so let it finish first: clearing while it still writes can leave PNGs in the
    directory the next episode reuses, since it gets the same episode index."""
    if dataset.image_writer is not None:
        dataset.image_writer.wait_until_done()
    dataset.clear_episode_buffer()


def finalize(dataset):
    """Close the dataset and add what GR00T needs on top of LeRobot v2.1: modality.json,
    written here, and the two stats files, computed by gr00t's own stats code."""
    if dataset.episode_buffer["size"] > 0:  # interrupted mid-episode
        discard_episode(dataset)
    dataset.stop_image_writer()
    root = dataset.root
    if dataset.meta.total_episodes == 0:
        print(f"no episode kept; {root} is not a usable dataset")
        return
    slices = {key: {"start": a, "end": b} for key, (a, b) in DROID_SLICES.items()}
    modality = {
        "state": slices,
        "action": slices,
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in CAMERAS},
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
    }
    (root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2))
    generate_stats(root)
    generate_rel_stats(root, EmbodimentTag.resolve(EMBODIMENT))
    print(f"{dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames in {root}")


def run_episode(i, viewer, dataset=None, renderer=None):
    """Episode i, seeded 1000 + i: the same index always gives the same layout and start pose.
    With a dataset, the episode is saved when the cube ends MIN_LIFT up, else discarded.
    Returns how far the cube ends above its resting height, in m."""
    rng = np.random.default_rng(1000 + i)
    layout = random_layout(rng)
    start = DROID_HOME + rng.uniform(-start_jtter, start_jtter, 7)
    reset(start, None, objects=layout)
    IK_CONFIG.update(data.qpos)
    posture_task.set_target_from_configuration(IK_CONFIG)
    # The face normal nearest to where the fingers already close along: any of the four
    # grips the cube, and the nearest needs the least wrist rotation (never a 180 deg spin).
    closing = data.site("panda/robotiq/pinch").xmat.reshape(3, 3) @ CLOSING_AXIS
    roll_task.set_target(FACE_NORMALS[np.argmax(FACE_NORMALS @ closing)])
    # Read once: a live cube target would follow the cube up with the gripper, so the lift
    # would ask for "stay where you are" and never happen.
    cube = data.body("cube").xpos.copy()
    phases = [  # (target, gripper ctrl, frames)
        (cube, 0, FRAMES),                        # reach, open
        (cube, 255, CLOSE_FRAMES),                # close, arm holds still
        (cube + [0, 0, LIFT_HEIGHT], 255, LIFT_FRAMES),  # lift
    ]

    for p, (target, grip, n) in enumerate(phases):
        data.ctrl[-1] = grip
        for frame in range(n):
            if dataset is not None:
                # Before the command: DROID pairs each observation with the command sent next.
                state = droid_state(data)
                images = render_images(renderer)
            data.ctrl[:7] = ik_step_mink(target)
            if dataset is not None:
                action = np.concatenate([commanded_eef(data.ctrl[:7]), [data.ctrl[-1] / 255], data.ctrl[:7]])
                dataset.add_frame(
                    {"observation.state": state, "action": action.astype(np.float32), **images}, task=TASK
                )
            for j in range(STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
            # Measured after the physics steps: where the arm actually got to this frame.
            IK_CONFIG.update(data.qpos)
            pos_err = np.linalg.norm(data.site("panda/robotiq/pinch").xpos - target)
            axis_err = np.linalg.norm(axis_task.compute_error(IK_CONFIG))
            print(f"  phase {p} frame {frame:3d}  pos {pos_err * 1000:6.1f} mm  axis {axis_err:.4f}")
            if viewer is not None:
                viewer.sync()
                time.sleep(1 / FPS)
    lift = data.body("cube").xpos[2] - CUBE_REST_Z
    print(f"episode {i}: cube lifted {lift * 100:.1f} cm")
    if dataset is not None:
        if lift >= MIN_LIFT:
            dataset.save_episode()
            # What check_dataset.py's replay needs to rebuild the start, since dropped episodes
            # make the dataset's episode index differ from i.
            row = {"episode_index": dataset.meta.total_episodes - 1, "seed": 1000 + i,
                   "start_qpos": start.tolist(), "layout": layout, "lift": float(lift)}
            with open(dataset.root / "meta" / "generation.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        else:
            discard_episode(dataset)
            print(f"  dropped: below {MIN_LIFT * 100:.0f} cm")
    return lift


def run_episodes(first, count, viewer, dataset, renderer):
    for i in range(first, first + count):
        run_episode(i, viewer, dataset, renderer)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1, help="episodes per batch")
    parser.add_argument("--headless", action="store_true", help="no viewer: run one batch and exit")
    parser.add_argument("--record", type=Path,
                        help="new directory to write kept episodes to as a LeRobot dataset; omitted, nothing is written")
    args = parser.parse_args()

    dataset = create_dataset(args.record) if args.record else None
    renderer = mujoco.Renderer(model, 180, 320) if dataset is not None else None
    viewer = None if args.headless else mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    try:
        first = 0
        run_episodes(first, args.episodes, viewer, dataset, renderer)
        while viewer is not None and viewer.is_running():
            if pending["rerun"]:
                pending["rerun"] = False
                first += args.episodes
                run_episodes(first, args.episodes, viewer, dataset, renderer)
            time.sleep(0.1)
    finally:
        if dataset is not None:
            finalize(dataset)


if __name__ == "__main__":
    main()
