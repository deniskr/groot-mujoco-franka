from scene import CUBE_REST_Z, DROID_HOME, data, model, random_layout, reset
import time

import mujoco
import mujoco.viewer
import numpy as np
import mink

episodes = 1
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

viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)

def run_episodes(first=0):
    for i in range(first, first + episodes):
        rng = np.random.default_rng(1000 + i) if i is not None else np.random.default_rng()
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
                data.ctrl[:7] = ik_step_mink(target)
                for j in range(STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
                # Measured after the physics steps: where the arm actually got to this frame.
                IK_CONFIG.update(data.qpos)
                pos_err = np.linalg.norm(data.site("panda/robotiq/pinch").xpos - target)
                axis_err = np.linalg.norm(axis_task.compute_error(IK_CONFIG))
                print(f"  phase {p} frame {frame:3d}  pos {pos_err * 1000:6.1f} mm  axis {axis_err:.4f}")
                viewer.sync()
                time.sleep(1 / FPS)
        print(f"episode {i}: cube lifted {(data.body('cube').xpos[2] - CUBE_REST_Z) * 100:.1f} cm")

first = 0
run_episodes(first)
while viewer.is_running():
    if pending["rerun"]:
        pending["rerun"] = False
        first += episodes
        run_episodes(first)
    time.sleep(0.1)

# run_episodes()
# while viewer.is_running():
#     time.sleep(0.1)

