"""MuJoCo scene: a Franka Panda with a Robotiq 2F-85, standing beside a robosuite table.

Geometry, cameras and lighting are chosen to keep rendered observations inside the
distribution the GR00T DROID head was trained on. See README.md for why each number is
what it is; changing them changes what the policy does.
"""

from pathlib import Path

import mujoco
import numpy as np
from robot_descriptions import panda_mj_description, robotiq_2f85_mj_description
from robosuite.models.arenas import TableArena
from scipy.spatial.transform import Rotation

TABLE_TOP = 0.8
BASE_POS = [-0.45, 0, 0.63]  # 0.17 m below the tabletop: on DROID the work surface is above the base plate
CUBE_SIZE = [0.02, 0.02, 0.02]
CAMERAS = ("exterior_image_1_left", "wrist_image_left")

arm = mujoco.MjSpec.from_file(str(Path(panda_mj_description.MJCF_PATH).with_name("panda_nohand.xml")))
arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # 2F-85 contact tuning
arm.option.impratio = 10
arm.attach(
    mujoco.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH),
    site="attachment_site",
    prefix="robotiq/",
)

# TableArena rather than arenas/table_arena.xml: the raw XML is a template whose top is a
# solid floor-to-surface block and whose legs are stacked at the body origin. Only
# configure_location() turns it into a 5 cm slab on four legs.
scene = mujoco.MjSpec.from_string(
    TableArena(table_full_size=(0.8, 1.2, 0.05), table_offset=(0.2, 0, TABLE_TOP)).get_xml()
)
scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
scene.option.impratio = 10
scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
scene.attach(arm, frame=scene.worldbody.add_frame(pos=BASE_POS), prefix="panda/")
scene.worldbody.add_geom(
    name="robot_stand",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[BASE_POS[0], BASE_POS[1], BASE_POS[2] / 2],
    size=[0.12, 0.12, BASE_POS[2] / 2],
    rgba=[0.25, 0.25, 0.28, 1],
)
# robosuite recolours collision geoms olive; this one is coincident with the visual slab and
# z-fights it. Group 3 is not drawn by default. Rendering only, contacts are untouched.
next(g for g in scene.geoms if g.name == "table_collision").group = 3


def rest_z(gtype, size):
    """Spawn height at which a geom of this size rests exactly on the tabletop."""
    half = {mujoco.mjtGeom.mjGEOM_BOX: 2, mujoco.mjtGeom.mjGEOM_CYLINDER: 1, mujoco.mjtGeom.mjGEOM_SPHERE: 0}
    return TABLE_TOP + size[half[gtype]]


CUBE_REST_Z = rest_z(mujoco.mjtGeom.mjGEOM_BOX, CUBE_SIZE)

# Target cube plus three distractors, all inside DROID's observed workspace and clear of
# the cube's randomization box so nothing spawns interpenetrating.
for name, gtype, size, xy, rgba in [
    ("cube", mujoco.mjtGeom.mjGEOM_BOX, CUBE_SIZE, [0.10, 0.00], [0.80, 0.20, 0.20, 1]),
    ("dist_box", mujoco.mjtGeom.mjGEOM_BOX, [0.025, 0.025, 0.025], [0.20, -0.17], [0.20, 0.45, 0.85, 1]),
    ("dist_cyl", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.030, 0.035, 0], [0.05, -0.20], [0.20, 0.65, 0.30, 1]),
    ("dist_sphere", mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], [-0.10, 0.05], [0.90, 0.75, 0.20, 1]),
]:
    body = scene.worldbody.add_body(name=name, pos=[*xy, rest_z(gtype, size)])
    body.add_freejoint(name=name)
    body.add_geom(type=gtype, size=size, rgba=rgba)


def lookat_quat(pos, target, up=(0, 0, 1)):
    z = np.subtract(pos, target)
    z /= np.linalg.norm(z)  # a camera looks along its own -z
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, np.cross(z, x), z]).flatten())
    return quat


# Exterior view. DROID tripods moved every session, so no single pose is "correct"; what
# every real frame shares is the arm and the objects both being in view.
EXTERIOR_POS = np.add(BASE_POS, [0.20, -0.85, 0.55])
scene.worldbody.add_camera(
    name="exterior_image_1_left",
    pos=EXTERIOR_POS,
    quat=lookat_quat(EXTERIOR_POS, np.add(BASE_POS, [0.55, -0.05, 0.20])),
    fovy=54,
    resolution=[320, 180],
)

# The wrist mount is constructed, not derived: orbit the flange approach axis at radius R,
# sit Z along it, aim at a point AIM ahead. phi=270 puts the finger pads side by side along
# the bottom of the frame, as they are in real DROID wrist video, and leaves the cube
# unoccluded. Rotating about the flange z axis cannot move the fingertips, so phi has to be
# chosen from the off-axis pads.
WRIST_PHI, WRIST_R, WRIST_Z, WRIST_AIM = np.radians(270), 0.12, 0.01, 0.35
wrist_p = np.array([WRIST_R * np.cos(WRIST_PHI), WRIST_R * np.sin(WRIST_PHI), WRIST_Z])
wrist_z_ax = wrist_p - [0, 0, WRIST_AIM]
wrist_z_ax /= np.linalg.norm(wrist_z_ax)
wrist_x_ax = np.cross([0, 0, 1.0], wrist_z_ax)
wrist_x_ax /= np.linalg.norm(wrist_x_ax)
wrist_R = np.column_stack([wrist_x_ax, np.cross(wrist_z_ax, wrist_x_ax), wrist_z_ax])

# base_mount relative to the flange is measured from a probe compile, not assumed, so this
# survives menagerie revisions. The soft-light array below needs the same probe.
probe = scene.compile()
probe_data = mujoco.MjData(probe)
mujoco.mj_forward(probe, probe_data)
site, mount = probe_data.site("panda/attachment_site"), probe_data.body("panda/robotiq/base_mount")
R_mount_flange = mount.xmat.reshape(3, 3).T @ site.xmat.reshape(3, 3)
p_mount_flange = mount.xmat.reshape(3, 3).T @ (site.xpos - mount.xpos)
scene.body("panda/robotiq/base_mount").add_camera(
    name="wrist_image_left",
    pos=p_mount_flange + R_mount_flange @ wrist_p,
    quat=Rotation.from_matrix(R_mount_flange @ wrist_R).as_quat()[[3, 0, 1, 2]],  # xyzw -> wxyz
    fovy=58,
    resolution=[320, 180],
)

# Soft shadows. One light physically cannot cast a penumbra -- MuJoCo's shadow test is
# binary -- so reproduce the cause instead: several dim copies of panda/top spread over a
# disc, each casting its own hard shadow. Splitting the intensity keeps total brightness
# unchanged. The renderer uses at most 7 lights and the arena supplies one.
SOFT_LIGHTS, LIGHT_RADIUS = 6, 0.12
spot = next(light for light in scene.lights if light.name == "panda/top")
spot_id = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_LIGHT, "panda/top")
spot_pos = probe_data.light_xpos[spot_id].copy()  # world pose, not the attached spec's [0, 0, 2]
spot_dir = probe_data.light_xdir[spot_id] / np.linalg.norm(probe_data.light_xdir[spot_id])
disc_u = np.cross(spot_dir, [0, 1.0, 0])
disc_u /= np.linalg.norm(disc_u)
disc_v = np.cross(spot_dir, disc_u)
spot.diffuse = list(np.array(spot.diffuse) / SOFT_LIGHTS)
spot.specular = list(np.array(spot.specular) / SOFT_LIGHTS)
golden = np.pi * (3 - np.sqrt(5))
for i in range(1, SOFT_LIGHTS):
    r = np.sqrt(i / (SOFT_LIGHTS - 1)) * LIGHT_RADIUS  # sqrt radii on a golden spiral fill the disc
    scene.worldbody.add_light(
        name=f"soft_{i}",
        pos=list(spot_pos + r * (np.cos(i * golden) * disc_u + np.sin(i * golden) * disc_v)),
        dir=list(spot_dir),
        cutoff=spot.cutoff,
        exponent=spot.exponent,
        diffuse=spot.diffuse,
        specular=spot.specular,
    )

model = scene.compile()
model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
model.vis.quality.shadowsize = 8192
model.vis.map.shadowclip = 0.15  # shadow frustum over the workspace, not the 10.6 m arena
model.vis.map.znear = 0.0005  # a fraction of model extent: the default clips the whole gripper
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)


def reset(qpos7, cube_xy=None, settle=150):
    """Arm at an in-distribution DROID pose, gripper open, free objects settled on the table."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:7] = data.ctrl[:7] = qpos7
    data.qvel[:] = 0
    data.ctrl[-1] = 0
    if cube_xy is not None:
        adr = int(np.ravel(model.joint("cube").qposadr)[0])
        data.qpos[adr : adr + 7] = [*cube_xy, CUBE_REST_Z, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    for _ in range(settle):
        mujoco.mj_step(model, data)
