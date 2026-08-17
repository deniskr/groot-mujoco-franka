import os
import mujoco.viewer

import torch, run
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from scene import CAMERAS, data, model


run.policy = Gr00tPolicy(                     
    model_path="nvidia/GR00T-N1.7-3B",
    embodiment_tag=EmbodimentTag.resolve("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"),
    device="mps",
    strict=True,
)

viewer = mujoco.viewer.launch_passive(model, data)

