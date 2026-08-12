# Third-party notices

This repository vendors no third-party source code or asset files. The robot models,
table arena and textures are fetched at install or run time from their own distributions
(`robot-descriptions`, `robosuite`), and the GR00T checkpoint is downloaded from the
Hugging Face hub.

Notices are reproduced here because the repository *does* commit rendered media —
`docs/*.gif` and `videos/*.mp4` — and those renders depict the third-party robot models
and embed robosuite's texture images pixel for pixel.

---

## Robotiq 2F-85 gripper model

From [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), obtained via
`robot-descriptions`. Depicted in the committed renders.

```
Copyright (c) 2013, ROS-Industrial
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## robosuite — table arena and textures

[ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite). `TableArena`
builds the table; its textures (`ceramic.png`, `steel-brushed.png`,
`light-gray-floor-tile.png`, `light-gray-plaster.png`) appear in every rendered frame.

```
MIT License

Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Franka Emika Panda model

From [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), obtained via
`robot-descriptions`. Depicted in the committed renders. Licensed under the Apache License
2.0; the upstream model directory ships no `NOTICE` file. A copy of the licence text is in
[LICENSE](LICENSE), which is the same Apache-2.0 text this project uses.

## MuJoCo

[google-deepmind/mujoco](https://github.com/google-deepmind/mujoco), Apache License 2.0.
Used as a dependency; not redistributed here.

## NVIDIA Isaac GR00T

Two separate licences, which do not travel together:

- **Code** — [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), Apache License 2.0.
  Used as a dependency; not redistributed here.
- **Model weights** — [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B),
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
  No weights are contained in this repository; they are downloaded by the user at run time,
  under that licence.

## DROID

[droid-dataset.github.io](https://droid-dataset.github.io/). The sample episodes
distributed with Isaac-GR00T supply the arm's start pose and the end-effector frame
conventions reproduced in `run.py`. No dataset files are contained in this repository;
refer to the dataset's own terms.
