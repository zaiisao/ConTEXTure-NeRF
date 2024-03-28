import sys

import torch
import torchvision

import math
import tempfile
import pyrallis

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

mesh_paths = ["human.obj"]

for mesh_path in mesh_paths:
    with tempfile.NamedTemporaryFile(mode='w+') as fp:
        fp.write(f"""log:
  exp_name: test
guide:
    text: "nothing"
    shape_path: shapes/human.obj""")
        fp.flush()

        @pyrallis.wrap(config_path=fp.name)
        def main(cfg: TrainConfig):
            trainer = TEXTure(cfg)
            background = torch.Tensor([1, 1, 1]).to(trainer.device)

            # for theta in [60, 90, ]
            outputs = trainer.mesh_model.render(
                theta=math.radians(60),
                phi=math.radians(50),
                radius=1.5,
                background=background
            )

            depth_render = outputs['depth']
            print(depth_render)
            torchvision.utils.save_image(depth_render, "/home/jaehoon/test.png")

        main()