import sys
import tempfile
import pyrallis

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

for i in reversed(range(1, 8, 2)):
  for t in reversed(range(1, 8, 2)):
    with tempfile.NamedTemporaryFile(mode='w+') as fp:
      fp.write(f"""log:
  exp_name: spiderman_c0123_{i}_{t}
guide:
  text: "A photo of Spiderman, {'{}'} view"
  append_direction: True
  shape_path: shapes/human.obj
  guidance_scale: 10
  second_model_type: "control_zero123"

  individual_control_of_conditions: True
  guidance_scale_i: {i}
  guidance_scale_t: {t}""")
      fp.flush()

      @pyrallis.wrap(config_path=fp.name)
      def main(cfg: TrainConfig):
        trainer = TEXTure(cfg)
        if cfg.log.eval_only:
          trainer.full_eval()
        else:
          trainer.paint()

      main()