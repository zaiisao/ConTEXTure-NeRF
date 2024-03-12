import sys
import tempfile
import pyrallis

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

for concat_control in range(1, 10, 2):
    for crossattn_control in range(1, 10, 2):
        for crossattn_concat in range(1, 10, 2):
            with tempfile.NamedTemporaryFile(mode='w+') as fp:
                fp.write(f"""log:
  exp_name: spiderman_{concat_control}_{crossattn_control}_{crossattn_concat}
guide:
  text: "A photo of Spiderman, {'{}'} view"
  append_direction: True
  shape_path: shapes/human.obj
  guidance_scale: 10
  second_model_type: "control_zero123"

  individual_control_of_conditions: True
  guidance_scale_concat_control: {concat_control}
  guidance_scale_crossattn_control: {crossattn_control}
  guidance_scale_crossattn_concat: {crossattn_concat}

  guess_mode: True
  use_inpainting: False""")
                fp.flush()

                @pyrallis.wrap(config_path=fp.name)
                def main(cfg: TrainConfig):
                    trainer = TEXTure(cfg)
                    if cfg.log.eval_only:
                        trainer.full_eval()
                    else:
                        trainer.paint()

                main()

                # os.system(f'python -m scripts.run_texture --config_path={fp.name}')
