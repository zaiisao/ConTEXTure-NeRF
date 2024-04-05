import sys
import tempfile
import pyrallis

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

pairs = [
    {
        "prompts": [
            "white humanoid robot, movie poster, main character of a science fiction movie",
            "comic book superhero, red body suit",
            "white humanoid robot, movie poster, villain character of a science fiction movie"
        ],
        "path": "texfusion_dataset/Text2Mesh/person.obj",
        "front_offset": 90.0
    },
    {
        "prompts": [
            "person wearing black shirt and white pants",
            "person wearing white t-shirt with a peace sign"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_alvin_rigged_003_yup_a.obj"
    },
    {
        "prompts": [
            "person in red sweater, blue jeans",
            "person in white sweater with a red logo, yoga pants"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_alexandra_rigged_004_yup_a.obj"
    },
    {
        "prompts": [
            "nunn in a black dress",
            "nunn in a white dress, black headscarf"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_adanna_rigged_007_yup_a.obj"
    },
    {
        "prompts": [
            "railroad worker wearing high-vis vest",
            "biker wearing red jacket and black pants"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_aaron_rigged_001_yup_a.obj"
    },
    {
        "prompts": [
            "a photo of spiderman",
            "a minecraft character"
        ],
        "path": "shapes/human.obj"
    },

    # Age49-LoganWade -- WE DO NOT HAVE IT YET
    # Age26-AngelicaCollins -- WE DO NOT HAVE IT YET
]

for i in reversed(range(3, 7)):
  for t in reversed(range(1, 11)):
    with tempfile.NamedTemporaryFile(mode='w+') as fp:
      fp.write(f"""log:
  exp_name: spiderman_{i}_{t}
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