import os
import sys
import tempfile
import pyrallis

from pathlib import Path

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

pairs = [
    # {
    #     "prompts": [
    #         "white humanoid robot, movie poster, main character of a science fiction movie",
    #         "comic book superhero, red body suit",
    #         "white humanoid robot, movie poster, villain character of a science fiction movie",
    #         "futuristic soldier, glowing armor, protagonist of an action game",
    #         "medieval knight in shining armor, fantasy movie hero",
    #         "steampunk adventurer, leather attire with brass accessories",
    #         "astronaut in a sleek space suit, exploring alien worlds",
    #         "cyberpunk hacker, neon-lit clothing, main character in a dystopian cityscape"
    #     ],
    #     "path": "texfusion_dataset/Text2Mesh/person.obj",
    #     "front_offset": -90.0
    # },
    # {
    #     "prompts": [
    #         "person wearing black shirt and white pants",
    #         "person wearing white t-shirt with a peace sign",
    #         "person wearing a classic detective trench coat and fedora",
    #         "surfer wearing board shorts with a tropical pattern",
    #         "mountaineer in a thermal jacket and snow goggles",
    #         "chef in a white jacket and checkered pants",
    #         "pilot in a vintage leather jacket with aviator sunglasses"
    #     ],
    #     "path": "texfusion_dataset/Renderpeople/rp_alvin_rigged_003_yup_a.obj"
    # },
    # {
    #     "prompts": [
    #         "person in red sweater, blue jeans",
    #         "person in white sweater with a red logo, yoga pants",
    #         "professional gamer in a team jersey and headphones",
    #         "ballet dancer in a pink tutu and ballet slippers",
    #         "rock star with leather jacket",
    #         "vintage 1950s dress with polka dots and sunglasses",
    #         "athlete in a running outfit with a marathon number"
    #     ],
    #     "path": "texfusion_dataset/Renderpeople/rp_alexandra_rigged_004_yup_a.obj"
    # },
    # {
    #     "prompts": [
    #         "nunn in a black dress",
    #         "nunn in a white dress, black headscarf",
    #         "professional in a suit jacket, skirt, and elegant headscarf",
    #         "athlete in sportswear with a sporty hijab",
    #         "artist in a paint-splattered apron and a stylish hijab",
    #         "student in a denim jacket, casual dress, and a colorful headscarf",
    #         "doctor in a lab coat with a simple, modest hijab"
    #     ],
    #     "path": "texfusion_dataset/Renderpeople/rp_adanna_rigged_007_yup_a.obj"
    # },
    # {
    #     "prompts": [
    #         "railroad worker wearing high-vis vest",
    #         "biker wearing red jacket and black pants",
    #         "firefighter in full gear with reflective stripes",
    #         "plumber in a blue jumpsuit",
    #         "electrician with a tool belt and safety goggles",
    #         "carpenter in overalls with a hammer in pocket",
    #         "landscape gardener in a green t-shirt and cargo pants"
    #     ],
    #     "path": "texfusion_dataset/Renderpeople/rp_aaron_rigged_001_yup_a.obj"
    # },
    {
        "prompts": [
            # "a photo of spiderman",
            # "a caricature of a pirate with a large hat and eye patch",
            "a whimsical wizard with a pointed hat, dark shadow"#,
            # "a cartoon astronaut with a bubbly space helmet",
            # "a ninja turtle with a colorful mask",
            # "a cartoon zombie in tattered clothes"
        ],
        "path": "shapes/human.obj"
    }

    # Age49-LoganWade -- WE DO NOT HAVE IT YET
    # Age26-AngelicaCollins -- WE DO NOT HAVE IT YET
]

puredepth = False

for pair in pairs:
  for prompt in pair["prompts"]:
    path_stem = Path(pair["path"]).stem
    exp_name = f"{path_stem}_{prompt}"

    if puredepth:
      exp_name += "_puredepth"
      print("PURE DEPTH")

    # if os.path.isdir(os.path.join(os.getcwd(), "experiments", exp_name)):
    #   print(f"skipping {exp_name}")
    #   continue

    success = False
    while success == False:
      try:
        with tempfile.NamedTemporaryFile(mode='w+') as fp:
          if not puredepth:
            fp.write(f"""log:
  exp_name: "{exp_name}"
guide:
  text: "{prompt}"
  shape_path: {pair["path"]}
  guidance_scale: 10
  second_model_type: "control_zero123"
  use_inpainting: False

  individual_control_of_conditions: True
  guidance_scale_i: 5
  guidance_scale_t: 5
  
{"render:" if "front_offset" in pair else ""}
  {("front_offset: " + str(pair["front_offset"])) if "front_offset" in pair else ""}""")
          else:
            print(4.1)
            fp.write(f"""log:
  exp_name: "{Path(pair["path"]).stem}_{prompt}_puredepth"
guide:
  text: "{prompt}"
  shape_path: {pair["path"]}
  guidance_scale: 10
    
{"render:" if "front_offset" in pair else ""}
  {("front_offset: " + str(pair["front_offset"])) if "front_offset" in pair else ""}""")
          fp.flush()

          @pyrallis.wrap(config_path=fp.name)
          def main(cfg: TrainConfig):
            print(5)
            trainer = TEXTure(cfg)
            if cfg.log.eval_only:
              trainer.full_eval()
            else:
              trainer.paint()

          main()
          print(6)
        success = True
        print(7)
      except KeyboardInterrupt:
        sys.exit(0)
      except Exception as error:
        print(error)
        pass
