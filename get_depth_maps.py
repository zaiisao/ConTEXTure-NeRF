import os
# import sys

# import torch
# import torchvision

# import math
# import tempfile
# import pyrallis

# from src.training.trainer import TEXTure
# from src.configs.train_config import TrainConfig

# sys.path.append("./src/zero123/zero123")
# sys.path.append("./src/zero123/ControlNet")

pairs = [
    {
        "prompts": [
            "lamborghini urus",
            "pink porsche cayenne",
            "white mercedes benz SUV",
            "green ambulance with red cross"
        ],
        "path": "shapenet/1a64bf1e658652ddb11647ffa4306609/model.obj"
    },
    {
        "prompts": [
            "silver porsche 911",
            "blue bmw m5 with white stripes",
            "red ferrari with orange headlights",
            "beautiful yellow sports car"
        ],
        "path": "shapenet/1a7b9697be903334b99755e16c4a9d21/model.obj"
    },
    {
        "prompts": [
            "black pickup truck",
            "old toyota pickup truck",
            "red pickup truck with black trunk"
        ],
        "path": "shapenet/1a48d03a977a6f0aeda0253452893d75/model.obj"
    },
    {
        "prompts": [
            "blue luggage box",
            "black luggage with a yellow smiley face"
        ],
        "path": "shapenet/133c16fc6ca7d77676bb31db0358e9c6/model.obj"
    },
    {
        "prompts": [
            "white handbag",
            "turquoise blue handbag",
            "black handbag with gold trims"
        ],
        "path": "shapenet/1b9ef45fefefa35ed13f430b2941481/model.obj"
    },
    {
        "prompts": [
            "red backpack",
            "camper bag, camouflage",
            "black backpack with red accents"
        ],
        "path": "shapenet/54cd45b275f551b276bb31db0358e9c6/model.obj"
    },
    {
        "prompts": [
            "crocodile skin handbag",
            "blue handbag with silver trims",
            "linen fabric handbag"
        ],
        "path": "shapenet/e49f6ae8fa76e90a285e5a1f74237618/model.obj"
    },
    {
        "prompts": [
            "leather lounge chair",
            "red velvet lounge chair"
        ],
        "path": "shapenet/2c6815654a9d4c2aa3f600c356573d21/model.obj"
    },
    {
        "prompts": [
            "soft pearl fabric sofa",
            "modern building in the shape of a sofa"
        ],
        "path": "shapenet/2fa970b5c40fbfb95117ae083a7e54ea/model.obj"
    },
    {
        "prompts": [
            "yellow plastic stool with white seat",
            "silver metallic stool"
        ],
        "path": "shapenet/5bfee410a492af4f65ba78ad9601cf1b/model.obj"
    },
    {
        "prompts": [
            "wooden dinning chair with leather seat",
            "cast iron dinning chair"
        ],
        "path": "shapenet/97cd4ed02e022ce7174150bd56e389a8/model.obj"
    },
    {
        "prompts": [
            "yellow school bus"
        ],
        "path": "shapenet/5b04b836924fe955dab8f5f5224d1d8a/model.obj"
    },
    {
        "prompts": [
            "new york taxi, yellow cab",
            "taxi from tokyo, black toyota crown"
        ],
        "path": "shapenet/7fc729def80e5ef696a0b8543dac6097/model.obj"
    },
    {
        "prompts": [
            "green ambulance with red cross",
            "ambulance, white paint with red accents",
            "pink van with blue top"
        ],
        "path": "shapenet/85a8ee0ef94161b049d69f6eaea5d368/model.obj"
    },
    {
        "prompts": [
            "old and rusty volkswagon beetle",
            "red volkswagon beetle, cartoon style"
        ],
        "path": "shapenet/a3d77c6b58ea6e75e4b68d3b17c43658/model.obj"
    },
    {
        "prompts": [
            "classic red farm truck",
            "farm truck from cars movie, brown, rusty"
        ],
        "path": "shapenet/b4a86e6b096bb93eb7727d322e44e79b/model.obj"
    },
    {
        "prompts": [
            "batmobile",
            "blue bugatti chiron"
        ],
        "path": "shapenet/fc86bf465674ec8b7c3c6f82a395b347/model.obj"
    },
    {
        "prompts": [
            "white humanoid robot, movie poster, main character of a science fiction movie",
            "comic book superhero, red body suit",
            "white humanoid robot, movie poster, villain character of a science fiction movie"
        ],
        "path": "Text2Mesh/person.obj"
    },
    {
        "prompts": [
            "person wearing black shirt and white pants",
            "person wearing white t-shirt with a peace sign"
        ],
        "path": "Renderpeople/rp_alvin_rigged_003_yup_a.obj"
    },
    {
        "prompts": [
            "person in red sweater, blue jeans",
            "person in white sweater with a red logo, yoga pants"
        ],
        "path": "Renderpeople/rp_alexandra_rigged_004_yup_a.obj"
    },
    {
        "prompts": [
            "nunn in a black dress",
            "nunn in a white dress, black headscarf"
        ],
        "path": "Renderpeople/rp_adanna_rigged_007_yup_a.obj"
    },
    {
        "prompts": [
            "railroad worker wearing high-vis vest",
            "biker wearing red jacket and black pants"
        ],
        "path": "Renderpeople/rp_aaron_rigged_001_yup_a.obj"
    },

    # Age49-LoganWade -- WE DO NOT HAVE IT YET
    # Age26-AngelicaCollins -- WE DO NOT HAVE IT YET

    {
        "prompts": [
            "medieval celtic House, stone bricks, wooden roof",
            "minecraft house, bricks, rock, grass, stone",
            "colonial style house, white walls, blue ceiling"
        ],
        "path": "Turbosquid/house.obj"
    },
    {
        "prompts": [
            "white house by the dock, green ceiling, cartoon style",
            "minecraft house, bricks, rock, grass, stone",
            "white house by the dock, green ceiling, impressionist painting"
        ],
        "path": "Turbosquid/casa.obj"
    },
    {
        "prompts": [
            "brown rabbit",
            "purple rabbit",
            "tiger with yellow and black stripes"
        ],
        "path": "Turbosquid/Rabbit.obj"
    },
    {
        "prompts": [
            "cartoon dog",
            "lion dance, red and green",
            "brown bull dog"
        ],
        "path": "Turbosquid/LionGames_obj.obj"
    },
    {
        "prompts": [
            "brown mountain goat",
            "black goat with white hoofs",
            "milk cow"
        ],
        "path": "Turbosquid/Goat.obj"
    },
    {
        "prompts": [
            "cartoon milk cow",
            "giant panda"
        ],
        "path": "Turbosquid/Cow_High.obj"
    },
    {
        "prompts": [
            "cartoon fox",
            "brown wienner dog",
            "white fox"
        ],
        "path": "Turbosquid/Red_fox.obj"
    },
    {
        "prompts": [
            "white bunny"
        ],
        "path": "Stanford/bun_zipper.obj"
    },
    {
        "prompts": [
            "black and white dragon in chinese ink art style",
            "cartoon dragon, red and green"
        ],
        "path": "Stanford/dragon_vrip.obj"
    },
    {
        "prompts": [
            "sandstone statue of hermanubis",
            "portrait of greek-egyptian deity hermanubis, lapis skin and gold clothing"
        ],
        "path": "3DScans/Hermanubis.obj"
    },
    {
        "prompts": [
            "portrait of Provost, oil paint",
            "marble statue of Provost"
        ],
        "path": "3DScans/Statue-Provost_1M_Poly.obj"
    }
]

for pair in pairs:
    for prompt in pair["prompts"]:
        path = os.path.join(os.getcwd(), "texfusion_dataset", pair["path"])
        print(os.path.isfile(path))

            # depth_render = outputs['depth']
            # print(depth_render)
            # torchvision.utils.save_image(depth_render, "./test.png")

# for mesh_path in mesh_paths:
#     with tempfile.NamedTemporaryFile(mode='w+') as fp:
#         fp.write(f"""log:
#   exp_name: test
# guide:
#     text: "nothing"
#     shape_path: shapes/human.obj""")
#         fp.flush()

#         @pyrallis.wrap(config_path=fp.name)
#         def main(cfg: TrainConfig):
#             trainer = TEXTure(cfg)
#             background = torch.Tensor([1, 1, 1]).to(trainer.device)

#             # for theta in [60, 90, ]
#             outputs = trainer.mesh_model.render(
#                 theta=math.radians(60),
#                 phi=math.radians(50),
#                 radius=1.5,
#                 background=background
#             )

#             depth_render = outputs['depth']
#             print(depth_render)
#             torchvision.utils.save_image(depth_render, "/home/jaehoon/test.png")

#         main()