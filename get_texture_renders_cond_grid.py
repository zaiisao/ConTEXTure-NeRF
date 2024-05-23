import os
import sys
import torch
import torchvision
from pytorch_lightning import seed_everything
from einops import repeat, rearrange
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from pathlib import Path

sys.path.append("../stablediffusion")

from transformers import pipeline
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from diffusers import DiffusionPipeline

import math
import tempfile
import pyrallis
# from diffusers import StableDiffusionDepth2ImgPipeline

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig
from src.utils import make_path, tensor2numpy, pad_tensor_to_size, split_zero123plus_grid

# torch.set_grad_enabled(False)


# from scripts.gradio.depth2img import predict

pairs = [
    {
        "prompts": [
            "white humanoid robot, movie poster, main character of a science fiction movie",
            "comic book superhero, red body suit",
            "white humanoid robot, movie poster, villain character of a science fiction movie",
            "futuristic soldier, glowing armor, protagonist of an action game",
            "medieval knight in shining armor, fantasy movie hero",
            "steampunk adventurer, leather attire with brass accessories",
            "astronaut in a sleek space suit, exploring alien worlds",
            "cyberpunk hacker, neon-lit clothing, main character in a dystopian cityscape"
            "blue humanoid robot, low quality, blurry, noisy",
            "a person in a red shirt and blue pants"
        ],
        "path": "texfusion_dataset/Text2Mesh/person.obj",
        "front_offset": -90.0
    },
    {
        "prompts": [
            "person wearing black shirt and white pants",
            "person wearing white t-shirt with a peace sign",
            "person wearing a classic detective trench coat and fedora",
            "surfer wearing board shorts with a tropical pattern",
            "mountaineer in a thermal jacket and snow goggles",
            "chef in a white jacket and checkered pants",
            "pilot in a vintage leather jacket with aviator sunglasses"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_alvin_rigged_003_yup_a.obj"
    },
    {
        "prompts": [
            "person in red sweater, blue jeans",
            "person in white sweater with a red logo, yoga pants",
            "professional gamer in a team jersey and headphones",
            "ballet dancer in a pink tutu and ballet slippers",
            "rock star with leather jacket",
            "vintage 1950s dress with polka dots and sunglasses",
            "athlete in a running outfit with a marathon number"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_alexandra_rigged_004_yup_a.obj"
    },
    {
        "prompts": [
            "nunn in a black dress",
            "nunn in a white dress, black headscarf",
            "professional in a suit jacket, skirt, and elegant headscarf",
            "athlete in sportswear with a sporty hijab",
            "artist in a paint-splattered apron and a stylish hijab",
            "student in a denim jacket, casual dress, and a colorful headscarf",
            "doctor in a lab coat with a simple, modest hijab"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_adanna_rigged_007_yup_a.obj"
    },
    {
        "prompts": [
            "railroad worker wearing high-vis vest",
            "biker wearing red jacket and black pants",
            "firefighter in full gear with reflective stripes",
            "plumber in a blue jumpsuit",
            "electrician with a tool belt and safety goggles",
            "carpenter in overalls with a hammer in pocket",
            "landscape gardener in a green t-shirt and cargo pants"
        ],
        "path": "texfusion_dataset/Renderpeople/rp_aaron_rigged_001_yup_a.obj"
    },
    {
        "prompts": [
            "a photo of spiderman",
            "a caricature of a pirate with a large hat and eye patch",
            "a whimsical wizard with a pointed hat, dark shadow",
            "a cartoon astronaut with a bubbly space helmet",
            "a ninja turtle with a colorful mask",
            "a cartoon zombie in tattered clothes"
        ],
        "path": "shapes/human.obj"
    }
]

def get_nonzero_region(mask: torch.Tensor):
    # Get the indices of the non-zero elements
    nz_indices = mask.nonzero()
    # Get the minimum and maximum indices along each dimension
    min_h, max_h = nz_indices[:, 0].min(), nz_indices[:, 0].max()
    min_w, max_w = nz_indices[:, 1].min(), nz_indices[:, 1].max()

    # Calculate the size of the square region
    size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
    # Calculate the upper left corner of the square region
    h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
    w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

    min_h = max(0, int(h_start))
    min_w = max(0, int(w_start))
    max_h = min(mask.shape[0], int(min_h + size))
    max_w = min(mask.shape[1], int(min_w + size))

    return min_h, min_w, max_h, max_w

def crop_image_tensor(input_tensor, min_h, min_w, max_h, max_w):
    return input_tensor[:, :, min_h:max_h, min_w:max_w]

def pad_tensor_to_size(tensor, target_height, target_width):
    if tensor.dim() == 3:  # 만약 텐서가 [C, H, W] 형태라면
        tensor = tensor.unsqueeze(0)  # 배치 차원 추가
    _, _, h, w = tensor.shape
    pad_height = (target_height - h) // 2
    pad_width = (target_width - w) // 2
    padded_tensor = torch.nn.functional.pad(tensor, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=1)
    return padded_tensor

def resize_image_to_320x320(tensor):
    # 이미지를 320x320 크기로 조정
    resized_image = TF.resize(tensor, [320, 320])
    return resized_image

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def save_image(tensor, filename):
    # 텐서의 차원을 확인
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        # 배치 차원을 제거 (첫 번째 이미지 선택)
        tensor = tensor.squeeze(0)
    # 이미지 저장
    img = TF.to_pil_image(tensor)
    img.save(filename)

for pair in pairs:
  for prompt in pair["prompts"]:
    path_stem = Path(pair["path"]).stem
    exp_name = f"{path_stem}_{prompt}"
    obj_path = os.path.join(os.getcwd(), pair["path"])

    # try:
    if True:
        with tempfile.NamedTemporaryFile(mode='w+') as fp:
            fp.write(f"""log:
  exp_name: "{exp_name}"
guide:
  text: "{prompt}"
  shape_path: {pair["path"]}
  guidance_scale: 10
  use_zero123plus: True
  
{"render:" if "front_offset" in pair else ""}
  {("front_offset: " + str(pair["front_offset"])) if "front_offset" in pair else ""}
optim:
  learn_max_z_normals: True""")
            fp.flush()

            @pyrallis.wrap(config_path=fp.name)
            def main(cfg: TrainConfig):
                trainer = TEXTure(cfg)
                
                if cfg.log.eval_only:
                    trainer.full_eval()
                else:
                    trainer.paint()

                background = torch.Tensor([0.5, 0.5, 0.5]).to(trainer.device)

                renders = []

                for phi in [0, 30, 90, 150, 210, 270, 330]:
                    # Set elevation angle based on the azimuth angle (phi)
                    if phi in [30, 150, 270]:
                        theta = math.radians(90 - 30)  # Elevation is 30 degrees for these azimuth angles
                    elif phi in [90, 210, 330]:
                        theta = math.radians(90 + 20)  # Elevation is -20 degrees for these azimuth angles
                    elif phi == 0:
                        theta = math.radians(60)  # Default elevation, can be adjusted as needed

                    outputs = trainer.mesh_model.render(
                        theta=theta,
                        phi=math.radians(phi),
                        radius=1.5,
                        background=background
                    )

                    renders.append({ 
                        "image": outputs["image"],
                        "mask": outputs["mask"],
                        "phi": phi
                    })

                del trainer

                max_height, max_width = 0, 0
                cropped_images = []

                for render in renders:
                    image_rgba = (render["image"] * render["mask"]) + (torch.ones_like(render["image"]) * (1 - render["mask"]))
                    min_h, min_w, max_h, max_w = get_nonzero_region(render["mask"][0, 0])  # Assuming mask is [1, H, W]
                    cropped_image = crop_image_tensor(image_rgba, min_h, min_w, max_h, max_w)
                    cropped_images.append(cropped_image)
                    max_height = max(max_height, cropped_image.shape[-2])
                    max_width = max(max_width, cropped_image.shape[-1])
                
                uniform_cropped_images = []

                for image in cropped_images:
                    if image.shape[-2] != max_height or image.shape[-1] != max_width:
                        padded_image = pad_tensor_to_size(image, max_height, max_width)
                        uniform_cropped_images.append(padded_image)
                    else:
                        uniform_cropped_images.append(image)
                
                if "name" in pair:
                    mesh_name = pair["name"]
                else:
                    mesh_name = os.path.basename(obj_path)

                createDirectory(f"/home/sogang/jaehoon/texture_test/contexture/{mesh_name}/{prompt}")

                for i, uniform_cropped_image in enumerate(uniform_cropped_images):
                    uniform_cropped_image = resize_image_to_320x320(uniform_cropped_image)
                    save_image(uniform_cropped_image, f"/home/sogang/jaehoon/texture_test/contexture/{mesh_name}/{prompt}/rendered_image_{i}.png")

            main()
    # except KeyboardInterrupt:
    #     sys.exit(0)
    # except Exception as error:
    #     print(error)
    #     pass

