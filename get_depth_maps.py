import os
import sys

import torch
import torchvision
from pytorch_lightning import seed_everything
from einops import repeat, rearrange
import numpy as np
from PIL import Image

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import math
import tempfile
import pyrallis
# from diffusers import StableDiffusionDepth2ImgPipeline

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig

torch.set_grad_enabled(False)

sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")
sys.path.append("../stablediffusion")

# from scripts.gradio.depth2img import predict

pairs = [
    {
        "prompts": [
            "lamborghini urus",
            "pink porsche cayenne",
            "white mercedes benz SUV",
            "green ambulance with red cross"
        ],
        "path": "shapenet/1a64bf1e658652ddb11647ffa4306609/model.obj",
        "name": "1a64bf1e658652ddb11647ffa4306609"
    },
    {
        "prompts": [
            "silver porsche 911",
            "blue bmw m5 with white stripes",
            "red ferrari with orange headlights",
            "beautiful yellow sports car"
        ],
        "path": "shapenet/1a7b9697be903334b99755e16c4a9d21/model.obj",
        "name": "1a7b9697be903334b99755e16c4a9d21"
    },
    {
        "prompts": [
            "black pickup truck",
            "old toyota pickup truck",
            "red pickup truck with black trunk"
        ],
        "path": "shapenet/1a48d03a977a6f0aeda0253452893d75/model.obj",
        "name": "1a48d03a977a6f0aeda0253452893d75"
    },
    {
        "prompts": [
            "blue luggage box",
            "black luggage with a yellow smiley face"
        ],
        "path": "shapenet/133c16fc6ca7d77676bb31db0358e9c6/model.obj",
        "name": "133c16fc6ca7d77676bb31db0358e9c6"
    },
    {
        "prompts": [
            "white handbag",
            "turquoise blue handbag",
            "black handbag with gold trims"
        ],
        "path": "shapenet/1b9ef45fefefa35ed13f430b2941481/model.obj",
        "name": "1b9ef45fefefa35ed13f430b2941481"
    },
    {
        "prompts": [
            "red backpack",
            "camper bag, camouflage",
            "black backpack with red accents"
        ],
        "path": "shapenet/54cd45b275f551b276bb31db0358e9c6/model.obj",
        "name": "54cd45b275f551b276bb31db0358e9c6"
    }
    {
        "prompts": [
            "crocodile skin handbag",
            "blue handbag with silver trims",
            "linen fabric handbag"
        ],
        "path": "shapenet/e49f6ae8fa76e90a285e5a1f74237618/model.obj",
        "name": "e49f6ae8fa76e90a285e5a1f74237618"
    },
    {
        "prompts": [
            "leather lounge chair",
            "red velvet lounge chair"
        ],
        "path": "shapenet/2c6815654a9d4c2aa3f600c356573d21/model.obj",
        "name": "2c6815654a9d4c2aa3f600c356573d21"
    },
    {
        "prompts": [
            "soft pearl fabric sofa",
            "modern building in the shape of a sofa"
        ],
        "path": "shapenet/2fa970b5c40fbfb95117ae083a7e54ea/model.obj",
        "name": "2fa970b5c40fbfb95117ae083a7e54ea"
    },
    {
        "prompts": [
            "yellow plastic stool with white seat",
            "silver metallic stool"
        ],
        "path": "shapenet/5bfee410a492af4f65ba78ad9601cf1b/model.obj",
        "name": "5bfee410a492af4f65ba78ad9601cf1b"
    },
    {
        "prompts": [
            "wooden dinning chair with leather seat",
            "cast iron dinning chair"
        ],
        "path": "shapenet/97cd4ed02e022ce7174150bd56e389a8/model.obj",
        "name": "97cd4ed02e022ce7174150bd56e389a8"
    },
    {
        "prompts": [
            "yellow school bus"
        ],
        "path": "shapenet/5b04b836924fe955dab8f5f5224d1d8a/model.obj",
        "name": "5b04b836924fe955dab8f5f5224d1d8a"
    },
    {
        "prompts": [
            "new york taxi, yellow cab",
            "taxi from tokyo, black toyota crown"
        ],
        "path": "shapenet/7fc729def80e5ef696a0b8543dac6097/model.obj",
        "name": "7fc729def80e5ef696a0b8543dac6097"
    },
    {
        "prompts": [
            "green ambulance with red cross",
            "ambulance, white paint with red accents",
            "pink van with blue top"
        ],
        "path": "shapenet/85a8ee0ef94161b049d69f6eaea5d368/model.obj",
        "name": "85a8ee0ef94161b049d69f6eaea5d368"
    },
    {
        "prompts": [
            "old and rusty volkswagon beetle",
            "red volkswagon beetle, cartoon style"
        ],
        "path": "shapenet/a3d77c6b58ea6e75e4b68d3b17c43658/model.obj",
        "name": "a3d77c6b58ea6e75e4b68d3b17c43658"
    },
    {
        "prompts": [
            "classic red farm truck",
            "farm truck from cars movie, brown, rusty"
        ],
        "path": "shapenet/b4a86e6b096bb93eb7727d322e44e79b/model.obj",
        "name": "b4a86e6b096bb93eb7727d322e44e79b"
    },
    {
        "prompts": [
            "batmobile",
            "blue bugatti chiron"
        ],
        "path": "shapenet/fc86bf465674ec8b7c3c6f82a395b347/model.obj",
        "name": "fc86bf465674ec8b7c3c6f82a395b347"
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

def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler

def paint(sampler, depth, prompt, t_enc, seed, scale, num_samples=1, callback=None,
          do_full_sample=False):
    model = sampler.model
    seed_everything(seed)

    with torch.no_grad(),\
            torch.autocast("cuda"):
        c = model.cond_stage_model.encode(prompt)

        c_cat = list()
        cc = torch.nn.functional.interpolate(
            depth,
            size=(depth.shape[-2] // 8, depth.shape[-1] // 8),
            mode="bicubic",
            align_corners=False,
        )
        depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                        keepdim=True)
        cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
        c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        z_enc = torch.randn((1, 4, depth.shape[-2] // 8, depth.shape[-1] // 8)).to(model.device)
        # decode it
        samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc_full, callback=callback)
        x_samples_ddim = model.decode_first_stage(samples)
        result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255

    return [Image.fromarray(img.astype(np.uint8)) for img in result]


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def pad_image_tensor(input_tensor):
    # Assuming input_tensor is of shape [C, H, W]
    B, C, H, W = input_tensor.shape
    pad_h = max(2, -H % 64)  # Ensuring the padding makes the height a multiple of 64
    pad_w = max(2, -W % 64)  # Ensuring the padding makes the width a multiple of 64

    # Padding format is (left, right, top, bottom) for last two dimensions
    padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode='replicate')
    
    return padded_tensor

# def predict(input_image, prompt, steps, num_samples, scale, seed, eta, strength):
def predict(depth, prompt, steps, num_samples, scale, seed, eta, strength):
    # depth = input_image.convert("RGB")
    # depth = pad_image(depth)  # resize to integer multiple of 32
    depth = pad_image_tensor(depth)

    sampler = initialize_model(
        "/home/sogang/jaehoon/stablediffusion/configs/stable-diffusion/v2-midas-inference.yaml",
        "/home/sogang/jaehoon/stablediffusion/512-depth-ema.ckpt"
    )

    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    do_full_sample = strength == 1.
    t_enc = min(int(strength * steps), steps-1)
    result = paint(
        sampler=sampler,
        # image=input_image, #image,
        depth=depth,
        prompt=prompt,
        t_enc=t_enc,
        seed=seed,
        scale=scale,
        num_samples=num_samples,
        callback=None,
        do_full_sample=do_full_sample
    )
    return result

for pair in pairs:
    # try:
    with tempfile.NamedTemporaryFile(mode='w+') as fp:
        obj_path = os.path.join(os.getcwd(), "texfusion_dataset", pair["path"])

        fp.write(f"""log:
    exp_name: test
guide:
    text: "nothing"
    shape_path: {obj_path}""")
        fp.flush()

        @pyrallis.wrap(config_path=fp.name)
        def main(cfg: TrainConfig):
            trainer = TEXTure(cfg)
            background = torch.Tensor([1, 1, 1]).to(trainer.device)

            renders = []
            for phi in [0, 45, -45, 90, -90, 135, -135, 180]:
                outputs = trainer.mesh_model.render(
                    theta=math.radians(60),
                    phi=math.radians(phi),
                    radius=1.5,
                    background=background
                )

                renders.append({
                    "depth": outputs["depth"],
                    "mask": outputs["mask"],
                    "phi": phi
                })

            del trainer

            for prompt in pair["prompts"]:
                for render in renders:
                    # pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
                    #     "stabilityai/stable-diffusion-2-depth",
                    #     torch_dtype=torch.float16,
                    # ).to("cuda")

                    depth = render["depth"]
                    mask = render["mask"] # JA: depth and mask both have batch dimension
                    phi = render["phi"]

                    # min_h, min_w, max_h, max_w = get_nonzero_region(outputs['mask'][0, 0])
                    # crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
                    # cropped_depth = crop(depth)

                    # cropped_image = pipe(
                    #     prompt=prompt,
                    #     image=torch.cat([depth, depth, depth], dim=1),
                    #     depth_map=depth[0],
                    #     strength=1
                    # ).images[0]


                    original_depth_height, original_depth_width = depth.shape[-2], depth.shape[-1]

                    image = predict(depth, prompt, 50, 1, 9.0, 0, 0, 1)[0]

                    # height, width = outputs['mask'].shape[-2], outputs['mask'].shape[-1]
                    # image = np.ones((height, width, 3))
                    # image[min_h:max_h, min_w:max_w, :] = np.array(cropped_image)
                    # image = np.transpose(np.array(cropped_image), (2, 0, 1))

                    image = torchvision.transforms.functional.pil_to_tensor(image).to(depth.device)

                    image = image[:, :original_depth_height, :original_depth_width]

                    image = torch.where(
                        torch.cat([mask[0], mask[0], mask[0]], dim=0) == 1,
                        image,
                        torch.ones_like(image) * 255
                    )

                    image = torchvision.transforms.functional.to_pil_image(image)

                    if "name" in pair:
                        mesh_name = pair["name"]
                    else:
                        mesh_name = os.path.basename(obj_path)

                    image.save(f"./render_dataset/{mesh_name}_{prompt}_{phi}.png")

        main()
    # except:
    #     if "name" in pair:
    #         mesh_name = pair["name"]
    #     else:
    #         mesh_name = os.path.basename(obj_path)

        # print(f"Pair {mesh_name} could not be generated; skipping")