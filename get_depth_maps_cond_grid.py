import os
import sys
import torch
import torchvision
from pytorch_lightning import seed_everything
from einops import repeat, rearrange
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

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

# sys.path.append("./src/zero123/zero123")
# sys.path.append("./src/zero123/ControlNet")

from src.training.trainer import TEXTure
from src.configs.train_config import TrainConfig
from src.utils import make_path, tensor2numpy, pad_tensor_to_size, split_zero123plus_grid

torch.set_grad_enabled(False)


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

def create_rgba_image(depth, mask):
    depth_rgba = torch.cat((depth, depth, depth, mask), dim=1)  # Creating RGBA image
    return depth_rgba

def crop_image_tensor(input_tensor, min_h, min_w, max_h, max_w):
    return input_tensor[:, :, min_h:max_h, min_w:max_w]

def pad_tensor_to_size(tensor, target_height, target_width):
    if tensor.dim() == 3:  # 만약 텐서가 [C, H, W] 형태라면
        tensor = tensor.unsqueeze(0)  # 배치 차원 추가
    _, _, h, w = tensor.shape
    pad_height = (target_height - h) // 2
    pad_width = (target_width - w) // 2
    padded_tensor = torch.nn.functional.pad(tensor, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)
    return padded_tensor

def resize_image_to_320x320(tensor):
    # 이미지를 320x320 크기로 조정
    resized_image = TF.resize(tensor, [320, 320])
    return resized_image

def save_image(tensor, filename):
    # 텐서의 차원을 확인
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        # 배치 차원을 제거 (첫 번째 이미지 선택)
        tensor = tensor.squeeze(0)
    # 이미지 저장
    img = TF.to_pil_image(tensor)
    img.save(filename)


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

def create_grid(cropped_images):
    cropped_depth_grid = torch.cat((
            torch.cat((cropped_images[0], cropped_images[1]), dim=3),
            torch.cat((cropped_images[2], cropped_images[3]), dim=3),
            torch.cat((cropped_images[4], cropped_images[5]), dim=3),
        ), dim=2)
    return cropped_depth_grid

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

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

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
        # obj_path = os.path.join(os.getcwd(), "texfusion_dataset", pair["path"])
        obj_path = os.path.join(os.getcwd(), pair["path"])

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
                    "depth": outputs["depth"],
                    "mask": outputs["mask"],
                    "phi": phi
                })

            del trainer

            max_height, max_width = 0, 0
            cropped_images = []

            for render in renders:  # Skipping the front view
                depth_rgba = create_rgba_image(1 - render["depth"], render["mask"])
                min_h, min_w, max_h, max_w = get_nonzero_region(render["mask"][0, 0])  # Assuming mask is [1, H, W]
                cropped_image = crop_image_tensor(depth_rgba, min_h, min_w, max_h, max_w)
                cropped_images.append(cropped_image)
                max_height = max(max_height, cropped_image.shape[-2])
                max_width = max(max_width, cropped_image.shape[-1])
            
            uniform_cropped_images = []

            for image in cropped_images[1:]:
                if image.shape[-2] != max_height or image.shape[-1] != max_width:
                    padded_image = pad_tensor_to_size(image, max_height, max_width)
                    uniform_cropped_images.append(padded_image)
                else:
                    uniform_cropped_images.append(image)
            
            for i in range(6):
                uniform_cropped_images[i] = resize_image_to_320x320(uniform_cropped_images[i])
            
            final_grid = create_grid(uniform_cropped_images)

            # condition image 만들기 
            # 프롬프트를 기반으로 front view map에 texture 생성
            for prompt in pair["prompts"]: 
                render = renders[0]
                depth = render["depth"]
                mask = render["mask"] # JA: depth and mask both have batch dimension
                phi = render["phi"]

                original_depth_height, original_depth_width = depth.shape[-2], depth.shape[-1]

                image = predict(depth, prompt, 50, 1, 9.0, 0, 0, 1)[0]
                
                image = torchvision.transforms.functional.pil_to_tensor(image).to(depth.device)

                image = image[:, :original_depth_height, :original_depth_width]

                image = torch.where(
                    torch.cat([mask[0], mask[0], mask[0]], dim=0) == 1,
                    image,
                    torch.ones_like(image) * 255
                )

                image = torchvision.transforms.functional.to_pil_image(image)
                min_h, min_w, max_h, max_w = get_nonzero_region(mask[0, 0])  # Assuming mask is [1, H, W]
                
                image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(depth.device)
                image_tensor = image_tensor.unsqueeze(0)
                # image_tensor = torch.concat((image_tensor, mask), dim=1)

                # 크롭 영역 계산 및 크롭 실행
                cropped_cond_image = crop_image_tensor(image_tensor, min_h, min_w, max_h, max_w)

                # 이미지 패딩
                if cropped_cond_image.shape[-2] != max_height or cropped_cond_image.shape[-1] != max_width:
                    padded_cond_image = pad_tensor_to_size(cropped_cond_image, max_height, max_width)
                else:
                    padded_cond_image = cropped_cond_image

                final_cond_image = padded_cond_image.squeeze(0) 

                # PIL 이미지로 변환
                final_cond_image = torchvision.transforms.functional.to_pil_image(resize_image_to_320x320(final_cond_image).float() / 255)


                if "name" in pair:
                    mesh_name = pair["name"]

                else:
                    mesh_name = os.path.basename(obj_path)

                prompt = prompt.replace(" ", "_")
                createDirectory(f"/home/sogang/jaehoon/texture_test/test_set/{mesh_name}/{prompt}")

                final_cond_image.save(f"/home/sogang/jaehoon/texture_test/test_set/{mesh_name}/{prompt}/cond_image.png")
                print({prompt}, " Complete.")

            save_image(final_grid, f"/home/sogang/jaehoon/texture_test/test_set/{mesh_name}/depth_map_grid.png")
            

        main()
