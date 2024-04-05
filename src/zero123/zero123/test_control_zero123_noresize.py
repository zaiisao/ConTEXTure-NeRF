'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''

import os
import random
import sys
sys.path.append("../ControlNet")

import diffusers  # 0.12.1
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import time
import torch
import torchvision
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
# from ldm.models.diffusion.ddim import DDIMSampler # Removed by JA: We wish to use the custom DDIMSampler in ControlNet
from cldm.ddim_hacked import DDIMSampler

# from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config, add_margin
from ldm.util import create_carvekit_interface, instantiate_from_config, add_margin
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
import PIL
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms

from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
import cv2

_GPU_INDEX = 0

apply_midas = MidasDetector()

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, depth_map_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z, guess_mode):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            if depth_map_im is not None:
                depth_map = np.array(depth_map_im.resize(input_im.shape[2:]).convert(mode='RGB'))
                depth_map = torch.from_numpy(depth_map.copy()).float().cuda()

                depth_min = torch.amin(depth_map, dim=[0, 1, 2], keepdim=True)
                depth_max = torch.amax(depth_map, dim=[0, 1, 2], keepdim=True)

                control = 2. * (depth_map - depth_min) / (depth_max - depth_min) - 1.
                control = torch.stack([control for _ in range(n_samples)], dim=0)
                control = rearrange(control, 'b h w c -> b c h w').clone()

            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])

            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)]
            
            if depth_map_im is not None:
                cond['c_control'] = [control]

            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]

                if depth_map_im is not None:
                    uc['c_control'] = None if guess_mode else [control]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        
def load_and_preprocess(interface, input_im):
    '''
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    '''
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert('RGB')

    image_without_background = interface([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, : , -1].astype(np.bool_)
    image[~foreground] = [255., 255., 255.]

    # Commented by JA: We do not want to reposition the object to be in the center in the
    # ControlNet-finetuned zero123 (because if the object is offset from the center, we want
    # to keep this information (because the rotation "pivot" takes place at the center of
    # the image, NOT the center of the (possibly) off-centered object).

    # x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    # image = image[y:y+h, x:x+w, :]
    image = PIL.Image.fromarray(np.array(image))
    
    # resize image such that long edge is 512
    image.thumbnail([450, 450], Image.Resampling.LANCZOS)
    image = add_margin(image, (255, 255, 255), size=512)
    image = np.array(image)
    
    return image, est_seg

def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im, est_seg = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im, est_seg

def get_depth(filename, midas, device):
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    transform = midas_transforms.dpt_transform
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    return output

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

def main_run(models, device,
             x=0.0, y=0.0, z=0.0,
             raw_im=None, target_im=None, depth_map=None, preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=50, guess_mode=False, ddim_eta=1.0,
             precision='fp32', h=256, w=256):
    '''
    :param raw_im (PIL Image).
    '''

    # raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    input_im, _ = preprocess_image(models, raw_im, preprocess)
    _, est_seg = preprocess_image(models, target_im, preprocess)

    if depth_map is not None:
        depth_map = np.array(depth_map)

        depth_min = depth_map.min()
        depth_max = depth_map.max()

        depth_map = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        depth_map *= est_seg[:, : , -1].astype(np.bool_)

        depth_map = transforms.ToTensor()(depth_map).unsqueeze(0).to(device)
        depth_map = pad_image_tensor(depth_map)

        depth_map = torchvision.transforms.functional.to_pil_image(depth_map[0])

        # # resize image such that long edge is 512
        # depth_map.thumbnail([200, 200], Image.Resampling.LANCZOS)
        # # depth_map = add_margin(depth_map, (255, 255, 255), size=256)
        # depth_map = add_margin(depth_map, 0, size=256)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    # input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(models['turncam'])
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    used_x = x  # NOTE: Set this way for consistency.

    # input_im = pad_image_tensor(input_im)
    # h = input_im.shape[-2]
    # w = input_im.shape[-1]
    
    x_samples_ddim = sample_model(input_im, depth_map, models['turncam'], sampler, precision, h, w,
                                    ddim_steps, n_samples, scale, ddim_eta, used_x, y, z, guess_mode)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    return output_ims

def process_images(image1, image2):
    # Placeholder for your actual processing function
    # This function should return a PIL image object
    # For demonstration, we'll just blend the two images
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    result = Image.blend(img1, img2, alpha=0.5)
    return result

def get_angle_difference(image1_name, image2_name):
    # Extracting the numbers from the filenames and calculating the angle difference
    angle1 = int(image1_name.split('.')[0]) * 90
    angle2 = int(image2_name.split('.')[0]) * 90
    angle_difference = angle2 - angle1
    if angle_difference < 0:
        angle_difference += 360  # Ensuring the difference is within 0 to 360 degrees
    return angle_difference if angle_difference <= 180 else angle_difference - 360

def run_demo(use_depth=True):
    device_idx = _GPU_INDEX
    ckpt = "/home/sogang/jaehoon/TEXTureWithZero123/epoch=19-step=6359.ckpt"

    if use_depth:
        config = "/home/sogang/jaehoon/TEXTureWithZero123/src/zero123/ControlNet/models/cldm_zero123.yaml"
    else:
        config = "/home/sogang/jaehoon/TEXTureWithZero123/src/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml"

    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1:
        print('old device_idx:', device_idx)
        device_idx = int(sys.argv[1])
        print('new device_idx:', device_idx)

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.

    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    root_dir = "/home/sogang/jaehoon/google_test"  # Starting directory

    # JA: Evaluate the performance of the Control Zero123 using the dataset used to evaluate Zero123
    for subdir, dirs, files in os.walk(root_dir):
        if os.path.basename(subdir) == "thumbnails":
            images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
            selected_images = random.sample(images, 2)  # Select two random images
            angle_difference = get_angle_difference(*selected_images)

            if use_depth:
                depth_map = Image.fromarray(get_depth(os.path.join(subdir, selected_images[1]), midas, device))
            else:
                depth_map = None

            results = main_run(
                models, device,
                x=0.0, y=angle_difference, z=0.0,
                raw_im=Image.open(os.path.join(subdir, selected_images[0])),
                target_im=Image.open(os.path.join(subdir, selected_images[1])),
                depth_map=depth_map,
                preprocess=True,
                scale=3.0, n_samples=1, ddim_steps=50, guess_mode=False, ddim_eta=1.0,
                precision='fp32', h=512, w=512
            )

            processed_image = results[0]
            
            # Construct the filename to indicate the chosen images and angle difference
            suffix = "_c0123" if use_depth else "_0123"
            new_filename = f"{selected_images[0].split('.')[0]}_{selected_images[1].split('.')[0]}_angle{angle_difference}{suffix}_TEST.jpg"
            save_path = os.path.join(subdir, new_filename)
            processed_image.save(save_path)
            print(f"Processed and saved: {save_path}")
            break

    # results = main_run(
    #     models, device,
    #     x=0.0, y=45.0, z=0.0,
    #     raw_im=Image.open("/home/sogang/jaehoon/0001_full_output.jpg"), depth_map=Image.open("/home/sogang/jaehoon/converted_image.png"), preprocess=True,
    #     scale=3.0, n_samples=1, ddim_steps=50, guess_mode=False, ddim_eta=1.0,
    #     precision='fp32', h=256, w=256
    # )

    results[0].save("/home/sogang/jaehoon/test.png")

if __name__ == "__main__":
    # JA: When use_depth is True, we use Control Zero123
    run_demo(use_depth=False)