import time
from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import types
import random
# from torchviz import make_dot

from torch_scatter import scatter_max

import torchvision
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel, DDPMScheduler, EulerAncestralDiscreteScheduler
from diffusers.training_utils import cast_training_params

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import Zero123PlusDataset, ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy, pad_tensor_to_size, split_zero123plus_grid, split_3x2_grid_to_tensor_with_6_elements
from src.run_nerf_helpers import *
from src.optimizer import Adan

from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from src.scheduling_euler_ancestral_discrete import StatelessEulerAncestralDiscreteScheduler

# JA: scale_latents, unscale_latents, scale_image, and unscale_image are from the Zero123++ pipeline code:
# https://huggingface.co/sudo-ai/zero123plus-pipeline/blob/main/pipeline.py
def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def scale_image(image):
    image = image * 0.5 / 0.8
    return image

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image
# Add this import at the top of your file
import matplotlib.pyplot as plt

class DreamTimeScheduler:
    def __init__(self, alphas_cumprod, total_iterations, m=750, s=125):
        """
        Initializes the Time Prioritized SDS scheduler.

        Args:
            alphas_cumprod (torch.Tensor): The cumulative product of alphas from the diffusion model.
            total_iterations (int): The total number of training iterations (N).
            m (int): The mean (center) of the Gaussian perception prior.
            s (int): The standard deviation of the Gaussian perception prior.
        """
        self.device = alphas_cumprod.device
        self.total_iterations = total_iterations
        self.T = len(alphas_cumprod)

        # Pre-compute the weights W(t) for all timesteps t in [0, T-1]
        
        # 1. Diffusion Prior W_d(t) based on SNR (Eq. 6-7, Source 313, 317)
        # w_d = torch.sqrt(1 - alphas_cumprod)
        w_d = torch.sqrt((1 - alphas_cumprod) / (alphas_cumprod + 1e-9))

        # 2. Perception Prior W_p(t), a Gaussian bell curve (Source 417)
        timesteps = torch.arange(self.T, device=self.device)
        w_p = torch.exp(-((timesteps - m) ** 2) / (2 * (s ** 2)))

        # 3. Combined weights W(t) (Source 403)
        weights = w_d * w_p
        
        # 4. Normalize the weights to sum to 1
        weights /= weights.sum()

        # 5. Pre-compute the cumulative survival function (sum from t' to T)
        # This is used for the deterministic mapping from iteration to timestep.
        self.cumulative_survival = torch.flip(torch.cumsum(torch.flip(weights, dims=[0]), dim=0), dims=[0])

    def get_t(self, i):
        """
        Gets the deterministic timestep 't' for the current iteration 'i'.

        Args:
            i (int): The current training iteration.

        Returns:
            int: The calculated timestep for this iteration.
        """
        # Calculate the target cumulative weight based on training progress (i/N)
        target_cumulative_weight = i / self.total_iterations
        
        # Find the timestep t' where the cumulative survival is closest to the target
        # This implements Eq. 5 from the paper (Source 398)
        diffs = torch.abs(self.cumulative_survival - target_cumulative_weight)
        t = torch.argmin(diffs).item()
        
        return t

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        if not isinstance(self.running_average, torch.Tensor):
            self.running_average = update_value
        else:
            new_average = self.momentum * self.running_average.detach()
            self.running_average = update_value + new_average

def project(
    v0: torch.Tensor, # [B, C, H, W]
    v1: torch.Tensor, # [B, C, H, W]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def adaptive_projected_guidance(
    pred_cond: torch.Tensor, # [B, C, H, W]
    pred_uncond: torch.Tensor, # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided

class ConTEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        # JA: From run_nerf_helpers.py
        # The positional embedder for 2D UV coordinates
        self.prolificdreamer = True

        mlp_args = {
            'D': 2,
            'W': 512,
            # 'input_ch': 2,
            'output_ch': 3,
            'skips': []
        }
        if self.prolificdreamer:
            self.uv_embedder = None
            
            self.texture_mlp = NeRF2DNetwork(n_particles=1, **mlp_args)
        else:
            # self.uv_embedder, input_ch_uv = get_embedder(multires=8) #MJ: input_ch_uv = the dim of the Fourier embedding vector of (u,v), say 60
            self.uv_embedder, output_dim = get_grid_encoder()

            # The 2D NeRF model, with input dimensions matching the embedder's output
            
            self.texture_mlp = NeRF2D(input_ch=output_dim, **mlp_args).to(self.device)
            if torch.cuda.device_count() > 1:
                self.texture_mlp = nn.DataParallel(self.texture_mlp)

        # You should also pass these new components to your mesh model
        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom'] # self.view_dirs[dir] when dir = [4] = [right]
        
        self.mesh_model = self.init_mesh_model(texture_mlp=self.texture_mlp, uv_embedder=self.uv_embedder)
        #MJ: self.mesh_model.texture_mlp is g(theta) for the texture generation
        
        self.diffusion = self.init_diffusion()

        if self.cfg.guide.use_zero123plus:
            self.zero123plus = self.init_zero123plus()

        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0, 1) / 255.0

        self.zero123_front_input = None
       
    def create_face_view_map(self, face_idx):
        num_views, _, H, W = face_idx.shape  # Assume face_idx shape is (B, 1, H, W)

        # Flatten the face_idx tensor to make it easier to work with
        face_idx_flattened_2d = face_idx.view(num_views, -1)  # Shape becomes (num_views, H*W)

        # Get the indices of all elements
        # JA: From ChatGPT:
        # torch.meshgrid is used to create a grid of indices that corresponds to each dimension of the input tensor,
        # specifically in this context for the view indices and pixel indices. It allows us to pair each view index
        # with every pixel index, thereby creating a full coordinate system that can be mapped directly to the values
        # in the tensor face_idx.
        view_by_pixel_indices, pixel_by_view_indices = torch.meshgrid(
            torch.arange(num_views, device=face_idx.device),
            torch.arange(H * W, device=face_idx.device),
            indexing='ij'
        )

        # Flatten indices tensors
        view_by_pixel_indices_flattened = view_by_pixel_indices.flatten()
        pixel_by_view_indices_flattened = pixel_by_view_indices.flatten()

        faces_idx_view_pixel_flattened = face_idx_flattened_2d.flatten()

        # Convert pixel indices back to 2D indices (i, j)
        pixel_i_indices = pixel_by_view_indices_flattened // W
        pixel_j_indices = pixel_by_view_indices_flattened % W

        # JA: The original face view map is made of nested dictionaries, which is very inefficient. Face map information
        # is implemented as a single tensor which is efficient. Only tensors can be processed in GPU; dictionaries cannot
        # be processed in GPU.
        # The combined tensor represents, for each pixel (i, j), its view_idx 
        combined_tensor_for_face_view_map = torch.stack([
            faces_idx_view_pixel_flattened,
            view_by_pixel_indices_flattened,
            pixel_i_indices,
            pixel_j_indices
        ], dim=1)

        # Filter valid faces
        faces_idx_valid_mask = faces_idx_view_pixel_flattened >= 0

        # JA:
        # [[face_id_1, view_1, i_1, j_1]
        #  [face_id_1, view_1, i_2, j_2]
        #  [face_id_1, view_1, i_3, j_3]
        #  [face_id_1, view_2, i_4, j_4]
        #  [face_id_1, view_2, i_5, j_5]
        #  ...
        #  [face_id_2, view_1, i_k, j_l]
        #  [face_id_2, view_1, i_{k + 1}, j_{l + 1}]
        #  [face_id_2, view_2, i_{k + 2}, j_{l + 2}]]
        #  ...
        # The above example shows face_id_1 is projected, under view_1, to three pixels (i_1, j_1), (i_2, j_2), (i_3, j_3)
        # Shape is Nx4 where N is the number of pixels (no greater than H*W*num_views = 1200*1200*7) that projects the
        # valid face ID.
        return combined_tensor_for_face_view_map[faces_idx_valid_mask]
    # You can place this helper function inside your ConTEXTure class
    def visualize_gradients(self, model_name: str, parameter: torch.Tensor, iteration: int):
        if iteration % 100 == 0 and parameter.grad is not None: # Log every 100 iterations
            grad_data = parameter.grad.cpu().numpy()
            
            # Squeeze dimensions for visualization if needed (e.g., for bias terms)
            if grad_data.ndim > 2:
                grad_data = grad_data.mean(axis=tuple(range(grad_data.ndim - 2)))
            
            plt.figure(figsize=(8, 6))
            plt.title(f"Gradient Heatmap: {model_name} (Iter {iteration})")
            img = plt.imshow(grad_data, cmap='viridis', aspect='auto')
            plt.colorbar(img)
            
            # Make sure the directory exists
            viz_path = self.train_renders_path / "grad_viz"
            viz_path.mkdir(exist_ok=True)
            
            plt.savefig(viz_path / f"{model_name}_iter_{iteration:04d}.png")
            plt.close()

    def log_lora_output(self, vae, scheduler, noisy_latents, v_prediction, t, iteration):
        """Decodes the LoRA UNet's v-prediction into an image and saves it."""
        if iteration % 10 != 0: # Only run this periodically
            return
            
        with torch.no_grad():
            # Use the scheduler to step backwards and get the predicted clean latents
            pred_original_sample = scheduler.step(v_prediction, t, noisy_latents).pred_original_sample

            # Unscale the latents using your custom function
            latents_unscaled = unscale_latents(pred_original_sample)

            # Decode the latents with the VAE
            latents_for_vae = latents_unscaled / vae.config.scaling_factor
            decoded_image = vae.decode(latents_for_vae.half(), return_dict=False)[0]

            # Unscale the image from the custom function and normalize from [-1, 1] to [0, 1]
            image_unscaled = unscale_image(decoded_image)
            image_normalized = (image_unscaled.clamp(-1, 1) + 1) / 2

            # Save the image using your existing logger
            self.log_train_image(image_normalized, f'lora_output_iter_{iteration:06d}')

    def log_teacher_guidance(self, vae, scheduler, noisy_latents, v_prediction, t, iteration):
        """Decodes the frozen UNet's v-prediction to visualize the guidance signal."""
        if iteration % 10 != 0: # Only run this periodically
            return
            
        with torch.no_grad():
            # Use the scheduler to get the predicted clean latents from the teacher's guidance
            pred_original_sample = scheduler.step(v_prediction, t, noisy_latents).pred_original_sample

            # Unscale the latents
            latents_unscaled = unscale_latents(pred_original_sample)
            latents_for_vae = latents_unscaled / vae.config.scaling_factor
            
            # Decode into an image
            decoded_image = vae.decode(latents_for_vae.half(), return_dict=False)[0]

            # Normalize and save
            image_unscaled = unscale_image(decoded_image)
            image_normalized = (image_unscaled.clamp(-1, 1) + 1) / 2
            self.log_train_image(image_normalized, f'teacher_guidance_iter_{iteration:06d}')


    def compare_face_normals_between_views(self,face_view_map, face_normals, face_idx):
        num_views, _, H, W = face_idx.shape
        weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool, device=face_idx.device)

        face_ids = face_view_map[:, 0] # JA: face_view_map.shape = (H*W*num_views, 4) = (1200*1200*7, 4) = (10080000, 4)
        views = face_view_map[:, 1]
        i_coords = face_view_map[:, 2]
        j_coords = face_view_map[:, 3]
        z_normals = face_normals[views, 2, face_ids] # JA: The shape of face_normals is (num_views, 3, num_faces)
                                                     # For example, face_normals can be (7, 3, 14232)
                                                     # z_normals is (N,)

        # Scatter z-normals into the tensor, ensuring each index only keeps the max value
        # JA: z_normals is the source/input tensor, and face_ids is the index tensor to scatter_max function.
        max_z_normals_over_views, _ = scatter_max(z_normals, face_ids, dim=0) # JA: N is a subset of length H*W*num_views
        # The shape of max_z_normals_over_N is the (num_faces,). The shape of the scatter_max output is equal to the
        # shape of the number of distinct indices in the index tensor face_ids.

        # Map the gathered max normals back to the respective face ID indices
        # JA: max_z_normals_over_views represents the max z normals over views for every face ID.
        # The shape of face_ids is (N,). Therefore the shape of max_z_normals_over_views_per_face is also (N,).
        max_z_normals_over_views_per_face = max_z_normals_over_views[face_ids]

        # Calculate the unworthy mask where current z-normals are less than the max per face ID
        unworthy_pixels_mask = z_normals < max_z_normals_over_views_per_face

        # JA: Update the weight masks. The shapes of face_view_map, whence views, i_coords, and j_coords were extracted
        # from, all have the shape of (N,), which represents the number of valid pixel entries. Therefore,
        # weight_masks[views, 0, i_coords, j_coords] will also have the shape of (N,) which allows the values in
        # weight_masks to be set in an elementwise manner.
        #
        # weight_masks[views[0], 0, i_coords[0], j_coords[0]] = ~(unworthy_pixels_mask[0])
        # The above variable represents whether the pixel (i_coords[0], j_coords[0]) under views[0] is worthy to
        # contribute to the texture atlas.
        weight_masks[views, 0, i_coords, j_coords] = ~(unworthy_pixels_mask)

        return weight_masks

    def init_mesh_model(self, texture_mlp, uv_embedder) -> nn.Module:
        # fovyangle = np.pi / 6 if self.cfg.guide.use_zero123plus else np.pi / 3
        fovyangle = np.pi / 3
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide,
                                  
                                  texture_mlp=texture_mlp,
                                  uv_embedder=uv_embedder,
                                  
                                  device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False,
                                  fovyangle=fovyangle)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        # JA: The StableDiffusion class composes a pipeline by using individual components such as VAE encoder,
        # CLIP encoder, and UNet
        second_model_type = self.cfg.guide.second_model_type
        if self.cfg.guide.use_zero123plus:
            second_model_type = "zero123plus"

        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True,
                                          second_model_type=self.cfg.guide.second_model_type,
                                          guess_mode=self.cfg.guide.guess_mode)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model
    
    def init_zero123plus(self) -> DiffusionPipeline:
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",#custom_pipeline="src/zero123plus.py",
            torch_dtype=torch.float16
        )

        #MJ: The SDS training loop does NOT use the scheduler; so it is not relevant to define the scheduler for the SDS
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

        pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
        ), conditioning_scale=2)

        # pipeline._callback_tensor_inputs += ["noise_pred"]

        pipeline.prepare()
        pipeline.to(self.device)

        if self.prolificdreamer:
            import copy
            from peft import LoraConfig

            _unet = copy.deepcopy(pipeline.unet)  #MJ: Check if _unet refers to the zero123++ which uses v_parameterization
            _unet.requires_grad_(False)

            device = self.device

            unet_lora_config = LoraConfig(
                r=32,  # Rank
                lora_alpha=32,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            _unet.add_adapter(unet_lora_config)
            cast_training_params(_unet, dtype=torch.float32)

            # text_input = self.guidance.tokenizer("", padding='max_length', max_length=self.guidance.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            # with torch.no_grad():
            #     text_embeddings = self.guidance.text_encoder(text_input.input_ids.to(self.guidance.device))[0]
            
            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32
                    # self.text_embeddings = text_embeddings

                def forward(self, *args, **kwargs):
                    # textemb = einops.repeat(self.text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
                    return self.unet(*args, **kwargs)

            self.lora_unet = LoraUnet().to(device)

            logger.info("--- Verifying Trainable Parameters ---")

            print("\n--- Trainable Parameters in texture_mlp (for mlp_optimizer) ---")
            for name, param in self.mesh_model.texture_mlp.named_parameters():
                if param.requires_grad:
                    print(f"{name} | {param.shape}")

            print("\n--- Trainable Parameters in lora_unet (for lora_optimizer) ---")
            for name, param in self.lora_unet.named_parameters():
                if param.requires_grad:
                    print(f"{name} | {param.shape}")
            print("\n" + "-"*40 + "\n")

        pipeline.inpaint_unet = self.diffusion.inpaint_unet

        return pipeline

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if self.cfg.guide.use_zero123plus:
            assert not self.cfg.guide.append_direction, "append_direction should be False when use_zero123plus is True"

            text_z = []
            text_string = []

            text_string.append(ref_text)
            text_string.append(ref_text + ", front view")
            
            for text in text_string:
                negative_prompt = None
                text_z.append(self.diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        elif not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            for d in self.view_dirs:
                text = ref_text.format(d)
                text_string.append(text)
                logger.info(text)
                negative_prompt = None
                logger.info(negative_prompt)
                text_z.append(self.diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        return text_z, text_string # JA: text_z contains the embedded vectors of the six view prompts

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        if self.cfg.guide.use_zero123plus:
            init_train_dataloader = Zero123PlusDataset(self.cfg.render, device=self.device).dataloader()
        else:
            init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        self.paint_zero123plus()

    def define_view_weights(self):
        # Set the camera poses:
        self.thetas = []
        self.phis = []
        self.radii = []
       
        for i, data in enumerate(self.dataloaders['train']):
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)

            self.thetas.append(theta)
            self.phis.append(phi)
            self.radii.append(radius)

        augmented_vertices = self.mesh_model.mesh.vertices

        batch_size = len(self.dataloaders['train'])

        # JA: We need to repeat several tensors to support the batch size.
        # For example, with a tensor of the shape, [1, 3, 1200, 1200], doing
        # repeat(batch_size, 1, 1, 1) results in [1 * batch_size, 3 * 1, 1200 * 1, 1200 * 1]
        _, _, _, face_normals, face_idx = self.mesh_model.render_face_normals_face_idx(
            augmented_vertices[None].repeat(batch_size, 1, 1),
            self.mesh_model.mesh.faces, # JA: the faces tensor can be shared across the batch and does not require its own batch dimension.
            self.mesh_model.face_attributes.repeat(batch_size, 1, 1, 1),
            elev=torch.tensor(self.thetas).to(self.device), # MJ: elev, azim, and radius should be tensors
            azim=torch.tensor(self.phis).to(self.device),
            radius=torch.tensor(self.radii).to(self.device),
            
            look_at_height=self.mesh_model.dy,
            background_type='none'
        )
        
        logger.info(f'Generating face view map')

        #MJ: get the binary masks for each view which indicates how much the image rendered from each view
        # should contribute to the texture atlas over the mesh which is the cause of the image
        face_view_map = self.create_face_view_map(face_idx)

        # logger.info(f'Creating weight masks for each view')
        weight_masks = self.compare_face_normals_between_views(face_view_map, face_normals, face_idx)

        self.view_weights = weight_masks

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def get_cropped_rgb_renders(self, rgb_renders, object_masks):
        B, _, _, _ = object_masks.shape
        cropped_rgb_renders_list = []
        for i in range(B):
            mask_i = object_masks[i, 0]
            min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(mask_i) #MJ: outputs["mask"][0, 0]: shape (1,1,H,W)
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_rgb_render = crop(rgb_renders[i][None])
            cropped_rgb_renders_list.append(cropped_rgb_render)

        return cropped_rgb_renders_list

    def compute_view_consistency(self, rendered_views, faces, all_face_idx, all_face_vertices_image):
        num_views, C, h, w = rendered_views.shape
        num_vertices = faces.max().item() + 1

        # Create the vertex-to-pixel lookup map for coordinate mapping
        # JA: Create a map to project all faces to their pixel coordinates
        vertex_to_pixel_map = torch.full((num_views, num_vertices, 2), -1, dtype=torch.long, device=self.device)
        flat_faces = faces.flatten()

        for i in range(num_views):
            # JA: face_vertices_image is provided by prepare_vertices and refers to the vertices' pixel positions.
            # Because it has a shape of (B, F, 3, 2), the 3 refers to the positions of the three vertices that make
            # up a single face, and the 2 refers to the pixel coordinate in normalized space [-1, 1]

            # JA: Change domain of coordinates [-1, 1] -> [0, 1] -> [0, W (or H)]
            coords_normalized = (all_face_vertices_image[i].reshape(-1, 2) + 1) / 2

            coords_xy = (coords_normalized * torch.tensor([w, h], device=self.device, dtype=torch.float32)).long()
            coords_yx = coords_xy[:, [1, 0]]
            vertex_to_pixel_map[i, flat_faces] = coords_yx

        # JA: vertex_visiblity is a visibility map for each vertex for all views. This information can be extracted
        # from retrieving face_idx, which is provided by the rasterize function
        vertex_visibility = torch.zeros((num_vertices, num_views), dtype=torch.bool, device=self.device)
        for j in range(num_views):
            visible_faces_in_view = torch.unique(all_face_idx[j])
            visible_faces_in_view = visible_faces_in_view[visible_faces_in_view != -1]
            if visible_faces_in_view.numel() > 0:
                visible_vertices = faces[visible_faces_in_view].flatten()
                vertex_visibility[visible_vertices, j] = True

        visibility_mask_tensor = torch.zeros((num_views, num_views, h, w), dtype=torch.bool, device=self.device)
        coord_map_tensor = torch.full((num_views, num_views, h, w, 2), -1, dtype=torch.long, device=self.device)
        color_diff_tensor = torch.full((num_views, num_views, h, w), -1.0, dtype=torch.float32, device=self.device)

        # JA: Here, j is the source view index where the visibility information is retrived from, and i is the
        # target view which is the view we are creating a mask for
        for j in range(num_views):
            is_visible_in_view_j = vertex_visibility[:, j]
            source_image_j = rendered_views[j]

            for i in range(num_views):
                # JA: If the target view i does not have any valid face indices (that is, there are no valid pixels),
                # then it should skip this view. This usually should not be the case.
                target_view_face_idx = all_face_idx[i]
                valid_pixels = target_view_face_idx != -1
                if not torch.any(valid_pixels):
                    continue

                # JA: The faces can be retrieved based on the valid pixels of view i, and these faces can be used to
                # check if they are also visible in view j
                pixel_vertices = faces[target_view_face_idx[valid_pixels]]
                pixel_vertex_status = is_visible_in_view_j[pixel_vertices]
                has_shared_vertex = torch.any(pixel_vertex_status, dim=1)

                # JA: Create a mask displaying parts of the mesh from view i that are also visible in view j
                pairwise_mask = torch.zeros((h, w), dtype=torch.bool, device=self.device)
                pairwise_mask[valid_pixels] = has_shared_vertex
                visibility_mask_tensor[j, i] = pairwise_mask

                if not torch.any(has_shared_vertex):
                    continue # JA: No shared pixels, so no coordinates to map

                # Find representative vertices ONLY for the shared pixels
                first_visible_v_idx = torch.argmax(pixel_vertex_status[has_shared_vertex].int(), dim=1)
                num_shared_pixels = has_shared_vertex.sum()
                representative_v_ids = pixel_vertices[has_shared_vertex][torch.arange(num_shared_pixels), first_visible_v_idx]

                # Look up their coordinates in the source view `j`
                corresponding_coords = vertex_to_pixel_map[j, representative_v_ids]
                
                # Get the (y,x) locations of the shared pixels to place the new coords
                shared_pixel_locations = valid_pixels.nonzero(as_tuple=False)[has_shared_vertex]
                y_indices, x_indices = shared_pixel_locations[:, 0], shared_pixel_locations[:, 1]
                
                coord_map_tensor[j, i, y_indices, x_indices] = corresponding_coords

                # Get the target image for this specific pair
                target_image_i = rendered_views[i]
                
                # Get coordinates for source and target pixels
                source_y, source_x = corresponding_coords[:, 0], corresponding_coords[:, 1]
                target_y, target_x = y_indices, x_indices
                
                gathered_colors = source_image_j[:, source_y, source_x]
                target_colors = target_image_i[:, target_y, target_x]

                diff = 1 - torch.abs(target_colors.float() - gathered_colors.float()).sum(dim=0) / C
                
                color_diff_tensor[j, i, target_y, target_x] = diff

        pair_mask = ~torch.eye(num_views, num_views, dtype=torch.bool, device=self.device)
        relevant_similarities = color_diff_tensor[pair_mask]

        valid_similarity_values = relevant_similarities[relevant_similarities >= 0]

        if valid_similarity_values.numel() > 0:
            mean_similarity = torch.mean(valid_similarity_values)
        else:
            # If no pixels overlap, there is no inconsistency, so the similarity is 0.
            mean_similarity = torch.tensor(0.0, device=self.device)

        return mean_similarity

    def to_rgb_image(self, maybe_rgba: Image.Image):
        if maybe_rgba.mode == 'RGB':
            return maybe_rgba
        elif maybe_rgba.mode == 'RGBA':
            rgba = maybe_rgba
            img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
            img = Image.fromarray(img, 'RGB')
            img.paste(rgba, mask=rgba.getchannel('A'))
            return img
        else:
            raise ValueError("Unsupported image type.", maybe_rgba.mode)

    def total_variation_loss(self, texture_map):
        """
        Computes the Total Variation loss for a texture map based on the formula:
        TV(z) = sum(|z_{u+1,v} - z_{u,v}| + |z_{u,v+1} - z_{u,v}|)

        Args:
            texture_map (torch.Tensor): The texture map tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The scalar Total Variation loss.
        """
        # Horizontal variation (differences between adjacent columns)
        dw = torch.abs(texture_map[:, :, :, 1:] - texture_map[:, :, :, :-1])
        
        # Vertical variation (differences between adjacent rows)
        dh = torch.abs(texture_map[:, :, 1:, :] - texture_map[:, :, :-1, :])
        
        # Sum of absolute differences
        return (torch.sum(dw) + torch.sum(dh)) / texture_map.numel()

    def paint_zero123plus(self):
        """
        Generates the texture map using Score Distillation Sampling (SDS)
        with the Zero123++ model as the teacher.
        """
        logger.info('Starting SDS Texture Generation ^_^')

        self.define_view_weights()
        self.mesh_model.train()
        background_gray = torch.tensor([0.5, 0.5, 0.5], device=self.device)

        # Generate the front view to be used as the condition image
        frontview_data = next(iter(self.dataloaders['train']))
        with torch.no_grad():
            rgb_output_front, object_mask_front = self.paint_viewpoint(frontview_data, should_project_back=False)

        # Render all 7 views to get depth maps and object masks
        outputs_all_views = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii, background=background_gray)

        object_masks = outputs_all_views['mask']
        depth_maps = 1.0 - outputs_all_views['depth']
        render_cache = outputs_all_views['render_cache']
        B = object_masks.shape[0]

        # Prepare the condition image (front view)
        min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(object_mask_front[0, 0])
        front_image_rgba = torch.cat((rgb_output_front, object_mask_front), dim=1)
        cropped_front_image_rgba = front_image_rgba[:, :, min_h:max_h, min_w:max_w]
        cond_image_pil_rgba = torchvision.transforms.functional.to_pil_image(cropped_front_image_rgba[0]).resize((320, 320))

        # JA: to_rgb_image is a helper from Zero123++ which turns the background of cond_image_pil and depth_image_pil
        # gray
        cond_image_pil_rgb = self.to_rgb_image(cond_image_pil_rgba)

        # JA: Prepare the 3x2 depth grid for the 6 novel views
        # object_masks is used as the alpha channel for depth_rgba, resulting in a tensor corresponding to an RGBA image
        # in which the backround regions have an alpha channel of 1. This is then made into a 3x2 grid. It is then
        # converted to a PIL image which is processed by to_rgb_image, which turns all transparent parts gray.
        depth_rgba = torch.cat((depth_maps, depth_maps, depth_maps, object_masks), dim=1)
        cropped_depths_small_list = []
        for i in range(1, B):
            min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(object_masks[i, 0])
            cropped_depth = F.interpolate(depth_rgba[i:i+1, :, min_h:max_h, min_w:max_w], (320, 320), mode='bilinear', align_corners=False)
            cropped_depths_small_list.append(cropped_depth)
        
        cropped_depth_grid = torch.cat((
            torch.cat((cropped_depths_small_list[0], cropped_depths_small_list[3]), dim=3),
            torch.cat((cropped_depths_small_list[1], cropped_depths_small_list[4]), dim=3),
            torch.cat((cropped_depths_small_list[2], cropped_depths_small_list[5]), dim=3),
        ), dim=2)

        self.log_train_image(cropped_depth_grid, 'cropped_depth_grid', file_type="png")

        depth_image_pil_rgba = torchvision.transforms.functional.to_pil_image(cropped_depth_grid[0])
        depth_image_pil_rgb = self.to_rgb_image(depth_image_pil_rgba)

        # Setup SDS loop
        logger.info("Setting up SDS optimization loop...")
        
        
        if not self.prolificdreamer:
            param_groups = [
                {"params": self.uv_embedder.parameters(), "lr": 1e-2},
                {"params": self.texture_mlp.parameters(), "lr": 5e-4}
            ] #MJ: param_groups = theta for g(theta)

            mlp_optimizer = Adan(
                param_groups,
                lr=5e-4,
                eps=1e-18,
                weight_decay=2e-5,
                max_grad_norm=5.0,
                foreach=False
            )
            
        else: #MJ: Define mlp_optimizer for the 3  particles
            # param_groups = [
            #     # {"params": self.mesh_model.texture_mlp.parameters()},
            # ] 

            # for particle in self.mesh_model.texture_mlp.particles:
            #     param_groups.append({"params": particle.encoder.parameters(), "lr": 1e-2})
            #     param_groups.append({"params": particle.mlp.parameters(), "lr": 1e-2})


            # mlp_optimizer = Adan(
            #     param_groups,
            #     eps=1e-18,
            #     weight_decay=2e-5,
            #     max_grad_norm=5.0,
            #     foreach=False
            # ) 

            param_groups = []
            for particle in self.mesh_model.texture_mlp.particles:
                # Group 1: Encoder parameters WITHOUT weight decay
                param_groups.append({"params": particle.encoder.parameters(), "lr": 1e-2})
                # Group 2: MLP parameters WITH weight decay
                param_groups.append({"params": particle.mlp.parameters(), "lr": 1e-4, "weight_decay": 1e-6})

            mlp_optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.99),
                eps=1e-15 
            )

           #MJ: PyTorch treats each dict as a parameter group. All groups share the same optimizer object opt.
           #mlp_optimizer.step() will optimizer.step() walks through all parameter groups and applies updates independently to each group’s parameters,
           # but only if they have non-zero gradients.
           #   
        if self.prolificdreamer:
            params = [
                {'params': self.lora_unet.parameters()}
            ] 

            lora_optimizer = torch.optim.AdamW(
                params, 
                lr=1e-4, 
                betas=(0.9, 0.99),
                eps=1e-15
            )

        scheduler = self.zero123plus.scheduler
        unet = self.zero123plus.unet
        vae = self.zero123plus.vae
        
        with torch.no_grad():
            # JA: cond_image_pil is the front view image with a gray background
            #MJ: Get the image feature of the condition image for "reference [cross] attention"
            cond_image_vae = self.zero123plus.feature_extractor_vae(
                images=cond_image_pil_rgb,
                return_tensors="pt"
            ).pixel_values.to(device=self.device, dtype=vae.dtype)
            
            #MJ: Get the image feature of  the condition image for "usual [semantic] cross attention"
            
            cond_image_clip = self.zero123plus.feature_extractor_clip(
                images=cond_image_pil_rgb,
                return_tensors="pt"
            ).pixel_values.to(device=self.device, dtype=unet.dtype)

            # JA: cond_lat is from the front view image: MJ: get the latent image of the reference image
            cond_lat = vae.encode(cond_image_vae).latent_dist.sample()
            
            # JA: Get unconditional latent the reference image [for reference attention]
            negative_lat = vae.encode(torch.zeros_like(cond_image_vae)).latent_dist.sample()
            
            #MJ: get the latent image of the semantic cross-attention  image
            encoded = self.zero123plus.vision_encoder(cond_image_clip, output_hidden_states=False)
            global_embeds = encoded.image_embeds.unsqueeze(-2)
            
            # JA: Get text embeddings (for empty prompt) and combine with vision embeddings
            text_embeds = self.zero123plus.encode_prompt("", self.device, 1, False)[0] #MJ: there is a real text_embeds, but in the current release of zero123++, it is omitted
            ramp = global_embeds.new_tensor(self.zero123plus.config.ramping_coefficients).unsqueeze(-1)
            cond_encoder_hidden_states = text_embeds + global_embeds * ramp
            
            # JA: Get unconditional text embeddings: 
            uncond_embeds = self.zero123plus.encode_prompt("", self.device, 1, True)[1]
            
            # JA: Concatenate for classifier-free guidance:
            #MJ: For the semantic cross attention
            encoder_hidden_states = torch.cat([uncond_embeds, cond_encoder_hidden_states])
            #MJ: For the reference image cross attention
            clean_cond_lat = torch.cat([negative_lat, cond_lat])

            # JA: Prepare depth map tensor for ControlNet
            depth_tensor = self.zero123plus.depth_transforms_multi(depth_image_pil_rgb).to(device=self.device, dtype=unet.dtype)

        num_timesteps = 1000
        scheduler.set_timesteps(num_timesteps, device=self.device)
        timesteps = scheduler.timesteps

        alphas_cumprod = scheduler.alphas_cumprod.to(self.device)

        iterations = 15001
        ikl_running_avg = None

        import wandb
        run = wandb.init(
            project="ConTEXTure-NeRF"
        )

        # momentum_buffer = MomentumBuffer(momentum=-1.25)

        # --- 3. MAIN SDS OPTIMIZATION LOOP ---
        with tqdm(range(iterations), desc='SDS Texture Optimization') as pbar:
            for i in pbar:
                # Sample a random timestep for each iteration

                timestep_scheme = "dreamtime"

                if self.prolificdreamer:
                    timestep_scheme = "prolificdreamer"

                assert timestep_scheme in ["basic_annealing", "random", "dreamtime", "linear_decrease", "prolificdreamer"]

                if timestep_scheme == "basic_annealing":
                    t_max_start = 980  # Start with high-noise timesteps (coarse details).
                    t_max_end = 50     # End with low-noise timesteps (fine details).
                    annealing_period = 750 # Number of iterations to perform the annealing over.

                    # Calculate the current progress through the annealing period.
                    progress = min(i / annealing_period, 1.0)

                    # Linearly interpolate the max timestep.
                    current_t_max = int(t_max_start * (1 - progress) + t_max_end * progress)

                    # Sample a random timestep from the annealed range.
                    t = torch.randint(t_max_end, current_t_max + 1, (1,), device=self.device).long()
                    # t = timesteps[t]
                elif timestep_scheme == "random":
                    t = torch.randint(int(num_timesteps * 0.3), int(num_timesteps * 0.6), (1,), device=self.device).long()
                    # t = timesteps[t]
                elif timestep_scheme == "dreamtime":
                    dreamtime_scheduler = DreamTimeScheduler(alphas_cumprod, iterations, m=500, s=125)

                    t_int = dreamtime_scheduler.get_t(i)
                    t = torch.tensor([t_int], device=self.device)
                elif timestep_scheme == "linear_decrease":
                    progress = min(i / iterations, 1.0)
                    t = torch.tensor([int((iterations - 1) * (1 - progress))], device=self.device)
                elif timestep_scheme == "prolificdreamer":
                    anneal_point = 5000
                    if i < anneal_point:
                        min_step_percent = 0.02
                        max_step_percent = 0.98
                    else:
                        min_step_percent = 0.02
                        max_step_percent = 0.50

                    t_min = int(min_step_percent * num_timesteps)
                    t_max = int(max_step_percent * num_timesteps)
                    t = torch.randint(t_min, t_max, (1,), device=self.device).long()

               #MJ: When you call loss.backward(), PyTorch traverses the computation graph and computes gradients of all tensors with requires_grad=True.
               # These gradients are added (accumulated) into each parameter’s .grad field. In this code, we do not want this default behavior:
               # So call mlp_optimizer.zero_grad():
               
                # --- Render Student and Prepare Latents ---
                outputs = self.mesh_model.render(render_cache=render_cache, background=background_gray)
                
                camera_transform = outputs['render_cache']['camera_transform']
                rendered_six_views_clean = outputs['image'][1:]
                six_depth_maps = outputs['depth'][1:]
                six_raw_depth_maps = outputs['render_cache']['raw_depth_map'][1:]
                six_view_weights = self.view_weights[1:]

                gray_bg = torch.full_like(rendered_six_views_clean, 0.5)

                cropped_renders_small_list = []
                cropped_depths_small_list = []
                
                for j in range(B - 1):
                    min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(object_masks[j + 1, 0])

                    cropped_render = F.interpolate(rendered_six_views_clean[j:j+1, :, min_h:max_h, min_w:max_w], (320, 320), mode='bilinear', align_corners=False)
                    cropped_depth = F.interpolate(six_depth_maps[j:j+1, :, min_h:max_h, min_w:max_w], (320, 320), mode='bilinear', align_corners=False)
                    cropped_renders_small_list.append(cropped_render)
                    cropped_depths_small_list.append(cropped_depth)
                
                rendered_grid_0 = torch.cat((
                    torch.cat((cropped_renders_small_list[0], cropped_renders_small_list[3]), dim=3),
                    torch.cat((cropped_renders_small_list[1], cropped_renders_small_list[4]), dim=3),
                    torch.cat((cropped_renders_small_list[2], cropped_renders_small_list[5]), dim=3),
                ), dim=2) #MJ: The final tensor rendered_grid_clean will have a shape corresponding to a single image that contains a 3x2 grid of the original images

                rendered_grid_0 = rendered_grid_0 * 2 - 1  #MJ:   rendered_grid_clean = x0 = g(theta)
                rendered_grid_0 = scale_image(rendered_grid_0)

                latent_render_0 = vae.encode(rendered_grid_0.to(vae.dtype)).latent_dist.sample() #MJ: z0 = vae.encode(x0=g(theta)) 
                latent_render_0 = latent_render_0 * vae.config.scaling_factor

                scaled_latent_render_0 = scale_latents(latent_render_0)

                # JA: Calculate SDS loss gradient
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t.cpu().long()]).to(self.device).reshape(-1, 1, 1, 1)
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t.cpu().long()]).to(self.device).reshape(-1, 1, 1, 1)

                #MJ: with torch.no_grad():
                #It temporarily sets all NEWLY  created tensors to have requires_grad=False (unless you explicitly override)
                
                noise_true_t = torch.randn_like(scaled_latent_render_0)
                    # sigma = sigmas[t.cpu().long()]

                # JA: Forward diffusion: z_t = sqrt(α_t) * xz0 + sqrt(1 - α_t) * ε
                scaled_latent_render_t = scheduler.add_noise(scaled_latent_render_0, noise_true_t, t.unsqueeze(-1))
                #MJ: latents_noisy=z_t: latents_noisy.grad=False, but it would cause no problem, because dz_t/dz_0 is computed by other means and
                # incoporated into grad below
                
                scaled_latent_render_t = scaled_latent_render_t.half() #MJ: zero123++ is trained using latents with half precision

                #MJ: We have parallel  batches for the unet, one for the uncond prompt and the other for the cond promt;
                # So, provide 
                latent_model_input_t = torch.cat([scaled_latent_render_t] * 2) # JA: latents_noisy is the x_t obtained from the rendered image x_0
                latent_model_input_t = scheduler.scale_model_input(latent_model_input_t, t)
                latent_model_input_t = latent_model_input_t.half()
                
                
                if not self.prolificdreamer:

                    with torch.no_grad():
                        v_pred_t = unet(
                            latent_model_input_t, t,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=dict(cond_lat=clean_cond_lat, control_depth=depth_tensor),
                            return_dict=False,
                        )[0]

                        # Perform guidance
                        v_pred_t_uncond, v_pred_t_text = v_pred_t.chunk(2) # v_pred = (B * 2, 4, H // 8, W // 8)

                        use_cfg = True

                        if use_cfg:
                            guidance_scale = 7.5
                            v_pred_t = v_pred_t_uncond + guidance_scale * (v_pred_t_text - v_pred_t_uncond)
                        else:
                            v_pred_t = adaptive_projected_guidance(
                                pred_cond=v_pred_t_text,
                                pred_uncond=v_pred_t_uncond,
                                guidance_scale=500.0,
                                # momentum_buffer=momentum_buffer
                            )

                        self.log_teacher_guidance(vae, scheduler, scaled_latent_render_t, v_pred_t, t, i)
                
                    
                        v_true_t = sqrt_alphas_cumprod * noise_true_t - sqrt_one_minus_alphas_cumprod * scaled_latent_render_0
                    

                    
                    
                        # For v-prediction, the score difference is (v_pred - v) / (alpha_t * sigma_t)
                        
                        # Avoid division by zero at t=0
                        sigma_t = sqrt_one_minus_alphas_cumprod.clamp(min=1e-8)
                        alpha_t = sqrt_alphas_cumprod.clamp(min=1e-8)

                        # divergence_at_t = torch.sum(((v_pred.detach() - v.detach()) / (alpha_t * sigma_t)) ** 2)
                        fisher_divergence_t = torch.sum((alpha_t / sigma_t) ** 2 * torch.abs(v_pred_t - v_true_t) ** 2)

                        # Update the running average (Exponential Moving Average)
                        if ikl_running_avg is None:
                            ikl_running_avg = fisher_divergence_t.item()
                        else:
                            beta = 0.99  # Smoothing factor
                            ikl_running_avg = beta * ikl_running_avg + (1 - beta) * fisher_divergence_t.item()
                
                    
                        grad_scale = 1
                        w = (1 - alphas_cumprod[t.cpu().long()])

                    # JA: The original formula for the grad is as follows:
                    # grad = grad_scale * w[:, None, None, None] * sqrt_alphas_cumprod * (v_pred - v)
                    #MJ: eps_pred = c1*v_pred + c2 * z_t; eps = c1*v^{star} + c2*z_t, where c1=sqrt_alphas_cumprod ,
                    # c2= sqrt(1-alphas_cumprod) => eps_pred - eps = c1*(v_pred - v^{star}),
                    # where v^{star} is the true target= c1*eps - c2*z0,  given (z0, eps)

                                      
                        grad_zt = grad_scale *  (v_pred_t - v_true_t)
                        #MJ: 
                        #MJ: grad_z0 =  sqrt_alphas_cumprod * * (v_pred_t - v_true_t)
                        #MJ: #MJ: noisy_v.grad = True?

                        # 2. Define the target gradient: stop-grad on the whole target
                        targets_zt = (scaled_latent_render_t - grad_zt).float().detach()

                    #MJ: .detach():
                        # z = y.detach()
                        # z shares the same storage as y, but no longer has a grad_fn.
                        # Graph for z stops there. Backprop through z won’t reach x.
                        # But note: y is still connected to x. If you backprop through y or loss = y**2, the original x is still updated.
                    
                    #MJ: Pseudo-loss OUTSIDE no_grad (so it tracks grads through z_t -> z0 -> theta)                    
                    
                    sds_pseudo_loss = 0.5 * F.mse_loss( # [B, C, 120, 80]
                        scaled_latent_render_t.float(),
                        targets_zt,
                        reduction='mean'
                    )
                # MJ: ∂L/∂z_t = (1/N) * grad_zt; ∂z_t/∂θ = √α_t * ∂z_0/∂θ 
                # => ∇θ L = ( 1 / N ) * * grad_zt *  √α_t *(∂z_0/∂θ) 

                #MJ: The equiv to the above is:
                #   # scaled_latent_render_t = z_t:
                    # L_theta = (grad_zt.detach() * z_t).mean() ⇒ ∂L/∂z_t = grad_zt / N.
                    
                    # => dL/dtheta = dL/dz_t *  dz_t/dtheta =  (grad_zt/N) *(dz_t/dtheta) 
                    #  = (grad_zt/N) * sqrt_alpha_t (dz_0/dtheta) 

                # consistency_reward = 0#self.compute_view_consistency(
                # #     rendered_six_views_clean,
                # #     self.mesh_model.mesh.faces,
                # #     render_cache['face_idx'][1:],
                # #     render_cache['face_vertices_image'][1:]
                # # )

                # graph = make_dot(loss, params=dict(self.lora_unet.named_parameters()))

                # print(f"SDS: {sds_loss:.2f}, VC: {vc_loss:.2f}")

                    mlp_optimizer.zero_grad()
             
                    sds_pseudo_loss.backward()  ## autograd computes ∂L/∂θ
                    
                    mlp_optimizer.step()
                
                    mlp_param_to_watch = list(self.mesh_model.texture_mlp.parameters())[-2]
                    self.visualize_gradients("MLP_Final_Layer", mlp_param_to_watch, i)

                   
                    # torch.nn.utils.clip_grad_norm_(self.mesh_model.texture_mlp.parameters(), 1.0)

                    # =========================================================================
                    # <<< START DEBUGGING BLOCK: CHECK WEIGHT UPDATES >>>
                    # =========================================================================

                    # Check MLP (Texture Generator) Weights ---
                    mlp_weight_before = None
                    # Create a list of ONLY the trainable parameters
                    trainable_mlp_params = [p for p in self.mesh_model.texture_mlp.parameters() if p.requires_grad]
                    if trainable_mlp_params:
                        # Select the first one from the guaranteed-to-be-trainable list
                        mlp_param_to_watch = trainable_mlp_params[0]
                        mlp_weight_before = mlp_param_to_watch.clone().detach()
                    else:
                        # This would be a major problem, but our check will catch it
                        print("CRITICAL DEBUG: No trainable NeRF parameters were found!")
                # if not self.prolificdreamer:
                
                else:    #self.prolificdreamer == True:
                # Check LoRA (Diffusion UNet) Weights (only if prolificdreamer is active) ---
                    lora_weight_before = None
            
                    # Create a list of ONLY the trainable parameters
                    trainable_lora_params = [p for p in self.lora_unet.parameters() if p.requires_grad]
                    if trainable_lora_params:
                        # Select the first one from the guaranteed-to-be-trainable list
                        lora_param_to_watch = trainable_lora_params[0]
                        lora_weight_before = lora_param_to_watch.clone().detach()
                    else:
                        # This would be a major problem, but our check will catch it
                        print("CRITICAL DEBUG: No trainable LoRA parameters were found!")
                         
                    v_pred_t_q = self.lora_unet(
                        scaled_latent_render_t, #MJ: Jaehoon, you used latent_render_t here
                            t,
                        encoder_hidden_states=cond_encoder_hidden_states,
                        cross_attention_kwargs=dict(cond_lat=cond_lat, control_depth=depth_tensor),
                        return_dict=False,
                    )[0]
                    
                    v_true_t = sqrt_alphas_cumprod * noise_true_t - sqrt_one_minus_alphas_cumprod * scaled_latent_render_0

                    with torch.no_grad():
                        v_pred_t = unet(
                            latent_model_input_t, t,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=dict(cond_lat=clean_cond_lat, control_depth=depth_tensor),
                            return_dict=False,
                        )[0]

                        # Perform guidance
                        v_pred_t_uncond, v_pred_t_text = v_pred_t.chunk(2) # v_pred = (B * 2, 4, H // 8, W // 8)

                        use_cfg = True

                        if use_cfg:
                            guidance_scale = 7.5
                            v_pred_t = v_pred_t_uncond + guidance_scale * (v_pred_t_text - v_pred_t_uncond)
                        else:
                            v_pred_t = adaptive_projected_guidance(
                                pred_cond=v_pred_t_text,
                                pred_uncond=v_pred_t_uncond,
                                guidance_scale=200.0,
                                # momentum_buffer=momentum_buffer
                            )
                        
                        self.log_teacher_guidance(vae, scheduler, scaled_latent_render_t, v_pred_t, t, i)
                        self.log_lora_output(vae, scheduler, scaled_latent_render_t, v_pred_t_q, t, i)
                                                    
                        # For v-prediction, the score difference is (v_pred - v) / (alpha_t * sigma_t)
                        
                        # Avoid division by zero at t=0
                        sigma_t = sqrt_one_minus_alphas_cumprod.clamp(min=1e-8)
                        alpha_t = sqrt_alphas_cumprod.clamp(min=1e-8)

                        # divergence_at_t = torch.sum(((v_pred.detach() - v.detach()) / (alpha_t * sigma_t)) ** 2)
                        fisher_divergence_t = torch.sum((alpha_t / sigma_t) ** 2 * torch.abs(v_pred_t - v_true_t) ** 2)

                        # Update the running average (Exponential Moving Average)
                        if ikl_running_avg is None:
                            ikl_running_avg = fisher_divergence_t.item()
                        else:
                            beta = 0.99  # Smoothing factor
                            ikl_running_avg = beta * ikl_running_avg + (1 - beta) * fisher_divergence_t.item()
                
                    
                        grad_scale = 1
                        w = (1 - alphas_cumprod[t.cpu().long()])

                    # JA: The original formula for the grad is as follows:
                    # grad = grad_scale * w[:, None, None, None] * sqrt_alphas_cumprod * (v_pred - v)
                    #MJ: eps_pred = c1*v_pred + c2 * z_t; eps = c1*v^{star} + c2*z_t, where c1=sqrt_alphas_cumprod ,
                    # c2= sqrt(1-alphas_cumprod) => eps_pred - eps = c1*(v_pred - v^{star}),
                    # where v^{star} is the true target= c1*eps - c2*z0,  given (z0, eps)

                                    
                        grad_zt = grad_scale *  (v_pred_t -   v_pred_t_q)
                    
                        #MJ: grad_z0 =  sqrt_alphas_cumprod * * (v_pred_t - v_true_t)
                        

                        # 2. Define the target gradient: stop-grad on the whole target
                        targets_zt = (scaled_latent_render_t - grad_zt).float().detach()

                        #MJ: .detach():
                        # z = y.detach()
                        # z shares the same storage as y, but no longer has a grad_fn.
                        # Graph for z stops there. Backprop through z won’t reach x.
                        # But note: y is still connected to x. If you backprop through y or loss = y**2, the original x is still updated.
                    
                    
                    #END with torch.no_grad():
             
                
                               
                #MJ: Pseudo-loss OUTSIDE no_grad (so it tracks grads through z_t -> z0 -> theta)                    
                
                    vsd_pseudo_loss = 0.5 * F.mse_loss( # [B, C, 120, 80]
                        scaled_latent_render_t.float(),
                        targets_zt,
                        reduction='mean'
                    )
                # MJ: ∂L/∂z_t = (1/N) * grad_zt; ∂z_t/∂θ = √α_t * ∂z_0/∂θ 
                # => ∇θ L = ( 1 / N ) * * grad_zt *  √α_t *(∂z_0/∂θ) 

                #MJ: The equiv to the above is:
                #   # scaled_latent_render_t = z_t:
                    # L_theta = (grad_zt.detach() * z_t).mean() ⇒ ∂L/∂z_t = grad_zt / N.
                    
                    # => dL/dtheta = dL/dz_t *  dz_t/dtheta =  (grad_zt/N) *(dz_t/dtheta) 
                    #  = (grad_zt/N) * sqrt_alpha_t (dz_0/dtheta) 
                

                # graph = make_dot(loss, params=dict(self.lora_unet.named_parameters()))

                # print(f"SDS: {sds_loss:.2f}, VC: {vc_loss:.2f}")
                
                  
                
                    lora_loss = 0.5 * F.mse_loss( # [B, C, 120, 80]
                        v_pred_t_q,
                        v_true_t.to(v_pred_t_q.dtype),
                        reduction='mean'
                    )
                                        
                    #MJ: Choose the particle randomly .
                    self.mesh_model.texture_mlp.set_idx() 

                    # print("--- NeRF Model Gradients (Iter {}) ---".format(i))
                    # max_grad_nerf = 0.0
                    # for name, p in self.mesh_model.texture_mlp.named_parameters():
                    #     if p.grad is not None:
                    #         param_max_grad = p.grad.abs().max().item()
                    #         print(f"{name}: max_grad = {param_max_grad:.6f}")
                    #         if param_max_grad > max_grad_nerf:
                    #             max_grad_nerf = param_max_grad

                    # # Check gradients for the LoRA model (student 2)
                    # print("--- LoRA Model Gradients (Iter {}) ---".format(i))
                    # max_grad_lora = 0.0
                    # for name, p in self.lora_unet.named_parameters():
                    #     if p.requires_grad and p.grad is not None:
                    #         param_max_grad = p.grad.abs().max().item()
                    #         print(f"{name}: max_grad = {param_max_grad:.6f}")
                    #         if param_max_grad > max_grad_lora:
                    #             max_grad_lora = param_max_grad

                    # print(f"OVERALL MAX GRADIENT: NeRF={max_grad_nerf:.6f}, LoRA={max_grad_lora:.6f}\n")

                    
                    mlp_optimizer.zero_grad()
                    lora_optimizer.zero_grad()
            
                    vsd_pseudo_loss.backward(retain_graph=True)  ## autograd computes ∂L/∂θ
                    lora_loss.backward()

                    mlp_optimizer.step()    
                    lora_optimizer.step()
                #END else  #self.prolificdreamer == True

                # Collect all gradients into a list of flat 1D tensors
                grads = [p.grad.view(-1) for p in self.mesh_model.texture_mlp.parameters() if p.grad is not None]

                # Concatenate into one big vector
                if grads:
                    grad_vector = torch.cat(grads)
                else:
                    grad_vector = torch.tensor(0.0, device=self.device)

                grad_norm = torch.linalg.norm(grad_vector)

                # wandb.log({
                #     "grad_norm": grad_norm,
                #     "fisher_divergence_t": fisher_divergence_t,
                #     "ikl_running_avg": ikl_running_avg,
                #     "sds_loss": sds_pseudo_loss,
                #     #"tv_loss": tv_loss,
                #     #"consistency_reward": consistency_reward,
                #     "t": t
                # })

                if (i % 10 == 0 and i < 1000) or (i % 100 == 0):
                    self.log_texture_map(i)
                    self.log_train_image((unscale_image(rendered_grid_0) + 1) / 2, f'rendered_grid_clean_{i}')

                if i % 50 == 0:
                    # Get the gradient of the last layer of your MLP
                    #MJ: The parameters list would be ordered as: [layer1_weights, layer1_bias, layer2_weights, layer2_bias, ..., final_layer_weights, final_layer_bias]. 
                    # Therefore, [-2] correctly accesses the weights of the final layer
                    final_layer = list(self.mesh_model.texture_mlp.parameters())[-2] # Usually the weight, not the bias
                    
                    if final_layer.grad is not None:
                        grad_norm = final_layer.grad.norm().item()
                        logits = outputs['mlp_output']
                        
                        # Log the values
                        logger.info(f"--- Iteration {i} Debug Info ---")
                        logger.info(f"Logits > min: {logits.min().item():.2f}, max: {logits.max().item():.2f}, mean: {logits.mean().item():.2f}")
                        logger.info(f"Gradient Norm of Final Layer: {grad_norm}")
                        logger.info(f"---------------------------------")


                if (i % 10 == 0 and i < 1000) or (i % 100 == 0):
                    self.log_texture_map(i)
                    self.log_train_image((unscale_image(rendered_grid_0) + 1) / 2, f'rendered_grid_0_{i}')

                # pbar.set_description(f"SDS Texture Optimization: Iter {i}, Loss: {loss_for_logging:.4f}")




        self.mesh_model.change_default_to_median()
        logger.info('Finished SDS Painting ^_^')
        self.full_eval()

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False): #MJ: dataloader=self.dataloaders['val']
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video: 
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data) #MJ: preds, textures, depths, normals = rgb_render, texture_rgb, depth_render, pred_z_normals
            #MJ: normals =  pred_z_normals = meta_output['image'][:, :1].detach() #MJ: pred_z_normals refers to max_z_normals
            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"eval:rendered_image:{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'eval:normal_map:{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"eval:depth_map:{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"eval:texture_atlas:texture.png")
        
       
        
        
          
        if save_as_video:  #np.cat: Shape Change: If the input arrays have shape (A, B, C), the concatenated array will have shape (NA, B, C) if axis=0 (where N is the number of arrays).
            all_preds = np.stack(all_preds, axis=0) # combine a sequence of arrays along a new axis:  If the input arrays have shape (A, B, C), the stacked array will have shape (N, A, B, C) if axis=0 (where N is the number of arrays).

            
            dump_vid = lambda video, name: imageio.mimsave(save_path / f"eval:constructed_video:{name}_{self.cfg.optim.seed}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'all_rendered_rgb')
        logger.info('Eval Done!')

    def full_eval(self, output_dir: Path = None):

        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\t Full Eval Done!")

    # JA: paint_viewpoint computes a portion of the texture atlas for the given viewpoint
    def paint_viewpoint(self, data: Dict[str, Any], should_project_back=True):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius'] # JA: data represents a viewpoint which is stored in the dataset
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if  self.cfg.guide.second_model_type in ["zero123", "control_zero123"]: #self.view_dirs[data['dir']] != "front":
            # JA: For Zero123, the input image background is always white
            background = torch.Tensor([1, 1, 1]).to(self.device)
        elif self.cfg.guide.use_background_color: # JA: When use_background_color is True, set the background to the green color
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else: # JA: Otherwise, set the background to the brick image
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
        render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)
        rgb_render_raw = outputs['image']  #MJ: The rendered image without using use-median = True 
        depth_render = outputs['depth']
        object_mask_bchw = outputs['mask']
        
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        
     
       
        outputs = self.mesh_model.render(background=background,
                                          render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        
        # meta_output = self.mesh_model.render(background=background,
        #                                     use_meta_texture=True, render_cache=render_cache)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        # z_normals_cache = meta_output['image'].clamp(0, 1)
        # edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

          
        self.log_train_image(rgb_render, 'paint_viewpoint:rgb_render')
        self.log_train_image(depth_render[0, 0], 'paint_viewpoint:depth', colormap=True)
        # self.log_train_image(z_normals[0, 0], 'paint_viewpoint:z_normals', colormap=True)
        # self.log_train_image(z_normals_cache[0, 0], 'paint_viewpoint:z_normals_cache', colormap=True)

        # text embeddings
        if self.cfg.guide.use_zero123plus:
            text_z = self.text_z[1]
            text_string = self.text_string[1]
            view_dir = "front"
        elif self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs] # JA: dirs is one of the six directions. text_z is the embedding vector of the specific view prompt
            text_string = self.text_string[dirs]
            view_dir = self.view_dirs[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
            view_dir = None
        logger.info(f'text: {text_string}')

        # Crop to inner region based on object mask
        object_mask_hw = object_mask_bchw[0, 0] # JA: object_mask_bchw.shape = [1, 1, 1200, 1200]
        min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(object_mask_hw)
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render) # JA: This is rendered image which is denoted as Q_0.
                                              # In our experiment, 1200 is cropped to 827
        cropped_depth_render = crop(depth_render)
        cropped_object_mask_bchw = crop(object_mask_bchw)
     
        self.log_train_image(cropped_rgb_render, name='paint_viewpoint:cropped_rgb_render')
        self.log_train_image(cropped_depth_render.repeat_interleave(3, dim=1), name='paint_viewpoint:cropped_depth')

        start_time = time.perf_counter()  # Record the start time

        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(),
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=cropped_object_mask_bchw,
                                                                    fixed_seed=self.cfg.optim.seed,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps)
        

        
        end_time = time.perf_counter()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"Elapsed time in self.diffusion.img2img_step in TEXTureWithZero123: {elapsed_time:0.4f} seconds")
        
        self.log_train_image(cropped_rgb_output, name='paint_viewpoint:cropped_rgb_output (result of img2img) (magenta boundary?)')
        self.log_diffusion_steps(steps_vis)
        # JA: cropped_rgb_output always has a shape of (512, 512); recover the resolution of the nonzero rendered image (e.g. (827, 827))
        cropped_rgb_output = F.interpolate(cropped_rgb_output, 
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        # JA: After the image is generated, we insert it into the original RGB output
        rgb_output = rgb_render.clone() # JA: rgb_render shape is 1200x1200
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output # JA: For example, (189, 1016, 68, 895) refers to the nonzero region of the render image
        self.log_train_image(rgb_output, name='full_output')

        # Project back
        object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200
        # JA: Compute a part of the texture atlas corresponding to the target render image of the specific viewpoint
        if should_project_back:
            if not self.cfg.guide.use_zero123plus:  
               fitted_pred_rgb = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask,  z_normals=z_normals,
                                                z_normals_cache=z_normals_cache
                                                )
            else:
               fitted_pred_rgb = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask,  z_normals=None,
                                                z_normals_cache=None
                                                )                                                                                  
            self.log_train_image(fitted_pred_rgb, name='paint_viewpoint:fitted_pred_rgb rendered using the texture map learned from the front view image')
            
            

        # JA: Zero123 needs the input image without the background
        # rgb_output is the generated and uncropped image in pixel space
        zero123_input = crop(
            rgb_output * object_mask
            + torch.ones_like(rgb_output, device=self.device) * (1 - object_mask)
        )   # JA: In the case of front view, the shape is (930,930).
            # This rendered image will be compressed to the shape of (512, 512) which is the shape of the diffusion
            # model.

        if view_dir == "front":
            self.zero123_front_input = zero123_input
        
        # if self.zero123_inputs is None:
        #     self.zero123_inputs = []
        
        # self.zero123_inputs.append({
        #     'image': zero123_input,
        #     'phi': data['phi'],
        #     'theta': data['theta']
        # })

        self.log_train_image(zero123_input, name='paint_viewpoint:zero123_cond_image')

        return rgb_output, object_mask

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        
        #Now, self.texture_img has been learned fully (when we call eval_render even when self.texture_img is partially learned)
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        
        
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        
        #rgb_render.register_hook(self.print_hook) #MJ: for debugging with loss.backward(retrain_graph=True)
        
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask
        #MJ: In case when  self.texture_img is not learned (still with the default magenta color), 
        # fill that with the mean color of the learned part
        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), #MJ: use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        
        pred_z_normals = meta_output['image'][:, :1].detach() #MJ: pred_z_normals refers to max_z_normals
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    def print_hook(self, grad):
           print(f"Gradient: {grad}")  

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False, file_type="jpg"):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            
            if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
    #     # Raise an exception if there are any NaNs or infinite values
    #      tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
    #      Image.fromarray( (tensor * 255).astype(np.uint8) ).save('experiments'/f'debug:NanOrInf.jpg')

                raise ValueError("Tensor contains NaNs or infinite values")
            
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'debug:{name}.{file_type}')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)

    def log_texture_map(self, iter: int):
        """
        Generates and saves the current texture map from the 2D NeRF model.
        """
        # Put model in evaluation mode to ensure no gradients are computed
        self.mesh_model.eval()

        # Get the texture map from the 2D NeRF
        with torch.no_grad():
            texture_tensor, _ = self.mesh_model.get_texture_map()
            # texture_tensor = (texture_tensor + 1) / 2
            
        # [cite_start]Convert tensor to a NumPy array in the HWC format [cite: 399]
        # The get_texture_map() returns (1, 3, H, W)
        texture_np = einops.rearrange(texture_tensor, 'b c h w -> b h w c')[0].cpu().numpy()
        
        # Scale values to 0-255 and convert to uint8
        texture_np = (texture_np * 255).astype(np.uint8)

        # Save the image to the specified path
        save_path = self.train_renders_path / f'texture_map_iter_{iter:06d}.png'
        Image.fromarray(texture_np).save(save_path)
        
        # Restore model to training mode
        self.mesh_model.train()

        logger.info(f"Saved texture map to {save_path}")