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

from torch_scatter import scatter_max

import torchvision
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import Zero123PlusDataset, ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy, pad_tensor_to_size, split_zero123plus_grid
from src.run_nerf_helpers import *

from PIL import Image, ImageDraw
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
        self.uv_embedder, input_ch_uv = get_embedder(multires=10) 

        # The 2D NeRF model, with input dimensions matching the embedder's output
        self.texture_mlp = NeRF2D(D=8, W=256, input_ch=input_ch_uv, output_ch=3, skips=[4]).to(self.device)
        if torch.cuda.device_count() > 1:
            self.texture_mlp = nn.DataParallel(self.texture_mlp)

        # You should also pass these new components to your mesh model
        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom'] # self.view_dirs[dir] when dir = [4] = [right]
        self.mesh_model = self.init_mesh_model(texture_mlp=self.texture_mlp, uv_embedder=self.uv_embedder)
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

        pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
        ), conditioning_scale=2)

        pipeline.to(self.device)

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
        outputs_all_views = self.mesh_model.render(
            theta=self.thetas, phi=self.phis, radius=self.radii, 
            background=background_gray, use_median=True
        )
        object_masks = outputs_all_views['mask']
        depth_maps = 1.0 - outputs_all_views['depth']
        render_cache = outputs_all_views['render_cache']
        B = object_masks.shape[0]

        # Prepare the condition image (front view)
        min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(object_mask_front[0, 0])
        front_image_rgba = torch.cat((rgb_output_front, object_mask_front), dim=1)
        cropped_front_image_rgba = front_image_rgba[:, :, min_h:max_h, min_w:max_w]
        cond_image_pil = torchvision.transforms.functional.to_pil_image(cropped_front_image_rgba[0]).resize((320, 320))

        # Prepare the 3x2 depth grid for the 6 novel views
        depth_rgba = torch.cat((depth_maps, depth_maps, depth_maps), dim=1)
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

        self.log_train_image(cropped_depth_grid, 'cropped_depth_grid')

        depth_image_pil = torchvision.transforms.functional.to_pil_image(cropped_depth_grid[0])

        # Setup SDS loop
        logger.info("Setting up SDS optimization loop...")
        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=1e-5, betas=(0.9, 0.99), eps=1e-15)
        scheduler = self.zero123plus.scheduler
        unet = self.zero123plus.unet
        vae = self.zero123plus.vae
        
        with torch.no_grad():
            cond_image_vae = self.zero123plus.feature_extractor_vae(images=cond_image_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=vae.dtype)
            cond_image_clip = self.zero123plus.feature_extractor_clip(images=cond_image_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=unet.dtype)

            cond_lat = vae.encode(cond_image_vae).latent_dist.sample()
            
            # JA: Get unconditional latent for guidance
            negative_lat = vae.encode(torch.zeros_like(cond_image_vae)).latent_dist.sample()
            
            encoded = self.zero123plus.vision_encoder(cond_image_clip, output_hidden_states=False)
            global_embeds = encoded.image_embeds.unsqueeze(-2)
            
            # JA: Get text embeddings (for empty prompt) and combine with vision embeddings
            text_embeds = self.zero123plus.encode_prompt("", self.device, 1, False)[0]
            ramp = global_embeds.new_tensor(self.zero123plus.config.ramping_coefficients).unsqueeze(-1)
            encoder_hidden_states = text_embeds + global_embeds * ramp
            
            # JA: Get unconditional text embeddings
            uncond_embeds = self.zero123plus.encode_prompt("", self.device, 1, True)[1]
            
            # JA: Concatenate for classifier-free guidance
            final_encoder_hidden_states = torch.cat([uncond_embeds, encoder_hidden_states])
            final_cond_lat = torch.cat([negative_lat, cond_lat])

            # JA: Prepare depth map tensor for ControlNet
            depth_tensor = self.zero123plus.depth_transforms_multi(depth_image_pil).to(device=self.device, dtype=unet.dtype)

        num_timesteps = 1000
        timesteps = torch.arange(0, num_timesteps, dtype=torch.float64, device=self.device).flip(0)

        # JA: Where α = 1 - β, β represents standard deviations and ranges from sqrt(1e-4) to sqrt(2e-2). This is the same
        # setting used by Stable Diffusion. β consists of num_timesteps number of equidistant standard deviations between
        # the aforementioned two numbers.
        alphas = 1. - (torch.linspace(
            1e-4 ** 0.5, 2e-2 ** 0.5, num_timesteps, device=self.device, dtype=torch.float64
        ) ** 2)

        alphas_cumprod = torch.cumprod(alphas, dim=0)

        epochs = 30000

        # --- 3. MAIN SDS OPTIMIZATION LOOP ---
        with tqdm(range(epochs), desc='SDS Texture Optimization') as pbar:
            for i in pbar:
                # Sample a random timestep for each iteration
                t = torch.randint(0, num_timesteps, (1,), device=self.device).long()
                t = timesteps[t]

                optimizer.zero_grad()

                # --- Render Student and Prepare Latents ---
                outputs = self.mesh_model.render(render_cache=render_cache, background=background_gray)
                rendered_six_views_clean = outputs['image'][1:]
                six_depth_maps = outputs['depth'][1:]
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
                
                rendered_grid_clean = torch.cat((
                    torch.cat((cropped_renders_small_list[0], cropped_renders_small_list[3]), dim=3),
                    torch.cat((cropped_renders_small_list[1], cropped_renders_small_list[4]), dim=3),
                    torch.cat((cropped_renders_small_list[2], cropped_renders_small_list[5]), dim=3),
                ), dim=2)

                rendered_grid_clean = rendered_grid_clean * 2 - 1
                rendered_grid_clean = scale_image(rendered_grid_clean)

                latents_clean = vae.encode(rendered_grid_clean.to(vae.dtype)).latent_dist.sample()
                latents_clean = latents_clean * vae.config.scaling_factor

                scaled_latents_clean = scale_latents(latents_clean)

                alpha_cumprod_t = alphas_cumprod[t.cpu().long()].to(scaled_latents_clean.device)
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t).reshape(1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alpha_cumprod_t).reshape(1, 1, 1, 1)

                with torch.no_grad():
                    noise = torch.randn_like(scaled_latents_clean)

                    # JA: Forward diffusion: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
                    latents_noisy = sqrt_alpha_cumprod_t * scaled_latents_clean + sqrt_one_minus_alpha_cumprod_t * noise
                    latents_noisy = latents_noisy.half()

                    latent_model_input = torch.cat([latents_noisy] * 2)
                    latent_model_input = self.zero123plus.scheduler.scale_model_input(latent_model_input, t)
                    
                    # JA: Prepare cross-attention kwargs, which is how Zero123++ pipeline passes conditioning
                    cross_attention_kwargs = {
                        "cond_lat": final_cond_lat,
                        "control_depth": depth_tensor #depth_grid.half()
                    }

                    v_pred = unet(
                        latent_model_input, t, 
                        encoder_hidden_states=final_encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False
                    )[0]
                
                    # Perform guidance
                    v_pred_uncond, v_pred_text = v_pred.chunk(2)
                    guidance_scale = 10
                    v_pred = v_pred_uncond + guidance_scale * (v_pred_text - v_pred_uncond)

                # JA: Calculate SDS loss gradient
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t.cpu().long()]).to(self.device).reshape(-1, 1, 1, 1)
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t.cpu().long()]).to(self.device).reshape(-1, 1, 1, 1)
                v_target = sqrt_alphas_cumprod * noise - sqrt_one_minus_alphas_cumprod * scaled_latents_clean

                grad_scale = 1
                w = (1 - alphas_cumprod[t.cpu().long()])
                grad = grad_scale * w[:, None, None, None] * (v_pred - v_target)
                grad = torch.nan_to_num(grad)

                # 2. Define the target latent
                targets = (scaled_latents_clean - grad).float().detach()

                # 3. Calculate the MSE loss
                loss = 0.5 * F.mse_loss(
                    scaled_latents_clean.float(),
                    targets,
                    reduction='sum'
                ) / scaled_latents_clean.shape[0]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mesh_model.texture_mlp.parameters(), 1.0)

                if i % 50 == 0:
                    # Get the gradient of the last layer of your MLP
                    final_layer = list(self.mesh_model.texture_mlp.parameters())[-2] # Usually the weight, not the bias
                    
                    if final_layer.grad is not None:
                        grad_norm = final_layer.grad.norm().item()
                        logits = outputs['mlp_output']
                        
                        # Log the values
                        logger.info(f"--- Iteration {i} Debug Info ---")
                        logger.info(f"Logits > min: {logits.min().item():.2f}, max: {logits.max().item():.2f}, mean: {logits.mean().item():.2f}")
                        logger.info(f"Gradient Norm of Final Layer: {grad_norm}")
                        logger.info(f"---------------------------------")

                optimizer.step()

                if (i % 10 == 0 and i < 1000) or (i % 100 == 0):
                    self.log_texture_map(i)
                    self.log_train_image((unscale_image(rendered_grid_clean) + 1) / 2, f'rendered_grid_clean_{i}')

                # pbar.set_description(f"SDS Texture Optimization: Iter {i}, Loss: {loss_for_logging:.4f}")
                pbar.update(1)

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
      
    def project_back_max_z_normals(self):
        optimizer = torch.optim.Adam(self.mesh_model.get_params_max_z_normals(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                        eps=1e-15)

        #End  for v in range( len(self.thetas) )
        with  tqdm(range(300), desc='project_back_max_z_normals:fitting max_z_normals') as pbar:
            render_cache = None
            for iter in pbar:
                #MJ: Render the max_z_normals (self.meta_texure_img) which has been learned using the previous view z_normals
                # At the beginning of the for loop, self.meta_texture_img is set to 0
                if render_cache is None:
                    meta_output = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii,
                                                    background=torch.Tensor([0, 0, 0]).to(self.device),
                                                            use_meta_texture=True, render_cache=None)
                    render_cache = meta_output["render_cache"]
                else:
                    meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device), use_meta_texture=True, render_cache=render_cache)

                max_z_normals_projected = meta_output['image'][:,0:1,:,:]
                #MJ: meta_output['image'] is the projected meta_texture_img; The first channel refers to the max_z_normals in the current view
                z_normals = meta_output['normals'][:,2:3,:,:]   #MJ: Get the z component of the face normal in the current view
                #MJ: z_normals is the z component of the normal vectors of the faces seen by each view
                z_normals_mask = meta_output['mask']   #MJ: shape = (1,1,1200,1200)
                #MJ: Try blurring the object-mask "curr_z_mask" with Gaussian blurring:
                # The following code is a simply  cut and paste from project-back:
                object_mask = z_normals_mask
                # #MJ: erode the boundary of the mask
                # object_mask_v = torch.from_numpy( cv2.erode(object_mask_v[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8)) ).to(
                #                      object_mask_v.device).unsqueeze(0).unsqueeze(0)
                # # object_mask = torch.from_numpy(
                # #     cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
                # #     object_mask.device).unsqueeze(0).unsqueeze(0)
                # # render_update_mask = object_mask.clone()
                render_update_mask =  object_mask.clone()
                # #MJ: dilate the bounary of the mask
                # blurred_render_update_mask_v = torch.from_numpy(
                #      cv2.dilate(render_update_mask_v[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
                #      render_update_mask_v.device).unsqueeze(0).unsqueeze(0)
                # blurred_render_update_mask_v = utils.gaussian_blur(blurred_render_update_mask_v, 21, 16)
                # # Do not get out of the object
                # blurred_render_update_mask_v[object_mask_v == 0] = 0
                max_z_normals_projected  = max_z_normals_projected.clone()  *    render_update_mask.float()
                z_normals = z_normals.clone() *  render_update_mask.float()
                delta =  max_z_normals_projected -   z_normals
            # Compute the ReLU of the negative differences
                loss_v = F.relu(-delta)  # Shape: (B, 1, h, w)
                # Sum the loss over all pixels and add to total loss

                total_loss = loss_v.sum()

                optimizer.zero_grad()
                total_loss.backward() # JA: Compute the gradient vector of the loss with respect 
                                # to the trainable parameters of the network, that is, the pixel value of the
                                # texture atlas
                optimizer.step()

                pbar.set_description(f"project_max_z_normals: Fitting z_normals -Epoch {iter}, Loss: {total_loss.item():.7f}")

                if total_loss == 0:
                    print(f"max_z_normals training reached 0 loss at epoch {iter}")
                    break
        #End for _ in tqdm(range(300), desc='fitting max_z_normals')
                
         
    def compute_view_weights(self, z_normals, max_z_normals, alpha=-10.0 ):        
        
        """
        Compute view weights where the weight increases exponentially as z_normals approach max_z_normals.
        
        Args:
            z_normals (torch.Tensor): The tensor containing the z_normals data.
            max_z_normals (torch.Tensor): The tensor containing the max_z_normals data.
            alpha (float): A scaling parameter that controls how sharply the weight increases (should be negative).
        
        Returns:
            torch.Tensor: The computed weights with the same shape as the input tensors (B, 1, H, W).
        """
        # Ensure inputs have the same shape
        assert z_normals.shape == max_z_normals.shape, "Input tensors must have the same shape"
        
        # Compute the difference between max_z_normals and z_normals

        delta = max_z_normals - z_normals
        # for i in range( delta.shape[0]):
        #     print(f'min delta for view-{i}:{delta[i].min()}')
        #     print(f'max  delta for view-{i}:{delta[i].max()}')
        #MJ: delta is supposed to be greater than 0; But sometimes, z_normals is greater than max_z_normals.
        # It means that project_back_max_z_normals() was not fully successful.
        
        max_z_normals = torch.where( delta >=0, max_z_normals, z_normals)
        delta_new = max_z_normals - z_normals
        # Calculate the weights using an exponential function, multiplying by negative alpha
        weights = torch.exp(alpha * delta_new)  #MJ: the max value of torch.exp(alpha * delta)   will be torch.exp(alpha * 0) = 1 
        #debug: for i in range( weights.shape[0]):
        #     print(f'min weights  for view-{i}:{weights[i].min()}')
        #     print(f'max  weights for view-{i}:{weights[i].max()}')
        # Normalize to have the desired shape (B, 1, H, W)
        #weights = weights.view(weights.size(0), 1, weights.size(1), weights.size(2))
        
        return weights
       
    
    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
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
                self.train_renders_path / f'debug:{name}.jpg')

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