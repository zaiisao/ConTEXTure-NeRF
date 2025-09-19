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
        w_d = torch.sqrt(1 - alphas_cumprod)

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
        
        self.uv_embedder, input_ch_uv = get_embedder(multires=10) #MJ: input_ch_uv = the dim of the Fourier embedding vector of (u,v), say 60

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

        pipeline.prepare()
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
        cond_image_pil = torchvision.transforms.functional.to_pil_image(cropped_front_image_rgba[0]).resize((320, 320))

        # JA: to_rgb_image is a helper from Zero123++ which turns the background of cond_image_pil and depth_image_pil
        # gray
        cond_image_pil = self.to_rgb_image(cond_image_pil)

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

        depth_image_pil = torchvision.transforms.functional.to_pil_image(cropped_depth_grid[0])
        depth_image_pil = self.to_rgb_image(depth_image_pil)

        # Setup SDS loop
        logger.info("Setting up SDS optimization loop...")
        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=1e-5, betas=(0.9, 0.99), eps=1e-15)
        scheduler = self.zero123plus.scheduler
        unet = self.zero123plus.unet
        vae = self.zero123plus.vae
        
        with torch.no_grad():
            # JA: cond_image_pil is the front view image with a gray background
            cond_image_vae = self.zero123plus.feature_extractor_vae(images=cond_image_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=vae.dtype)
            cond_image_clip = self.zero123plus.feature_extractor_clip(images=cond_image_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=unet.dtype)

            # JA: cond_lat is from the front view image
            cond_lat = vae.encode(cond_image_vae).latent_dist.sample()
            
            # JA: Get unconditional latent for guidance
            negative_lat = vae.encode(torch.zeros_like(cond_image_vae)).latent_dist.sample()
            
            encoded = self.zero123plus.vision_encoder(cond_image_clip, output_hidden_states=False)
            global_embeds = encoded.image_embeds.unsqueeze(-2)
            
            # JA: Get text embeddings (for empty prompt) and combine with vision embeddings
            text_embeds = self.zero123plus.encode_prompt("", self.device, 1, False)[0]
            ramp = global_embeds.new_tensor(self.zero123plus.config.ramping_coefficients).unsqueeze(-1)
            cond_encoder_hidden_states = text_embeds + global_embeds * ramp
            
            # JA: Get unconditional text embeddings
            uncond_embeds = self.zero123plus.encode_prompt("", self.device, 1, True)[1]
            
            # JA: Concatenate for classifier-free guidance
            encoder_hidden_states = torch.cat([uncond_embeds, cond_encoder_hidden_states])
            clean_cond_lat = torch.cat([negative_lat, cond_lat])

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

        # sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        # sigmas = torch.cat([torch.flip(sigmas, dims=[0]), torch.zeros(1).to(sigmas.device)])

        iterations = 15000
        ikl_running_avg = None

        import wandb
        run = wandb.init(
            project="ConTEXTure-NeRF"
        )

        dreamtime_scheduler = DreamTimeScheduler(alphas_cumprod, iterations, m=500, s=125)

        # --- 3. MAIN SDS OPTIMIZATION LOOP ---
        with tqdm(range(iterations), desc='SDS Texture Optimization') as pbar:
            for i in pbar:
                # Sample a random timestep for each iteration
                # t = torch.randint(0, num_timesteps, (1,), device=self.device).long()
                # t_max_start = 980  # Start with high-noise timesteps (coarse details).
                # t_max_end = 50     # End with low-noise timesteps (fine details).
                # annealing_period = 7500 # Number of iterations to perform the annealing over.

                # # Calculate the current progress through the annealing period.
                # progress = min(i / annealing_period, 1.0)
                
                # # Linearly interpolate the max timestep.
                # current_t_max = int(t_max_start * (1 - progress) + t_max_end * progress)
                
                # # Sample a random timestep from the annealed range.
                # t = torch.randint(t_max_end, current_t_max + 1, (1,), device=self.device).long()

                t_int = dreamtime_scheduler.get_t(i)
                t = torch.tensor([t_int], device=self.device) # Ensure t is a tensor

                # t = timesteps[t]

                optimizer.zero_grad()

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
                
                rendered_grid_clean = torch.cat((
                    torch.cat((cropped_renders_small_list[0], cropped_renders_small_list[3]), dim=3),
                    torch.cat((cropped_renders_small_list[1], cropped_renders_small_list[4]), dim=3),
                    torch.cat((cropped_renders_small_list[2], cropped_renders_small_list[5]), dim=3),
                ), dim=2) #MJ: The final tensor rendered_grid_clean will have a shape corresponding to a single image that contains a 3x2 grid of the original images

                rendered_grid_clean = rendered_grid_clean * 2 - 1
                rendered_grid_clean = scale_image(rendered_grid_clean)

                latents_clean = vae.encode(rendered_grid_clean.to(vae.dtype)).latent_dist.sample() #MJ: z0 = latents_clean
                latents_clean = latents_clean * vae.config.scaling_factor

                scaled_latents_clean = scale_latents(latents_clean)

                alpha_cumprod_t = alphas_cumprod[t.cpu().long()].to(scaled_latents_clean.device)
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t).reshape(1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alpha_cumprod_t).reshape(1, 1, 1, 1)

                with torch.no_grad():
                    noise = torch.randn_like(scaled_latents_clean)
                    # sigma = sigmas[t.cpu().long()]

                    # JA: Forward diffusion: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
                    latents_noisy = sqrt_alpha_cumprod_t * scaled_latents_clean + sqrt_one_minus_alpha_cumprod_t * noise #MJ: zt = latents_noisy
                    latents_noisy = latents_noisy.half() #MJ: zero123++ is trained using latents with half precision

                    latent_model_input = torch.cat([latents_noisy] * 2)
                    latent_model_input = self.zero123plus.scheduler.scale_model_input(latent_model_input, t)

                    # latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
                    latent_model_input = latent_model_input.half()

                    # JA: Prepare cross-attention kwargs, which is how Zero123++ pipeline passes conditioning
                    down_block_res_samples, mid_block_res_sample = unet.controlnet(
                        latent_model_input, t, #MJ: latent_model_input = torch.cat([latents_noisy] * 2)
                        encoder_hidden_states=encoder_hidden_states, 
                        #MJ: encoder_hidden_states = torch.cat([uncond_embeds, cond_encoder_hidden_states])
                        #  uncond_embeds = self.zero123plus.encode_prompt("", self.device, 1, True)[1]
           
                        controlnet_cond=depth_tensor,
                        conditioning_scale=2,
                        return_dict=False,
                    )
                    
                    noise_cond = torch.randn_like(clean_cond_lat)
                    noisy_cond_lat = sqrt_alpha_cumprod_t * clean_cond_lat + sqrt_one_minus_alpha_cumprod_t * noise_cond
                    noisy_cond_lat = self.zero123plus.scheduler.scale_model_input(noisy_cond_lat, t)

                    # noisy_cond_lat = noisy_cond_lat / ((sigma ** 2 + 1) ** 0.5)
                    noisy_cond_lat = noisy_cond_lat.half()

                    # JA: The Zero123++ pipeline follows a unique approach of performing two forward passes through the UNet
                    # in a single denoising step: one to extract features from a reference image, and a second to apply those
                    # features to the image being generated
                    #
                    # unet = DepthControlUNet
                    # unet.unet = RefOnlyNoisedUNet (custom wrapper in the Zero123++ pipeline for streamlining this approach)
                    # unet.unet.unet = UNet2DConditionModel
                    #
                    # Note: We call the forward function of the UNet2DConditionModel instead of the forward function of
                    # RefOnlyNoisedNet because the noise adding process is included within it.

                    ref_dict = {}
                    unet.unet.unet(
                        noisy_cond_lat, t, #MJ:  noisy_cond_lat from  clean_cond_lat = torch.cat([negative_lat, cond_lat]); noisy_cond_lat = sample_t
                        encoder_hidden_states=encoder_hidden_states, 
                        # MJ: #MJ: encoder_hidden_states = torch.cat([uncond_embeds, cond_encoder_hidden_states])
                        #  uncond_embeds = self.zero123plus.encode_prompt("", self.device, 1, True)[1]
                        # 
                        class_labels=None,
                        cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict), 
                        #MJ: In “write” mode, the processor runs each cross-attn layer and mutates the Python dict you gave (e.g., ref_dict[layer_id] = (K_ref, V_ref)). 
                        return_dict=False #MJ: the return type of the Unet call (tuple vs. a dataclass)
                    )

                    weight_dtype = unet.unet.unet.dtype

                    v_pred = unet.unet.unet(
                        latent_model_input, t, #MJ: latent_model_input = torch.cat([latents_noisy] * 2)
                        encoder_hidden_states=encoder_hidden_states, 
                        #MJ: encoder_hidden_states = torch.cat([uncond_embeds, cond_encoder_hidden_states])
                        #  uncond_embeds = self.zero123plus.encode_prompt("", self.device, 1, True)[1]
                        #
                        
                        cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=False),
                        
                        return_dict=False,
                        
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ] if down_block_res_samples is not None else None,
                        
                        mid_block_additional_residual=(
                            mid_block_res_sample.to(dtype=weight_dtype)
                            if mid_block_res_sample is not None else None
                        ),
                    )[0]
                
                    # Perform guidance
                    v_pred_uncond, v_pred_text = v_pred.chunk(2)
                    #MJ:  (torch.cat([latents_noisy]*2)) =>  Both v_pred_uncond and v_pred_text will have the values you expect?
                    # I ask this question, because  you set is_cfg_guidance=False). 
                    # I would guess that  the “uncond” branch is not truly unconditional; 
                    # Check what happens when is_cfg_guidance=True (in the first call of unet) and False (in the second call of the unet)
                    guidance_scale = 10
                    v_pred = v_pred_uncond + guidance_scale * (v_pred_text - v_pred_uncond)

                # JA: Calculate SDS loss gradient
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t.cpu().long()]).to(self.device).reshape(-1, 1, 1, 1)
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t.cpu().long()]).to(self.device).reshape(-1, 1, 1, 1)
                
                v = sqrt_alphas_cumprod * noise - sqrt_one_minus_alphas_cumprod * scaled_latents_clean #MJ: v = alpha_t * eps - sigma_t*z0
                with torch.no_grad():
                    #MJ: with torch.no_grad():
                    # Disables Gradient Tracking: When you enter the with torch.no_grad(): block, 
                    # PyTorch stops building the computational graph. It doesn't track the history of operations. 
                    # This means that when you later call loss.backward(), the gradients won't be calculated for any tensors,
                    # e.g., fisher_divergence_t,  created inside the no_grad block.
                    
                    # Calculate the Fisher Divergence, which is the squared distance between the scores.
                    # For v-prediction, the score difference is (v_pred - v) / (alpha_t * sigma_t)
                    # alpha_t = sqrt_alphas_cumprod
                    # sigma_t = sqrt_one_minus_alphas_cumprod
                    
                    # Avoid division by zero at t=0
                    sigma_t = sqrt_one_minus_alphas_cumprod.clamp(min=1e-8)
                    alpha_t = sqrt_alphas_cumprod.clamp(min=1e-8)

                    # divergence_at_t = torch.sum(((v_pred.detach() - v.detach()) / (alpha_t * sigma_t)) ** 2)
                    fisher_divergence_t = torch.sum((alpha_t / sigma_t) ** 2 * torch.abs(v_pred - v) ** 2)

                    # Update the running average (Exponential Moving Average)
                    if ikl_running_avg is None:
                        ikl_running_avg = fisher_divergence_t.item()
                    else:
                        beta = 0.99  # Smoothing factor
                        ikl_running_avg = beta * ikl_running_avg + (1 - beta) * fisher_divergence_t.item()

                grad_scale = 1
                w = (1 - alphas_cumprod[t.cpu().long()])
                grad = grad_scale * w[:, None, None, None] * sqrt_alphas_cumprod * (v_pred - v)
                # grad = grad_scale * w[:, None, None, None] * (v_pred - v_target)  #MJ: (eps_pred - eps): score_t = - eps/sigma_t; score_t = -xt - alpha_t/sigma_t *v_hat(xt,t)
                #MJ: grad = grad_scale * w[:, None, None, None] * (v_pred - v) #: eps_pred-  eps = alpha_t * (v_pred -v)
                grad = torch.nan_to_num(grad)

                # 2. Define the target gradient
                targets = (scaled_latents_clean - grad).float().detach()

                # 3. Calculate the MSE loss: dloss/dtheta
                sds_loss = 0.5 * F.mse_loss( # [B, C, 120, 80]
                    # scaled_latents_clean[:, :, :40, :40].float(), #MJ: z0 = scaled_latents_clean
                    # targets[:, :, :40, :40],                       #MJ: [z0 * (z0 - grad)]^2 => dloss/dtheta = (eps_pred- eps)*dz0/dtheta
                    scaled_latents_clean.float(),
                    targets,
                    reduction='sum'
                ) / scaled_latents_clean.shape[0]

                consistency_reward = 0#self.compute_view_consistency(
                #     rendered_six_views_clean,
                #     self.mesh_model.mesh.faces,
                #     render_cache['face_idx'][1:],
                #     render_cache['face_vertices_image'][1:]
                # )

                loss = sds_loss - 500 * consistency_reward
                # print(f"SDS: {sds_loss:.2f}, VC: {vc_loss:.2f}")

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.mesh_model.texture_mlp.parameters(), 1.0)

                grad_vector = torch.nn.utils.parameters_to_vector(
                    p.grad for p in self.mesh_model.texture_mlp.parameters() if p.grad is not None
                )

                grad_norm = torch.linalg.norm(grad_vector)

                wandb.log({
                    "grad_norm": grad_norm,
                    "fisher_divergence_t": fisher_divergence_t,
                    "ikl_running_avg": ikl_running_avg,
                    "sds_loss": sds_loss,
                    "consistency_reward": consistency_reward,
                    "t": t
                })

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