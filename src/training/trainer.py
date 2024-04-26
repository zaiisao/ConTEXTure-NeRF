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

import torchvision
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import Zero123PlusDataset, ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy, pad_tensor_to_size, split_zero123plus_grid


class TEXTure:
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

        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom'] # self.view_dirs[dir] when dir = [4] = [right]
        self.mesh_model = self.init_mesh_model()
        self.diffusion = self.init_diffusion()
        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        self.zero123_front_input = None

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_mesh_model(self) -> nn.Module:
        # fovyangle = np.pi / 6 if self.cfg.guide.use_zero123plus else np.pi / 3
        fovyangle = np.pi / 3
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
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

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            for d in self.view_dirs:
                text = ref_text.format(d)
                if d != 'front':
                    text = "" # JA: For all non-frontal views, we wish to use a null string prompt
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
        if self.cfg.guide.use_zero123plus:
            self.paint_zero123plus()
        else:
            self.paint_legacy()

    def paint_zero123plus(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        render_outputs = []
        viewpoint_data = []
        depths_rgba = []

        background = torch.Tensor([0.5, 0.5, 0.5]).to(self.device)

        front_image = None

        for i, data in enumerate(self.dataloaders['train']):
            if i == 0:
                # JA: The first viewpoint should always be frontal. It creates the extended version of the cropped
                # front view image.
                rgb_output_front, object_mask_front = self.paint_viewpoint(data, should_project_back=False)

                # JA: The object mask is multiplied by the output to erase any generated part of the image that
                # "leaks" outside the boundary of the mesh from the front viewpoint. This operation turns the
                # background black, but we would like to use a white background, which is why we set the inverse 
                # of the mask to a ones tensor (the tensor is normalized to be between 0 and 1).
                front_image = rgb_output_front * object_mask_front \
                    + torch.ones_like(rgb_output_front, device=self.device) * (1 - object_mask_front)

            # JA: Even though the legacy function calls self.mesh_model.render for a similar purpose as for what
            # we do below, we still do the rendering again for the front viewpoint outside of the function for
            # the sake of brevity.

            # JA: Similar to what the original paint_viewpoint function does, we save all render outputs, that is,
            # the results from the renderer. In paint_viewpoint, rendering happened at the start of each viewpoint
            # and the images were generated using the depth/inpainting pipeline, one by one.

            theta, phi, radius = data['theta'], data['phi'], data['radius']
            phi = phi - np.deg2rad(self.cfg.render.front_offset)    # JA: The front azimuth angles of some meshes are
                                                                    # not 0. In that case, we need to add the front
                                                                    # azimuth offset
            phi = float(phi + 2 * np.pi if phi < 0 else phi) # JA: We convert negative phi angles to positive

            outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
            render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)

            outputs = self.mesh_model.render(
                background=torch.Tensor([0.5, 0.5, 0.5]).to(self.device),
                render_cache=render_cache,
                use_median=True
            )

            render_outputs.append(outputs)
            viewpoint_data.append({
                "render_outputs": outputs,
                "update_mask": outputs["mask"]
            })
        # END for i, data in enumerate(self.dataloaders['train'])

        for data in viewpoint_data:
            render_outputs = data["render_outputs"]

            # JA: In the depth controlled Zero123++ code example, the test depth map is found here:
            # https://d.skis.ltd/nrp/sample-data/0_depth.png
            # As it can be seen here, the foreground is closer to 0 (black) and background closer to 1 (white).
            # This is opposite of the SD 2.0 pipeline and the TEXTure internal renderer and must be inverted
            # (i.e. 1 minus the depth map, since the depth map is normalized to be between 0 and 1)
            depth = 1 - render_outputs['depth']
            mask = render_outputs['mask']

            # JA: The generated depth only has one channel, but the Zero123++ pipeline requires an RGBA image.
            # The mask is the object mask, such that the background has value of 0 and the foreground a value of 1.
            depth_rgba = torch.cat((depth, depth, depth, mask), dim=1)
            depths_rgba.append(depth_rgba)

        zero123plus_cond = pad_tensor_to_size(self.zero123_front_input[0], 1200, 1200, value=1) # JA: pad the front view image with ones so that the resulting image will be 1200x1200. This makes the background white
                                                                                                # This is new code. zero123_front_input is the cropped version. In control zero123 pipeline, zero123_front_input is used without padding

        # JA: depths_rgba is a list that arranges the rows of the depth map, row by row
        # These depths are not cropped versions
        depth_grid = torch.cat((
            torch.cat((depths_rgba[1], depths_rgba[4]), dim=3),
            torch.cat((depths_rgba[2], depths_rgba[5]), dim=3),
            torch.cat((depths_rgba[3], depths_rgba[6]), dim=3),
        ), dim=2)

        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        )
        pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
        ), conditioning_scale=2)
        # Feel free to tune the scheduler
        # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        #     pipeline.scheduler.config, timestep_spacing='trailing'
        # )
        pipeline.to(self.device)
        # Run the pipeline

        # JA: Zero123++ was trained with 320x320 images: https://github.com/SUDO-AI-3D/zero123plus/issues/70
        cond_image = torchvision.transforms.functional.to_pil_image(zero123plus_cond).resize((320, 320))

        # JA: From: https://pytorch.org/vision/main/generated/torchvision.transforms.ToPILImage.html
        # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image
        # while adjusting the value range depending on the mode.
        # From: https://www.geeksforgeeks.org/python-pil-image-resize-method/
        # Parameters: 
        # size – The requested size in pixels, as a 2-tuple: (width, height).
        depth_image = torchvision.transforms.functional.to_pil_image(depth_grid[0]).resize((640, 960))

        # @torch.enable_grad
        # def on_step_end(pipeline, i, t, callback_kwargs):
            # grid_latent = callback_kwargs["latents"]

            # check_mask_iters = 0.5

            # latents = split_zero123plus_grid(grid_latent, 320 // pipeline.vae_scale_factor)
            # blended_latents = []
            # rgb_outputs = []

            # for viewpoint_index, data in enumerate(self.dataloaders['train']):
            #     if viewpoint_index == 0:
            #         continue

            #     theta, phi, radius = data['theta'], data['phi'], data['radius']
            #     phi = phi - np.deg2rad(self.cfg.render.front_offset)
            #     phi = float(phi + 2 * np.pi if phi < 0 else phi)

            #     outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
            #     render_cache = outputs['render_cache']
            #     outputs = self.mesh_model.render(
            #         background=torch.Tensor([0.5, 0.5, 0.5]).to(self.device),
            #         render_cache=render_cache,
            #         use_median=True
            #     )

            #     rgb_render = outputs['image']

            #     image_row_index = (viewpoint_index - 1) % 3
            #     image_col_index = (viewpoint_index - 1) // 3

            #     latent = latents[image_row_index][image_col_index]

            #     update_mask = viewpoint_data[viewpoint_index - 1]["update_mask"]
            #     refine_mask = viewpoint_data[viewpoint_index - 1]["refine_mask"]
            #     generate_mask = viewpoint_data[viewpoint_index - 1]["generate_mask"]

            #     check_mask = self.generate_checkerboard(update_mask, refine_mask, generate_mask)

            #     rgb_render_small = F.interpolate(rgb_render, (320, 320), mode='bilinear', align_corners=False)
            #     gt_latents = pipeline.vae.encode(
            #         rgb_render_small.half(),
            #         return_dict=False
            #     )[0].sample() * pipeline.vae.config.scaling_factor
            #     noise = torch.randn_like(gt_latents)
            #     noised_truth = pipeline.scheduler.add_noise(gt_latents, noise, t[None])

            #     if check_mask is not None and i < int(pipeline.num_timesteps * check_mask_iters):
            #         curr_mask = F.interpolate(
            #             check_mask,
            #             (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
            #             mode='nearest'
            #         )
            #     else:
            #         curr_mask = F.interpolate(
            #             update_mask,
            #             (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
            #             mode='nearest'
            #         )

            #     def scale_latents(latents):
            #         latents = (latents - 0.22) * 0.75
            #         return latents

            #     blended_latent = latent * curr_mask + scale_latents(noised_truth) * (1 - curr_mask) # JA: latent will be unscaled after generation. To make noised_truth unscaled as well, we scale them.
            #     blended_latents.append(blended_latent) # blended_latent = latent * curr_mask + noised_truth * (1 - curr_mask)
            
            # callback_kwargs["latents"] = torch.cat((
            #     torch.cat((blended_latents[0], blended_latents[3]), dim=3),
            #     torch.cat((blended_latents[1], blended_latents[4]), dim=3),
            #     torch.cat((blended_latents[2], blended_latents[5]), dim=3),
            # ), dim=2).half()

            # return callback_kwargs

        def on_step_end_project_back(pipeline, i, t, callback_kwargs):
            grid_latent = callback_kwargs["latents"]

            check_mask_iters = 0.5

            latents = split_zero123plus_grid(grid_latent, 320 // pipeline.vae_scale_factor)
            blended_latents = []
            rgb_outputs = []

            thetas, phis, radii = [], [], []

            for viewpoint_index, data in enumerate(self.dataloaders['train']):
                if viewpoint_index == 0:
                    continue

                theta, phi, radius = data['theta'], data['phi'], data['radius']
                phi = phi - np.deg2rad(self.cfg.render.front_offset)
                phi = float(phi + 2 * np.pi if phi < 0 else phi)

                thetas.append(theta)
                phis.append(phi)
                radii.append(radius)

            # JA: The following render uses the texture atlas inferred by the previous invocation of project_back
            outputs = self.mesh_model.render(theta=thetas, phi=phis, radius=radii, background=background)
            render_cache = outputs['render_cache']
            # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
            outputs = self.mesh_model.render(
                background=torch.Tensor([0.5, 0.5, 0.5]).to(self.device),
                render_cache=render_cache,
                use_median=i > 0
            )

            object_masks = outputs['mask']
            # rgb_render = outputs['image']
            # Render meta texture map
            # meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
            #                                     use_meta_texture=True, render_cache=render_cache)
            # z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
            # z_normals_cache = meta_output['image'].clamp(0, 1)

            for viewpoint_index, data in enumerate(self.dataloaders['train']):
                if viewpoint_index == 0:
                    continue

                image_row_index = (viewpoint_index - 1) % 3
                image_col_index = (viewpoint_index - 1) // 3

                latent = latents[image_row_index][image_col_index]

                # object_mask = outputs['mask'][viewpoint_index] # JA: mask has a shape of 1200x1200
                # rgb_render = outputs['image'][viewpoint_index - 1][None] # JA: rgb_render is needed for blending the latent image with the ground truth rendered image
                # rgb_render_small = F.interpolate(rgb_render, (320, 320), mode='bilinear', align_corners=False)
                # gt_latents = pipeline.vae.encode(
                #     rgb_render_small.half(),
                #     return_dict=False
                # )[0].sample() * pipeline.vae.config.scaling_factor
                # noise = torch.randn_like(gt_latents)
                # noised_truth = pipeline.scheduler.add_noise(gt_latents, noise, t[None])

                # update_mask = viewpoint_data[viewpoint_index - 1]["update_mask"] # viewpoint_data[viewpoint_index - 1]["render_"]
                # refine_mask = viewpoint_data[viewpoint_index - 1]["refine_mask"]
                # generate_mask = viewpoint_data[viewpoint_index - 1]["generate_mask"]
                # check_mask = self.generate_checkerboard(update_mask, refine_mask, generate_mask)

                # if check_mask is not None and i < int(pipeline.num_timesteps * check_mask_iters):
                #     curr_mask = F.interpolate(
                #         check_mask,
                #         (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
                #         mode='nearest'
                #     )
                # else:
                #     curr_mask = F.interpolate(
                #         update_mask,
                #         (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
                #         mode='nearest'
                #     )

                # curr_mask = F.interpolate(
                #     update_mask,
                #     (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
                #     mode='nearest'
                # )

                def scale_latents(latents):
                    latents = (latents - 0.22) * 0.75
                    return latents

                blended_latent = latent #* curr_mask + scale_latents(noised_truth) * (1 - curr_mask)
                blended_latents.append(blended_latent)

                rgb_output = F.interpolate(
                    pipeline.vae.decode(blended_latent.half() / pipeline.vae.config.scaling_factor, return_dict=False)[0],
                    (1200, 1200),
                    mode='bilinear',
                    align_corners=False
                )

                rgb_outputs.append(rgb_output)

            # end of train dataloader for loop

            # num_epochs = num_epochs = int(200 * (i + 1) / 36)

            # update_masks = torch.cat([
            #     viewpoint_data[viewpoint_index]["update_mask"] for viewpoint_index in range(len(viewpoint_data))
            # ])

            # self.project_back(
            #     render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
            #     object_mask=object_masks, update_mask=update_masks, z_normals=z_normals,
            #     z_normals_cache=z_normals_cache
            # )
            self.project_back(
                render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
                object_mask=object_masks, update_mask=object_masks, z_normals=None, z_normals_cache=None
            )
            
            callback_kwargs["latents"] = torch.cat((
                torch.cat((blended_latents[0], blended_latents[3]), dim=3),
                torch.cat((blended_latents[1], blended_latents[4]), dim=3),
                torch.cat((blended_latents[2], blended_latents[5]), dim=3),
            ), dim=2).half()

            return callback_kwargs
        # end of on_step_end_project_back

        # JA: Here we call the Zero123++ pipeline
        result = pipeline(
            cond_image,
            depth_image=depth_image,
            num_inference_steps=36#,
            # callback_on_step_end=on_step_end_project_back
        ).images[0]

        grid_image = torchvision.transforms.functional.pil_to_tensor(result).to(self.device).float() / 255

        images = split_zero123plus_grid(grid_image, 320)

        thetas, phis, radii = [], [], []
        update_masks = []
        rgb_outputs = []
        for i, data in enumerate(self.dataloaders['train']):
            if i == 0:
                image = front_image
            else:
                image_row_index = (i - 1) % 3
                image_col_index = (i - 1) // 3

                image = images[image_row_index][image_col_index][None]

            rgb_output = F.interpolate(image, (1200, 1200), mode='bilinear', align_corners=False)
            rgb_outputs.append(rgb_output)

            theta, phi, radius = data['theta'], data['phi'], data['radius']
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)

            thetas.append(theta)
            phis.append(phi)
            radii.append(radius)

            # JA: Create trimap of keep, refine, and generate using the render output
            update_masks.append(viewpoint_data[i]["update_mask"])

        outputs = self.mesh_model.render(theta=thetas, phi=phis, radius=radii, background=background)

        render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                        render_cache=render_cache, use_median=True)

        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                            use_meta_texture=True, render_cache=render_cache)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1) # JA: Get the Z component of the face normal vectors relative to the camera
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200

        # fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
        #                                     object_mask=object_mask, update_mask=update_masks, z_normals=z_normals,
        #                                     z_normals_cache=z_normals_cache)

        self.project_back(
            render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
            object_mask=object_mask, update_mask=object_mask, z_normals=z_normals, z_normals_cache=z_normals_cache
        )

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    def paint_legacy(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # JA: The following loop computes the texture atlas for the given mesh using ten render images. In other words,
        # it is the inverse rendering process. Each of the ten views is one of the six view images.
        for data in self.dataloaders['train']:
            self.paint_step += 1
            pbar.update(1)
            self.paint_viewpoint(data) # JA: paint_viewpoint computes the part of the texture atlas by using a specific view image
            self.evaluate(self.dataloaders['val'], self.eval_renders_path)  # JA: This is the validation step for the current
                                                                            # training step
            self.mesh_model.train() # JA: Set the model to train mode because the self.evaluate sets the model to eval mode.

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

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

            logger.info(f"\tDone!")

    # JA: paint_viewpoint computes a portion of the texture atlas for the given viewpoint
    def paint_viewpoint(self, data: Dict[str, Any], should_project_back=True):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius'] # JA: data represents a viewpoint which is stored in the dataset
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if  True: #self.cfg.guide.second_model_type in ["zero123", "control_zero123"]: #self.view_dirs[data['dir']] != "front":
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
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        self.log_train_image(rgb_render, 'rendered_input')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs] # JA: dirs is one of the six directions. text_z is the embedding vector of the specific view prompt
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')

        # JA: Create trimap of keep, refine, and generate using the render output
        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])

        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return

        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
        self.log_train_image(rgb_render * refine_mask, name='refine_regions')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render) # JA: This is rendered image which is denoted as Q_0.
                                              # In our experiment, 1200 is cropped to 827
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
        self.log_train_image(cropped_rgb_render, name='cropped_input')
        self.log_train_image(cropped_depth_render.repeat_interleave(3, dim=1), name='cropped_depth')

        checker_mask = None
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            # JA: generate_checkerboard is defined in formula 2 of the paper
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
                                 'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        # JA: self.zero123_front_input has been added for Zero123 integration
        if self.zero123_front_input is None:
            resized_zero123_front_input = None
        else: # JA: Even though zero123 front input is fixed, it will be resized to the rendered image of each viewpoint other than the front view
            resized_zero123_front_input = F.interpolate(
                self.zero123_front_input,
                (cropped_rgb_render.shape[-2], cropped_rgb_render.shape[-1]) # JA: (H, W)
            )

        condition_guidance_scales = None
        if self.cfg.guide.individual_control_of_conditions:
            if self.cfg.guide.second_model_type != "control_zero123":
                raise NotImplementedError

            assert self.cfg.guide.guidance_scale_i is not None
            assert self.cfg.guide.guidance_scale_t is not None

            condition_guidance_scales = {
                "i": self.cfg.guide.guidance_scale_i,
                "t": self.cfg.guide.guidance_scale_t
            }

        # JA: Compute target image corresponding to the specific viewpoint, i.e. front, left, right etc. image
        # In the original implementation of TEXTure, the view direction information is contained in text_z. In
        # the new version, text_z 
        # D_t (depth map) = cropped_depth_render, Q_t (rendered image) = cropped_rgb_render.
        # Trimap is defined by update_mask and checker_mask. cropped_rgb_output refers to the result of the
        # Modified Diffusion Process.

        # JA: So far, the render image was created. Now we generate the image using the SD pipeline
        # Our pipeline uses the rendered image in the process of generating the image.
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(), # JA: We use the cropped rgb output as the input for the depth pipeline
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=cropped_update_mask,
                                                                    fixed_seed=self.cfg.optim.seed,
                                                                    check_mask=checker_mask,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps,

                                                                    # JA: The following were added to use the view image
                                                                    # created by Zero123
                                                                    view_dir=self.view_dirs[dirs], # JA: view_dir = "left", this is used to check if the view direction is front
                                                                    front_image=resized_zero123_front_input,
                                                                    phi=data['phi'],
                                                                    theta=data['base_theta'] - data['theta'],
                                                                    condition_guidance_scales=condition_guidance_scales)

        self.log_train_image(cropped_rgb_output, name='direct_output')
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
            fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                                z_normals_cache=z_normals_cache)
            self.log_train_image(fitted_pred_rgb, name='fitted')

        # JA: Zero123 needs the input image without the background
        # rgb_output is the generated and uncropped image in pixel space
        zero123_input = crop(
            rgb_output * object_mask
            + torch.ones_like(rgb_output, device=self.device) * (1 - object_mask)
        )   # JA: In the case of front view, the shape is (930,930).
            # This rendered image will be compressed to the shape of (512, 512) which is the shape of the diffusion
            # model.

        if self.view_dirs[dirs] == "front":
            self.zero123_front_input = zero123_input
        
        # if self.zero123_inputs is None:
        #     self.zero123_inputs = []
        
        # self.zero123_inputs.append({
        #     'image': zero123_input,
        #     'phi': data['phi'],
        #     'theta': data['theta']
        # })

        self.log_train_image(zero123_input, name='zero123_input')

        return rgb_output, object_mask

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask

        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(update_mask)
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        if self.cfg.guide.initial_texture is None:
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
            refine_mask[z_normals < 0.4] = 0
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0

        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask

            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'shaded_input')
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask

    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard,
                                     (512, 512))
        checker_mask = F.interpolate(update_mask_inner, (512, 512))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (512, 512))
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        eroded_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            eroded_mask = cv2.erode(object_mask[i, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
            eroded_masks.append(torch.from_numpy(eroded_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        eroded_object_mask = torch.cat(eroded_masks, dim=0)
        # object_mask = torch.from_numpy(
        #     cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
        #     object_mask.device).unsqueeze(0).unsqueeze(0)
        # render_update_mask = object_mask.clone()
        render_update_mask = eroded_object_mask.clone()

        # render_update_mask[update_mask == 0] = 0
        render_update_mask[update_mask == 0] = 0

        # blurred_render_update_mask = torch.from_numpy(
        #     cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
        #     render_update_mask.device).unsqueeze(0).unsqueeze(0)
        dilated_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            dilated_mask = cv2.dilate(render_update_mask[i, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))
            dilated_masks.append(torch.from_numpy(dilated_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        blurred_render_update_mask = torch.cat(dilated_masks, dim=0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            if z_normals is not None and z_normals_cache is not None:
                z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
                blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask
        for i in range(rgb_output.shape[0]):
            self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back_input_{i}')

        # Update the normals
        if z_normals is not None and z_normals_cache is not None:
            z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            if z_normals is not None and z_normals_cache is not None:
                meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                    use_meta_texture=True, render_cache=render_cache)
                current_z_normals = meta_outputs['image']
                current_z_mask = meta_outputs['mask'].flatten()
                masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                        current_z_mask == 1][:, :1]
                masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                        current_z_mask == 1][:, :1]
                loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            # losses.append(loss.cpu().detach().numpy())
            loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                            # the network, that is, the pixel value of the texture atlas
            optimizer.step()

        if z_normals is not None and z_normals_cache is not None:
            return rgb_render, current_z_normals
        else:
            return rgb_render
        
    def project_back_only_texture_atlas(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        eroded_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            eroded_mask = cv2.erode(object_mask[i, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
            eroded_masks.append(torch.from_numpy(eroded_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        eroded_object_mask = torch.cat(eroded_masks, dim=0)
        render_update_mask = eroded_object_mask.clone()
        render_update_mask[update_mask == 0] = 0

        dilated_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            dilated_mask = cv2.dilate(render_update_mask[i, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))
            dilated_masks.append(torch.from_numpy(dilated_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        blurred_render_update_mask = torch.cat(dilated_masks, dim=0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0

        render_update_mask = blurred_render_update_mask
        for i in range(rgb_output.shape[0]):
            self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back_input_{i}')

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
            rgb_render = outputs['image']

            # loss = (render_update_mask * (rgb_render - rgb_output.detach()).pow(2)).mean()
            loss = (render_update_mask * z_normals * (rgb_render - rgb_output.detach()).pow(2)).mean()

            loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                            # the network, that is, the pixel value of the texture atlas
            optimizer.step()

        return rgb_render

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')

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
