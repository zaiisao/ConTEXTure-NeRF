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

        if self.cfg.guide.use_zero123plus:
            self.zero123plus = self.init_zero123plus()

        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        self.zero123_front_input = None

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')
        
         
       #MJ: Create the mask for each view based on the value of the z-normals
        
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
        
        background_type = 'none'
        use_render_back = False
        batch_size = 7
        # JA: We need to repeat several tensors to support the batch size.
        # For example, with a tensor of the shape, [1, 3, 1200, 1200], doing
        # repeat(batch_size, 1, 1, 1) results in [1 * batch_size, 3 * 1, 1200 * 1, 1200 * 1]
        mask, depth_map, normals_image,face_normals,face_idx = self.mesh_model.render_face_normals_face_idx(
            augmented_vertices[None].repeat(batch_size, 1, 1),
            self.mesh_model.mesh.faces, # JA: the faces tensor can be shared across the batch and does not require its own batch dimension.
            self.mesh_model.face_attributes.repeat(batch_size, 1, 1, 1),
            elev=torch.tensor(self.thetas).to(self.device),
            azim=torch.tensor(self.phis).to(self.device),
            radius=torch.tensor(self.radii).to(self.device),
            look_at_height=self.mesh_model.dy,
         
            # dims=self.cfg.render.train_grid_size, # MJ: 1200,
            background_type=background_type
        )
        
        self.mask = mask #MJ: object_mask of the mesh
        self.depth_map = depth_map
        self.normals_image = normals_image
        self.face_normals = face_normals
        self.face_idx = face_idx
        
        #MJ: get the binary masks for each view which indicates how much the image rendered from each view
        #should contribute to the texture atlas over the mesh which is the cause of the image
        
        #MJL three  versions and compare:
        # self.weight_masks = self.get_weight_masks_for_views_vectorized_over_ij( self.face_normals, self.face_idx )
        # self.weight_masks = self.get_weight_masks_for_views_ij_loop(face_normals, face_idx)
        #self.weight_masks =  self.get_weight_masks_for_views_vectorized(face_normals, face_idx )
        # self.weight_masks = self.get_weight_masks_for_views_loop_maxview1(face_normals, face_idx)
        # self.weight_masks = self.get_weight_masks_for_views_loop_maxview2(face_normals, face_idx)
        
        face_view_map = self.create_face_view_map( face_idx)
        self.weight_masks = self.compare_face_normals_between_views(face_view_map, face_normals, face_idx)
        
            # z_normals[0, 2, i, j] z_normals[1, 2, i, j] 
    #MJ: we will create a dict face_map which has the following:
    # {
    #     face_id_1: {
    #         view_1: [(i, j), (i, j), ...],
    #         view_2: [(i, j), ...],
    #         ...
    #     },
    #     face_id_2: {
    #         ...
    #     }
    # } 
    # (1) For each face_id, list one or more views, view_1, view_2,..., under which the face is projected on the view image;
    # (2) One face is projected onto a set of neighboring pixels (i,j)



    def create_face_view_map(self, face_idx):
        # Initialize a nested dictionary to hold face IDs with sub-dictionaries for views
        
        # Example usage:
        # Assuming face_idx is a tensor of shape (B, 1, H, W)
        # face_view_map = create_face_view_map(face_idx)
      
        face_view_map = {}
        num_views, _, H, W = face_idx.shape  # Assume face_idx shape is (B, 1, H, W)

      
        # Iterate over all views and pixel locations
        for v in range(num_views):
               
            for i in range(H):
                for j in range(W):
                     face_id = face_idx[v, 0, i, j].item()
                    
                     if face_id != -1:  # Only consider valid face IDs
                        if face_id not in face_view_map:
                            face_view_map[face_id] = {}
                        if v not in face_view_map[face_id]: #MJ: face_view_map[face_id] ={} initially
                            face_view_map[face_id][v] = []
                        face_view_map[face_id][v].append((i, j))

        return face_view_map


    def compare_face_normals_between_views(self,face_view_map, face_normals, face_idx):
        num_views, _, H, W = face_idx.shape
      

       
        # Overall Description:
        # Each face  over the whole mesh is projected on pixels (i,j) under each viewpoint v; 
        # We will make the pixels rendered under a given viewpoint contribute to the texture atlas, only when
        # those pixels are those whose z-normals are maximum among the views that cover the given face.
        # In actual implementation, we first initialize so that all pixels in each view are worthy to contribute to the
        # texture altas. Then each pixel in each view v is considered unworthy to contribute to the texture atlas,
        # if the z-component of the normal vector at the face_id of the pixel  is less than that of any other view. 
        
          
        weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool).to(self.device)
        # Create mask with True as default; It means that  all pixels in each view are worthy to contribute to the
        # texture altas; Later, each pixel in each view v is considered unworthy to contribute to the texture atlas, that is,
        #  weight_masks[v, 0, i,j] = False, if the z-component of the normal vector at the face_id of the pixel 
        # is less than that of any other view. 
        
        # Iterate through each face_id and its associated views/locations
        for face_id, view_locs_face_id in face_view_map.items(): #MJ: view_locs_face_id.keys() =dict_keys([0, 1]); face_id;92867;view_locs_face_id[0]=[(105, 586), (105, 587), (105, 588), (105, 589), (105, 590), (105, 591)]
            #MJ: view_locs_face_id is the view-locs info of each face_id of the whole mesh
            #MJ:
            #view_locs  = {
            #         view_1: [(i, j), (i, j), ...],
            #         view_2: [(i, j), ...],
            #         ...
            #     },
            
           
            if len(view_locs_face_id) > 1:  # Only consider face_ids that appear in more than one view
                
                # Find "max_z_normal_face_id", that is, the max among the z-normals of the given face_id projected onto the images undere different viewpoints
                # It is possible that the same face can be viewed under multiple viewpoints or only one viewpoint.
                # It is possible that a face is not seen by any viewpoint, depending on how the viewpoints are set up
                
                # Assume view_locs_face_id is a dictionary where keys represent view indices
                view_indices_face_id_list = list(view_locs_face_id.keys())

                # Now convert the list to a PyTorch tensor
                view_indices_face_id_tensor = torch.tensor(view_indices_face_id_list, dtype=torch.long)
                # Index into the face_normals tensor using the view_indices_tensor
                selected_z_normals_face_id = face_normals[view_indices_face_id_tensor, 2, face_id] #MJ:   selected_z_normals_face_id:shape=[2]; two views

                # Precompute the maximum z-normals per face ID across all views
                max_z_normal_face_id, _ = torch.max(selected_z_normals_face_id, dim=0)     #MJ: max_z_normal_face_id=tensor(0.5180, device='cuda:0') 
                #MJ: When you specify dim=0, it will return two things:
                    # The maximum values across the specified dimension.
                    # The indices of those maximum values.
                                    
                 
                # Update masks for views with z-normals less than the max
                for v,  ij_locs_view in view_locs_face_id.items(): #MJ: v = 0,1,2,3,4,5,6
                    ## Assuming  view_locs_face_id  is a list of tuples [(i1, j1), (i2, j2), ...]
                    # Check if the z_normals of face_id under v1 is
                    # not maximum, the pixels of face_id in view 1 is not considered worthy to contribute to the texture atlas.
                
                    if face_normals[v, 2, face_id] <  max_z_normal_face_id:
                        
                        # Extract all pixel coordinates of locs_v1 and apply advanced indexing
                        rows, cols = zip(*ij_locs_view)
                        rows_tensor = torch.tensor(rows, dtype=torch.long)
                        cols_tensor = torch.tensor(cols, dtype=torch.long)

                        weight_masks[v, 0, rows_tensor, cols_tensor] = False
                        #MJ: The above code is equivalent to the following:
                        # for i1, j1 in locs_v1:
                        #     weight_masks[v1, 0, i1, j1] = False
                            
                       #MJ: Advanced Indexing: Refers to indexing with lists or arrays to access non-contiguous indices or elements. 
                       # For instance:
                       # rows = [0, 1, 2]
                        # cols = [3, 4, 5]
                        # array[rows, cols]=> The result of array[rows, cols] will contain the values at the pairs (0, 3), (1, 4), and (2, 5).
                                        
                    
        return weight_masks
        
         
        
    def get_weight_masks_for_views_ij_loop(self, face_normals, face_idx ):
        
               
        #MJ: face_idx: shape = (B,1,H,W); # (B,H,W) => (B,1, H, W) to have the standard shape (B,C,H,W)
        # face_idx[:,0,:,:] refers to the face_idx from which the pixel (i,j) was projected
        
        #face_normals: shape = (B, 3, num_faces); face_normals[v,:, k] refers to the face normal vectors of face idx = k under view v;
                                   
        #face_normals[v1,:, face_idx]  refers to the face_normals of face_idx[v1,0,i,j] for every (i,j) 
        #MJ: weight_masks[b,0,i,j] indicates whether the face_idx[b,0,i,j] of  pixel (i,j) under view b should contribute to the texture atlas;
        #initialized to True by torch.full( face_idx.shape, True):  
        
        num_of_views = face_idx.shape[0] #MJ: The total num of views: from 0 to 6
        weight_masks = torch.full(face_idx.shape, True, dtype=torch.bool).to(self.device)  # face_idx.shape:  (B,1, H, W)

       # Iterate over each pixel in the HxW grid
        for i in range (face_idx.shape[2]):
            print(i)
            for j in range (face_idx.shape[3]):
                for v1 in range (num_of_views):
                    for v2 in range (num_of_views):
                        if v1 != v2: #Ensure different views are compared
                            #Check if
                            # face_normals[v1,2, face_idx[v1,0,i,j]] >= face_normals[v2,2, face_idx[v2,0]],
                            # where face_idx[v1,0,i,j] = face_idx[v2,0]  
                            # When you write c1[i, j] == c2, you are performing an element-wise comparison between a single value from the tensor c1 at position [i, j])
                            # and each element in the tensor c2 [broadcasting]
                            
                            # Get the face index for the current view and pixel
                            # Extract the index and then unsqueeze to add the dimensions back
                            face_index_v1_ij = face_idx[v1, 0, i, j].unsqueeze(0).unsqueeze(1)
                            # MJ: Or, Use slicing to keep dimensions: face_index_v1_ij = face_idx[v1, 0, i:i+1, j:j+1]
                            
                            face_index_v2 = face_idx[v2, 0] #MJ: face_index_v2 : shape =(H,W)
                            # Check if the face indices are the same and compare their z-normals

                            if (face_idx[v1, 0, i, j] == face_idx[v2, 0]).any():
                                z_normal_v1_ij = face_normals[v1, 2, face_index_v1_ij]
                                z_normal_v2 = face_normals[v2, 2, face_index_v2]
                                        
                            #MJ: matches will have the same shape of face_idx[v2,0] = (H,W)          
                            # If the z-normal of pixel (i,j), with face_idx[v1,0,i,j]  in v1 is less than that of any pixel in any other viewpoint v2 
                            # with the same face_idx, then the pixel (i,j) under v1 is not worthy to contribute to the texture atlas, because some worthy pixel exists in other viewpoints
                                if (torch.ones_like(z_normal_v2) * z_normal_v1_ij[0, 0] < z_normal_v2).any():   #MJ: broadcasting is done?                        
                                    weight_masks[v1, 0, i, j] = False
                                    break #MJ: Exit the loop earlier because we have found a higher z-normal:
                                        # if pixel (i,j) in v1 is not worthy to contribute in comparison with some other pixels in any view v2,
                                        # we do not need to compare (i,j) in v1 to pixels in any other viewpoints than v2; The existence
                                        # of such pixel in one viewpoint v2 is sufficient to make the pixel (i,j) in v1 unworthy:
    

        return weight_masks

    def get_weight_masks_for_views_loop_maxview1(self, face_normals, face_idx ):
        
               
        #MJ: face_idx: shape = (B,1,H,W); 
        # face_idx[:,0,:,:] refers to the face_idx from which the pixel (i,j) was projected
        
        #face_normals: shape = (B, 3, num_faces); face_normals[v,:, k] refers to the face normal vectors of face idx = k under view v;
                                   
        #face_normals[v1,:, face_idx]  refers to the face_normals of face_idx[v1,0,i,j] for every (i,j) 
        #MJ: weight_masks[b,0,i,j] indicates whether the face_idx[b,0,i,j] of  pixel (i,j) under view b should contribute to the texture atlas;
        #initialized to True by torch.full( face_idx.shape, True):  
        
        num_of_views = face_idx.shape[0] #MJ: The total num of views: from 0 to 6
        weight_masks = torch.full(face_idx.shape, True, dtype=torch.bool).to(self.device)  # face_idx.shape:  (B,1, H, W)

       # Iterate over each pixel in the HxW grid
        # for i in range ( face_idx.shape[2] ):
        #   for j in range (face_idx.shape[3]):
        # Iterate over each view
        for v1 in range (num_of_views):
                
            #   for v2 in range (num_of_views):
            #       if v1 != v2: #Ensure different views are compared
            #         #Check if
            #         # face_normals[v1,2, face_idx[v1,0,i,j]] >= face_normals[v2,2, face_idx[v2,0]],
            #         # where face_idx[v1,0,i,j] = face_idx[v2,0]  
            #         # When you write c1[i, j] == c2, you are performing an element-wise comparison between a single value from the tensor c1 at position [i, j])
            #         # and each element in the tensor c2 [broadcasting]
                    
            #         # Get the face index for the current view and pixel
            #         face_index_v1_ij = face_idx[v1, 0, i, j]
            #         face_index_v2 = face_idx[v2, 0]
            #         # Check if the face indices are the same and compare their z-normals
            #         if face_index_v1_ij == face_index_v2: #MJ: broadcasting is done?
            #             z_normal_v1_ij = face_normals[v1, 2, face_index_v1_ij]
            #             z_normal_v2 = face_normals[v2, 2, face_index_v2]
                                
            #         #MJ: matches will have the same shape of face_idx[v2,0] = (H,W)          
            #         # If the z-normal of pixel (i,j), with face_idx[v1,0,i,j]  in v1 is less than that of any pixel in any other viewpoint v2 
            #         # with the same face_idx, then the pixel (i,j) under v1 is not worthy to contribute to the texture atlas, because some worthy pixel exists in other viewpoints
            #             if z_normal_v1_ij <  z_normal_v2:   #MJ: broadcasting is done?                        
            #                weight_masks[v1, 0, i, j] = False
            #                break #MJ: Exit the loop earlier because we have found a higher z-normal:
            #                 # if pixel (i,j) in v1 is not worthy to contribute in comparison with some other pixels in any view v2,
            #                 # we do not need to compare (i,j) in v1 to pixels in any other viewpoints than v2; The existence
            #                 # of such pixel in one viewpoint v2 is sufficient to make the pixel (i,j) in v1 unworthy:
    
            face_index_v1 = face_idx[v1, 0]  # Indices for the current view: face_idx[v1, 0] =(H,W)
            face_index_allviews = face_idx[:, 0] #(7,H,W) => index to face:  # Z-normals for the current view
            z_normal_v1 = face_normals[v1, 2, face_index_v1]
            
            #MJ:[v1, 2, ...] from face_normals to get the z-component for all faces across all 7 views. 
            # The resulting shape after this partial indexing is (num_faces, H,W).
            
            # [:, 0] from face_idx reduces the face_idx tensor from (7, 1, H, W) to (7, H, W)
            # Prepare to compare with all other views
            max_normals = torch.zeros_like(z_normal_v1)
            
            for v2 in range(num_of_views):
                if v1 != v2:
                    z_normal_v2 = face_normals[v2, 2, face_idx[v2, 0]]  # Z-normals for view v2: shape =(1,H,W)
                    # Compare face indices and update max normals
                    same_faces = face_idx[v1, 0] == face_idx[v2, 0] #shape = (1, H, W)
                    max_normals = torch.where(same_faces, torch.max(z_normal_v1, z_normal_v2), max_normals) #If same_face => torch.max(z_normal_v1, z_normal_v2) 
                    #torch.where(condition, input, other, *, out=None) → Tensor:
                    

        # Update mask based on maximum normal comparison
        weight_masks[v1, 0] = z_normal_v1 >= max_normals

        return weight_masks


    def get_weight_masks_for_views_vectorized_over_ij(self, face_normals, face_idx ):
    # Assuming face_idx and face_normals are defined as described:
    # face_idx: shape = (B, 1, H, W); 
    # face_normals: shape = (B, 3, num_faces); 

        num_views = face_idx.shape[0]
        H, W = face_idx.shape[2], face_idx.shape[3]
        weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool).to(self.device)

        # Iterate over each pair of views
        for v1 in range(num_views):
            for v2 in range(num_views):
                if v1 != v2:
                    # Retrieve all normals for current view v1 based on face indices
                    normals_v1 = face_normals[v1, 2, face_idx[v1, 0].long()]  # shape (1, H, W)

                    # Retrieve all normals for view v2, same indexing approach
                    normals_v2 = face_normals[v2, 2, face_idx[v2, 0].long()]  # shape (1, H, W)

                    # Create mask where face_idx are equal and normals_v1 are not greater
                    same_face = face_idx[v1, 0] == face_idx[v2, 0]  # Broadcasting, shape (1, H, W)
                    lower_normal = normals_v1 < normals_v2           # element-wise comparison, shape (1, H, W)

                    # Update weight_masks where the same face has a lower normal in v1 compared to any v2
                    weight_masks[v1, 0] &= ~(same_face & lower_normal)  # Invert and update mask

        return weight_masks

    def get_weight_masks_for_views_vectorized_over_ij_2(self, face_normals, face_idx ):
    # Assuming face_idx and face_normals are defined as described:
    # face_idx: shape = (B, 1, H, W); 
    # face_normals: shape = (B, 3, num_faces); 

        num_views = face_idx.shape[0]
        H, W = face_idx.shape[2], face_idx.shape[3]
        weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool).to(self.device)

        # Iterate over each pair of views
        for v1 in range(num_views):
            for v2 in range(num_views):
                if v1 != v2:
                    
                    # Create a mask where face indices are equal between views v1 and v2
                    match_locations = face_idx[v1] == face_idx[v2]  #MJ: match_locations: shape=(1,H,W)
                   
                    matches = match_locations.any(keepdim=True)
                    
                    # Extract the z-normals for the matched indices
                    faceId_v1 = face_idx[v1, 0]  # (H, W) tensor of face indices for view v1
                    faceId_v2 = face_idx[v2, 0]  # (H, W) tensor of face indices for view v2

                    z_normal_v1 = face_normals[v1, 2, faceId_v1]  # Get z-normals for view v1
                    z_normal_v2 = face_normals[v2, 2, faceId_v2]  # Get z-normals for view v2
                    
                    # Find where the z-normal of v1 is less than z-normal of v2 at matching indices
                    lesser_normals = z_normal_v1 < z_normal_v2
                   
                    # Update weight_masks where the same face has a lower normal in v1 compared to any v2
                    weight_masks[v1] &=  matches & lesser_normals

        return weight_masks



    def get_weight_masks_for_views_vectorized(self, face_normals, face_idx):
        # Assuming initialization as described...

        num_views = face_idx.shape[0]
        H, W = face_idx.shape[2], face_idx.shape[3]
        weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool)

        # Expand face_idx and face_normals for all-to-all view comparisons
        num_faces = face_normals.shape[2]
        expanded_face_idx = face_idx.unsqueeze(0).expand(num_views, num_views, 1, H, W)
        expanded_normals = face_normals[:, 2, :].unsqueeze(0).expand(num_views, num_views, num_faces).gather(2, expanded_face_idx.long())

        # Compute masks
        same_faces = expanded_face_idx == expanded_face_idx.transpose(0, 1)
        lower_normals = expanded_normals < expanded_normals.transpose(0, 1)

        # Apply masks to weight_masks
        update_masks = same_faces & lower_normals
        weight_masks &= ~update_masks.any(dim=1)



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
    
    def init_zero123plus(self) -> DiffusionPipeline:
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        )

        pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
        ), conditioning_scale=2)

        pipeline.to(self.device)

        return pipeline

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

        viewpoint_data = []
        cropped_depths_rgba = []

        # JA: This color is the same as the background color of the image grid generated by Zero123++.
        background = torch.Tensor([0.5, 0.5, 0.5]).to(self.device)
        front_image = None

        max_cropped_image_height, max_cropped_image_width = 0, 0
        cropped_depth_sizes = []

        for i, data in enumerate(self.dataloaders['train']):
            if i == 0:
                # JA: The first viewpoint should always be frontal. It creates the extended version of the cropped
                # front view image.
                rgb_output_front, object_mask_front = self.paint_viewpoint(data, should_project_back=True)

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
                background=background,
                render_cache=render_cache,
                use_median=True
            )

            min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs["mask"][0, 0])
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_update_mask = crop(outputs["mask"])

            viewpoint_data.append({
                "render_outputs": outputs,
                "cropped_update_mask": cropped_update_mask,
                "nonzero_region": {
                    "min_h": min_h, "min_w": min_w,
                    "max_h": max_h, "max_w": max_w
                }
            })

            # JA: In the depth controlled Zero123++ code example, the test depth map is found here:
            # https://d.skis.ltd/nrp/sample-data/0_depth.png
            # As it can be seen here, the foreground is closer to 0 (black) and background closer to 1 (white).
            # This is opposite of the SD 2.0 pipeline and the TEXTure internal renderer and must be inverted
            # (i.e. 1 minus the depth map, since the depth map is normalized to be between 0 and 1)
            depth = 1 - outputs['depth']
            mask = outputs['mask']

            # JA: The generated depth only has one channel, but the Zero123++ pipeline requires an RGBA image.
            # The mask is the object mask, such that the background has value of 0 and the foreground a value of 1.
            depth_rgba = torch.cat((depth, depth, depth, mask), dim=1)

            cropped_depth_rgba = crop(depth_rgba)
            cropped_depth_sizes.append(cropped_depth_rgba.shape[-1])

            max_cropped_image_height = max(max_cropped_image_height, cropped_depth_rgba.shape[-2])
            max_cropped_image_width = max(max_cropped_image_width, cropped_depth_rgba.shape[-1])

            cropped_depths_rgba.append(cropped_depth_rgba)

        min_h, min_w, max_h, max_w = utils.get_nonzero_region(object_mask_front[0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_front_image = crop(front_image)

        should_pad = False

        # JA: We have to pad the depth tensors because cropped images from all viewpoints will vary in size from each
        # other. Without the padding, the tensors will not be concatenatable.
        for i, cropped_depth_rgba in enumerate(cropped_depths_rgba):
            if should_pad:
                cropped_depths_rgba[i] = pad_tensor_to_size(
                    cropped_depth_rgba,
                    max_cropped_image_height,
                    max_cropped_image_width,
                    value=0
                )
            else:
                cropped_depths_rgba[i] = F.interpolate(
                    cropped_depth_rgba,
                    (max_cropped_image_height, max_cropped_image_width),
                    mode='bilinear',
                    align_corners=False
                )

        # JA: cropped_depths_rgba is a list that arranges the rows of the depth map, row by row
        cropped_depth_grid = torch.cat((
            torch.cat((cropped_depths_rgba[1], cropped_depths_rgba[4]), dim=3),
            torch.cat((cropped_depths_rgba[2], cropped_depths_rgba[5]), dim=3),
            torch.cat((cropped_depths_rgba[3], cropped_depths_rgba[6]), dim=3),
        ), dim=2)

        self.log_train_image(cropped_front_image, 'cropped_front_image')
        self.log_train_image(cropped_depth_grid[:, 0:3], 'cropped_depth_grid')

        # JA: From: https://pytorch.org/vision/main/generated/torchvision.transforms.ToPILImage.html
        # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image
        # while adjusting the value range depending on the mode.
        # From: https://www.geeksforgeeks.org/python-pil-image-resize-method/
        # Parameters: 
        # size – The requested size in pixels, as a 2-tuple: (width, height).

        # JA: Zero123++ was trained with 320x320 images: https://github.com/SUDO-AI-3D/zero123plus/issues/70
        cond_image = torchvision.transforms.functional.to_pil_image(cropped_front_image[0]).resize((320, 320))
        depth_image = torchvision.transforms.functional.to_pil_image(cropped_depth_grid[0]).resize((640, 960))

        @torch.enable_grad
        def on_step_end(pipeline, i, t, callback_kwargs):
            grid_latent = callback_kwargs["latents"]

            latents = split_zero123plus_grid(grid_latent, 320 // pipeline.vae_scale_factor)
            blended_latents = []

            for viewpoint_index, data in enumerate(self.dataloaders['train']):
                if viewpoint_index == 0:
                    continue

                theta, phi, radius = data['theta'], data['phi'], data['radius']
                phi = phi - np.deg2rad(self.cfg.render.front_offset)
                phi = float(phi + 2 * np.pi if phi < 0 else phi)

                outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)

                phi = float(phi + 2 * np.pi if phi < 0 else phi)

                min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
                crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]

                if should_pad:
                    cropped_rgb_render_raw = pad_tensor_to_size(
                        crop(outputs['image']),
                        max_cropped_image_height,
                        max_cropped_image_width,
                        value=0.5
                    )
                else:
                    cropped_rgb_render_raw = crop(outputs['image'])

                render_cache = outputs['render_cache']
                outputs = self.mesh_model.render(
                    background=torch.Tensor([0.5, 0.5, 0.5]).to(self.device),
                    render_cache=render_cache,
                    use_median=True
                )

                if should_pad:
                    cropped_rgb_render = pad_tensor_to_size(
                        crop(outputs['image']),
                        max_cropped_image_height,
                        max_cropped_image_width,
                        value=0.5
                    )
                else:
                    cropped_rgb_render = crop(outputs['image'])

                image_row_index = (viewpoint_index - 1) % 3
                image_col_index = (viewpoint_index - 1) // 3

                latent = latents[image_row_index][image_col_index]

                diff = (cropped_rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
                    self.device)).abs().sum(axis=1)
                exact_generate_mask = (diff < 0.1).float().unsqueeze(0)

                # Extend mask
                generate_mask = torch.from_numpy(
                    cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
                    exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

                curr_mask = F.interpolate(
                    generate_mask,
                    (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
                    mode='nearest'
                )

                cropped_rgb_render_small = F.interpolate(
                    cropped_rgb_render,
                    (320, 320),
                    mode='bilinear',
                    align_corners=False
                )

                # JA: When the generated latent tensor is denoised entirely, the Zero123++ pipeline uniquely
                # performs operations in the process of turning the latent space tensor z into pixel space
                # tensor x in the following manner:
                #   x = postprocess(unscale_image(vae_decode(unscale_latents(z) / scaling_factor)))
                # In order to move pixel space tensor x into latent space tensor z, the inverse must be
                # applied in the following manner:
                #   z = scale_latents(vae_encode(scale_image(preprocess(x))) * scaling_factor)

                preprocessed_rgb_render_small = pipeline.image_processor.preprocess(cropped_rgb_render_small)

                scaled_rgb_render_small = scale_image(preprocessed_rgb_render_small.half())
                scaled_latent_render_small = pipeline.vae.encode(
                    scaled_rgb_render_small,
                    return_dict=False
                )[0].sample() * pipeline.vae.config.scaling_factor

                gt_latents = scale_latents(scaled_latent_render_small)

                noise = torch.randn_like(gt_latents)
                noised_truth = pipeline.scheduler.add_noise(gt_latents, noise, t[None])

                # JA: latent will be unscaled after generation. To make noised_truth unscaled as well, we scale them.
                # This blending equation is originally from TEXTure
                blended_latent = latent * curr_mask + noised_truth * (1 - curr_mask) 

                blended_latents.append(blended_latent)
            
            callback_kwargs["latents"] = torch.cat((
                torch.cat((blended_latents[0], blended_latents[3]), dim=3),
                torch.cat((blended_latents[1], blended_latents[4]), dim=3),
                torch.cat((blended_latents[2], blended_latents[5]), dim=3),
            ), dim=2).half()

            return callback_kwargs

        # JA: Here we call the Zero123++ pipeline
        result = self.zero123plus(
            cond_image,
            depth_image=depth_image,
            num_inference_steps=36,
            callback_on_step_end=on_step_end
        ).images[0]

        grid_image = torchvision.transforms.functional.pil_to_tensor(result).to(self.device).float() / 255

        self.log_train_image(grid_image[None], 'zero123plus_grid_image')

        images = split_zero123plus_grid(grid_image, 320)

        thetas, phis, radii = [], [], []
        rgb_outputs = []

        for i, data in enumerate(self.dataloaders['train']):
            if i == 0:
                rgb_output = front_image
            else:
                image_row_index = (i - 1) % 3
                image_col_index = (i - 1) // 3

                cropped_rgb_output_small = images[image_row_index][image_col_index][None]

                # JA: Since Zero123++ requires cond tensor and each depth tensor to be of size 320x320, we resize this
                # to match what it used to be prior to scaling down.
                cropped_rgb_output = F.interpolate(
                    cropped_rgb_output_small,
                    (max_cropped_image_height, max_cropped_image_width),
                    mode='bilinear',
                    align_corners=False
                )

                nonzero_region = viewpoint_data[i]["nonzero_region"]
                min_h, min_w = nonzero_region["min_h"], nonzero_region["min_w"]
                max_h, max_w = nonzero_region["max_h"], nonzero_region["max_w"]

                if should_pad:
                    padding_size = (cropped_rgb_output.shape[-1] - cropped_depth_sizes[i]) // 2
                    if padding_size > 0:
                        # JA: All the target depths and the corresponding results have a padding used to set the image size
                        # match the image size of the largest depth tensor. This padding must be removed so that we can
                        # ensure pixelwise alignment with other tensors upon extending the cropped tensor.
                        nonpadded_area_start_h, nonpadded_area_start_w = padding_size, padding_size
                        nonpadded_area_end_h = nonpadded_area_start_h + (max_h - min_h)
                        nonpadded_area_end_w = nonpadded_area_start_w + (max_w - min_w)

                        cropped_rgb_output = cropped_rgb_output[
                            :, :,
                            nonpadded_area_start_h:nonpadded_area_end_h,
                            nonpadded_area_start_w:nonpadded_area_end_w
                        ]
                else:
                    cropped_rgb_output = F.interpolate(
                        cropped_rgb_output,
                        (max_h - min_h, max_w - min_w),
                        mode='bilinear',
                        align_corners=False
                    )

                # JA: We initialize rgb_output, the image where cropped_rgb_output will be "pasted into." Since the
                # renderer produces tensors (depth maps, object mask, etc.) with a height and width of 1200, rgb_output
                # is initialized with the same size so that it aligns pixel-wise with the renderer-produced tensors.
                # Because Zero123++ generates non-transparent images, that is, images without an alpha channel, with
                # a background of rgb(0.5, 0.5, 0.5), we initialize the tensor using torch.ones and multiply by 0.5.
                rgb_output = torch.ones(
                    cropped_rgb_output.shape[0], cropped_rgb_output.shape[1], 1200, 1200
                ).to(rgb_output.device) * 0.5

                rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output

                rgb_output = F.interpolate(rgb_output, (1200, 1200), mode='bilinear', align_corners=False)

            rgb_outputs.append(rgb_output)

            theta, phi, radius = data['theta'], data['phi'], data['radius']
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)

            thetas.append(theta)
            phis.append(phi)
            radii.append(radius)

        outputs = self.mesh_model.render(theta=thetas, phi=phis, radius=radii, background=background)

        render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)

        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                        render_cache=render_cache, use_median=True)

        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                            use_meta_texture=True, render_cache=render_cache)

        # JA: Get the Z component of the face normal vectors relative to the camera
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200

        self.project_back_only_texture_atlas(
            render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
            object_mask=object_mask, update_mask=object_mask, z_normals=z_normals, z_normals_cache=z_normals_cache,
            # face_normals = self.face_normals, face_idx=self.face_idx
            # face_idx=self.face_idx
            weight_masks=self.weight_masks
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

        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
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
                     z_normals_cache: torch.Tensor
                     , weight_masks:torch.Tensor
                     ):
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
            self.log_train_image(
                torch.cat((z_normals[i][None], z_normals[i][None], z_normals[i][None]), dim=1),
                f'project_back_z_normals_{i}'
            )

        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []

        # JA: TODO: Add num_epochs hyperparameter
        with tqdm(range(200), desc='fitting mesh colors') as pbar:
            for i in pbar:
                optimizer.zero_grad()
                outputs = self.mesh_model.render(background=background,
                                                render_cache=render_cache)
                rgb_render = outputs['image']

                # loss = (render_update_mask * (rgb_render - rgb_output.detach()).pow(2)).mean()
                #loss = (render_update_mask * z_normals * (rgb_render - rgb_output.detach()).pow(2)).mean()
                #BY MJ:
                loss = (render_update_mask * weight_masks * (rgb_render - rgb_output.detach()).pow(2)).mean()
                loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                                # the network, that is, the pixel value of the texture atlas
                optimizer.step()

                pbar.set_description(f"Fitting mesh colors -Epoch {i + 1}, Loss: {loss.item():.4f}")

        return rgb_render

    def project_back_only_texture_atlas_max_z_normals(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor
                     , face_normals: torch.Tensor, face_idx: torch.Tensor, weight_masks:torch.Tensor
                     ):
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
            self.log_train_image(
                torch.cat((z_normals[i][None], z_normals[i][None], z_normals[i][None]), dim=1),
                f'project_back_z_normals_{i}'
            )

        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas_max_z_normal(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []

        batch_size,_, H,W  = face_idx.shape  # Number of views
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, H, W)
        # This generates a tensor with the shape (batch_size, H, W) containing the appropriate batch indices, 
        # ensuring each pixel is indexed with its corresponding batch.
        # JA: TODO: Add num_epochs hyperparameter
        with tqdm(range(200), desc='fitting mesh colors') as pbar:
            for i in pbar:
                optimizer.zero_grad()
                outputs = self.mesh_model.render(background=background,
                                                render_cache=render_cache)
                rgb_render = outputs['image']

                # loss = (render_update_mask * (rgb_render - rgb_output.detach()).pow(2)).mean()
                #loss = (render_update_mask * z_normals * (rgb_render - rgb_output.detach()).pow(2)).mean()
                #BY MJ:
                loss = (render_update_mask * weight_masks * (rgb_render - rgb_output.detach()).pow(2)).mean()
                
                
                if z_normals is not None: 
                    meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                        use_meta_texture=True, render_cache=render_cache)
                    current_z_normals = meta_outputs['image']
                    current_z_mask = meta_outputs['mask'].flatten()
                    masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                            current_z_mask == 1][:, :1]
                    # masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                    #                         current_z_mask == 1][:, :1]
                    #MJ: compute the max_z_normals[face_idx[i,j]]
                    #loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
                    

                    # Without batch_indices
                    try:
                        gathered_normals_without_batch = face_normals[:, 2, face_idx[:, 0, :,:]]
                    except Exception as e:
                        print("Error:", e)

                    gathered_z_normals = face_normals[batch_indices, 2, face_idx[:, 0, :,:]]
                    max_z_normals, _ = torch.max( gathered_z_normals, dim=0)
                    
                    loss += (masked_current_z_normals -  max_z_normals.detatch()).pow(2).mean()
                    
                loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                                # the network, that is, the pixel value of the texture atlas
                                
                                
                optimizer.step()

                pbar.set_description(f"Fitting mesh colors -Epoch {i + 1}, Loss: {loss.item():.4f}")

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
