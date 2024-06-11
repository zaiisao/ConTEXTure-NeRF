import random
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import einops
from matplotlib import cm
import torch.nn.functional as F


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    #MJ: added to handle bad pixel values 
    if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
    #     # Raise an exception if there are any NaNs or infinite values
    #      tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
    #      Image.fromarray( (tensor * 255).astype(np.uint8) ).save('experiments'/f'debug:NanOrInf.jpg')

          raise ValueError("Tensor contains NaNs or infinite values, which cannot be converted to np.uint8.")

    tensor = (tensor * 255).astype(np.uint8)
    #MJ: This line is syntactically correct for converting a tensor to a numpy array,
    # scaling its values by 255, and then converting it to an 8-bit unsigned integer format.
    # If it causes:  RuntimeWarning: invalid value encountered in cast
    # tensor = (tensor * 255).astype(np.uint8):
    #The warning often occurs when attempting to convert NaNs or infinities to integers,
    # as these values do not have direct integer representations.
    
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def save_colormap(tensor: torch.Tensor, path: Path):
    Image.fromarray((cm.seismic(tensor.cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(path)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    
    # Mixing random number generation from NumPy (numpy.random.normal()) and PyTorch (torch.randn_like(gt_latent)) in the same program involves using two separate random number generators,
    # each with its own state, behavior, and initialization methods.


def smooth_image(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
    """apply gaussian blur to an image tensor with shape [C, H, W]"""
    img = T.GaussianBlur(kernel_size=(51, 51), sigma=(sigma, sigma))(img)
    return img


def get_nonzero_region_tuple(mask: torch.Tensor): #MJ: mask: shape = (H,W)
    # Get the indices of the non-zero elements
    nz_indices = mask.nonzero() 
    #MJ:  nz_indices will have a shape of (N, 2), where N is the number of non-zero elements in mask.
    # The two columns in nz_indices represent the row index (height) and the column index (width) of each non-zero element, respectively.
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
    #return torch.tensor([min_h, min_w, max_h, max_w])

def get_nonzero_region_tensor(mask: torch.Tensor): #MJ: mask: shape = (H,W)
    # Get the indices of the non-zero elements
    nz_indices = mask.nonzero() 
    #MJ:  nz_indices will have a shape of (N, 2), where N = 175367 is the number of non-zero elements in mask.
    # The two columns in nz_indices represent the row index (height) and the column index (width) of each non-zero element, respectively.
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

    #return min_h, min_w, max_h, max_w
    return torch.tensor([min_h, min_w, max_h, max_w])

def get_nonzero_region_vectorized(mask: torch.Tensor):
    """
    Calculate bounding boxes for non-zero regions in each image of a batch, preserving the
    structure of the original function. Assumes mask has shape (B, 1, H, W).
    Returns a tensor of shape (B, 4), where each row contains [min_h, min_w, max_h, max_w].

    Parameters:
    mask (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: Bounding boxes for each image in the batch.
    """
    B, C, H, W = mask.size()
    output = torch.zeros((B, 4), dtype=torch.int32)  # Prepare output tensor

    for i in range(B):  # Processing each item in the batch
        # Flatten the (1, H, W) mask to (H, W) for processing
        single_mask = mask[i, 0, :, :]
        nz_indices = single_mask.nonzero()

        if nz_indices.nelement() == 0:  # Check if there are no non-zero entries
            continue  # Skip to next in batch if no non-zero entries

        # Process non-zero indices as in the original function
        min_h, max_h = nz_indices[:, 0].min(), nz_indices[:, 0].max()
        min_w, max_w = nz_indices[:, 1].min(), nz_indices[:, 1].max()

        # Calculate the size of the square region
        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1  # Increase size by 10%
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        # Calculate the bounding box, clamping values within image dimensions
        min_h = max(0, int(h_start))
        min_w = max(0, int(w_start))
        max_h = min(H, int(min_h + size))
        max_w = min(W, int(min_w + size))

        # Store results in the output tensor
        output[i, :] = torch.tensor([min_h, min_w, max_h, max_w])

    return output

# Example usage
# your_mask_tensor = torch.rand(5, 1, 100, 100) > 0.95  # Example tensor for demonstration
# result = get_nonzero_region_vectorized(your_mask_tensor)
# print(result)

import torch

def crop_img_to_bounding_box(img: torch.Tensor, bounding_boxes: torch.Tensor):
    """
    Crop each mask in a batch to the specified bounding box, assuming all channels have the same information.
    
    Parameters:
    image (torch.Tensor): The input image tensor with shape (B, C, H, W)
    bounding_boxes (torch.Tensor): The bounding boxes for each mask with shape (B, 4)
    
    Returns:
    torch.Tensor: The tensor containing cropped masks for each channel identically.
    """
    B, C, H, W = img.shape
    # Determine the maximum height and width needed to initialize the tensor
    max_height = max([box[2] - box[0] for box in bounding_boxes])
    max_width = max([box[3] - box[1] for box in bounding_boxes])

    # Initialize the tensor for cropped masks: backround should be set to 1
    cropped_imgs = torch.ones((B, C, max_height, max_width), dtype=img.dtype, device=img.device)
    
    for i in range(B):
        min_h, min_w, max_h, max_w = bounding_boxes[i]
        height = max_h - min_h
        width = max_w - min_w
        # Apply the same cropping to all channels since the information is repeated
        cropped_imgs[i, :, :height, :width] = img[i, :, min_h:max_h, min_w:max_w]
    
    return cropped_imgs

#crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
# cropped_rgb_render = crop(outputs['image'])

# # Example usage
# # Suppose outputs["mask"] is a tensor with shape (B, C, H, W)
# outputs = {
#     "mask": torch.rand(5, 4, 100, 100) > 0.95  # Example tensor for demonstration, with 4 channels
# }
# # Assuming bounding box calculations from one channel, apply to all due to repetition
# bounding_boxes = get_nonzero_region_vectorized(outputs["mask"][:,0:1,:,:])  # Use any single channel for bounding box calculation
# cropped_mask = crop_mask_to_bounding_box(outputs["mask"], bounding_boxes)

# print("Cropped Masks Shape:", cropped_mask.shape)
# print("Bounding Boxes:", bounding_boxes)


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


def gaussian_blur(image: torch.Tensor, kernel_size: int, std: int) -> torch.Tensor:
    gaussian_filter = gkern(kernel_size, std=std)
    gaussian_filter /= gaussian_filter.sum()

    image = F.conv2d(image,
                     gaussian_filter.unsqueeze(0).unsqueeze(0).cuda(), padding=kernel_size // 2)
    return image


def color_with_shade(color: List[float], z_normals: torch.Tensor, light_coef=0.7):
    normals_with_light = (light_coef + (1 - light_coef) * z_normals.detach())
    shaded_color = torch.tensor(color).view(1, 3, 1, 1).to(
        z_normals.device) * normals_with_light
    return shaded_color
#MJ:
# The detach() method is used to create a tensor that shares storage with z_normals
# but without tracking operations for gradient computation, meaning it is treated as a constant in this context.
# The calculation essentially scales the z_normals by a factor (1 - light_coef) and adds light_coef to it,
# modulating the normal values based on the light coefficient.
#If I  use z_normals rather than z_normals.detach(): 
#(1) Computation Graph: The computation graph will include additional nodes for the operations (1 - light_coef) * z_normals and the addition with light_coef. 
# This makes the graph more complex and larger, potentially increasing memory usage during training.
# (2) Training Dynamics:
# If the shading effect is not intended to affect the gradients and training dynamics, 
# not using .detach() might unintentionally allow the shading effect to influence the gradients. 
# This could lead to unintended consequences in the optimization process, as the gradients could propagate through the shading operation,
# potentially affecting the learning process in ways that are not desired.
#(3) BUT:
# If shaded_color and, consequently, normals_with_light are not part of the computation graph that influences the loss,
# the gradients will not be computed for these operations during backpropagation. This means that whether z_normals is detached or not,
# it will not affect the gradient computation since these operations do not participate in the loss calculation
# (4) In sum: The detachment is only significant when the operations involving z_normals are part of the loss computation graph.
def pad_tensor_to_size(input_tensor, target_height, target_width, value=1):
    # Get the current dimensions of the tensor
    current_height, current_width = input_tensor.shape[-2], input_tensor.shape[-1]
    
    # Calculate padding needed
    pad_height = target_height - current_height
    pad_width = target_width - current_width

    # Calculate padding for top/bottom and left/right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply the padding
    padded_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=value)
    
    return padded_tensor

def split_zero123plus_grid( grid_image_3x2, tile_size):
    images = []
    for row in range(3):
        images_col = []
        #MJ: create two columns for each row
        for col in range(2):
            # Calculate the start and end indices for the slices
            start_row = row * tile_size
            end_row = start_row + tile_size
            start_col = col * tile_size
            end_col = start_col + tile_size

            # Slice the tensor and add to the list
            if len(grid_image_3x2.shape) == 3:
                original_image = grid_image_3x2[:, start_row:end_row, start_col:end_col]
            elif len(grid_image_3x2.shape) == 4:
                original_image = grid_image_3x2[:, :, start_row:end_row, start_col:end_col]
            else:
                raise NotImplementedError

            images_col.append(original_image)

        images.append(images_col)

    return images


def merge_tensor_with_6_elements_to_3x2_grid(components, tile_size): #MJ: grid_image: (1,4,120,80);components: shape=(6,4,40,40)
    num_rows = 3
    num_cols = 2
    C = components.shape[1]
    
    grid_image = torch.empty(1,C, num_rows*components.shape[2],  num_cols*components.shape[3] )
    for col in range( num_cols):
        #MJ: images_col = []
        for row in range(num_rows):
            # Calculate the start and end indices for the slices
            start_row = row * tile_size
            end_row = start_row + tile_size
            start_col = col * tile_size
            end_col = start_col + tile_size

            idx =  num_rows * col + row
            grid_image[0,:, start_row:end_row, start_col:end_col] = components[  idx ]  
           
            #MJ: images: col0: img0, img1, img2
            #            col1: img3, img4, img5

    return  grid_image.to(components.device) #MJ: convert the list to the tensor along the first dim

def split_3x2_grid_to_tensor_with_6_elements(grid_image, tile_size): #MJ: grid_image: (1,4,120,80); tile_size of component
    num_rows = grid_image.shape[2] // tile_size
    num_cols =  grid_image.shape[3] // tile_size
    ## Initialize components tensor to hold the tiles; dimensions [number of tiles, channels, tile_size, tile_size]=[6,4,40,40]
    components = torch.empty( num_rows * num_cols, grid_image.shape[1], tile_size, tile_size )
   
    for col in range(num_cols):
        #MJ: images_col = []
        for row in range(num_rows):
            # Calculate the start and end indices for the tiles
            start_row = row * tile_size
            end_row = start_row + tile_size
            start_col = col * tile_size
            end_col = start_col + tile_size

           # Assign the tiled grid image to the corresponding position in components:  v = 3 * col + row; 3 = num_of_rows
            idx =  num_rows * col + row #MJ:   (3 x 2 matrix)  components[ idx ].shape:torch.Size([4, 40, 40])
            components[ idx ] = grid_image[0,:, start_row:end_row, start_col:end_col]
           
            #MJ: images: col0: img0, img1, img2
            #            col1: img3, img4, img5

    return  components.to(grid_image.device) 