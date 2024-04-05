import os
import shutil

from PIL import Image
import numpy as np
from torchvision import transforms
from einops import rearrange

from ldm.util import create_carvekit_interface, load_and_preprocess

models = {}
models['carvekit'] = create_carvekit_interface()

def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

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

    return input_im, est_seg

def copy_gt_images(source_base, dest_base, gt_base):
    for folder_name in os.listdir(source_base):  # Iterates over each folder in the source base (either 0123 or c0123)
        source_folder = os.path.join(source_base, folder_name)
        if os.path.isdir(source_folder):
            # Assuming there's only one image per folder in the pred structure
            for image_name in os.listdir(source_folder):
                if image_name.endswith('.jpg'):
                    # Extract the second number from the image name
                    _, second_number, *_ = image_name.split('_')
                    # Define the original image path in the google structure
                    original_image_path = os.path.join("/home/sogang/jaehoon/google", folder_name, "thumbnails", f"{second_number}.jpg")
                    # Define the destination path under gt
                    dest_folder = os.path.join(dest_base, folder_name)
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    dest_image_path = os.path.join(dest_folder, f"{second_number}.jpg")
                    # Copy the original image to the new location
                    shutil.copy2(original_image_path, dest_image_path)


                    image = Image.open(dest_image_path)
                    preprocessed_image, _ = preprocess_image(models, image, preprocess=True)

                    preprocessed_image = transforms.ToTensor()(preprocessed_image)
                    preprocessed_image = transforms.functional.resize(preprocessed_image, [256, 256])

                    x_sample = 255.0 * rearrange(preprocessed_image.cpu().numpy(), 'c h w -> h w c')
                    preprocessed_image = Image.fromarray(x_sample.astype(np.uint8))

                    preprocessed_image.save(dest_image_path)

                    print(f"Copied {original_image_path} to {dest_image_path}")

# Paths for the source 'pred' directories and destination 'gt' directories
source_paths = {
    "0123": "/home/sogang/jaehoon/fid/0123/pred",
    "c0123": "/home/sogang/jaehoon/fid/c0123/pred"
}
gt_paths = {
    "0123": "/home/sogang/jaehoon/fid/0123/gt",
    "c0123": "/home/sogang/jaehoon/fid/c0123/gt"
}

for key in source_paths:
    print(f"Processing {key}...")
    copy_gt_images(source_paths[key], gt_paths[key], "google")