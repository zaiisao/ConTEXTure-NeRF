import torch
import os
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)

pipeline.add_controlnet(ControlNetModel.from_pretrained(
    "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
), conditioning_scale=2)

pipeline.to('cuda:0')

base_dir = "/home/sogang/jaehoon/texture_test/test_set"

# Loop through the objects and prompts
for obj_name in os.listdir(base_dir):
    obj_path = os.path.join(base_dir, obj_name)
    if os.path.isdir(obj_path):
        depth_image_path = os.path.join(obj_path, "depth_map_grid.png")
        
        for prompt_name in os.listdir(obj_path):
            prompt_path = os.path.join(obj_path, prompt_name)

            if os.path.isdir(prompt_path):
                cond_image_path = os.path.join(prompt_path, "cond_image.png")

                if os.path.exists(cond_image_path) and os.path.exists(depth_image_path):
                    condition_image = Image.open(cond_image_path)
                    depth_image = Image.open(depth_image_path)
                    print(condition_image.size)

                    # Generate the image using the pipeline
                    result = pipeline(condition_image, depth_image=depth_image, num_inference_steps=36).images[0]

                    # Save output in slices
                    target_size = 320
                    vertical_cuts = 3  # Three vertical slices
                    horizontal_cuts = 2  # Two horizontal slices

                    for i in range(vertical_cuts):
                        for j in range(horizontal_cuts):
                            left = j * target_size
                            upper = i * target_size
                            right = left + target_size
                            lower = upper + target_size
                            crop = result.crop((left, upper, right, lower))
                            crop_filename = f"{i}_{j}.png"
                            crop_path = os.path.join(prompt_path, crop_filename)
                            crop.save(crop_path)
                            print(f"Saved {crop_path}")