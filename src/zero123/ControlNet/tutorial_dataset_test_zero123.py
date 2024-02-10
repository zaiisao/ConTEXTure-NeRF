import sys
import os

sys.path.append("../zero123")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ldm.data.simple import ObjaverseData
from omegaconf import OmegaConf
import torch
import torchvision
from ldm.util import instantiate_from_config
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler

config_path = './models/cldm_zero123.yaml'

config = OmegaConf.load(config_path)
# data
datamodule = instantiate_from_config(config.data)

# JA: When calling datamodule.train_dataloader, it complains that the default process group has
# not been initialized and to call init_process_group. Instead of calling the function to retrieve
# the dataloader, I decided to instantiate the dataset separately.
# train_dataloader = datamodule.train_dataloader() 

dataset = ObjaverseData(
    root_dir=datamodule.root_dir,
    total_view=datamodule.total_view,
    validation=False,
    image_transforms=datamodule.image_transforms
)

# sampler = DistributedSampler(dataset)

dataloader = wds.WebLoader(
    dataset,
    batch_size=datamodule.batch_size,
    num_workers=datamodule.num_workers,
    shuffle=False#,
    # sampler=sampler
)

# item = dataloader[300]

for batch_idx, batch in enumerate(dataloader):
    if batch_idx != 0:
        raise ValueError

    for item_idx in range(len(batch['image_target'])):
        item = {
            'image_target': batch['image_target'][item_idx],
            'image_cond': batch['image_cond'][item_idx],
            'hint': batch['hint'][item_idx],
        }

        print(item)
        # print(
        #     item['image_target'].permute(2, 0, 1).shape,
        #     item['image_cond'].permute(2, 0, 1).shape,
        #     item['hint'].permute(2, 0, 1).shape
        # )
        torchvision.utils.save_image(item['image_target'], f"image_target_{item_idx}.png")
        torchvision.utils.save_image(item['image_cond'], f"image_cond_{item_idx}.png")
        torchvision.utils.save_image(item['hint'], f"hint_{item_idx}.png")
    # print(data)
    break