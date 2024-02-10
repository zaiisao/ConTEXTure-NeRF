import sys

sys.path.append("../zero123")

from share import *

import pytorch_lightning as pl
from einops import rearrange
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from ldm.data.simple import ObjaverseData
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = '../ControlNet/models/control-zero123-xl.ckpt' # '../ControlNet/models/control-zero123-xl.ckpt'
config_path = './models/cldm_zero123.yaml'
# batch_size = 64
logger_freq = 300
learning_rate = 5e-5
sd_locked = True
only_mid_control = False
image_transforms_size = 256

def rearrange_func(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

if __name__ == '__main__':
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config_path).cpu() # JA: model is an object of ControlLDM
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Image transforms
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_transforms_size),
        transforms.ToTensor(),
        transforms.Lambda(rearrange_func)
        #transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
    ])

    # # Misc
    # dataset = ObjaverseData(
    #     "/home/jaehoon/.objaverse/hf-objaverse-v1/views_whole_sphere",
    #     total_view=4, validation=False, image_transforms=image_transforms)

    # dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    config = OmegaConf.load(config_path)
    # data
    dataloader = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though

    dataloader.prepare_data() # JA: In our case, prepare_data and setup are not defined in the child class (i.e. is empty)
    dataloader.setup()

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val/loss'#,
        # mode='max' # Commented by JA: ModelCheckpoint sets mode as 'min' by default, which is to be used when saving checkpoints based on loss
    )

    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(
        gpus=-1,
        precision=32,
        callbacks=[logger, checkpoint_callback],
        num_nodes=1,
        strategy="ddp"#,
        # resume_from_checkpoint="/home/jaehoon/repos/zero123/ControlNet/lightning_logs/version_66/checkpoints/epoch=0-step=174.ckpt"
    )

    # Train!
    trainer.fit(model, dataloader)  # JA: model has a special method called "training_step" used in the training loop
                                    # within PyTorch Lightning. training_step computes the loss of a given batch.