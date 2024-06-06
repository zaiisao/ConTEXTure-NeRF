# TEXTure: Text-Guided Texturing of 3D Shapes

## [[Project Page - Coming Soon]](https://google.com/)

> Abstract
> Coming soon

## Description :scroll:

Official Implementation for ConTEXTure.

## Recent Updates :newspaper:

## Getting Started with ConTEXTure 🐇

### Installation :floppy_disk:

#### 1. Install from requirements file

There is a basic list of packages to install. You can do so by installing from the requirements file:

```bash
pip install -r requirements.txt
```

#### 2. Install PyTorch

ConTEXTure uses [Kaolin](https://kaolin.readthedocs.io/en/latest/index.html) and [torch-scatter](https://pytorch-scatter.readthedocs.io/en/latest/). It is necessary to check both packages to check the latest version of both packages and see which they support. [The Kaolin documentation provides a table](https://kaolin.readthedocs.io/en/latest/notes/installation.html#quick-start-linux-windows) showing which combinations of PyTorch and CUDA versions are supported by the library. Instructions on installing previous versions of PyTorch can be found [here](https://pytorch.org/get-started/previous-versions/).

At the time of this writing, Kaolin is only supported up to PyTorch 2.1.1, which can be installed for a machine with CUDA 12.1 as follows:

```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install Kaolin and torch-scatter

Instructions for installing Kaolin can be found [here](https://kaolin.readthedocs.io/en/latest/notes/installation.html). At the time of this writing, the latest version of Kaolin is 0.15.0, which can be installed as follows:

```bash
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VER}_cu{CUDA_VER}.html
```

TORCH_VER and CUDA_VER should be replaced with your PyTorch and CUDA versions.

#### Note about Huggingface

Note that you also need a :hugs: token for StableDiffusion.
First accept conditions for the model you want to use, default one
is [`stabilityai/stable-diffusion-2-depth`]( https://huggingface.co/stabilityai/stable-diffusion-2-depth). Then, add a
TOKEN file [access token](https://huggingface.co/settings/tokens) to the root folder of this project, or use
the `huggingface-cli login` command

## Running 🏃

### Text Conditioned Texture Generation 🎨

Try out painting the [Napoleon](https://threedscans.com/nouveau-musee-national-de-monaco/napoleon-ler/)
from [Three D Scans](https://threedscans.com/) with a text prompt

```bash
python -m scripts.run_texture --config_path=configs/text_guided/napoleon.yaml
```

Or a next-gen NASCAR from [ModelNet40](https://modelnet.cs.princeton.edu/)

```bash
python -m scripts.run_texture --config_path=configs/text_guided/nascar.yaml
```

Configuration is managed using [pyrallis](https://github.com/eladrich/pyrallis) from either `.yaml` files or `cli`
