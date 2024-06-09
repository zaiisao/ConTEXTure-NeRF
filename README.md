# TEXTure: Text-Guided Texturing of 3D Shapes

## [[Project Page - Coming Soon]](https://google.com/)

> Abstract
> Coming soon

## Description :scroll:

Official Implementation for ConTEXTure.

## Recent Updates :newspaper:

## Getting Started with ConTEXTure 🐇

### Installation :floppy_disk:

#### 0. Create a new virtual environment

If you use Anaconda, a new virtual environment can be created and activated in the following manner:

```bash
conda create -n ConTEXTure python=3.9
conda activate ConTEXTure
```

#### 1. Install required packages

There is a basic list of packages to install. You can do so by installing from the requirements file:

```bash
pip install -r requirements.txt
pip install carvekit==4.1.2 --no-deps
```

`carvekit` is also required to run the code. It needs to be installed separately using the `--no-deps` flag because it is overly strict regarding its required packages, some of which go in conflict with the packages we already use.

#### 2. Install Kaolin and torch-scatter

Instructions for installing Kaolin can be found [here](https://kaolin.readthedocs.io/en/latest/notes/installation.html). At the time of this writing, the latest version of Kaolin is 0.15.0, which can be installed as follows:

```bash
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu121.html
```

The cu121 in both links should be replaced with your CUDA version.

#### 3. Add the Huggingface token

Note that you also need a :hugs: token for StableDiffusion.
First accept conditions for the [`stabilityai/stable-diffusion-2-depth`]( https://huggingface.co/stabilityai/stable-diffusion-2-depth) model. The [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/cli) can be used to apply the token, which will allow the ConTEXTure model to be runnable.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

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
