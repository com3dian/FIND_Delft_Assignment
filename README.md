# FIND Delft Assignment — DDIM sampling on a pretrained DDPM

This repository implements **DDPM** (ancestral) and **DDIM** sampling on top of a **fixed pretrained** diffusion model: no training. The default checkpoint is **`google/ddpm-cifar10-32`** on the Hugging Face Hub — a **32×32** unconditional CIFAR-10 model that stays small in memory while matching a strong published baseline (see the [model card](https://huggingface.co/google/ddpm-cifar10-32)).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools
pip install -e ".[dev]"
```

Optional notebook extras:

```bash
pip install -e ".[dev,colab]"
```

**CPU-only PyTorch:** if you do not want the default CUDA-enabled wheels from PyPI, install a CPU build of `torch` / `torchvision` first (see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)), then `pip install -e .` so this package does not pull a different `torch` build.

## Layout

| Path | Role |
|------|------|
| `src/find_delft_assignment/` | Package: load pretrained UNet + DDPM schedule, DDPM/DDIM loops, metrics, grids |
| `notebooks/ddim_experiment.ipynb` | Local or Colab workflow |
| `colab/README.md` | Colab install line and “Open in Colab” URL pattern |
| `outputs/` | Notebook outputs (grids, etc.; gitignored) |

## Reproduce

Open `notebooks/ddim_experiment.ipynb` and run all cells. On Colab, the first cell installs from [this repo](https://github.com/com3dian/FIND_Delft_Assignment) by default; change `COLAB_INSTALL_URL` there if you use a fork (see `colab/README.md`).

## Colab

[Open `ddim_experiment.ipynb` in Colab](https://colab.research.google.com/github/com3dian/FIND_Delft_Assignment/blob/main/notebooks/ddim_experiment.ipynb)

The first notebook cell installs the package from GitHub unless you are running locally. More detail: `colab/README.md`.

## Model choice

`google/ddpm-cifar10-32` is used because it is **small** (fast iteration on laptop/Colab) yet **high quality** for its size. To use another Hub model, change the `model_id` argument to `load_pretrained_ddpm()` in the notebook (the repo must expose a compatible `DDPMPipeline` and scheduler).

## License

See `LICENSE`.
