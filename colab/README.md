# Google Colab

## Install from GitHub

The notebooks default to this repo on `main`:

`git+https://github.com/com3dian/FIND_Delft_Assignment.git@main`

If you forked the project, change `COLAB_INSTALL_URL` in the first notebook cell to your URL and branch, for example:

`git+https://github.com/<your-user>/FIND_Delft_Assignment.git@main`

Equivalent one-shot install:

```python
import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/com3dian/FIND_Delft_Assignment.git@main",
    ]
)
```

Torch is usually preinstalled on Colab; after install, restart the runtime if the installer upgraded `torch`.

## Open the same notebook as locally

Use “Open in Colab” on `notebooks/ddim_experiment.ipynb` in your GitHub UI, or open:

`https://colab.research.google.com/github/com3dian/FIND_Delft_Assignment/blob/main/notebooks/ddim_experiment.ipynb`

The first notebook cell installs the package; the rest matches the local workflow.

## Checkpoints

The default model is downloaded automatically from the Hugging Face Hub (`google/ddpm-cifar10-32`). No manual weight upload is required unless you switch `--model-id`.
