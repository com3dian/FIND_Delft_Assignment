# FIND Delft Assignment — DDIM on a pretrained DDPM

Skeleton for optional component **4 — Experimental**: DDIM sampling with a **fixed pretrained DDPM** (no training), small sample sets, and a short **DDPM vs DDIM** comparison.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip setuptools
pip install -e ".[dev]"
```

## Layout

| Path | Role |
|------|------|
| `src/find_delft_assignment/` | Installable package: config, model stub, DDPM/DDIM sampling stubs, checkpoints helper, visuals |
| `notebooks/` | Local Jupyter workflow |
| `colab/` | Notes / mirror notebook for Google Colab (`pip install git+...`) |
| `checkpoints/` | Place pretrained weights here (large files gitignored; see `.gitignore`) |
| `outputs/` | Saved grids, metrics, run artifacts |
| `scripts/` | Optional shell helpers |

## Reproduce (after you implement sampling)

1. Add a pretrained checkpoint under `checkpoints/` (document URL in this README).
2. Run the CLI stub: `find-delft-sample --help`
3. Open `notebooks/ddim_experiment.ipynb` or the Colab workflow in `colab/README.md`.

## Colab

See `colab/README.md` for a copy-paste `pip install` line from your GitHub URL.

## License

See `LICENSE`.
