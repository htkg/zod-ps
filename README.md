# Alternating Diffusion for Proximal Sampling with Zeroth Order Queries (ICLR 2026)

This repository contains the code to reproduce the experiments in the paper "Alternating Diffusion for Proximal Sampling with Zeroth Order Queries", ICLR 2026.

## Setup from Scratch

We use virtual environment with Python 3.12 and the following packages:

```shell
$ conda create -y -n py312 --override-channels -c conda-forge python=3.12.11
$ conda activate py312
$ python -m venv .venv
$ source ./.venv/bin/activate
$ pip install torch numpy matplotlib scipy tqdm infomeasure
# or `pip install -r requirements.txt` if you want to follow the versions in requirements.txt
```

## Run Experiments on Gaussian Lasso Mixture

```shell
$ cd gaussian_lasso_mixture
$ python ${script_name}.py [--seed 0] [--outdir ./results]
```

Both options are optional:
- `--seed` sets the random seed
- `--outdir` specifies the output directory

Available Scripts
- Single run to check the objective function
    - `baseline_lmc.py`: Unadjusted Langevin Algorithm (ULA) as a simple baseline
    - `rgo_rs.py`: RGO with Rejection Sampling
- Collect $N$ samples with different methods
    - `baseline_lmc_loop.py`: ULA loop
    - `rgo_rs_loop.py`: RGO with Rejection Sampling for $N$ chains
    - `zodps.py`: Zeroth-Order Diffusive Proximal Sampler (ZOD-PS) with $N$ particles
- Ablation studies
    - `zodps_no_interaction.py`: ZOD-PS without interaction term (i.e. independent chains of $N=1$)
    - `zodps_shuffle.py`: ZOD-PS with replacing particles at each iteration, combined with the options below:
        - `--shuffle_Y`: $Y_{k+1/2}$ is sampled from Gaussian Mixture $\sum_i \mathcal{N}(X_k^i, hI_d)$
        - `--shuffle_x_t`: perturbed $X_{k+1}$ is sampled from Gaussian Mixture $\sum_i \mathcal{N}(X_k^i, hI_d)$
        - `--fix_Y`: fix all $Y_{k+1/2}$ to $Y_{1/2}$

## Run Experiments for Uniform Distribution

```shell
$ cd uniform_dist

# visualize sampling dynamics to check if the methods work
# see the code for details
$ python L-shape.py
$ python sparse_grid.py
$ python torus.py
```

To compare In-and-Out and our method on the uniform distribution over tori:

```shell
$ python eval_torus.py [--method I or Z] [--seed 0] [--outdir ./results]
```

Options are optional:
- `--method` selects the algorithm: `I` for In-and-Out, `Z` for ours, default is `I`
- Other options (`--seed`, `--outdir`) are the same as above

## Citation

Hirohane Takagi and Atsushi Nitanda. Alternating Diffusion for Proximal Sampling with Zeroth Order Queries. The 14th International Conference on Learning Representations (ICLR2026), 2026.

```bibtex
TBA
```
