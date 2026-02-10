import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from baseline_lmc import ArgParser, d, make_geometry
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1].resolve()))
from zod_ps import ZodPS


def make_torch_potential(Q: torch.Tensor, logdetQ: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def potential_fn(x: torch.Tensor) -> torch.Tensor:
        # x: (n, d)
        # ---- Gaussian component ----
        dx = x - torch.ones(d)  # (n, d)
        quad = torch.einsum("ni,ij,nj->n", dx, Q, dx)
        log_gauss = 0.5 * logdetQ - 0.5 * d * torch.log(torch.tensor(2.0 * torch.pi)) - 0.5 * quad

        # ---- Laplace component ----
        b = 0.25
        log_laplace = d * torch.log(torch.tensor(1.0 / (2.0 * b))) - torch.sum(torch.abs(x), dim=1) / b

        # ---- log-sum-exp for mixture ----
        lg = torch.log(torch.tensor(0.5)) + log_gauss
        ll = torch.log(torch.tensor(0.5)) + log_laplace
        m = torch.maximum(lg, ll)

        return -(m + torch.log(torch.exp(lg - m) + torch.exp(ll - m)))

    return potential_fn


if __name__ == "__main__":
    parser = ArgParser()
    seed = parser.seed

    np_rng = np.random.default_rng(seed)
    Q, Sigma, logdetQ = make_geometry(np_rng)
    Q_torch = torch.tensor(Q, dtype=torch.float32)
    logdetQ_torch = torch.tensor(logdetQ, dtype=torch.float32)
    potential_fn = make_torch_potential(Q_torch, logdetQ_torch)

    num_particles = 100
    k_end = 1000

    torch.manual_seed(seed)
    sampler = ZodPS(
        objective_fn=potential_fn,
        dim=d,
        num_particles=num_particles,
        prox_sigma=0.1,
        num_iter=k_end,
        diffusion_steps=10,
        interim_samples=4000,
        beta_min=0,
    )

    for k, (X, fn_eval_num) in tqdm(enumerate(sampler.sample())):
        np.save(parser.outdir / f"samples_k{k + 1:04d}_seed{seed}.npy", X.detach().cpu().numpy())
