import sys

import numpy as np
import torch
from baseline_lmc import ArgParser, d, make_geometry
from tqdm import tqdm
from zodps import make_torch_potential

sys.path.append("..")
from zod_ps import ZodPS_NoInteraction

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
    sampler = ZodPS_NoInteraction(
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
        np.save(parser.outdir / f"samples_no_interaction_k{k + 1:04d}_seed{seed}.npy", X.detach().cpu().numpy())
