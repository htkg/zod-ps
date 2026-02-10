import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from baseline_lmc import d, make_geometry
from tqdm import tqdm
from zodps import make_torch_potential

sys.path.append("..")
from zod_ps import ZodPS_Ablation


class Args(argparse.Namespace):
    seed: int
    outdir: Path
    shuffle_Y: bool
    shuffle_x_t: bool
    fix_Y: bool


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--seed", type=int, default=0, help="random seed")
        self.add_argument("--outdir", type=str, default=Path(__file__).parent, help="output directory")
        self.add_argument("--shuffle_Y", action="store_true", help="shuffle Y in each iteration")
        self.add_argument("--shuffle_x_t", action="store_true", help="shuffle x_t in each iteration")
        self.add_argument("--fix_Y", action="store_true", help="fix Y throughout the iterations")

        args = self.parse_args(namespace=Args())
        self.seed = args.seed
        self.outdir = Path(args.outdir)
        self.shuffle_Y = args.shuffle_Y
        self.shuffle_x_t = args.shuffle_x_t
        self.fix_Y = args.fix_Y

        self.outdir.mkdir(parents=True, exist_ok=True)


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
    sampler = ZodPS_Ablation(
        objective_fn=potential_fn,
        dim=d,
        num_particles=num_particles,
        prox_sigma=0.1,
        num_iter=k_end,
        diffusion_steps=10,
        interim_samples=4000,
        beta_min=0,
        shuffle_Y=parser.shuffle_Y,
        shuffle_x_t=parser.shuffle_x_t,
        update_Y=(not parser.fix_Y),
    )

    for k, (X, fn_eval_num) in tqdm(enumerate(sampler.sample())):
        np.save(parser.outdir / f"samples_shuffle_k{k + 1:04d}_seed{seed}.npy", X.detach().cpu().numpy())
