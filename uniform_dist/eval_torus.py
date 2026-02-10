import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from baseline import InAndOutParallel

sys.path.append(str(Path(__file__).parents[1].resolve()))
from zod_ps import ZodPS


def solid_torus_potential(
    x: torch.Tensor,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    R: float = 10.0,
    r: float = 1.0,
    V_out: float = 10.0,
) -> torch.Tensor:
    cx, cy, cz = center
    X = x[:, 0] - cx
    Y = x[:, 1] - cy
    Z = x[:, 2] - cz

    rho = torch.sqrt(X**2 + Y**2)
    s = (rho - R) ** 2 + Z**2
    inside = s <= (r**2)

    return torch.where(inside, torch.zeros_like(rho), torch.full_like(rho, float(V_out)))


def potential_fn(x: torch.Tensor) -> torch.Tensor:
    t1 = solid_torus_potential(x, center=(10.0, 0.0, 0.0), R=10.0, r=1.0, V_out=100.0)
    t2 = solid_torus_potential(x, center=(-13.0, 0.0, 0.0), R=3.0, r=1.0, V_out=100.0)
    return torch.minimum(t1, t2)


class Args(argparse.Namespace):
    seed: int
    outdir: Path
    method: str


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--seed", type=int, default=0, help="random seed")
        self.add_argument("--outdir", type=str, default=Path(__file__).parent, help="output directory")
        self.add_argument("--method", choices=["I", "Z"], default="I", help="sampling method, I: In-and-Out, Z: ZOD-PS")

        args = self.parse_args(namespace=Args())
        self.seed = args.seed
        self.outdir = Path(args.outdir)
        self.method = args.method

        self.outdir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = ArgParser()
    seed = parser.seed

    torch.manual_seed(seed)

    num_particles = 1000
    prox_sigma = 1.0

    match parser.method:
        case "I":
            sampler = InAndOutParallel(
                objective_fn=potential_fn,
                dim=3,
                num_particles=num_particles,
                prox_sigma=prox_sigma,
                num_iter=1000,
                max_trial=10000,
                report_worst=True,
            )
        case "Z":
            sampler = ZodPS(
                objective_fn=potential_fn,
                dim=3,
                num_particles=num_particles,
                prox_sigma=prox_sigma,
                num_iter=1000,
                diffusion_steps=10,
                interim_samples=300,
                beta_min=0.01,
            )
    initial_X = torch.randn(num_particles, 3)

    fn_eval_nums = []
    for k, (X, fn_eval_num) in enumerate(sampler.sample(initial_X)):
        np.save(parser.outdir / f"samples_{parser.method}_k{k + 1:04d}_seed{seed}.npy", X.detach().cpu().numpy())
        fn_eval_nums.append(fn_eval_num)

    np.save(parser.outdir / f"fn_eval_nums_{parser.method}_seed{seed}.npy", np.array(fn_eval_nums))
