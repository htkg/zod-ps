import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    I = "in-and-out"
    Z = "zod-ps"
    method = I  # "in-and-out" or "zod-ps"

    num_particles = 1000
    prox_sigma = 1.0

    match method:
        case "in-and-out":
            sampler = InAndOutParallel(
                objective_fn=potential_fn,
                dim=3,
                num_particles=num_particles,
                prox_sigma=prox_sigma,
                num_iter=1000,
                max_trial=10000,
                report_worst=True,
            )
            initial_X = torch.tensor([[0.0, 0.0, 0.0]]).repeat(num_particles, 1)
        case "zod-ps":
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

    plt.ion()
    fig, ax = plt.subplots()
    plt.show(block=False)
    cbar = fig.colorbar(ax.scatter(torch.empty(0), torch.empty(0)), ax=ax, label="Objective Value")

    for k, (X, fn_eval_num) in enumerate(sampler.sample(initial_X)):
        ax.clear()
        values = potential_fn(X)
        sc = ax.scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c=values.cpu().numpy(), vmax=100, cmap="viridis")
        cbar.update_normal(sc)

        X_mean = X[values <= 0].mean(dim=0)
        ax.scatter(X_mean[0].item(), X_mean[1].item(), c="red", marker="x", s=100)
        ax.set_title(f"{k=}, Valid Samples: {(values <= 0).sum()}/{values.shape[0]}, Num Eval: {fn_eval_num:,}")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
