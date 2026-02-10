import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from baseline import InAndOutParallel

sys.path.append(str(Path(__file__).parents[1].resolve()))
from zod_ps import ZodPS


# Constrained Region: L-shape
def potential_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Constrained Region: L-shape potential function in PyTorch.
    Input:
        x: Tensor of shape (..., 2), where last dimension is (x, y).
    Output:
        Tensor of shape (...,) with the potential values.
    """
    # Ensure shape (..., 2)
    x0 = x[..., 0:1]  # (..., 1)
    y0 = x[..., 1:2]  # (..., 1)
    width = 36.0

    # Rectangle 1: -2 <= x <= -1, -3 <= y <= 3
    x_plus1 = torch.clamp(x0 - (-1.0), min=0.0) ** 2
    x_minus1 = torch.clamp(-2.0 - x0, min=0.0) ** 2
    y_plus1 = torch.clamp(y0 - 3.0, min=0.0) ** 2
    y_minus1 = torch.clamp(-3.0 - y0, min=0.0) ** 2
    potential_rec1 = x_plus1 + x_minus1 + y_plus1 + y_minus1  # (..., 1)

    # Rectangle 2: -1 <= x <= -1+width, -3 <= y <= -2
    x_plus2 = torch.clamp(x0 - (-1.0 + width), min=0.0) ** 2
    x_minus2 = torch.clamp(-1.0 - x0, min=0.0) ** 2
    y_plus2 = torch.clamp(y0 - (-2.0), min=0.0) ** 2
    y_minus2 = torch.clamp(-3.0 - y0, min=0.0) ** 2
    potential_rec2 = x_plus2 + x_minus2 + y_plus2 + y_minus2  # (..., 1)

    potential = torch.where(potential_rec1 < potential_rec2, potential_rec1, potential_rec2)
    return 100 * potential.squeeze(-1)  # (...,)


if __name__ == "__main__":
    I = "in-and-out"
    Z = "zod-ps"
    method = Z  # "in-and-out" or "zod-ps"

    num_particles = 100

    match method:
        case "in-and-out":
            sampler = InAndOutParallel(
                objective_fn=potential_fn,
                dim=2,
                num_particles=num_particles,
                prox_sigma=3.0,
                num_iter=1000,
            )
            initial_X = torch.tensor([[-1.0, -2.0]]).repeat(num_particles, 1)
        case "zod-ps":
            sampler = ZodPS(
                objective_fn=potential_fn,
                dim=2,
                num_particles=num_particles,
                prox_sigma=3.0,
                num_iter=1000,
                diffusion_steps=25,
                beta_min=0.001,
            )
            initial_X = torch.tensor([[-1.0, -2.0]]).repeat(num_particles, 1) + torch.randn(num_particles, 2)

    plt.ion()
    fig, ax = plt.subplots()
    plt.show(block=False)
    cbar = fig.colorbar(ax.scatter(torch.empty(0), torch.empty(0)), ax=ax, label="Objective Value")

    for k, (X, fn_eval_num) in enumerate(sampler.sample(initial_X)):
        ax.clear()
        values = potential_fn(X).cpu().numpy()
        sc = ax.scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c=values, cmap="viridis")
        cbar.update_normal(sc)
        ax.scatter(X[:, 0].mean().item(), X[:, 1].mean().item(), c="red", marker="x", s=100)
        ax.set_title(f"{k=}, Valid Samples: {(values == 0).sum()}/{values.shape[0]}, Num Eval: {fn_eval_num:,}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
