import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from baseline import InAndOutParallel

sys.path.append(str(Path(__file__).parents[1].resolve()))
from zod_ps import ZodPS


# Constrained Region: Rectangle grid
def potential_fn(x: torch.Tensor) -> torch.Tensor:
    rect_w = 3
    rect_h = 2
    rect_l = 1
    gap = 10

    isin_x = (
        ((x[:, 0] >= -1) & (x[:, 0] <= rect_w - 1))
        | ((x[:, 0] >= rect_w + gap - 1) & (x[:, 0] <= 2 * rect_w + gap - 1))
        | ((x[:, 0] >= 2 * rect_w + 2 * gap - 1) & (x[:, 0] <= 3 * rect_w + 2 * gap - 1))
    )
    isin_y = (
        ((x[:, 1] >= -1) & (x[:, 1] <= rect_h - 1))
        | ((x[:, 1] >= rect_h + gap - 1) & (x[:, 1] <= 2 * rect_h + gap - 1))
        | ((x[:, 1] >= 2 * rect_h + 2 * gap - 1) & (x[:, 1] <= 3 * rect_h + 2 * gap - 1))
    )
    isin_z = (
        ((x[:, 2] >= -1) & (x[:, 2] <= rect_l - 1))
        | ((x[:, 2] >= rect_l + gap - 1) & (x[:, 2] <= 2 * rect_l + gap - 1))
        | ((x[:, 2] >= 2 * rect_l + 2 * gap - 1) & (x[:, 2] <= 3 * rect_l + 2 * gap - 1))
    )
    near_x = (x[:, 0] >= -1 - gap) & (x[:, 0] <= 3 * rect_w + 3 * gap - 1)
    near_y = (x[:, 1] >= -1 - gap) & (x[:, 1] <= 3 * rect_h + 3 * gap - 1)
    near_z = (x[:, 2] >= -1 - gap) & (x[:, 2] <= 3 * rect_l + 3 * gap - 1)

    potential = torch.ones_like(x[:, 0]) * 100
    potential[near_x & near_y & near_z] = 70
    potential[isin_x & isin_y & isin_z] = 0

    very_near_x = (
        ((x[:, 0] >= -1 - gap // 4) & (x[:, 0] <= rect_w - 1 + gap // 4))
        | ((x[:, 0] >= rect_w + gap - 1 - gap // 4) & (x[:, 0] <= 2 * rect_w + gap - 1 + gap // 4))
        | ((x[:, 0] >= 2 * rect_w + 2 * gap - 1 - gap // 4) & (x[:, 0] <= 3 * rect_w + 2 * gap - 1 + gap // 4))
    )
    very_near_y = (
        ((x[:, 1] >= -1 - gap // 4) & (x[:, 1] <= rect_h - 1 + gap // 4))
        | ((x[:, 1] >= rect_h + gap - 1 - gap // 4) & (x[:, 1] <= 2 * rect_h + gap - 1 + gap // 4))
        | ((x[:, 1] >= 2 * rect_h + 2 * gap - 1 - gap // 4) & (x[:, 1] <= 3 * rect_h + 2 * gap - 1 + gap // 4))
    )
    very_near_z = (
        ((x[:, 2] >= -1 - gap // 4) & (x[:, 2] <= rect_l - 1 + gap // 4))
        | ((x[:, 2] >= rect_l + gap - 1 - gap // 4) & (x[:, 2] <= 2 * rect_l + gap - 1 + gap // 4))
        | ((x[:, 2] >= 2 * rect_l + 2 * gap - 1 - gap // 4) & (x[:, 2] <= 3 * rect_l + 2 * gap - 1 + gap // 4))
    )
    potential[very_near_x & very_near_y & very_near_z] = 50
    potential[isin_x & isin_y & isin_z] = 0

    return potential


if __name__ == "__main__":
    I = "in-and-out"
    Z = "zod-ps"
    method = Z  # "in-and-out" or "zod-ps"

    num_particles = 900
    prox_sigma = 5.0

    match method:
        case "in-and-out":
            sampler = InAndOutParallel(
                objective_fn=potential_fn,
                dim=3,
                num_particles=num_particles,
                prox_sigma=prox_sigma,
                num_iter=1000,
                max_trial=10000,
            )
            initial_X = torch.tensor([[0.0, 0.0, 0.0]]).repeat(num_particles, 1)
        case "zod-ps":
            sampler = ZodPS(
                objective_fn=potential_fn,
                dim=3,
                num_particles=num_particles,
                prox_sigma=prox_sigma,
                num_iter=1000,
                diffusion_steps=100,
                interim_samples=100,
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

        # ax.set_xlim(-5, 40)
        # ax.set_ylim(-5, 25)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
