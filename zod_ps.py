import math
from typing import Callable, Generator

import torch


def logpdf_isotropic_gmm(x: torch.Tensor, mus: torch.Tensor, sigma2: float) -> torch.Tensor:
    """Compute log-density of an isotropic Gaussian mixture model.

    The mixture is defined as:
        p(x) = (1/N) * sum_i N(x ; mu_i, sigma^2 I)

    where `mus` provides the mixture means.

    Args:
        x (torch.Tensor):
            Query points of shape (M, D), where M is the number of points
            and D is the dimensionality.
        mus (torch.Tensor):
            Mixture component means of shape (N, D), where N is the number
            of components and D is the dimensionality (same D as `x`).
        sigma2 (float):
            Variance of the isotropic Gaussian components.
            Must be positive.

    Returns:
        torch.Tensor:
            Log-density values of shape (M,), one per query point in `x`.
    """
    N = mus.shape[0]
    D = x.shape[1]

    # Compute pairwise squared distances of all points ||x - mu||^2
    x2 = (x**2).sum(dim=1, keepdim=True)  # [M,1]
    mu2 = (mus**2).sum(dim=1, keepdim=True).T  # [1,N]
    dist2 = x2 - 2 * x @ mus.T + mu2  # [M,N]

    const = -0.5 * D * math.log(2 * math.pi * sigma2)
    log_comp = -0.5 * dist2 / sigma2  # [M,N]
    return torch.logsumexp(log_comp, dim=1) + const - math.log(N)  # [M,]


class ZodPS:
    def __init__(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        num_particles: int = 100,
        prox_sigma: float = 1.0,
        interim_samples: int = 30,
        num_iter: int = 100,
        diffusion_steps: int = 50,
        beta_min=0.0001,
    ):
        self.objective_fn = objective_fn
        self.dim = dim
        self.num_particles = num_particles
        self.prox_sigma = prox_sigma
        self.interim_samples = interim_samples
        self.num_iter = num_iter
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min

    def sample(
        self,
        initial_X: torch.Tensor | None = None,
    ) -> Generator[tuple[torch.Tensor, int], None, torch.Tensor]:
        if initial_X is not None:
            if initial_X.shape[0] != self.num_particles or initial_X.shape[1] != self.dim:
                raise ValueError(
                    f"Invalid initial_X shape: {initial_X.shape}, expected ({self.num_particles}, {self.dim})"
                )
            X = initial_X
        else:
            X = torch.randn(self.num_particles, self.dim)

        sigma2_list = torch.linspace(self.beta_min, self.prox_sigma, self.diffusion_steps + 1)

        fn_eval_num = 0

        for _ in range(self.num_iter):
            # corresponding to Y^{k+1/2}
            Y = X + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)

            log_comp_weights = -logpdf_isotropic_gmm(Y, X, self.prox_sigma)

            x_t = X + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)

            for t in range(self.diffusion_steps, 0, -1):
                # weights[i,j]: (x_t[i], Y[j]) similarity
                #   \propto w_j N(x_t[i]|Y[j],(h+\sigma_t^2)I_d)
                weights = torch.softmax(
                    log_comp_weights
                    - ((x_t.unsqueeze(1) - Y.unsqueeze(0)) ** 2).sum(dim=-1) / (2 * (self.prox_sigma + sigma2_list[t])),
                    dim=1,
                )

                dist_idx = torch.searchsorted(
                    torch.cumsum(weights, dim=-1),
                    torch.rand(self.num_particles, self.interim_samples),
                ).clamp(max=self.num_particles - 1)  # (num_particles, interim_samples)

                scale = 1 / ((1 / sigma2_list[t]) + (1 / self.prox_sigma))  # \bar{\sigma}^2
                mu_list = (
                    (1 / sigma2_list[t]) * x_t.unsqueeze(1)  # (num_particles ,1, dim)
                    + (1 / self.prox_sigma) * Y[dist_idx]  # (num_particles, interim_samples, dim)
                ) * scale
                # x_0|x_t (num_particles, interim_samples, dim)
                x_interims = mu_list + torch.randn(self.num_particles, self.interim_samples, self.dim).mul(scale**0.5)

                c_matrix = torch.softmax(
                    -self.objective_fn(x_interims.view(-1, self.dim)).view(self.num_particles, self.interim_samples),
                    dim=1,
                )  # (num_particles, interim_samples)

                fn_eval_num += x_interims.shape[0] * x_interims.shape[1]

                dt = sigma2_list[t] - sigma2_list[t - 1]
                x_t = (
                    x_t
                    + (x_interims - x_t.unsqueeze(1)).mul(c_matrix.unsqueeze(-1)).sum(dim=1) * dt / sigma2_list[t]
                    + torch.randn(self.num_particles, self.dim).mul(dt**0.5)
                )

            X = x_t
            yield X, fn_eval_num

        return X


class ZodPS_NoInteraction:
    def __init__(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        num_particles: int = 100,
        prox_sigma: float = 1.0,
        interim_samples: int = 30,
        num_iter: int = 100,
        diffusion_steps: int = 50,
        beta_min=0.0001,
    ):
        self.objective_fn = objective_fn
        self.dim = dim
        self.num_particles = num_particles
        self.prox_sigma = prox_sigma
        self.interim_samples = interim_samples
        self.num_iter = num_iter
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min

    def sample(
        self,
        initial_X: torch.Tensor | None = None,
    ) -> Generator[tuple[torch.Tensor, int], None, torch.Tensor]:
        if initial_X is not None:
            if initial_X.shape[0] != self.num_particles or initial_X.shape[1] != self.dim:
                raise ValueError(
                    f"Invalid initial_X shape: {initial_X.shape}, expected ({self.num_particles}, {self.dim})"
                )
            X = initial_X
        else:
            X = torch.randn(self.num_particles, self.dim)

        sigma2_list = torch.linspace(self.beta_min, self.prox_sigma, self.diffusion_steps + 1)

        fn_eval_num = 0

        for _ in range(self.num_iter):
            # corresponding to Y^{k+1/2}
            Y = X + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)
            x_t = X + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)

            for t in range(self.diffusion_steps, 0, -1):
                scale = 1 / ((1 / sigma2_list[t]) + (1 / self.prox_sigma))  # \bar{\sigma}^2
                mu_list = (
                    (1 / sigma2_list[t]) * x_t.unsqueeze(1)  # (num_particles ,1, dim)
                    + (1 / self.prox_sigma)
                    * Y.unsqueeze(1).repeat(1, self.interim_samples, 1)  # (num_particles, interim_samples, dim)
                ) * scale
                # x_0|x_t (num_particles, interim_samples, dim)
                x_interims = mu_list + torch.randn(self.num_particles, self.interim_samples, self.dim).mul(scale**0.5)

                c_matrix = torch.softmax(
                    -self.objective_fn(x_interims.view(-1, self.dim)).view(self.num_particles, self.interim_samples),
                    dim=1,
                )  # (num_particles, interim_samples)

                fn_eval_num += x_interims.shape[0] * x_interims.shape[1]

                dt = sigma2_list[t] - sigma2_list[t - 1]
                x_t = (
                    x_t
                    + (x_interims - x_t.unsqueeze(1)).mul(c_matrix.unsqueeze(-1)).sum(dim=1) * dt / sigma2_list[t]
                    + torch.randn(self.num_particles, self.dim).mul(dt**0.5)
                )

            X = x_t
            yield X, fn_eval_num

        return X


class ZodPS_Ablation:
    def __init__(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        num_particles: int = 100,
        prox_sigma: float = 1.0,
        interim_samples: int = 30,
        num_iter: int = 100,
        diffusion_steps: int = 50,
        beta_min=0.0001,
        shuffle_Y: bool = False,
        shuffle_x_t: bool = False,
        update_Y: bool = True,
    ):
        self.objective_fn = objective_fn
        self.dim = dim
        self.num_particles = num_particles
        self.prox_sigma = prox_sigma
        self.interim_samples = interim_samples
        self.num_iter = num_iter
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min
        self.shuffle_Y = shuffle_Y
        self.shuffle_x_t = shuffle_x_t
        self.update_Y = update_Y

    def sample(
        self,
        initial_X: torch.Tensor | None = None,
    ) -> Generator[tuple[torch.Tensor, int], None, torch.Tensor]:
        if initial_X is not None:
            if initial_X.shape[0] != self.num_particles or initial_X.shape[1] != self.dim:
                raise ValueError(
                    f"Invalid initial_X shape: {initial_X.shape}, expected ({self.num_particles}, {self.dim})"
                )
            X = initial_X
        else:
            X = torch.randn(self.num_particles, self.dim)

        sigma2_list = torch.linspace(self.beta_min, self.prox_sigma, self.diffusion_steps + 1)

        fn_eval_num = 0

        for _ in range(self.num_iter):
            # corresponding to Y^{k+1/2}
            if _ == 0 or self.update_Y:
                if self.shuffle_Y:
                    idx = torch.randint(0, self.num_particles, (self.num_particles,))
                    Y = X[idx] + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)
                else:
                    Y = X + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)

            log_comp_weights = -logpdf_isotropic_gmm(Y, X, self.prox_sigma)

            if self.shuffle_x_t:
                idx = torch.randint(0, self.num_particles, (self.num_particles,))
                x_t = X[idx] + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)
            else:
                x_t = X + torch.randn(self.num_particles, self.dim).mul(self.prox_sigma**0.5)

            for t in range(self.diffusion_steps, 0, -1):
                # weights[i,j]: (x_t[i], Y[j]) similarity
                #   \propto w_j N(x_t[i]|Y[j],(h+\sigma_t^2)I_d)
                weights = torch.softmax(
                    log_comp_weights
                    - ((x_t.unsqueeze(1) - Y.unsqueeze(0)) ** 2).sum(dim=-1) / (2 * (self.prox_sigma + sigma2_list[t])),
                    dim=1,
                )

                dist_idx = torch.searchsorted(
                    torch.cumsum(weights, dim=-1),
                    torch.rand(self.num_particles, self.interim_samples),
                ).clamp(max=self.num_particles - 1)  # (num_particles, interim_samples)

                scale = 1 / ((1 / sigma2_list[t]) + (1 / self.prox_sigma))  # \bar{\sigma}^2
                mu_list = (
                    (1 / sigma2_list[t]) * x_t.unsqueeze(1)  # (num_particles ,1, dim)
                    + (1 / self.prox_sigma) * Y[dist_idx]  # (num_particles, interim_samples, dim)
                ) * scale
                # x_0|x_t (num_particles, interim_samples, dim)
                x_interims = mu_list + torch.randn(self.num_particles, self.interim_samples, self.dim).mul(scale**0.5)

                c_matrix = torch.softmax(
                    -self.objective_fn(x_interims.view(-1, self.dim)).view(self.num_particles, self.interim_samples),
                    dim=1,
                )  # (num_particles, interim_samples)

                fn_eval_num += x_interims.shape[0] * x_interims.shape[1]

                dt = sigma2_list[t] - sigma2_list[t - 1]
                x_t = (
                    x_t
                    + (x_interims - x_t.unsqueeze(1)).mul(c_matrix.unsqueeze(-1)).sum(dim=1) * dt / sigma2_list[t]
                    + torch.randn(self.num_particles, self.dim).mul(dt**0.5)
                )

            X = x_t
            yield X, fn_eval_num

        return X
