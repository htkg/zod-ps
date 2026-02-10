from typing import Callable, Generator

import torch


class InAndOutParallel:
    def __init__(
        self,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        num_particles: int = 100,
        prox_sigma: float = 1.0,
        num_iter: int = 100,
        max_trial: int = 1000,
        report_worst: bool = False,
    ):
        self.objective_fn = objective_fn
        self.dim = dim
        self.num_particles = num_particles
        self.prox_sigma = prox_sigma
        self.num_iter = num_iter
        self.max_trial = max_trial
        self.report_worst = report_worst

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

        fn_eval_num = 0

        for _ in range(self.num_iter):
            Y = X + torch.randn(X.shape[0], self.dim).mul(self.prox_sigma**0.5)

            X_new = torch.zeros_like(X)
            require_update = torch.ones(X.shape[0], dtype=torch.bool)

            trial = 0
            while require_update.any() and trial < self.max_trial:
                proposal = Y[require_update] + torch.randn(int(require_update.sum().item()), self.dim).mul(
                    self.prox_sigma**0.5
                )

                is_accepted = self.objective_fn(proposal) == 0
                if not self.report_worst:
                    fn_eval_num += proposal.shape[0]

                if is_accepted.any():
                    idx = require_update.nonzero(as_tuple=True)[0]
                    accepted_idx = idx[is_accepted]
                    X_new[accepted_idx] = proposal[is_accepted]
                    require_update[accepted_idx] = False

                trial += 1

            if self.report_worst:
                fn_eval_num += trial

            X = X_new[~require_update]  # only keep accepted samples
            yield X, fn_eval_num

        return X
