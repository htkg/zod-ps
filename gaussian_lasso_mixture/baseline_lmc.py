# Reproduce Figure 2: Gaussian–Lasso mixture using LMC
# Liang, J., & Chen, Y. A Proximal Algorithm for Sampling. Transactions on Machine Learning Research.
import argparse
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------- mixture and geometry (Sec. 6) ----------
d = 5
S = np.diag([14, 15, 16, 17, 18])
one = np.ones(d)


def make_geometry(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    A = rng.normal(size=(d, d))
    U, _ = np.linalg.qr(A)
    Q = U @ S @ U.T
    Q = 0.5 * (Q + Q.T)  # symmetrize
    logdetQ = np.log(np.linalg.det(Q))
    Sigma = np.linalg.inv(Q)  # for 1D Gaussian marginal
    return Q, Sigma, logdetQ


def make_potential(
    Q: np.ndarray,
    logdetQ: float,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    def log_gauss_Q(x: np.ndarray) -> float:  # log φ_Q(x-1)
        dx = x - one
        return 0.5 * logdetQ - 0.5 * d * np.log(2 * np.pi) - 0.5 * dx @ Q @ dx

    def grad_log_gauss_Q(x: np.ndarray) -> np.ndarray:  # ∇ log φ_Q
        return -Q @ (x - one)

    def f(x: np.ndarray) -> float:
        lg = np.log(0.5) + log_gauss_Q(x)
        ll = np.log(0.5) + d * np.log(2.0) - np.sum(np.abs(4.0 * x))
        m = max(lg, ll)
        return -(m + np.log(np.exp(lg - m) + np.exp(ll - m)))

    def grad_f(x: np.ndarray) -> np.ndarray:
        lg = np.log(0.5) + log_gauss_Q(x)
        ll = np.log(0.5) + d * np.log(2.0) - np.sum(np.abs(4.0 * x))
        m = max(lg, ll)
        eg, el = np.exp(lg - m), np.exp(ll - m)
        Z = eg + el
        wg, wl = eg / Z, el / Z
        g1 = grad_log_gauss_Q(x)
        s = np.sign(x)
        s[np.abs(x) < 1e-12] = 0.0
        g2 = -4.0 * s  # grad of log Laplace with scale 1/4 is -4*sign(x)
        return -(wg * g1 + wl * g2)

    return f, grad_f


# ---------- step size η = 1/(M d), with M = L1, L1 = 27 (Sec. 6) ----------
L1 = 27.0
M = L1
eta = 1.0 / (M * d)


# ---------- LMC (Eq. (50)) ----------
def run_lmc(
    rng: np.random.Generator,
    grad_f: Callable[[np.ndarray], np.ndarray],
    T: int = 500_000,
    burnin: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    y = rng.normal(size=d)
    keep, trace3 = [], []
    for t in tqdm(range(T)):
        noise = math.sqrt(2.0 * eta) * rng.normal(size=d)
        y = y - eta * grad_f(y) + noise
        trace3.append(y[2])

        if t >= burnin:
            keep.append(y.copy())

    return np.array(keep), np.array(trace3)


# ---------- true 1D marginal of the 3rd coordinate (mixture) ----------
def pdf_mixture_coord3(grid: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    mu = 1.0
    std = math.sqrt(Sigma[2, 2])
    gauss = (1.0 / (math.sqrt(2.0 * math.pi) * std)) * np.exp(-0.5 * ((grid - mu) / std) ** 2)
    b = 0.25
    lap = (1.0 / (2.0 * b)) * np.exp(-np.abs(grid) / b)
    return 0.5 * gauss + 0.5 * lap


class Args(argparse.Namespace):
    seed: int
    outdir: Path


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--seed", type=int, default=0, help="random seed")
        self.add_argument("--outdir", type=str, default=Path(__file__).parent, help="output directory")

        args = self.parse_args(namespace=Args())
        self.seed = args.seed
        self.outdir = Path(args.outdir)

        self.outdir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = ArgParser()
    seed = parser.seed

    rng = np.random.default_rng(seed)
    Q, Sigma, logdetQ = make_geometry(rng)
    f, grad_f = make_potential(Q, logdetQ)

    T, burnin = 500_000, 100_000
    samples, trace3 = run_lmc(rng, grad_f, T=T, burnin=burnin)

    x3 = samples[:, 2]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))

    ax1.hist(x3, bins=100, density=True, alpha=0.7, edgecolor="none")
    xs = np.linspace(-4.0, 4.0, 800)
    ax1.plot(xs, pdf_mixture_coord3(xs, Sigma), lw=2.0)
    ax1.set_title("Gaussian–Lasso mixture using LMC")
    ax1.set_xlabel("x[3]")

    ax2.plot(trace3, lw=0.6)
    ax2.set_title("Trace of the 3rd coordinate")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("x[3]")

    plt.tight_layout()
    plt.savefig(parser.outdir / f"baseline_lmc_seed{seed}.pdf", dpi=300)
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}")
        pass

    plt.close()
