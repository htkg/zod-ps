# Figure 1 reproduction for:
# Liang, J., & Chen, Y. A Proximal Algorithm for Sampling. Transactions on Machine Learning Research.
# Section 6 Computational results (Gaussian–Laplace mixture)
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from baseline_lmc import (
    ArgParser,
    M,
    d,
    eta,
    make_geometry,
    make_potential,
    pdf_mixture_coord3,
)
from scipy.optimize import minimize
from tqdm import tqdm


# ----------------------------
# Proximal optimization: minimize f_eta_y(x) = f(x) + (1/(2η))||x - y||^2
# Return an approximate stationary point w s.t. ||∇ f_eta_y(w)|| small
# ----------------------------
def f_eta_y(f: Callable[[np.ndarray], float], x: np.ndarray, y: np.ndarray) -> float:
    dx = x - y
    return f(x) + 0.5 * np.vdot(dx, dx) / eta


def grad_f_eta_y(grad_f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return grad_f(x) + (x - y) / eta


def prox_stationary_point(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    maxit: int = 200,
) -> np.ndarray:
    def fun(z: np.ndarray):
        return f_eta_y(f, z, y)

    def jac(z: np.ndarray):
        return grad_f_eta_y(grad_f, z, y)

    res = minimize(fun, y if x0 is None else x0, jac=jac, method="L-BFGS-B", options=dict(maxiter=maxit, gtol=tol))
    return res.x


# ----------------------------
# RGO via approximate rejection sampling (Algorithm 2)
# ----------------------------
def h1_y(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    gy: np.ndarray,
    M: float,
    eta: float,
    alpha: float = 1.0,
    delta: float = 1.0,
):
    # (9)  h_{1,y}^w(x)
    return (
        f(w)
        + np.dot(gy, x - w)
        - 0.5 * M * np.linalg.norm(x - w) ** 2
        + 0.5 * np.linalg.norm(x - y) ** 2 / eta
        - 0.5 * (1.0 - alpha) * delta
    )


def rgo_sample(
    rng: np.random.Generator,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x_init: np.ndarray | None = None,
):
    # Step 1: approximate stationary point w of f_eta_y (f(x) + 1/(2η)||x - y||^2)
    w = prox_stationary_point(f, grad_f, y, x0=x_init, tol=1e-6, maxit=100)
    gy = grad_f(w)  # f'(w)
    # Proposal N(mu, Sigma) with mu = (y - η( f'(w)+M w ))/(1-ηM), Sigma = η/(1-ηM) I
    assert eta * M < 1.0, "need ηM < 1"
    mu = (y - eta * (gy + M * w)) / (1.0 - eta * M)
    std = math.sqrt(eta / (1.0 - eta * M))

    while True:
        X = mu + std * rng.normal(size=d)  # Step 2
        # Step 4: acceptance probability = exp(-fη^y(X)) / exp(-h1_y(X))
        log_num = -f(X) - 0.5 * np.linalg.norm(X - y) ** 2 / eta
        log_den = -h1_y(f, X, y, w, gy, M, eta, alpha=1.0, delta=1.0)
        acc_log = log_num - log_den
        if np.log(rng.random()) <= acc_log:
            return X, w


# ----------------------------
# Proximal sampler (ASF): iterate  X_t ~ π_{X|Y=y_{t-1}},  Y_t ~ N(X_t, η I)
# ----------------------------
def run_proximal(
    rng: np.random.Generator,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    T=500_000,
    burnin=100_000,
):
    y = rng.normal(size=d)  # initial Y_0 ~ N(0, I)
    samples = []
    x_warm = None
    trace3 = []
    for t in tqdm(range(T)):
        x, x_warm = rgo_sample(rng, f, grad_f, y, x_init=x_warm)
        # next Y ~ N(x, η I)
        y = x + math.sqrt(eta) * rng.normal(size=d)
        trace3.append(x[2])  # 3rd coordinate (index 2)

        if t >= burnin:
            samples.append(x.copy())

    return np.array(samples), np.array(trace3)


if __name__ == "__main__":
    parser = ArgParser()
    seed = parser.seed

    rng = np.random.default_rng(seed)
    Q, Sigma, logdetQ = make_geometry(rng)
    f, grad_f = make_potential(Q, logdetQ)

    T = 500_000
    burnin = 100_000
    samples, trace3 = run_proximal(rng, f, grad_f, T=T, burnin=burnin)

    np.save(parser.outdir / f"rgo_rs_samples_seed{seed}.npy", samples)

    x3 = samples[:, 2]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))

    ax1.hist(x3, bins=100, density=True, alpha=0.7, edgecolor="none")
    xmin, xmax = np.percentile(x3, [0.5, 99.5])
    xs = np.linspace(min(-4.0, xmin), max(4.0, xmax), 800)
    ax1.plot(xs, pdf_mixture_coord3(xs, Sigma), lw=2.0)
    ax1.set_title("Gaussian–Laplace mixture (Proximal sampler / ASF + RGO)")
    ax1.set_xlabel("x[3]")

    ax2.plot(trace3, lw=0.6)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("x[3]")
    ax2.set_title("Trace of the 3rd coordinate")

    plt.tight_layout()
    plt.savefig(parser.outdir / f"rgo_rs_seed{seed}.pdf", dpi=300)
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}")
        pass

    plt.close()
