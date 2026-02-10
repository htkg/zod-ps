import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("../uniform_dist")
from eval_torus import potential_fn, solid_torus_potential  # type: ignore

root_dir = Path(__file__).parents[1].resolve()
i_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_190120/samples_I*.npy"))
z_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_190157/samples_Z*.npy"))


def num_in_t1(x: np.ndarray) -> int:
    x_tensor = torch.from_numpy(x)
    val = solid_torus_potential(x_tensor, center=(10.0, 0.0, 0.0), R=10.0, r=1.0, V_out=100.0)
    return int((val == 0.0).sum().item())


def num_in_t2(x: np.ndarray) -> int:
    x_tensor = torch.from_numpy(x)
    val = solid_torus_potential(x_tensor, center=(-13.0, 0.0, 0.0), R=3.0, r=1.0, V_out=100.0)
    return int((val == 0.0).sum().item())


seed_results = []
for seed in range(10):
    i_in_t1_history = []
    i_in_t2_history = []
    z_in_t1_history = []
    z_in_t2_history = []

    for i_file in i_files:
        if f"seed{seed}" not in str(i_file):
            continue
        data = np.load(i_file)
        i_in_t1_history.append(num_in_t1(data))
        i_in_t2_history.append(num_in_t2(data))

    for z_file in z_files:
        if f"seed{seed}" not in str(z_file):
            continue
        data = np.load(z_file)
        z_in_t1_history.append(num_in_t1(data))
        z_in_t2_history.append(num_in_t2(data))

    seed_results.append([i_in_t1_history, i_in_t2_history, z_in_t1_history, z_in_t2_history])

all_results = np.array(seed_results)  # (num_seeds, 4, num_iters)
print("all_results shape:", all_results.shape)

res_mean = all_results.mean(axis=0)
res_std = all_results.std(axis=0)

plt.figure(figsize=(8, 4))

x = np.arange(len(res_mean[0])) + 1
plt.plot(x, res_mean[0], label="In-and-Out in T1", color="C0")
plt.fill_between(
    x,
    res_mean[0] - res_std[0],
    res_mean[0] + res_std[0],
    color="C0",
    alpha=0.3,
)
plt.plot(x, res_mean[1], label="In-and-Out in T2", color="C1")
plt.fill_between(
    x,
    res_mean[1] - res_std[1],
    res_mean[1] + res_std[1],
    color="C1",
    alpha=0.3,
)
plt.plot(x, res_mean[2], label="Outs in T1", color="C2")
plt.fill_between(
    x,
    res_mean[2] - res_std[2],
    res_mean[2] + res_std[2],
    color="C2",
    alpha=0.3,
)
plt.plot(x, res_mean[3], label="Ours in T2", color="C3")
plt.fill_between(
    x,
    res_mean[3] - res_std[3],
    res_mean[3] + res_std[3],
    color="C3",
    alpha=0.3,
)
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Number of particles in the torus")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("num_in_torus_comparison.pdf", dpi=300)
plt.close()


seed = 0
for k in [3, 10, 50, 200, 800]:
    for i_file in i_files:
        if f"seed{seed}" not in str(i_file):
            continue
        if f"k{k:04d}" not in str(i_file):
            continue
        data = np.load(i_file)
        plt.figure(figsize=(6, 4))
        plt.scatter(data[:, 0], data[:, 1], c=potential_fn(torch.from_numpy(data)).numpy(), cmap="viridis", s=8)
        plt.xlim(-20, 25)
        plt.ylim(-15, 15)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"In-and-Out: seed={seed}, k={k}")
        plt.tight_layout()
        plt.savefig(f"uniform_in_and_out_seed{seed}_k{k}.pdf", dpi=300)
        plt.close()

    for z_file in z_files:
        if f"seed{seed}" not in str(z_file):
            continue
        if f"k{k:04d}" not in str(z_file):
            continue
        data = np.load(z_file)
        plt.figure(figsize=(6, 4))
        plt.scatter(data[:, 0], data[:, 1], c=potential_fn(torch.from_numpy(data)).numpy(), cmap="viridis", s=8)
        plt.xlim(-20, 25)
        plt.ylim(-15, 15)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"Ours: seed={seed}, k={k}")
        plt.tight_layout()
        plt.savefig(f"uniform_ours_seed{seed}_k{k}.pdf", dpi=300)
        plt.close()
