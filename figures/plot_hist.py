import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
from gaussian_lasso_mixture.baseline_lmc import make_geometry, pdf_mixture_coord3

root_dir = Path(__file__).parents[1].resolve()
reference_samples_npy = root_dir.joinpath("exp_iter_1000/logs_20250917_184101/rgo_rs_samples_seed0.npy")

rgo_rs_loop_files = sorted(root_dir.glob("exp_iter_1000/logs_2025*/rgo_rs_loop_chain*_seed*.npy"))
zodps_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_183930/samples_k*_seed*.npy"))


rgo_rs_last_samples = []
for seed in range(10):
    rs_samples = []
    for chain_sample_file in rgo_rs_loop_files:
        if f"seed{seed}" not in str(chain_sample_file):
            continue
        samples = np.load(chain_sample_file)
        rs_samples.append(samples)

    rs_samples = np.stack(rs_samples)  # shape (100, 1000, 5)
    rs_samples = np.transpose(rs_samples, (1, 0, 2))  # shape (1000, 100, 5)
    rgo_rs_last_samples.append(rs_samples[-100:])  # last 100 steps

rgo_rs_last_samples = np.vstack(rgo_rs_last_samples).reshape(-1, 5)

zodps_interim_samples = []
for seed in range(10):
    zodps_samples = []
    for chain_sample_file in zodps_files:
        if f"seed{seed}" not in str(chain_sample_file):
            continue
        samples = np.load(chain_sample_file)
        zodps_samples.append(samples)

    zodps_samples = np.stack(zodps_samples)  # shape (1000, 100, 5)
    zodps_interim_samples.append(zodps_samples[50:150])  # 50-149 steps

zodps_interim_samples = np.vstack(zodps_interim_samples).reshape(-1, 5)

plt.figure(figsize=(6, 3))
nbins = 200
x3 = rgo_rs_last_samples[:, 2]
plt.hist(x3, bins=nbins, density=True, alpha=0.8, edgecolor="none", label="RGO (900-999iter)")
x3 = zodps_interim_samples[:, 2]
plt.hist(x3, bins=nbins, density=True, alpha=0.8, edgecolor="none", label="Ours (50-149iter)")

xmin, xmax = np.percentile(x3, [0.5, 99.5])
xs = np.linspace(min(-2.5, xmin), max(2.8, xmax), 1000)

Q, Sigma, logdetQ = make_geometry(np.random.default_rng(0))
plt.plot(xs, pdf_mixture_coord3(xs, Sigma), lw=2.0, color="red")  # red curve in the paper (scaled target density)
plt.xlabel("x[3]")
plt.xlim(-2.5, 2.8)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("hist_comparison_interim.pdf", dpi=300)
plt.close()


np.random.seed(0)
reference_samples: np.ndarray = np.load(reference_samples_npy)
reference_samples = np.random.permutation(reference_samples)
print("Loaded reference_samples shape:", reference_samples.shape)

zodps_interim_samples = []
for seed in range(10):
    zodps_samples = []
    for chain_sample_file in zodps_files:
        if f"seed{seed}" not in str(chain_sample_file):
            continue
        samples = np.load(chain_sample_file)
        zodps_samples.append(samples)

    zodps_samples = np.stack(zodps_samples)  # shape (1000, 100, 5)
    zodps_interim_samples.append(zodps_samples[200:300])  # 200-299 steps

zodps_interim_samples = np.vstack(zodps_interim_samples).reshape(-1, 5)

plt.figure(figsize=(6, 3))
nbins = 200
x3 = reference_samples[:, 2]
plt.hist(x3, bins=nbins, density=True, alpha=0.8, edgecolor="none", label="RGO (reference)")
x3 = zodps_interim_samples[:, 2]
plt.hist(x3, bins=nbins, density=True, alpha=0.8, edgecolor="none", label="Ours (200-299iter)")
xmin, xmax = np.percentile(x3, [0.5, 99.5])
xs = np.linspace(min(-2.5, xmin), max(2.8, xmax), 1000)
Q, Sigma, logdetQ = make_geometry(np.random.default_rng(0))
plt.plot(xs, pdf_mixture_coord3(xs, Sigma), lw=2.0, color="red")  # red curve in the paper (scaled target density)
plt.xlabel("x[3]")
plt.xlim(-2.5, 2.8)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("hist_comparison_later.pdf", dpi=300)
plt.close()
