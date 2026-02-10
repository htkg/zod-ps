from pathlib import Path

import infomeasure as im
import matplotlib.pyplot as plt
import numpy as np

root_dir = Path(__file__).parents[1].resolve()
reference_samples_npy = root_dir.joinpath("exp_iter_1000/logs_20250917_184101/rgo_rs_samples_seed0.npy")

# rgo_rs_loop_files = sorted(root_dir.glob("exp_iter_1000/logs_2025*/rgo_rs_loop_chain*_seed*.npy"))
zodps_files = sorted(root_dir.glob("logs/rebuttal/logs_20251123_110835/samples_k*_seed*.npy"))
n400_m1000_files = sorted(root_dir.glob("logs/rebuttal/logs_20251123_110955/samples_k*_seed*.npy"))
n1000_m400_files = sorted(root_dir.glob("logs/rebuttal/logs_20251123_111044/samples_k*_seed*.npy"))

np.random.seed(0)
reference_samples: np.ndarray = np.load(reference_samples_npy)
reference_samples_1000 = np.random.permutation(reference_samples)[:1000]
reference_samples_4000 = np.random.permutation(reference_samples)[:4000]
reference_samples_10000 = np.random.permutation(reference_samples)[:10000]
print(
    "Loaded reference_samples shape:",
    reference_samples.shape,
    reference_samples_1000.shape,
    reference_samples_4000.shape,
    reference_samples_10000.shape,
)


def process_samples(files: list[Path], seed: int, n=1000):
    if n == 1000:
        reference_samples = reference_samples_1000
    elif n == 4000:
        reference_samples = reference_samples_4000
    elif n == 10000:
        reference_samples = reference_samples_10000
    else:
        raise ValueError(f"n must be one of 1000, 4000, 10000, got {n}")

    all_samples = []
    for file in files:
        if f"seed{seed}" not in str(file):
            continue
        samples = np.load(file)
        all_samples.append(samples)

    all_samples = np.stack(all_samples)  # shape (1000, 100, 5)
    print(f"Loaded all_samples shape for seed {seed}:", all_samples.shape)

    kl_history = []
    np.random.seed(seed)
    initial_samples = np.random.randn(n, 5)
    kl_history.append(im.kld(initial_samples, reference_samples, approach="metric", k=4, minkowski_p=2))
    for i in range(0, 1000, 10):
        result_sample_batch = []
        for j in range(10):
            result_samples = all_samples[i + j]
            result_sample_batch.append(result_samples)
        result_samples = np.vstack(result_sample_batch)

        kl = im.kld(result_samples, reference_samples, approach="metric", k=4, minkowski_p=2)
        kl_history.append(kl)

    return kl_history


zodps_results = {}
n400_m1000_results = {}
n1000_m400_results = {}
for seed in range(10):
    # rs_samples = []
    # for chain_sample_file in rgo_rs_loop_files:
    #     if f"seed{seed}" not in str(chain_sample_file):
    #         continue
    #     samples = np.load(chain_sample_file)
    #     rs_samples.append(samples)
    # rs_samples = np.stack(rs_samples)  # shape (100, 1000, 5)
    # rs_samples = np.transpose(rs_samples, (1, 0, 2))  # shape (1000, 100, 5)
    # print("Loaded rs_samples shape:", rs_samples.shape)

    # rs_kl_history = []
    # np.random.seed(seed)
    # initial_samples = np.random.randn(1000, 5)
    # rs_kl_history.append(im.kld(initial_samples, reference_samples, approach="metric", k=4, minkowski_p=2))
    # for i in range(0, 1000, 10):
    #     result_sample_batch = []
    #     for j in range(10):
    #         result_samples = rs_samples[i + j]
    #         result_sample_batch.append(result_samples)
    #     result_samples = np.vstack(result_sample_batch)
    #     kl = im.kld(result_samples, reference_samples, approach="metric", k=4, minkowski_p=2)
    #     rs_kl_history.append(kl)
    # rgo_rs_loop_results[seed] = rs_kl_history

    zodps_results[seed] = process_samples(zodps_files, seed, n=1000)
    n400_m1000_results[seed] = process_samples(n400_m1000_files, seed, n=4000)
    n1000_m400_results[seed] = process_samples(n1000_m400_files, seed, n=10000)


zodps_results = np.array([zodps_results[seed] for seed in range(10)])
zodps_results_mean = zodps_results.mean(axis=0)
zodps_results_std = zodps_results.std(axis=0)

n400_m1000_results = np.array([n400_m1000_results[seed] for seed in range(10)])
n400_m1000_results_mean = n400_m1000_results.mean(axis=0)
n400_m1000_results_std = n400_m1000_results.std(axis=0)

n1000_m400_results = np.array([n1000_m400_results[seed] for seed in range(10)])
n1000_m400_results_mean = n1000_m400_results.mean(axis=0)
n1000_m400_results_std = n1000_m400_results.std(axis=0)


plt.plot(range(0, 1001, 10), zodps_results_mean, marker="o", markersize=3, label="N=100, M=4000", color="C1")
plt.fill_between(
    range(0, 1001, 10),
    zodps_results_mean - zodps_results_std,
    zodps_results_mean + zodps_results_std,
    color="C1",
    alpha=0.3,
)
plt.plot(
    range(0, 1001, 10),
    n400_m1000_results_mean,
    marker="s",
    linestyle="--",
    markersize=2,
    label="N=400, M=1000",
    color="C6",
    zorder=-2,
)
plt.fill_between(
    range(0, 1001, 10),
    n400_m1000_results_mean - n400_m1000_results_std,
    n400_m1000_results_mean + n400_m1000_results_std,
    color="C6",
    alpha=0.3,
    zorder=-2,
)
plt.plot(
    range(0, 1001, 10),
    n1000_m400_results_mean,
    marker="^",
    linestyle=":",
    markersize=2,
    label="N=1000, M=400",
    color="C7",
    zorder=-1,
)
plt.fill_between(
    range(0, 1001, 10),
    n1000_m400_results_mean - n1000_m400_results_std,
    n1000_m400_results_mean + n1000_m400_results_std,
    color="C7",
    alpha=0.3,
    zorder=-1,
)
plt.xlim(-10, 400)
plt.xlabel("Iteration")
plt.ylabel("Estimated KL divergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("kl_divergence_comparison_mn.pdf", dpi=300)
plt.close()

# exclude iteration 0
plt.figure(figsize=(6.4, 3.2))
plt.plot(range(10, 1001, 10), zodps_results_mean[1:], marker="o", markersize=3, label="N=100, M=4000", color="C1")
plt.fill_between(
    range(10, 1001, 10),
    zodps_results_mean[1:] - zodps_results_std[1:],
    zodps_results_mean[1:] + zodps_results_std[1:],
    color="C1",
    alpha=0.3,
)
plt.plot(
    range(10, 1001, 10),
    n400_m1000_results_mean[1:],
    marker="s",
    linestyle="--",
    markersize=2,
    label="N=400, M=1000",
    color="C6",
    zorder=-2,
)
plt.fill_between(
    range(10, 1001, 10),
    n400_m1000_results_mean[1:] - n400_m1000_results_std[1:],
    n400_m1000_results_mean[1:] + n400_m1000_results_std[1:],
    color="C6",
    alpha=0.3,
    zorder=-2,
)
plt.plot(
    range(10, 1001, 10),
    n1000_m400_results_mean[1:],
    marker="^",
    linestyle=":",
    markersize=2,
    label="N=1000, M=400",
    color="C7",
    zorder=-1,
)
plt.fill_between(
    range(10, 1001, 10),
    n1000_m400_results_mean[1:] - n1000_m400_results_std[1:],
    n1000_m400_results_mean[1:] + n1000_m400_results_std[1:],
    color="C7",
    alpha=0.3,
    zorder=-1,
)
plt.xlim(-10, 400)
plt.xlabel("Iteration")
plt.ylabel("Estimated KL divergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("mn_exclude_0.pdf", dpi=300)
plt.close()
