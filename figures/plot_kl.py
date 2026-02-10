from pathlib import Path

import infomeasure as im
import matplotlib.pyplot as plt
import numpy as np

root_dir = Path(__file__).parents[1].resolve()
reference_samples_npy = root_dir.joinpath("exp_iter_1000/logs_20250917_184101/rgo_rs_samples_seed0.npy")

rgo_rs_loop_files = sorted(root_dir.glob("exp_iter_1000/logs_2025*/rgo_rs_loop_chain*_seed*.npy"))
zodps_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_183930/samples_k*_seed*.npy"))
no_interaction_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_183948/samples_no_interaction_k*_seed*.npy"))

shuffle_y_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_184231/samples_shuffle_k*_seed*.npy"))
shuffle_x_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_184302/samples_shuffle_k*_seed*.npy"))
shuffle_xy_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_184328/samples_shuffle_k*_seed*.npy"))
fix_y_files = sorted(root_dir.glob("exp_iter_1000/logs_20250917_184401/samples_shuffle_k*_seed*.npy"))

np.random.seed(0)
reference_samples: np.ndarray = np.load(reference_samples_npy)
reference_samples = np.random.permutation(reference_samples)[:1000]
print("Loaded reference_samples shape:", reference_samples.shape)


def process_samples(files: list[Path], seed: int):
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
    initial_samples = np.random.randn(1000, 5)
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


rgo_rs_loop_results = {}
zodps_results = {}
no_interaction_results = {}
shuffle_y_results = {}
shuffle_x_results = {}
shuffle_xy_results = {}
fix_y_results = {}
for seed in range(10):
    rs_samples = []
    for chain_sample_file in rgo_rs_loop_files:
        if f"seed{seed}" not in str(chain_sample_file):
            continue
        samples = np.load(chain_sample_file)
        rs_samples.append(samples)
    rs_samples = np.stack(rs_samples)  # shape (100, 1000, 5)
    rs_samples = np.transpose(rs_samples, (1, 0, 2))  # shape (1000, 100, 5)
    print("Loaded rs_samples shape:", rs_samples.shape)

    rs_kl_history = []
    np.random.seed(seed)
    initial_samples = np.random.randn(1000, 5)
    rs_kl_history.append(im.kld(initial_samples, reference_samples, approach="metric", k=4, minkowski_p=2))
    for i in range(0, 1000, 10):
        result_sample_batch = []
        for j in range(10):
            result_samples = rs_samples[i + j]
            result_sample_batch.append(result_samples)
        result_samples = np.vstack(result_sample_batch)
        kl = im.kld(result_samples, reference_samples, approach="metric", k=4, minkowski_p=2)
        rs_kl_history.append(kl)
    rgo_rs_loop_results[seed] = rs_kl_history

    zodps_results[seed] = process_samples(zodps_files, seed)
    no_interaction_results[seed] = process_samples(no_interaction_files, seed)

    shuffle_y_results[seed] = process_samples(shuffle_y_files, seed)
    shuffle_x_results[seed] = process_samples(shuffle_x_files, seed)
    shuffle_xy_results[seed] = process_samples(shuffle_xy_files, seed)
    fix_y_results[seed] = process_samples(fix_y_files, seed)

rgo_rs_results = np.array([rgo_rs_loop_results[seed] for seed in range(10)])
rgo_rs_results_mean = rgo_rs_results.mean(axis=0)
rgo_rs_results_std = rgo_rs_results.std(axis=0)

zodps_results = np.array([zodps_results[seed] for seed in range(10)])
zodps_results_mean = zodps_results.mean(axis=0)
zodps_results_std = zodps_results.std(axis=0)

no_interaction_results = np.array([no_interaction_results[seed] for seed in range(10)])
no_interaction_results_mean = no_interaction_results.mean(axis=0)
no_interaction_results_std = no_interaction_results.std(axis=0)

plt.figure(figsize=(8, 4))
plt.plot(
    range(0, 1001, 10),
    rgo_rs_results_mean,
    marker="s",
    linestyle="--",
    markersize=2.5,
    label="RGO (thinning=10)",
    color="C0",
)
plt.fill_between(
    range(0, 1001, 10),
    rgo_rs_results_mean - rgo_rs_results_std,
    rgo_rs_results_mean + rgo_rs_results_std,
    color="C0",
    alpha=0.3,
)
plt.plot(range(0, 1001, 10), zodps_results_mean, marker="o", markersize=3, label="Ours (diffusion_step=10)", color="C1")
plt.fill_between(
    range(0, 1001, 10),
    zodps_results_mean - zodps_results_std,
    zodps_results_mean + zodps_results_std,
    color="C1",
    alpha=0.3,
)
plt.plot(
    range(0, 1001, 10),
    no_interaction_results_mean,
    marker="^",
    linestyle=":",
    markersize=2,
    label="Ours w/o Interaction",
    color="C2",
    zorder=-1,
)
plt.fill_between(
    range(0, 1001, 10),
    no_interaction_results_mean - no_interaction_results_std,
    no_interaction_results_mean + no_interaction_results_std,
    color="C2",
    alpha=0.3,
    zorder=-1,
)
plt.xlabel("Iteration")
plt.ylabel("Estimated KL divergence")
plt.title("KL divergence (100 particles vs 100 chains, aggregated every 10 steps)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("kl_divergence_comparison.pdf", dpi=300)
plt.close()


shuffle_y_results = np.array([shuffle_y_results[seed] for seed in range(10)])
shuffle_y_results_mean = shuffle_y_results.mean(axis=0)
shuffle_y_results_std = shuffle_y_results.std(axis=0)

shuffle_x_results = np.array([shuffle_x_results[seed] for seed in range(10)])
shuffle_x_results_mean = shuffle_x_results.mean(axis=0)
shuffle_x_results_std = shuffle_x_results.std(axis=0)

shuffle_xy_results = np.array([shuffle_xy_results[seed] for seed in range(10)])
shuffle_xy_results_mean = shuffle_xy_results.mean(axis=0)
shuffle_xy_results_std = shuffle_xy_results.std(axis=0)

plt.figure(figsize=(8, 4))
plt.plot(range(0, 1001, 10), zodps_results_mean, marker="o", markersize=3, label="Our setup", color="C1")
plt.fill_between(
    range(0, 1001, 10),
    zodps_results_mean - zodps_results_std,
    zodps_results_mean + zodps_results_std,
    color="C1",
    alpha=0.3,
)
plt.plot(range(0, 1001, 10), shuffle_xy_results_mean, label="Shuffle XY", color="C3", linestyle=":", zorder=-10)
plt.fill_between(
    range(0, 1001, 10),
    shuffle_xy_results_mean - shuffle_xy_results_std,
    shuffle_xy_results_mean + shuffle_xy_results_std,
    color="C3",
    alpha=0.3,
    zorder=-10,
)
plt.plot(range(0, 1001, 10), shuffle_x_results_mean, label="Shuffle X", color="C4", linestyle="-.", zorder=-9)
plt.fill_between(
    range(0, 1001, 10),
    shuffle_x_results_mean - shuffle_x_results_std,
    shuffle_x_results_mean + shuffle_x_results_std,
    color="C4",
    alpha=0.3,
    zorder=-9,
)
plt.plot(range(0, 1001, 10), shuffle_y_results_mean, label="Shuffle Y", color="C5", linestyle="--")
plt.fill_between(
    range(0, 1001, 10),
    shuffle_y_results_mean - shuffle_y_results_std,
    shuffle_y_results_mean + shuffle_y_results_std,
    color="C5",
    alpha=0.3,
)
plt.xlabel("Iteration")
plt.ylabel("Estimated KL divergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("kl_comp_shuffle.pdf", dpi=300)
plt.close()

fix_y_results = np.array([fix_y_results[seed] for seed in range(10)])
fix_y_results_mean = fix_y_results.mean(axis=0)
fix_y_results_std = fix_y_results.std(axis=0)

plt.figure(figsize=(8, 4))
plt.plot(fix_y_results_mean, label="Shuffle Y", color="C0")
plt.fill_between(
    np.arange(len(fix_y_results_mean)),
    fix_y_results_mean - fix_y_results_std,
    fix_y_results_mean + fix_y_results_std,
    color="C0",
    alpha=0.3,
)
plt.xlabel("Iteration")
plt.ylabel("KL divergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("kl_fix_y.pdf", dpi=300)
plt.close()
