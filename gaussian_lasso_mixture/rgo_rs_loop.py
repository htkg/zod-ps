import numpy as np
from baseline_lmc import ArgParser, make_geometry, make_potential
from rgo_rs import run_proximal
from tqdm import trange

if __name__ == "__main__":
    parser = ArgParser()
    seed = parser.seed

    rng = np.random.default_rng(seed)
    Q, Sigma, logdetQ = make_geometry(rng)
    f, grad_f = make_potential(Q, logdetQ)

    for i in trange(100):
        burnin = 0
        T = 10000
        skip = 10

        samples, trace3 = run_proximal(rng, f, grad_f, T=T, burnin=burnin)
        np.save(parser.outdir / f"rgo_rs_loop_chain{i}_skip{skip}_seed{seed}.npy", samples[::skip])
