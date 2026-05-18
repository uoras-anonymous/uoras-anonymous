import os
import time
import warnings
from datetime import datetime
from collections import OrderedDict
import numpy as np
import scipy
import scipy.sparse.linalg
import sksparse.cholmod
import pandas as pd
import torch
import pyamg
from utils_2_subdomains import construct_oras, Exp, set_seed
from utils_N_subdomains import decompose_domain, get_subdomains_endpoints


def build_oras_preconditioner(amatcsc_tilde_block, endpoints, num_dofs, size, num_subdomains):
    def matvec(r):
        rvec_ext = np.concatenate([r[s:e] for s, e in endpoints])
        evec_ext = scipy.sparse.linalg.spsolve(amatcsc_tilde_block, rvec_ext)

        corrvec = np.zeros(num_dofs)
        offset = 0
        for i, (idx_start, idx_end) in enumerate(endpoints):
            block_size = idx_end - idx_start
            evec_i = evec_ext[offset:offset + block_size]
            offset += block_size
            if i == 0:
                corrvec[idx_start:idx_end - size] = evec_i[:-size]
            elif i == num_subdomains - 1:
                corrvec[idx_start + size:idx_end] = evec_i[size:]
            else:
                corrvec[idx_start + size:idx_end - size] = evec_i[size:-size]
        return corrvec

    return scipy.sparse.linalg.LinearOperator(
        shape=(num_dofs, num_dofs),
        matvec=matvec,
        dtype=np.float64
    )


if __name__ == '__main__':
    set_seed(42)
    device = "cuda"
    NUM_SUBDOMAINS = 20
    ETA = 1
    TOL = 1e-6
    method = 't2'
    print(f"{NUM_SUBDOMAINS} subdomains, {method}, solver=gmres (pyamg)")

    data = np.load("data_N_subdomains/test_grids.npy", allow_pickle=True).item()
    NUM_SAMPLES = len(data['amatcsr_list'])

    if method == 'mlp':
        NUM_TRAIN_SUBDOMAINS = 10
        mlp_pars = OrderedDict(
            [
                ("fc1", torch.nn.Linear(2, 64, dtype=torch.float64)),
                ("nl1", torch.nn.Sigmoid()),
                ("fc2", torch.nn.Linear(64, 32, dtype=torch.float64)),
                ("nl2", torch.nn.Sigmoid()),
                ("fc3", torch.nn.Linear(32, 2, dtype=torch.float64)),
                ("nl3", Exp())
            ]
        )
        mlp = torch.nn.Sequential(mlp_pars)
        paramfile = f"results/train_stationary_10_subdomains_lr0.0001_20260423_205640/epoch_80"
        pars = torch.load(paramfile, weights_only=False)
        mlp.load_state_dict(pars['mlp'])
        mlp.to(device)
        expname = paramfile.split('_')[1]
        method = method + str(NUM_TRAIN_SUBDOMAINS) + expname
        print(mlp)

    rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results_test", exist_ok=True)
    log_path = os.path.join(
        "results_test",
        f"test_tol_gmres_{NUM_SUBDOMAINS}_subdomains_{method}_{run_id}.csv"
    )

    for s in range(NUM_SAMPLES):
        amat_csr = data['amatcsr_list'][s]
        hvalx, hvaly = data['hvalx_list'][s], data['hvaly_list'][s]
        etaval = data['etaval_list'][s]
        evecs = data['evecs_list'][s]
        num_dofs = amat_csr.shape[0]
        num_x, num_y = num_dofs ** (1 / 2), num_dofs ** (1 / 2)

        if num_x % NUM_SUBDOMAINS != 0:
            print(f"skipping {num_dofs}")
            continue

        overlap_pairs = decompose_domain(num_dofs, num_y, NUM_SUBDOMAINS)
        endpoints = get_subdomains_endpoints(overlap_pairs, num_dofs)
        i1, i2 = overlap_pairs[0]
        size = (i2 - i1) // 2
        overlaps = [idx for pair in overlap_pairs for idx in pair]
        overlaps = [0] + overlaps + [num_dofs]

        # Decompose system matrix into subdomain blocks
        amatcsr_list = []
        for i in range(NUM_SUBDOMAINS):
            idx_start, idx_end = endpoints[i]
            amat_csr_i = amat_csr[idx_start:idx_end, idx_start:idx_end]
            amatcsr_list.append(amat_csr_i)

        assert hvalx == hvaly

        errors, runtimes, num_iters_list = [], [], []

        for evec in evecs.T:
            if method == 't2':
                pval = np.sqrt(etaval)
                qval = 1 / (2 * np.sqrt(etaval))
            elif method == 't2b':
                interior_size = 1 / NUM_SUBDOMAINS - hvalx
                pval = np.sqrt(etaval) * 1 / np.tanh(np.sqrt(etaval) * interior_size)
                qval = 1 / np.tanh(interior_size * np.sqrt(etaval)) / np.sqrt(etaval) \
                       - interior_size / np.sinh(interior_size * np.sqrt(etaval)) ** 2
                qval = qval / 2
            else:
                pval = 2 ** (-3 / 5) * (np.pi ** 2 + etaval) ** (2 / 5) * hvalx ** (-1 / 5)
                qval = 2 ** (-1 / 5) * (np.pi ** 2 + etaval) ** (-1 / 5) * hvalx ** (3 / 5)

            if 'mlp' in method:
                etavct = torch.tensor(etaval, device=device).unsqueeze(0)
                hvctx = torch.tensor(hvalx, device=device).unsqueeze(0)
                featvct_tr = torch.cat([etavct, hvctx], dim=0).to(torch.float64)
                with torch.no_grad():
                    pscl_corr, qscl_corr = mlp(featvct_tr)
                pval = pval * pscl_corr.cpu().numpy()
                qval = qval * qscl_corr.cpu().numpy()

            if method != 'ras':
                amatcsr_tilde_list = []
                amatcsr_i = amatcsr_list[0]
                for j in range(1, NUM_SUBDOMAINS):
                    amatcsr_j = amatcsr_list[j]
                    amatcsr_tilde_i, amatcsr_tilde_j = construct_oras(
                        amatcsr_i, amatcsr_j, etaval, hvalx, size, pval, qval
                    )
                    amatcsr_tilde_list.append(amatcsr_tilde_i)
                    amatcsr_i = amatcsr_tilde_j
                amatcsr_tilde_list.append(amatcsr_tilde_j)
            else:
                amatcsr_tilde_list = amatcsr_list

            amatcsc_tilde_block = scipy.sparse.block_diag(amatcsr_tilde_list, format='csc')

            M = build_oras_preconditioner(
                amatcsc_tilde_block, endpoints, num_dofs, size, NUM_SUBDOMAINS
            )

            residuals = []
            time_start = time.time()
            x_sol, info = pyamg.krylov.gmres(
                amat_csr,
                np.zeros(num_dofs),
                x0=evec.copy(),
                tol=TOL,
                maxiter=500,
                M=M,
                residuals=residuals
            )
            runtime = time.time() - time_start

            error = residuals[-1]
            n_iters = len(residuals) - 1  # exclude the initial residual

            if info != 0:
                warnings.warn(
                    f"GMRES did not converge for sample {s} "
                    f"(info={info}, iters={n_iters}, error={error:.2e})"
                )
            errors.append(error)
            runtimes.append(runtime)
            num_iters_list.append(n_iters)

        errors_mean    = np.mean(errors)
        runtimes_mean  = np.mean(runtimes)
        num_iters_mean = np.mean(num_iters_list)
        errors_sd      = np.std(errors)
        runtimes_sd    = np.std(runtimes)
        num_iters_sd   = np.std(num_iters_list)

        print(f'{num_dofs}, {errors_mean:.3e}, {num_iters_mean:.1f}, {runtimes_mean:.4f}s')

        rows.append(
            {
                "num_dofs":           num_dofs,
                "num_iters_tol_mean": num_iters_mean,
                "method":             method,
                "errors_mean":        errors_mean,
                "runtimes_mean":      runtimes_mean,
                "num_iters_tol_sd":   num_iters_sd,
                "errors_sd":          errors_sd,
                "runtimes_sd":        runtimes_sd,
                "hvalx":              hvalx,
                "num_subdomains":     NUM_SUBDOMAINS,
                "solver":             "gmres",
            }
        )
        pd.DataFrame(rows).to_csv(log_path, index=False)