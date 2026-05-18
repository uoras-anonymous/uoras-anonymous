import os
import time
import warnings
from datetime import datetime
from collections import OrderedDict, deque
import numpy as np
import scipy
import sksparse.cholmod
import pandas as pd
import torch
from torch_sla import SparseTensor
from utils_2_subdomains import construct_oras, construct_oras_sparse_differentiable, set_seed, Exp
from utils_N_subdomains import decompose_domain, get_subdomains_endpoints

SEED = 42
set_seed(SEED)

if __name__ == '__main__':
    device = "cuda" 
    exp_name = 'meshrange-tolvalid5-exp'
    SAVE = True
    LEARNING_RATE = 0.001
    NUM_SUBDOMAINS = 2
    data_split = {
        'train': slice(5),
        'vd': slice(5, 10)
    }
    # Logs
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = os.path.join("results", f"train_{exp_name}_{NUM_SUBDOMAINS}_subdomains_lr{LEARNING_RATE}_{run_id}")
    os.makedirs(dirname, exist_ok=True)
    log_path = os.path.join(dirname, "log.csv")
    # Data
    data = np.load("data_N_subdomains/data_mesh_squares_6-50.npy", allow_pickle=True).item()
    amatcsr_tr_list, amatcsr_vd_list = data['amatcsr_list'][data_split['train']], data['amatcsr_list'][data_split['vd']]
    evecs_tr_list, evecs_vd_list = data['evecs_list'][data_split['train']], data['evecs_list'][data_split['vd']]
    etaval_tr_list, etaval_vd_list = data['etaval_list'][data_split['train']], data['etaval_list'][data_split['vd']]
    hvalx_tr_list, hvalx_vd_list = data['hvalx_list'][data_split['train']], data['hvalx_list'][data_split['vd']]
    hvaly_tr_list, hvaly_vd_list = data['hvaly_list'][data_split['train']], data['hvaly_list'][data_split['vd']]
    NUM_TRAIN_SAMPLES, NUM_VD_SAMPLES = len(amatcsr_tr_list), len(amatcsr_vd_list)

    # Precompute validation set
    amatcsr_dec_vd_list, size_vd_list, endpoints_vd_list = [], [], []
    etavct_vd_list, hvctx_vd_list = [], []
    for s in range(NUM_VD_SAMPLES):
        amatcsr_vd, etaval_vd, hvalx_vd = amatcsr_vd_list[s], etaval_vd_list[s], hvalx_vd_list[s]
        etavct_vd = torch.tensor(etaval_vd, device=device).unsqueeze(0)
        hvctx_vd = torch.tensor(hvalx_vd, device=device).unsqueeze(0)
        etavct_vd_list.append(etavct_vd)
        hvctx_vd_list.append(hvctx_vd)

        num_dofs_vd = amatcsr_vd.shape[0]
        num_x_vd = num_dofs_vd**(1/2)
        assert num_x_vd % NUM_SUBDOMAINS == 0 and num_x_vd > NUM_SUBDOMAINS
        overlap_pairs_vd = decompose_domain(num_dofs_vd, num_x_vd, NUM_SUBDOMAINS)
        endpoints_vd = get_subdomains_endpoints(overlap_pairs_vd, num_dofs_vd)
        i1_vd, i2_vd = overlap_pairs_vd[0]
        size_vd = (i2_vd-i1_vd)//2
        size_vd_list.append(size_vd)
        endpoints_vd_list.append(endpoints_vd)

        # Decompose system matrix
        amatcsr_dec_vd = []
        for j in range(NUM_SUBDOMAINS):
            idx_start, idx_end = endpoints_vd[j]            
            amat_csr_j = amatcsr_vd[idx_start:idx_end, idx_start:idx_end]
            amatcsr_dec_vd.append(amat_csr_j)
        amatcsr_dec_vd_list.append(amatcsr_dec_vd)
    
    # Precompute training set
    amtxcsr_tr_list, amtxsp_dec_tr_list, size_tr_list, endpoints_tr_list = [], [], [], []
    evcts_tr_list, etavct_tr_list, hvctx_tr_list = [], [], []
    for s in range(NUM_TRAIN_SAMPLES):
        amatcsr_tr, evecs_tr, etaval_tr, hvalx_tr = amatcsr_tr_list[s], evecs_tr_list[s], etaval_tr_list[s], hvalx_tr_list[s]
        evcts_tr = torch.tensor(evecs_tr, device=device)
        etavct_tr = torch.tensor(etaval_tr, device=device).unsqueeze(0)
        hvctx_tr = torch.tensor(hvalx_tr, device=device).unsqueeze(0)
        evcts_tr_list.append(evcts_tr)
        etavct_tr_list.append(etavct_tr)
        hvctx_tr_list.append(hvctx_tr)

        num_dofs_tr = amatcsr_tr.shape[0]
        num_x_tr = num_dofs_tr**(1/2)
        assert num_x_tr % NUM_SUBDOMAINS == 0 and num_x_tr > NUM_SUBDOMAINS
        overlap_pairs_tr = decompose_domain(num_dofs_tr, num_x_tr, NUM_SUBDOMAINS)
        endpoints_tr = get_subdomains_endpoints(overlap_pairs_tr, num_dofs_tr)
        i1_tr, i2_tr = overlap_pairs_tr[0]
        size_tr = (i2_tr-i1_tr)//2
        size_tr_list.append(size_tr)
        endpoints_tr_list.append(endpoints_tr)

        amatcoo_tr = amatcsr_tr.tocoo()
        amtxcoo_tr = torch.sparse_coo_tensor(
            torch.stack([torch.from_numpy(amatcoo_tr.row), torch.from_numpy(amatcoo_tr.col)]),
            torch.from_numpy(amatcoo_tr.data),
            size=amatcoo_tr.shape,
            device=device
        )
        amtxcsr_tr = amtxcoo_tr.to_sparse_csr()
        amtxcsr_tr_list.append(amtxcsr_tr)
        # Decompose system matrix
        amtxsp_dec_tr = []
        for i in range(NUM_SUBDOMAINS):
            idx_start, idx_end = endpoints_tr[i]
            amtxcoo_tr_i = amtxcoo_tr.index_select(0, torch.arange(idx_start, idx_end, device=device)).index_select(1, torch.arange(idx_start, idx_end, device=device))
            amtxsp_tr_i = SparseTensor.from_torch_sparse(amtxcoo_tr_i)
            amtxsp_dec_tr.append(amtxsp_tr_i)
        amtxsp_dec_tr_list.append(amtxsp_dec_tr)
    
    print(f"{exp_name}, {NUM_SUBDOMAINS} subdomains, lr={LEARNING_RATE}, device={device}")
    # Define MLP
    out_dim = 2
    mlp_pars = OrderedDict(
        [
        ("fc1", torch.nn.Linear(3, 64, dtype=torch.float64)),
        ("nl1", torch.nn.Sigmoid()),
        ("fc2", torch.nn.Linear(64, 32, dtype=torch.float64)),
        ("nl2", torch.nn.Sigmoid()),
        ("fc3", torch.nn.Linear(32, out_dim, dtype=torch.float64)),
        ("nl3", Exp())
        ]
    )
    mlp = torch.nn.Sequential(mlp_pars)
    mlp.to(device)
    print(mlp)
    torch.nn.init.normal_(mlp.fc3.weight, mean=0.0, std=1e-3)
    torch.nn.init.normal_(mlp.fc3.bias[0], 0, std=1e-3)
    torch.nn.init.normal_(mlp.fc3.bias[1], 0, std=1e-3)
    opt = torch.optim.Adam(list(mlp.parameters()), lr=LEARNING_RATE)
    # Non-stationary
    NUM_ITER_TRAIN_RANGE = (3, 15)
    PATIENCE = 50
    rolling_num_iters = deque(maxlen=PATIENCE)
    epoch, patience_count, best_num_iter = 0, 0, float("inf")
    rows = []
    while True:
        if epoch % 10 == 0:
            time_start = time.time()
            num_iter_vd_list = []
            for s in np.random.permutation(range(NUM_VD_SAMPLES)):
                amatcsr_vd, amatcsr_dec_vd, evecs_vd = amatcsr_vd_list[s], amatcsr_dec_vd_list[s], evecs_vd_list[s]
                etaval_vd, hvalx_vd, size_vd, endpoints_vd = etaval_vd_list[s], hvalx_vd_list[s], size_vd_list[s], endpoints_vd_list[s]
                etavct_vd, hvctx_vd = etavct_vd_list[s], hvctx_vd_list[s]
                for evec_vd in evecs_vd.T:
                    evec_vd_t = evec_vd.copy()
                    num_iter_vd = 0
                    while np.linalg.norm(evec_vd_t) > 1e-6:
                        if num_iter_vd >= 50:
                            break
                        rvec_vd = amatcsr_vd @ evec_vd_t
                        # Predict p and q
                        rnorm_logratio_vd = torch.log(torch.tensor(rvec_vd, device=device).norm(keepdim=True))
                        featvct_vd = torch.cat([rnorm_logratio_vd, etavct_vd, hvctx_vd], dim=0)
                        with torch.no_grad():
                            p_par_vd, q_par_vd = mlp(featvct_vd)
                        p_o2 = 2**(-3/5) * (np.pi**2 + etaval_vd)**(2/5) * hvalx_vd**(-1/5)
                        q_o2 = 2**(-1/5) * (np.pi**2 + etaval_vd)**(-1/5) * hvalx_vd**(3/5)
                        p_par_vd = p_par_vd.item()*p_o2
                        q_par_vd = q_par_vd.item()*q_o2
                        
                        # Modify subdomain matrices
                        amatcsr_tilde_vd_dec = []
                        amatcsr_vd_i = amatcsr_dec_vd[0]
                        for j in range(1, NUM_SUBDOMAINS):
                            amatcsr_vd_j = amatcsr_dec_vd[j]
                            amtx_tilde_vd_i, amtx_tilde_vd_j = construct_oras(amatcsr_vd_i, amatcsr_vd_j, etaval_vd, hvalx_vd, size_vd, p_par_vd, q_par_vd)
                            amatcsr_tilde_vd_dec.append(amtx_tilde_vd_i)
                            amatcsr_vd_i = amtx_tilde_vd_j
                        amatcsr_tilde_vd_dec.append(amtx_tilde_vd_j)

                        amatcsc_tilde_vd_block = scipy.sparse.block_diag(amatcsr_tilde_vd_dec, format='csc')
                        rvec_vd_ext = np.concatenate([rvec_vd[s:e] for s, e in endpoints_vd])

                        try:
                            factor = sksparse.cholmod.cholesky(amatcsc_tilde_vd_block)
                            evec_vd_ext = factor(rvec_vd_ext)
                        except sksparse.cholmod.CholmodNotPositiveDefiniteError:
                            warnings.warn(
                                f"Matrix with {amatcsr_vd} DoFs is not SPD. Skipping Cholesky decomposition.",
                                RuntimeWarning
                            )
                            evec_vd_ext = scipy.sparse.spsolve(amatcsc_tilde_vd_block, rvec_vd_ext)
                        corrvec_vd = np.zeros_like(evec_vd)
                        offset = 0
                        for i, (idx_start, idx_end) in enumerate(endpoints_vd):
                            block_size = idx_end - idx_start
                            evec_vd_i = evec_vd_ext[offset:offset + block_size]
                            offset += block_size
                            if i == 0:
                                corrvec_vd[idx_start:idx_end-size_vd] = evec_vd_i[:-size_vd]
                            elif i == NUM_SUBDOMAINS-1:
                                corrvec_vd[idx_start+size_vd:idx_end] = evec_vd_i[size_vd:]
                            else:
                                corrvec_vd[idx_start+size_vd:idx_end-size_vd] = evec_vd_i[size_vd:-size_vd]
                        evec_vd_t = evec_vd_t - corrvec_vd
                        num_iter_vd += 1
                    num_iter_vd_list.append(num_iter_vd)
            runtime_vd = time.time() - time_start
            print("===Validation===")
            num_iter_vd_sd, num_iter_vd_mean = np.std(num_iter_vd_list), np.mean(num_iter_vd_list)
            print(f"[{epoch}] num iter till 1e-6:  {num_iter_vd_mean:.4f}±{num_iter_vd_sd:.4f}, time: {runtime_vd:.3f}")
            print("=====")
            if num_iter_vd_mean < best_num_iter:
                best_num_iter = num_iter_vd_mean
                pars = {
                    "mlp": mlp.state_dict()
                }
                if SAVE:
                    torch.save(pars, f"{dirname}/epoch_{epoch}")
                print(f"New best num iter till 1e-6: {best_num_iter:.3f}")
                patience_count = 0
            if SAVE:
                pd.DataFrame(rows).to_csv(log_path, index=False)
            patience_count += 1
            if all(norm < num_iter_vd_mean for norm in rolling_num_iters):
                patience_count += 1
            else:
                patience_count = 0
            rolling_num_iters.append(num_iter_vd_mean)
            if patience_count >= PATIENCE:
                break

        errors_tr = []
        time_start = time.time()
        for s in np.random.permutation(range(NUM_TRAIN_SAMPLES)):
            opt.zero_grad()
            amtxcsr_tr, amtxsp_dec_tr, evcts_tr = amtxcsr_tr_list[s], amtxsp_dec_tr_list[s], evcts_tr_list[s]
            etaval_tr, hvalx_tr, size_tr, endpoints_tr = etaval_tr_list[s], hvalx_tr_list[s], size_tr_list[s], endpoints_tr_list[s]
            etavct_tr, hvctx_tr = etavct_tr_list[s], hvctx_tr_list[s]
            errors_batch = []
            batch_size = evcts_tr.shape[1]
            for k in np.random.permutation(range(batch_size)):
                evct_t = evcts_tr[:, k].clone()
                num_iter = np.random.randint(*NUM_ITER_TRAIN_RANGE)
                for n_tr in range(num_iter):
                    rvct_tr = amtxcsr_tr @ evct_t
                    # Predict p and q
                    rnorm_logratio_tr = torch.log(rvct_tr.norm(keepdim=True))
                    featvct_tr = torch.cat([rnorm_logratio_tr, etavct_tr, hvctx_tr], dim=0).to(device)
                    p_par_tr, q_par_tr = mlp(featvct_tr)
                    p_o2 = 2**(-3/5) * (torch.pi**2 + etavct_tr.squeeze())**(2/5) * hvctx_tr.squeeze()**(-1/5)
                    q_o2 = 2**(-1/5) * (torch.pi**2 + etavct_tr.squeeze())**(-1/5) * hvctx_tr.squeeze()**(3/5)
                    p_par_tr = p_par_tr*p_o2
                    q_par_tr = q_par_tr*q_o2
                    # Modify subdomain matrices
                    amtxsp_tilde_tr_dec = []
                    amtxsp_tr_i = amtxsp_dec_tr[0]
                    for j in range(1, NUM_SUBDOMAINS):
                        amtxsp_tr_j = amtxsp_dec_tr[j]
                        amtxsp_tilde_tr_i, amtxsp_tilde_tr_j = construct_oras_sparse_differentiable(amtxsp_tr_i, amtxsp_tr_j, etaval_tr, hvalx_tr, hvalx_tr, size_tr, p_par_tr, q_par_tr, device=device)
                        amtxsp_tilde_tr_dec.append(amtxsp_tilde_tr_i)
                        amtxsp_tr_i = amtxsp_tilde_tr_j
                    amtxsp_tilde_tr_dec.append(amtxsp_tilde_tr_j)

                    # Construct block diagonal modified system matrix
                    val_list, row_list, col_list, rvct_list = [], [], [], []
                    subdomain_sizes = []
                    offset = 0
                    for j in range(NUM_SUBDOMAINS):
                        amtxsp_tilde_tr_dec_j = amtxsp_tilde_tr_dec[j]
                        idx_start, idx_end = endpoints_tr[j]
                        rvct_j = rvct_tr[idx_start:idx_end]
                        
                        n_rows = amtxsp_tilde_tr_dec_j.shape[0]
                        subdomain_sizes.append(n_rows)
                        rvct_list.append(rvct_j)
                        
                        vals = amtxsp_tilde_tr_dec_j.values
                        val_list.append(vals)
                        
                        row_list.append(amtxsp_tilde_tr_dec_j.row_indices + offset)
                        col_list.append(amtxsp_tilde_tr_dec_j.col_indices + offset)
                        offset += n_rows

                    block_vals = torch.cat(val_list)
                    block_rows = torch.cat(row_list)
                    block_cols = torch.cat(col_list)
                    
                    amtxsp_tilde_tr_block = SparseTensor(block_vals, block_rows, block_cols, (offset, offset))
                    rvct_block = torch.cat(rvct_list, dim=0)

                    evct_block = amtxsp_tilde_tr_block.solve(rvct_block)

                    split_evct = torch.split(evct_block, subdomain_sizes)
                    corrvct_tr_list = []
                    for i in range(NUM_SUBDOMAINS):
                        evct_i = split_evct[i]
                        if i == 0:
                            corrvct_tr_list.append(evct_i[:-size_tr])
                        elif i == NUM_SUBDOMAINS-1:
                            corrvct_tr_list.append(evct_i[size_tr:])
                        else:
                            corrvct_tr_list.append(evct_i[size_tr:-size_tr])
                            
                    corrvct = torch.cat(corrvct_tr_list, dim=0)
                    evct_t = evct_t - corrvct
                    if evct_t.norm() < 1e-6:
                        break
                errors_batch.append(evct_t.norm())
            loss = torch.stack(errors_batch).mean()
            loss.backward()
            opt.step()
            errors_tr.extend(errors_batch)
        runtime_tr = time.time() - time_start
        tr_norm_sd, tr_norm_mean = torch.std_mean(torch.stack(errors_tr))
        tr_norm_sd, tr_norm_mean = tr_norm_sd.item(), tr_norm_mean.item()
        if epoch % 1 == 0:
            print(f"[{epoch}] error {num_iter} iter: {tr_norm_mean:.4e}±{tr_norm_sd:.4e}, time: {runtime_tr:.3f}")
        rows.append(
            {
                "epoch": epoch,
                "num_iter_train": num_iter,
                "error_train_mean": tr_norm_mean,
                "error_train_sd": tr_norm_sd,
                "num_iter_valid_mean": num_iter_vd_mean,
                "num_iter_valid_sd": num_iter_vd_sd,
                "runtime_train": runtime_tr,
                "runtime_valid": runtime_vd,
                "num_subdomains": NUM_SUBDOMAINS,
                "lr": LEARNING_RATE,
                "seed": SEED
            }
        )
        epoch += 1
    print(f"Done. Best validation norm: {best_num_iter:.3e}")