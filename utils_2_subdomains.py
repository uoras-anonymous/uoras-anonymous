import copy
import random
import scipy
import numpy as np
import torch
from torch_sla import SparseTensor


class Exp(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def construct_optimal_schwarz(amat_sp, i1, i2):
    bmat1 = amat_sp[i1:i2, i2:]
    cmat1 = amat_sp[:i1, i1:i2]
    bmat2 = amat_sp[i1:i2, :i1]
    cmat2 = amat_sp[i2:, i1:i2]
    amat1i = amat_sp[:i1, :i1]
    amat2i = amat_sp[i2:, i2:]
    amat_gamma = amat_sp[i1:i2, i1:i2]
    bottom = np.concatenate([bmat2, amat_gamma - bmat1 @ np.linalg.inv(amat2i) @ cmat2], axis=1)
    top = np.concatenate([amat1i, cmat1], axis=1)
    amat1_tilde = np.concatenate([top, bottom], axis=0)
    top = np.concatenate([amat_gamma - bmat2 @ np.linalg.inv(amat1i) @ cmat1, bmat1], axis=1)
    bottom = np.concatenate([cmat2, amat2i], axis=1)
    amat2_tilde = np.concatenate([top, bottom], axis=0)
    return amat1_tilde, amat2_tilde

def construct_partitioning(i1, i2, NUM_DOFS):
    size = (i2-i1)//2
    rmat1 = np.eye(i2, NUM_DOFS)
    rmat2 = np.eye(NUM_DOFS-i1, NUM_DOFS, k=i1)
    rmat1_tilde = copy.deepcopy(rmat1)
    rmat2_tilde = copy.deepcopy(rmat2)
    rmat1_tilde[i1+size:] = 0
    rmat2_tilde[:size] = 0
    return rmat1, rmat2, rmat1_tilde, rmat2_tilde

def construct_tmat_tilde(eta, h, size, p, q):
    ovec = np.ones(size)
    imat_csr = scipy.sparse.eye(size)
    tmat_zero = scipy.sparse.spdiags([-ovec, 4*ovec, -ovec], [-1, 0, 1], size, size)
    tmat_eta = tmat_zero + imat_csr * eta*h**2
    tmat_tilde = 1/2 * tmat_eta + p*h*imat_csr + q/h*(tmat_zero - 2*imat_csr)
    tmat_tilde = tmat_tilde/h**2
    return tmat_tilde

def construct_oras(amat1, amat2, eta, h, size, p, q=0):
    amat1 = copy.deepcopy(amat1)
    amat2 = copy.deepcopy(amat2)
    tmat_tilde = construct_tmat_tilde(eta, h, size, p, q)
    amat1[-size:, -size:] = tmat_tilde
    amat2[:size, :size] = tmat_tilde
    return amat1, amat2

def construct_tmtx_tilde(eta: float, h: float, size: int, p_par: torch.Tensor, q_par: torch.Tensor = 0):
    ovct = torch.ones(size, dtype=torch.float64)
    imtx = torch.eye(size, dtype=torch.float64)
    tmtx_zero = torch.diag(-ovct[1:], -1) + torch.diag(4 * ovct, 0) + torch.diag(-ovct[:-1], 1)
    tmtx_eta = tmtx_zero + imtx*eta*h**2
    tmtx_tilde = 1/2 * tmtx_eta + p_par*h*imtx + q_par/h*(tmtx_zero - 2*imtx)
    tmtx_tilde = tmtx_tilde/h**2
    return tmtx_tilde

def construct_tmtx_tilde_sparse(eta: float, h_x: float, h_y: float, size: int, p_par: torch.Tensor, q_par: torch.Tensor = 0, device: str = "cpu"):
    idx_main = torch.arange(size)
    idx_sub  = torch.arange(1, size)
    
    indices_main = torch.stack([idx_main, idx_main])
    indices_low = torch.stack([idx_sub, idx_sub - 1])
    indices_up = torch.stack([idx_sub - 1, idx_sub])
    all_indices = torch.cat([indices_low, indices_main, indices_up], dim=1).to(device)

    val_trimtx = torch.cat([
        -torch.ones(size - 1, dtype=torch.float64),
        2 * torch.ones(size, dtype=torch.float64),
        -torch.ones(size - 1, dtype=torch.float64)
    ]).to(device)
    # Reconstructing tmtx
    diag_scale = 2*(h_x**2+h_y**2) + h_x**2*h_y**2 * eta
    offdiag_scale = h_x**2
    val_tmtx = torch.cat([
        -torch.ones(size - 1, dtype=torch.float64) * offdiag_scale,
        torch.ones(size, dtype=torch.float64) * diag_scale,
        -torch.ones(size - 1, dtype=torch.float64) * offdiag_scale
    ]) / h_y**2
    val_tmtx = val_tmtx.to(device)
    
    tmtx_tilde_values = 0.5 * val_tmtx + h_x*(q_par / h_y**2) * val_trimtx
    tmtx_tilde_values[size-1:2*size-1] += p_par * h_x
    tmtx_tilde_values = tmtx_tilde_values / h_x**2
    
    tmtx_tilde_sparse = torch.sparse_coo_tensor(
        all_indices, tmtx_tilde_values, (size, size)
    ).coalesce()
    return tmtx_tilde_sparse

def construct_tmtx_tilde_sparse_sla(eta: float, h_x: float, h_y: float, size: int, p_par: torch.Tensor, q_par: torch.Tensor = 0, device: str = "cpu"):
    idx_main = torch.arange(size)
    idx_sub  = torch.arange(1, size)
    
    row_low = idx_sub
    col_low = idx_sub - 1
    
    row_main = idx_main
    col_main = idx_main
    
    row_up = idx_sub - 1
    col_up = idx_sub
    
    rows = torch.cat([row_low, row_main, row_up]).to(device)
    cols = torch.cat([col_low, col_main, col_up]).to(device)

    val_trimtx = torch.cat([
        -torch.ones(size - 1, dtype=torch.float64),
        2 * torch.ones(size, dtype=torch.float64),
        -torch.ones(size - 1, dtype=torch.float64)
    ]).to(device)
    
    diag_scale = 2*(h_x**2+h_y**2) + h_x**2*h_y**2 * eta
    offdiag_scale = h_x**2
    val_tmtx = torch.cat([
        -torch.ones(size - 1, dtype=torch.float64) * offdiag_scale,
        torch.ones(size, dtype=torch.float64) * diag_scale,
        -torch.ones(size - 1, dtype=torch.float64) * offdiag_scale
    ]) / h_y**2
    val_tmtx = val_tmtx.to(device)
    
    tmtx_tilde_values = 0.5 * val_tmtx + h_x*(q_par / h_y**2) * val_trimtx
    tmtx_tilde_values[size-1:2*size-1] += p_par * h_x
    tmtx_tilde_values = tmtx_tilde_values / h_x**2
    
    return SparseTensor(tmtx_tilde_values, rows, cols, (size, size))
    

def construct_oras_differentiable(amtx1: torch.Tensor, amtx2: torch.Tensor, eta: float, h: float, size: int, p_par: torch.Tensor, q_par: torch.Tensor):
    amtx1_tilde, amtx2_tilde = amtx1.clone(), amtx2.clone()
    tmtx_tilde = construct_tmtx_tilde(eta, h, size, p_par, q_par)
    amtx1_tilde[-size:, -size:] = tmtx_tilde
    amtx2_tilde[:size, :size] = tmtx_tilde
    return amtx1_tilde, amtx2_tilde

def replace_sparse_corner_sla(large_mtx, small_mtx, corner='bottom_right', device="cpu"):
    """
    Replaces a corner of large_mtx with small_mtx in a differentiable way.
    large_mtx: torch_sla.SparseTensor (N x N)
    small_mtx: torch_sla.SparseTensor (size x size)
    """
    num_large = large_mtx.shape[0]
    num_small = small_mtx.shape[0]
    
    l_val = large_mtx.values.to(device)
    l_row = large_mtx.row_indices.to(device)
    l_col = large_mtx.col_indices.to(device)
    
    s_val = small_mtx.values.to(device)
    s_row = small_mtx.row_indices.to(device)
    s_col = small_mtx.col_indices.to(device)
    
    if corner == 'bottom_right':
        mask = ~((l_row >= num_large - num_small) & (l_col >= num_large - num_small))
        shift = num_large - num_small
    elif corner == 'top_left':
        mask = ~((l_row < num_small) & (l_col < num_small))
        shift = 0
    else:
        raise ValueError("Corner must be 'bottom_right' or 'top_left'")
        
    keep_row = l_row[mask]
    keep_col = l_col[mask]
    keep_val = l_val[mask]
    
    new_row = s_row + shift
    new_col = s_col + shift
    new_val = s_val
    
    final_row = torch.cat([keep_row, new_row])
    final_col = torch.cat([keep_col, new_col])
    final_val = torch.cat([keep_val, new_val])
    
    return SparseTensor(final_val, final_row, final_col, large_mtx.shape)

def replace_sparse_corner(large_mtx, small_mtx, corner='bottom_right', device="cpu"):
    """
    Replaces a corner of large_mtx with small_mtx in a differentiable way.
    large_mtx: Sparse COO tensor (N x N)
    small_mtx: Sparse COO tensor (size x size)
    """
    large_mtx = large_mtx.to(device)
    small_mtx = small_mtx.to(device)
    num_large = large_mtx.size(0)
    num_small = small_mtx.size(0)
    
    large_mtx = large_mtx.coalesce()
    indices = large_mtx.indices()
    values = large_mtx.values()
    
    if corner == 'bottom_right':
        mask = ~((indices[0] >= num_large - num_small) & (indices[1] >= num_large - num_small))
        shift = num_large - num_small
    elif corner == 'top_left':
        mask = ~((indices[0] < num_small) & (indices[1] < num_small))
        shift = 0
        
    keep_indices = indices[:, mask]
    keep_values = values[mask]
    
    new_indices = small_mtx.indices() + shift
    new_values = small_mtx.values()
    
    final_indices = torch.cat([keep_indices, new_indices], dim=1)
    final_values = torch.cat([keep_values, new_values])
    
    return torch.sparse_coo_tensor(final_indices, final_values, (num_large, num_large)).coalesce()

def construct_oras_sparse_differentiable(amtx1, amtx2, eta, h_x, h_y, size, p_par, q_par, device="cpu"):
    tmtx_tilde = construct_tmtx_tilde_sparse_sla(eta, h_x, h_y, size, p_par, q_par, device=device)
    amtx1_tilde = replace_sparse_corner_sla(amtx1, tmtx_tilde, corner='bottom_right', device=device)
    amtx2_tilde = replace_sparse_corner_sla(amtx2, tmtx_tilde, corner='top_left', device=device)
    return amtx1_tilde, amtx2_tilde

def construct_tmtx_tilde_regression(eta: float, h: float, size: int, a1_par: torch.Tensor, b1_par: torch.Tensor, c1_par: torch.Tensor, a2_par: torch.Tensor, b2_par: torch.Tensor, c2_par: torch.Tensor):
    ovct = torch.ones(size, dtype=torch.float64)
    imtx = torch.eye(size, dtype=torch.float64)
    tmtx_zero = torch.diag(-ovct[1:], -1) + torch.diag(4 * ovct, 0) + torch.diag(-ovct[:-1], 1)
    tmtx_eta = tmtx_zero + imtx*eta*h**2
    p = a1_par*(torch.pi**2+eta)**b1_par * h**c1_par
    q = a2_par*(torch.pi**2+eta)**b2_par * h**c2_par
    if p.ndim == 0:
        tmtx_tilde = 1/2 * tmtx_eta + p*h*imtx + q/h*(tmtx_zero - 2*imtx)
    else:
        tmtx_tilde = 1/2 * tmtx_eta + p*h + q@(tmtx_zero - 2*imtx)/h
    tmtx_tilde = tmtx_tilde/h**2
    return tmtx_tilde

def construct_oras_differentiable_regression(amtx1: torch.Tensor, amtx2: torch.Tensor, eta: float, h: float, size: int,
                                              a1_par: torch.Tensor, b1_par: torch.Tensor, c1_par: torch.Tensor, a2_par: torch.Tensor, b2_par: torch.Tensor, c2_par: torch.Tensor):
    amtx1_tilde, amtx2_tilde = amtx1.clone(), amtx2.clone()
    tmtx_tilde = construct_tmtx_tilde_regression(eta, h, size, a1_par, b1_par, c1_par, a2_par, b2_par, c2_par)
    amtx1_tilde[-size:, -size:] = tmtx_tilde
    amtx2_tilde[:size, :size] = tmtx_tilde
    return amtx1_tilde, amtx2_tilde

def construct_tmtx_tilde_sparse_regression(eta: float, h: float, size: int, 
                                          a1_par, b1_par, c1_par, 
                                          a2_par, b2_par, c2_par):
    idx_main = torch.arange(size)
    idx_sub  = torch.arange(1, size)
    
    indices = torch.cat([
        torch.stack([idx_sub, idx_sub - 1]),
        torch.stack([idx_main, idx_main]),  
        torch.stack([idx_sub - 1, idx_sub]) 
    ], dim=1)

    val_zero = torch.cat([
        -torch.ones(size - 1, dtype=torch.float64),
         4 * torch.ones(size, dtype=torch.float64),
        -torch.ones(size - 1, dtype=torch.float64)
    ])
    
    p = a1_par * (torch.pi**2 + eta)**b1_par * h**c1_par
    q = a2_par * (torch.pi**2 + eta)**b2_par * h**c2_par

    tmtx_values = 0.5 * val_zero.clone()
    tmtx_values[size-1:2*size-1] += 0.5 * eta * h**2
    
    tmtx_values[size-1:2*size-1] += p.squeeze() * h
    
    q_term_values = val_zero.clone()
    q_term_values[size-1:2*size-1] -= 2.0 # The -2*I part
    tmtx_values += (q.squeeze() / h) * q_term_values
    
    tmtx_values = tmtx_values / h**2
    return torch.sparse_coo_tensor(indices, tmtx_values, (size, size)).coalesce()

def construct_oras_sparse_differentiable_regression(amtx1_coo, amtx2_coo, eta, h, size, 
                                     a1_p, b1_p, c1_p, a2_p, b2_p, c2_p):
    tmtx_tilde_sparse = construct_tmtx_tilde_sparse_regression(
        eta, h, size, a1_p, b1_p, c1_p, a2_p, b2_p, c2_p
    )
    amtx1_tilde = replace_sparse_corner(amtx1_coo, tmtx_tilde_sparse, corner='bottom_right')
    amtx2_tilde = replace_sparse_corner(amtx2_coo, tmtx_tilde_sparse, corner='top_left')
    return amtx1_tilde, amtx2_tilde

def interpolate(x_interp, x_orig, vec, kind='linear'):
    if kind == 'linear':
        vec_interp = np.interp(x_interp, x_orig, vec)
    elif kind == 'cubic':
        cs = scipy.interpolate.CubicSpline(x_orig, vec)
        vec_interp = cs(x_interp)
    return vec_interp

def soft_bins(x, num_center, sigma, space='lin'):
    if space == 'lin':
        centersvec = torch.linspace(0, 1, num_center)
    elif space == 'log':
        start_log, end_log = -27.631, 0.0
        centersvec = torch.linspace(start_log, end_log, num_center)
    featvec = torch.exp(-torch.square(centersvec - x) / (2 * sigma**2))
    return featvec

def decompose_domain(amatsp, overlap_size=1):
    num_dofs = amatsp.shape[0]
    num_x = np.sqrt(amatsp.shape[0])
    i1, i2 = num_dofs//2 - overlap_size*num_x, num_dofs//2 + overlap_size*num_x
    i1, i2 = int(i1), int(i2)
    assert i1 >= 0, i2 >= 0
    return i1, i2

def run_additive_schwarz(mmat, amat_sp, uvec, fvec, method='stationary', NUM_ITER=20):
    assert method == 'stationary'
    is_nonstationary = isinstance(mmat, list)
    uvec_t = copy.deepcopy(uvec)
    uvecs1, uvecs2 = [], []
    uvec_star = np.linalg.solve(amat_sp, fvec)
    errors = [np.linalg.norm(uvec_t - uvec_star)]
    rvec = fvec - amat_sp@uvec_t
    residuals = [np.linalg.norm(rvec)]
    for i in range(NUM_ITER):
        if method == 'matrix':
            fvec_gamma1 = fvec_gamma - bmat1 @ uvec2i
            uvec1 = np.linalg.inv(amat1) @ np.concatenate((fvec1i, fvec_gamma1), axis=0)
            fvec_gamma2 = fvec_gamma - bmat2 @ uvec1i
            uvec2 = np.linalg.inv(amat2) @ np.concatenate((fvec_gamma2, fvec2i), axis=0)
            uvec1i = uvec1[:i1]
            uvec2i = uvec2[i2-i1:]
            uvecs1.append(uvec1)
            uvecs2.append(uvec2)
            uvec_t = np.concatenate((uvec1[:i1+2], uvec2[2:]), axis=0)
        elif method == 'stationary':
            if is_nonstationary:
                uvec_t = uvec_t + mmat[i] @ rvec
            else:
                uvec_t = uvec_t + mmat @ rvec
            rvec = fvec - amat_sp@uvec_t
        residual = np.linalg.norm(rvec)
        error = np.linalg.norm(uvec_t - uvec_star)
        residuals.append(residual)
        errors.append(error)
        # print(f'{i}: {error:.5e}')
    return errors, residuals

def build_mmtx(amtxcoo_tilde_list, endpoints):
    all_indices = []
    all_values = []
    num_subdomains = len(amtxcoo_tilde_list)
    num_dofs = endpoints[-1][-1]
    size = (endpoints[0][1] - endpoints[1][0])//2
    for i in range(num_subdomains):
        local_inv = torch.linalg.inv(amtxcoo_tilde_list[i].to_dense())
        n_local_rows, n_local_cols = local_inv.shape
        
        global_start, global_end = endpoints[i]
        
        if i == 0:
            local_row_slice = slice(0, n_local_rows - size)
            global_row_range = torch.arange(global_start, global_end - size)
        elif i == num_subdomains - 1:
            local_row_slice = slice(size, n_local_rows)
            global_row_range = torch.arange(global_start + size, global_end)
        else:
            local_row_slice = slice(size, n_local_rows - size)
            global_row_range = torch.arange(global_start + size, global_end - size)
            
        restricted_inv = local_inv[local_row_slice, :]
        
        col_start, col_end = endpoints[i]
        global_col_range = torch.arange(col_start, col_end)
        
        grid_r, grid_c = torch.meshgrid(global_row_range, global_col_range, indexing='ij')
        
        all_indices.append(torch.stack((grid_r.flatten(), grid_c.flatten()), dim=0))
        all_values.append(restricted_inv.flatten())

    indices_glob = torch.cat(all_indices, dim=1)
    values_glob = torch.cat(all_values)

    mmtx = torch.sparse_coo_tensor(
        indices_glob, 
        values_glob, 
        size=(num_dofs, num_dofs)
    ).coalesce()
    return mmtx