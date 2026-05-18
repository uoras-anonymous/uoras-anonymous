import numpy as np
import scipy


def get_rdiag_indices(overlaps, NUM_SUBDOMAINS):
    rdiag_indices_list = []
    cursor = 0
    for i in range(NUM_SUBDOMAINS):
        if i == 0 or i == NUM_SUBDOMAINS-1:
            rdiag_indices_i = np.arange(overlaps[cursor], overlaps[cursor+2])
            cursor += 1
        else:
            rdiag_indices_i = np.arange(overlaps[cursor], overlaps[cursor+3])
            cursor += 2
        rdiag_indices_list.append(rdiag_indices_i)
    return rdiag_indices_list

def get_rmat(rdiag_indices_list, NUM_DOFS):
    rmat_list = []
    for rdiag_indices in rdiag_indices_list:
        ovec = np.ones_like(rdiag_indices)
        size_subd = len(rdiag_indices)
        indices = (range(size_subd), rdiag_indices)
        shape = (size_subd, NUM_DOFS)
        rmat = scipy.sparse.coo_array((ovec, indices), shape=shape)
        rmat_list.append(rmat)
    return rmat_list

def get_rmat_tilde(rdiag_indices_list, size, NUM_SUBDOMAINS, NUM_DOFS):
    rmat_tilde_list = []
    for i, rdiag_indices in enumerate(rdiag_indices_list):
        size_subd = len(rdiag_indices)
        if i == 0:
            rdiag_indices_tilde = rdiag_indices[:-size]
            size_interior = len(rdiag_indices_tilde)
            indices = (range(size_interior), rdiag_indices_tilde)
        elif i == NUM_SUBDOMAINS-1:
            rdiag_indices_tilde = rdiag_indices[size:]
            size_interior = len(rdiag_indices_tilde)
            indices = (range(size, size_interior+size), rdiag_indices_tilde)
        else:
            rdiag_indices_tilde = rdiag_indices[size:-size]
            size_interior = len(rdiag_indices_tilde)
            indices = (range(size, size_interior+size), rdiag_indices_tilde)
        ovec = np.ones_like(rdiag_indices_tilde)
        shape = (size_subd, NUM_DOFS)
        rmat_tilde = scipy.sparse.coo_array((ovec, indices), shape=shape)
        rmat_tilde_list.append(rmat_tilde)
    return rmat_tilde_list

def decompose_domain(num_dofs, num_y, num_subdomains=2):
    overlap_pairs = []
    for i in range(1, num_subdomains):
        i1, i2 = i*num_dofs/num_subdomains-num_y, i*num_dofs/num_subdomains+num_y
        i1, i2 = int(i1), int(i2)
        overlap_pairs.append((i1, i2))
    return overlap_pairs

def get_subdomains_endpoints(overlap_pairs, num_dofs):
    endpoints = []
    num_subdomains = len(overlap_pairs)+1
    for i in range(num_subdomains):
        if i == 0:
            idx_start, idx_end = 0, overlap_pairs[i][1]
        elif i == num_subdomains-1:
            idx_start, idx_end = overlap_pairs[i-1][0], num_dofs
        else:
            idx_start, idx_end = overlap_pairs[i-1][0], overlap_pairs[i][1]
        endpoints.append((idx_start, idx_end))
    return endpoints

def decompose_amat(amat_csr, rmat_list):
    amats = []
    for i in range(NUM_SUBDOMAINS):
        rmat_i = rmat_list[i]
        amat_csr[np.ix_(rmat_i.col, rmat_i.col)]
        amats.append(amat_csr)
    return amats