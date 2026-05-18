import os
import time
import warnings
from datetime import datetime
from collections import OrderedDict
import numpy as np
import scipy
import sksparse.cholmod
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils_2_subdomains import construct_oras, Exp, set_seed, build_mmat
from utils_N_subdomains import decompose_domain, get_subdomains_endpoints
from scipy.sparse.linalg import eigsh

if __name__ == '__main__':
    data = np.load("data_N_subdomains/data_mesh_squares_20-1000.npy", allow_pickle=True).item()
    amat_csr = data['amatcsr_list'][30]
    assert (amat_csr != amat_csr.T).nnz == 0
    eigvals_A, eigvecs_A = scipy.sparse.linalg.eigsh(amat_csr, k=5000, which='SM', tol=1e-10)
    np.save('eigvals_A', eigvals_A)
    np.save('eigvecs_A', eigvecs_A)
    print("done")
