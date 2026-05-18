import numpy as np
import scipy    

np.random.seed(42)


if __name__ == '__main__':
    etaval = 1.0
    num_xs = list(range(16, 50, 2))
    num_ys = list(range(16, 50, 2))
    etaval_list = [etaval]*len(num_xs)
    amatcsr_list = []
    hvalx_list = []
    hvaly_list = []
    evecs_list = []
    for num_x, num_y in zip(num_xs, num_ys):
        assert num_x >= num_y
        num_samples = len(num_xs)
        hval_x, hval_y = 1/(num_x+1), 1/(num_y+1)
        num_dofs = num_x*num_y
        ovec_x, ovec_y = np.ones(num_x), np.ones(num_y)
        imat_x, imat_y = scipy.sparse.eye(num_x), scipy.sparse.eye(num_y)
        lapmat_x = scipy.sparse.spdiags([-ovec_x, 2*ovec_x, -ovec_x], [-1, 0, 1])/hval_x**2
        lapmat_y = scipy.sparse.spdiags([-ovec_y, 2*ovec_y, -ovec_y], [-1, 0, 1])/hval_y**2
        amat = scipy.sparse.kron(imat_x, lapmat_y) + scipy.sparse.kron(lapmat_x, imat_y)
        amat = amat + scipy.sparse.eye(num_dofs) * etaval
        amat_csr = amat.tocsr()
        amatcsr_list.append(amat_csr)
        hvalx_list.append(hval_x)
        hvaly_list.append(hval_y)
        evecs = np.random.randn(num_dofs, 5)
        evecs = evecs/np.linalg.norm(evecs, axis=0)
        evecs_list.append(evecs)
        print(num_x)
    data = {
        "amatcsr_list": amatcsr_list,
        "evecs_list": evecs_list,
        "etaval_list": etaval_list,
        "hvalx_list": hvalx_list,
        "hvaly_list": hvaly_list,
    }
    np.save(f"./data_N_subdomains/data_mesh_squares_16-50_test.npy", data, allow_pickle=True)