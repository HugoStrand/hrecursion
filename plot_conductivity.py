
import itertools
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt


def build_2d_nn_tight_binding_matrix(t, Nx, Ny):

    x = np.arange(Nx)
    y = np.arange(Ny)

    X, Y = np.meshgrid(x, y)
    r = np.vstack((X.flatten(), Y.flatten())).T

    RmR = r[:,None,:] - r[None,:,:]

    for idx, N in enumerate((Nx, Ny)):
        RmR[:, :, idx] = np.mod(RmR[:, :, idx] + N//2, N) - N//2

    L = np.linalg.norm(RmR, axis=2)

    H = -t*(L == 1)
    return H


def build_2d_tight_binding_sparse_matrix(hopping, Nx, Ny):

    def get_idx(i, j):
        i = np.mod(i, Nx)
        j = np.mod(j, Ny)
        return i + Nx * j

    M = Nx * Ny
    S = sp.dok_array((M, M), dtype=float)

    for i, j in itertools.product(range(Nx), range(Ny)):
        for (di, dj), c in hopping.items():
            a = get_idx(i, j)
            b = get_idx(i + di, j + dj)
            S[a, b] = c

    return S.tocsr()


def build_2d_velocity_tight_binding_sparse_matrix(hopping, Nx, Ny, idx):

    def get_idx(i, j):
        i = np.mod(i, Nx)
        j = np.mod(j, Ny)
        return i + Nx * j

    M = Nx * Ny
    S = sp.dok_array((M, M), dtype=float)

    for i, j in itertools.product(range(Nx), range(Ny)):
        for (di, dj), c in hopping.items():
            a = get_idx(i, j)
            b = get_idx(i + di, j + dj)
            if idx == 0 and di != 0:
                S[a, b] = di * c
            if idx == 1 and dj != 0:
                S[a, b] = dj * c

    return S.tocsr()


if __name__ == '__main__':

    t = 1.0

    hopping = {
        (+1, 0) : -t,
        (-1, 0) : -t,
        (0, +1) : -t,
        (0, -1) : -t,
        }
    
    N = 3
    Nx, Ny = N, N
    #Nx, Ny = 11, 16

    x = np.arange(Nx)
    y = np.arange(Ny)

    X, Y = np.meshgrid(x, y)
    r = np.vstack((X.flatten(), Y.flatten())).T
    
    #H = build_2d_nn_tight_binding_matrix(t, Nx, Ny)
    H_r = build_2d_tight_binding_sparse_matrix(hopping, Nx, Ny)
    print(H_r.shape)
    print(f'H_r = \n{H_r.todense()}')

    kx = 2*np.pi* np.arange(Nx) / Nx
    ky = 2*np.pi* np.arange(Ny) / Ny

    KX, KY = np.meshgrid(kx, ky)
    k = np.vstack((KX.flatten(), KY.flatten())).T

    eps_k = -2*t*np.sum(np.cos(k), axis=1)
    #print(eps_k.shape)

    M = Nx * Ny
    D = np.exp(1j * np.sum(r[:, None, :] * k[None, :, :], axis=2)) / np.sqrt(M)

    print(D.shape)
    H_k = D.T.conj() @ H_r @ D
    print(f'H_k = \n{H_k}')

    print(f'diag(H_k) = {np.diag(H_k).real}')

    vx = 2 * np.sin(k[:, 0])
    vy = 2 * np.sin(k[:, 1])

    print(f'eps_k = {eps_k}')
    E_ref = np.sort(eps_k)
    E = np.linalg.eigvalsh(H_r.todense())
    print(f'E = {E}')

    assert( np.max(np.abs(E - E_ref)) < 1e-10 )

    vx_ij = D @ np.diag(vx) @ D.T.conj()
    vy_ij = D @ np.diag(vy) @ D.T.conj()

    def cut_zeros(arr, tol=1e-10):
        idx = np.abs(arr) < tol
        arr[idx] = 0.
        return arr

    vx_ij = cut_zeros(vx_ij)
    vy_ij = cut_zeros(vy_ij)
    
    print(f'vx_ij = \n{vx_ij.imag}')
    print(f'vy_ij = \n{vy_ij.imag}')

    vx_ij_ref = build_2d_velocity_tight_binding_sparse_matrix(hopping, Nx, Ny, 0)
    vy_ij_ref = build_2d_velocity_tight_binding_sparse_matrix(hopping, Nx, Ny, 1)

    print(f'vx_ij_ref = \n{vx_ij_ref.todense()}')
    print(f'vy_ij_ref = \n{vy_ij_ref.todense()}')

    np.testing.assert_array_almost_equal(vx_ij.imag, vx_ij_ref.todense())
    np.testing.assert_array_almost_equal(vy_ij.imag, vy_ij_ref.todense())
