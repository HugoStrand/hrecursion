"""
Implementation of conductivity tensor calculation
using Chebyshev moment evaluation.

Comparing with conductivity from finite differences
in the applied vector potential.

Garcia, Covaci, Rappoport, PRL 114, 116602 (2015)
https://doi.org/10.1103/PhysRevLett.114.116602

Author: Hugo U. R. Strand (2024)
"""

import itertools
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt


def build_real_space_hamiltonian(hopping, Nx, Ny, A):

    """ A is the applied vector poential """

    def get_idx(i, j):
        i = np.mod(i, Nx)
        j = np.mod(j, Ny)
        return i + Nx * j

    M = Nx * Ny
    S = sp.dok_array((M, M), dtype=complex)

    for i, j in itertools.product(range(Nx), range(Ny)):
        for (di, dj), c in hopping.items():
            a = get_idx(i, j)
            b = get_idx(i + di, j + dj)
            dr = np.array([di, dj])
            S[a, b] = c * np.exp(-1j * np.sum(dr * A))

    return S.tocsr()


def build_real_space_velocity(hopping, Nx, Ny, A, idx):

    """ A is the applied vector poential """

    def get_idx(i, j):
        i = np.mod(i, Nx)
        j = np.mod(j, Ny)
        return i + Nx * j

    M = Nx * Ny
    S = sp.dok_array((M, M), dtype=complex)

    for i, j in itertools.product(range(Nx), range(Ny)):
        for (di, dj), c in hopping.items():
            a = get_idx(i, j)
            b = get_idx(i + di, j + dj)
            if idx == 0 and di != 0:
                #S[a, b] = di * c * 1.j * np.exp(-1j * di * A[0])
                S[a, b] = di * c * 1.j
            if idx == 1 and dj != 0:
                #S[a, b] = dj * c * 1.j * np.exp(-1j * dj * A[1])
                S[a, b] = dj * c * 1.j 

    return S.tocsr()


def get_real_space_hamiltoian_and_velocity_operators(t, A, mu, Nx, Ny):

    hopping = {
        ( 0, 0) : mu,
        (+1, 0) : -t,
        (-1, 0) : -t,
        (0, +1) : -t,
        (0, -1) : -t,
        }
            
    H_r = build_real_space_hamiltonian(hopping, Nx, Ny, A)
    v_x = build_real_space_velocity(hopping, Nx, Ny, A, 0)
    v_y = build_real_space_velocity(hopping, Nx, Ny, A, 1)
    
    return H_r, v_x, v_y


def get_real_space_vectors(Nx, Ny):
    
    rx = np.arange(Nx)
    ry = np.arange(Ny)

    RX, RY = np.meshgrid(rx, ry)
    r = np.vstack((RX.flatten(), RY.flatten())).T

    return r, RX, RY


def get_momentum_space_vectors(Nx, Ny):

    kx = 2*np.pi* np.arange(Nx) / Nx
    ky = 2*np.pi* np.arange(Ny) / Ny

    KX, KY = np.meshgrid(kx, ky)
    k = np.vstack((KX.flatten(), KY.flatten())).T

    return k, KX, KY


def get_momentum_dispersion_and_velocity_operators(t, A, mu, Nx, Ny):

    k, KX, KY = get_momentum_space_vectors(Nx, Ny)
    
    eps_k = -2*t*np.sum(np.cos(k - A[None,:]), axis=1) + mu

    #v_kx = 2 * np.sin(k[:, 0] - A[0])
    #v_ky = 2 * np.sin(k[:, 1] - A[1])

    v_kx = 2 * t * np.sin(k[:, 0])
    v_ky = 2 * t * np.sin(k[:, 1])

    return eps_k, v_kx, v_ky


def get_fourier_matrix(Nx, Ny):

    r, _, _ = get_real_space_vectors(Nx, Ny)
    k, _, _ = get_momentum_space_vectors(Nx, Ny)

    M = Nx * Ny

    D = np.exp(1j * np.sum(r[:, None, :] * k[None, :, :], axis=2))
    D /= np.sqrt(M)

    return D


def test_real_and_momentum_space_repr():

    # Inverse temperature
    beta = 1.5
    
    # Chemical potential
    mu = 0.5
    
    # External vector potential
    A = np.array([np.pi/2, 0.0])
    
    t = 1.0 # nn hopping

    for Nx, Ny in [(3, 3), (5, 5), (4, 3), (3, 4), (9, 8)]:
        driver_test_real_and_momentum_space_repr(t, mu, A, beta, Nx, Ny)
    

def driver_test_real_and_momentum_space_repr(t, mu, A, beta, Nx, Ny):

    H_r, v_x, v_y = \
        get_real_space_hamiltoian_and_velocity_operators(t, A, mu, Nx, Ny)
    
    eps_k, v_kx, v_ky = \
        get_momentum_dispersion_and_velocity_operators(t, A, mu, Nx, Ny)    

    D = get_fourier_matrix(Nx, Ny)

    H_k = D.T.conj() @ H_r @ D

    np.testing.assert_array_almost_equal(np.diag(H_k), eps_k)

    v_x_ref = D @ np.diag(v_kx) @ D.T.conj()
    v_y_ref = D @ np.diag(v_ky) @ D.T.conj()
    
    np.testing.assert_array_almost_equal(v_x.todense(), v_x_ref)
    np.testing.assert_array_almost_equal(v_y.todense(), v_y_ref)

    rho = real_space_density_matrix(H_r, beta)
    n = density_from_density_matrix(rho)
    n_ref = density_from_momentum(eps_k, beta)

    np.testing.assert_array_almost_equal(n, n_ref)
    
    j_kx = current_from_momentum(eps_k, v_kx, beta)
    j_ky = current_from_momentum(eps_k, v_ky, beta)

    j_x = operator_trace(rho, v_x).real
    j_y = operator_trace(rho, v_y).real
    
    np.testing.assert_array_almost_equal(j_x, j_kx)
    np.testing.assert_array_almost_equal(j_y, j_ky)


def fermi_function(E, beta):

    f = np.zeros_like(E)

    pidx = E > 0
    Ep = E[pidx]
    f[pidx] = 1 / (1 + np.exp(-beta * Ep))

    midx = E <= 0
    Em = E[midx]
    f[midx] = np.exp(beta * Em) / (1 + np.exp(beta * Em))

    return f
    

def density_from_momentum(eps_k, beta):
    n_k = fermi_function(eps_k, beta)
    n = np.sum(n_k) / len(eps_k)
    return n


def current_from_momentum(eps_k, v_k, beta):
    n_k = fermi_function(eps_k, beta)
    j = np.sum(n_k * v_k) / len(eps_k)
    return j


def real_space_density_matrix(H_r, beta):

    E, U = np.linalg.eigh(H_r.todense())

    H_r_ref = U @ np.diag(E) @ U.T.conj()
    np.testing.assert_array_almost_equal(H_r.todense(), H_r_ref)
    
    n = fermi_function(E, beta)

    rho = U @ np.diag(n) @ U.T.conj()
    
    return rho


def density_from_density_matrix(rho):
    n = np.sum(np.diag(rho)).real / rho.shape[0]
    return n


def operator_trace(rho, op):
    return np.sum(np.diag( rho @ op )) / rho.shape[0]


def chebyshev_matrix_recursion(H, N):

    assert( N > 2 )

    H = H.todense()

    T = np.zeros([N] + list(H.shape), dtype=H.dtype)

    T[0] = np.eye(H.shape[0])
    T[1] = H

    for idx in range(2, N):
        T[idx] = 2 * H @ T[idx-1] - T[idx-2]

    return T


class ChebyshevVectorRecursion:

    def __init__(self, H, v, Nmax):
        self.H = H
        self.v = v
        self.n = 0
        self.Nmax = Nmax

    def __iter__(self):
        return self
    
    def __next__(self):

        #print(f'ChebRec: n = {self.n}')
        
        if self.n >= self.Nmax:
            raise StopIteration
        
        if self.n > 1:
            v2 = 2 * self.H @ self.v1 - self.v0
            self.v0 = self.v1
            self.v1 = v2
        elif self.n == 0:
            self.v1 = self.v.copy()
        elif self.n == 1:
            self.v0 = self.v1
            self.v1 = self.H @ self.v0
        else:
            raise ValueError

        self.n += 1
        
        return self.v1
        

def get_mu_tensor_stochastic_trace(v_a, v_b, H, N, R, Rblocks):

    """
    H : scipy.sparse.xxx_matrix
        Hamiltonian matrix
    N : int
        Chebyshev max order
    R : int
        Number of concomitant random vectors.
    Rblocks : int
        Number of repretitions of R random vectors.

    The total number of random vectors used is: `R * Rblocks`.
    
    """

    M = H.shape[0]

    mu = np.zeros((N, N), dtype=complex)

    from tqdm import tqdm # For ascii art progress bar...

    #for r in range(Rblocks):
    for r in tqdm(range(Rblocks)):
        
        v = np.exp(2j*np.pi * np.random.random((M, R)))
        v_conj = np.conj(v)

        for n, vn in enumerate(ChebyshevVectorRecursion(H, v, N)):
            w0 = v_b @ vn
            for m, wm in enumerate(ChebyshevVectorRecursion(H, w0, N)):
                #print(f'n, m = {n}, {m}')
                mu[n, m] += np.sum(v_conj * (v_a @ wm)) / R

    mu /= Rblocks
    
    return mu


def get_W(N):
    W = np.ones(N)
    W[0] = 0.5
    return W


def get_mu_tensor(v_a, v_b, H, N):
    
    T = chebyshev_matrix_recursion(H, N)

    mu = np.einsum('ij,njk,kl,mli->nm', v_a.todense(), T, v_b.todense(), T)
    
    return mu


def eval_Gamma_nm(x, N):

    n = np.arange(N)
    x = x[:, None]
    n = n[None, :]

    Cn = (x - 1j*n*np.sqrt(1 - x**2)) * np.exp(+1j * n * np.arccos(x))
    Cm = (x + 1j*n*np.sqrt(1 - x**2)) * np.exp(-1j * n * np.arccos(x))

    x = x.flatten()
    T = np.polynomial.chebyshev.chebvander(x, N-1)

    Gamma = Cn[:, :, None] * T[:, None, :] + Cm[:, None, :] * T[:, :, None]
    Gamma /= (1 - x[:, None, None]**2)**2 # Eq. 4 PRL 114, 116602

    W = get_W(N)
    g = jackson_kernel(N)

    Gamma = np.einsum('xnm,n,m,n,m->xnm', Gamma, g, g, W, W)
    
    return Gamma


if __name__ == '__main__':

    test_real_and_momentum_space_repr()

    t = 1.45    # nn hopping
    beta = 2.0 # Inverse temperature
    mu = 0.0   # Chemical potential
    
    A = np.array([0.0, 0.0]) # External vector potential

    #N = 4
    N = 4
    Nx, Ny = N, N
    
    r, RX, RY = get_real_space_vectors(Nx, Ny)
    k, KX, KY = get_momentum_space_vectors(Nx, Ny)

    H_r, v_x, v_y = \
        get_real_space_hamiltoian_and_velocity_operators(t, A, mu, Nx, Ny)

    print(f'Hamiltonian done {H_r.shape}')
 
    # ----------------------------------------------------------------

    print('--> Cheb dos')
    
    from plot_lanczos_vs_chebyshev import *

    recursion_steps = 128 * 8

    H = H_r
    
    Emin, Emax = sparse_eigsh_emin_emax(H)
    print(f'Emin = {Emin}, Emax = {Emax}')

    eps = 0.1
    eps_w = 10. / beta
    w_min, w_max = Emin - eps, Emax + eps
    w = np.linspace(w_min, w_max, num=recursion_steps*10)
    
    shift, scale = chebyshev_shift_scale(Emin, Emax, eps=eps)
    print(f'shift = {shift}, scale = {scale}')

    v = np.zeros((H.shape[0]), dtype=complex)
    v[0] = 1.

    mu_n = chebyshev_recursion(H, v, recursion_steps, shift, scale)
    gmu_n = mu_n * jackson_kernel(len(mu_n))
    dos_cheb = evaluate_chebyshev(w, gmu_n, shift, scale)
    
    # ----------------------------------------------------------------
    
    print('--> Cheb cond')

    #N_cheb = 256
    #N_cheb = 128
    N_cheb = 64
    #N_cheb = 32
    #N_cheb = 16
    #N_cheb = 3
    
    H_s = H - sp.identity(H.shape[0]) * shift
    H_s /= scale

    # -- Compute matrix recursion storing all matrices
    # -- TODO: replace with stochastic vector sampling
    
    T = chebyshev_matrix_recursion(H_s, N_cheb)

    # -- Cf with DOS moments
    
    v = np.zeros((H.shape[0]), dtype=complex)
    v[0] = 1.
    mu_ref = np.einsum('inm,m,n->i', T, v, v)
    np.testing.assert_array_almost_equal(mu_ref, mu_n[:N_cheb])

    # -- Conductivity tensor

    Rblocks = 10
    R = 100
    
    mu_xx_ref = get_mu_tensor_stochastic_trace(v_x, v_x, H_s, N_cheb, R, Rblocks)
    #mu_xx = mu_xx_ref
    mu_xx = get_mu_tensor(v_x, v_x, H_s, N_cheb)
    print(f'mu_xx.shape = {mu_xx.shape}')
    #mu_xx = mu_xx_ref
    
    #print(f'Re[mu_xx] =\n{mu_xx.real}')
    #print(f'Im[mu_xx] =\n{mu_xx.imag}')

    #print(f'Re[mu_xx_ref] =\n{mu_xx_ref.real}')
    #print(f'Im[mu_xx_ref] =\n{mu_xx_ref.imag}')

    diff = np.max(np.abs(mu_xx - mu_xx_ref))
    print(f'diff = {diff:2.2E}')

    #exit()
    
    x = (w - shift) / scale

    Gamma = eval_Gamma_nm(x, N_cheb)
    print(f'Gamma.shape = {Gamma.shape}')

    I = np.einsum('xnm,nm->x', Gamma, mu_xx)

    #mus_cheb = np.linspace(-6, 6, num=16+1)
    mus_cheb = np.linspace(w_min, w_max, num=128)
    sigmas_cheb = np.empty_like(mus_cheb, dtype=complex)

    def eval_sigma_integral(x, I, scale, mu, beta):
        f = fermi_function(x - mu/scale, beta*scale)
        sigma = np.trapz(I * f, x=x)

        # FIXME: Figure out scale factor
        N_gamma = H_s.shape[0]
        
        #sigma *= -1/scale**2 / N_gamma / N_cheb * 8 / np.pi
        #sigma *= -1/scale / N_gamma / N_cheb
        sigma *= -1 / N_gamma / N_cheb * 2 / np.pi
        
        return sigma
    
    for idx, mu in enumerate(mus_cheb):
        sigmas_cheb[idx] = eval_sigma_integral(x, I, scale, mu, beta)
        #print(f'sigma = {sigmas[idx]}')

    norm = np.trapz(sigmas_cheb, x=mus_cheb).real
    print(f'norm = {norm}')

    print(f'mus_cheb = {mus_cheb}')
    print(f'sigmas_cheb = {sigmas_cheb.real}')

    if True:
        # -- Linear response current-current response function in imaginary time

        from pydlr import dlr

        d = dlr(lamb=beta * 2 * 4 * 10, eps=1e-12)
        print(f'N_dlr = {len(d)}')
        tau_l = d.get_tau(beta)
        tau_l_rev = beta - tau_l
        #dlr_tau_integral_weight = -np.tanh(d.dlrrf/2) * beta / d.dlrrf / H.shape[0] / 2

        sigmas_dlr = np.empty_like(mus_cheb, dtype=complex)

        Id = sp.identity(H.shape[0])
        for idx, mu in enumerate(mus_cheb):
            print(f'idx = {idx}, mu = {mu}')
            H_aa = (H - mu * Id).todense()

            G_laa = d.free_greens_function_tau(H_aa, beta, xi=-1)                                      
            G_xaa = d.dlr_from_tau(G_laa)
            G_laa_rev = d.eval_dlr_tau(G_xaa, tau_l_rev, beta)

            chi_l = np.einsum('ab,lac,lbd,cd->l', v_x.todense(), G_laa, G_laa_rev, v_x.todense())        
            chi_x = d.dlr_from_tau(chi_l)

            #sigma_dlr_ref = np.sum(chi_x * dlr_tau_integral_weight).real
            sigma_dlr = -d.eval_dlr_freq(chi_x, np.array([0.]), beta, xi=+1) / H.shape[0]

            #diff = np.max(np.abs(sigma_dlr - sigma_dlr_ref))
            #print(f'dlr diff = {diff:+2.2E} -- {sigma_dlr}, {sigma_dlr_ref}')
            sigmas_dlr[idx] = sigma_dlr

        print(f'sigmas_dlr = {sigmas_dlr}')

        if False:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(mus_cheb, sigmas_cheb.real, 'o-', label='cheb')
            plt.plot(mus_cheb, sigmas_dlr.real, 'x-', label='dlr')
            plt.legend()
            plt.tight_layout()
    
            #plt.show(); exit()
    
    #exit()

    # ----------------------------------------------------------------

    plt.figure(figsize=(6, 10))

    subp = [3, 1, 1]

    plt.subplot(*subp); subp[-1] += 1

    n = np.arange(N_cheb)
    nn = n[:, None] + n[None, :]

    mu_plot = np.abs(mu_xx.flatten().real)
    idx = mu_plot < 1e-10
    mu_plot[idx] = 0

    mu_plot_ref = np.abs(mu_xx_ref.flatten().real)
    idx = mu_plot_ref < 1e-10
    mu_plot_ref[idx] = 0
    
    plt.plot(nn.flatten(), mu_plot, 'o', alpha=0.5, label='full Tr')
    plt.plot(nn.flatten(), mu_plot_ref, '.', alpha=0.5, label='rng Tr')
    plt.semilogy([], [])
    plt.legend(loc='upper right')
    plt.xlabel(r'$n+m$')
    plt.ylabel(r'$\mu^{(xx)}_{nm}$')
    plt.grid(True)

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(x, I.real, label='re')
    #plt.plot(x, I.imag, label='im')
    #plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'Integrand $I(x)$')
    plt.grid(True)

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(mus_cheb, sigmas_cheb.real)
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\sigma_{xx}$')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure_conductivity_moments.svg')
    #plt.show(); exit()
    
    # ----------------------------------------------------------------

    if False:

        subp = [2, 2, 1]

        plt.subplot(*subp); subp[-1] += 1    
        plt.pcolormesh(KX, KY, eps_k.reshape((Nx, Ny)))
        plt.axis('square')

        plt.subplot(*subp); subp[-1] += 1    
        plt.pcolormesh(KX, KY, n_k.reshape((Nx, Ny)))
        plt.axis('square')

        plt.subplot(*subp); subp[-1] += 1    
        plt.pcolormesh(KX, KY, v_kx.reshape((Nx, Ny)))
        plt.axis('square')

        plt.subplot(*subp); subp[-1] += 1    
        plt.pcolormesh(KX, KY, v_ky.reshape((Nx, Ny)))
        plt.axis('square')

        plt.show()

        exit()
    
    # ----------------------------------------------------------------
    # -- Compute density and current from "non-equilibrium" finite diffs
    
    print('--> Density')
    
    plt.figure(figsize=(6, 10))
    
    subp = [3, 1, 1]

    plt.subplot(*subp); subp[-1] += 1    
    plt.plot(w, dos_cheb.real)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'DOS')
    
    # ---

    plt.subplot(*subp); subp[-1] += 1    

    mu = 0.0 # Chemical potential
    A = np.array([0.0, 0.0]) # External vector potential

    #mus = np.linspace(-6, 6, num=128)
    mus = np.linspace(w_min, w_max, num=128)
    ns = np.zeros_like(mus)
    ns_ref = np.zeros_like(mus)

    djs = np.zeros_like(mus)

    for idx, mu in enumerate(mus):
        
        eps_k, v_kx, v_ky = \
            get_momentum_dispersion_and_velocity_operators(t, A, mu, Nx, Ny)    
        ns_ref[idx] = density_from_momentum(eps_k, beta)

        #H_r, _, _ = \
        #    get_real_space_hamiltoian_and_velocity_operators(t, A, mu, Nx, Ny)
        #rho = real_space_density_matrix(H_r, beta)
        #ns[idx] = density_from_density_matrix(rho)

        # -- Conductivity from finite difference in current :)
        
        dA = 0.0001
        
        Am = np.array([-dA, 0])
        eps_k, v_kx, v_ky = \
            get_momentum_dispersion_and_velocity_operators(t, Am, mu, Nx, Ny)    
        jm = current_from_momentum(eps_k, v_kx, beta)

        Ap = np.array([+dA, 0])
        eps_k, v_kx, v_ky = \
            get_momentum_dispersion_and_velocity_operators(t, Ap, mu, Nx, Ny)    
        jp = current_from_momentum(eps_k, v_kx, beta)
        
        djs[idx] = (jm - jp)/(2*dA)

    plt.plot(mus, ns_ref, '-', label='momentum space')
    #plt.plot(mus, ns, '--', label='real space')

    #plt.plot(w, dos_cheb.real / np.max(dos_cheb.real), alpha=0.5)
    
    plt.legend(loc='upper left')
    plt.ylabel(r'$\langle n \rangle$')
    plt.xlabel(r'$\mu$')
    plt.grid(True)
    
    # ---

    plt.subplot(*subp); subp[-1] += 1    

    plt.plot(mus, djs, '-', label='momentum space')
    plt.plot(mus_cheb, sigmas_cheb.real, '-', label='chebyshev $\sigma_{xx}$')
    plt.plot(mus_cheb, sigmas_dlr.real, '--', label='dlr')
    plt.legend(loc='upper left')
    plt.ylabel(r'$d \langle j_x \rangle / d A_x$')
    plt.xlabel(r'$\mu$')
    plt.grid(True)

    # --
    
    plt.tight_layout()
    plt.savefig('figure_conductivity_comparison.svg')
    plt.show()
    
    exit()
