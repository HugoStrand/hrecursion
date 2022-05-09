""" 
Comparison between Lanczos recursion
and Chebyshev recursion (a.k.a. the Kernel Polynomial Method)
for computing the spectral function of a 1D chain
with nearest and next-nearest neighbour hopping.

Author: Hugo U.R. Strand (2022)
"""

import numpy as np
import scipy.sparse as sp


def lanczos_recursion(H, v, steps):
    """ Returns the tridiagonal Lanczos coefficients
    of the matrix `H` using the starting vector `v`
    using the given steps of Lanczos recursions. """

    v_1 = v
    wp_1 = H * v_1
    a = np.dot(wp_1, v_1)
    w_1 = wp_1 - a * v_1

    A, B = np.zeros(steps), np.zeros(steps)

    A[0] = a

    for i in range(1, steps):
        v_0 = v_1
        w_0 = w_1

        b = np.linalg.norm(w_0)
        v_1 = w_0 / b
        wp_1 = H * v_1
        a = np.dot(wp_1, v_1)
        w_1 = wp_1 - a * v_1 - b * v_0

        A[i], B[i] = a, b

    return A, B
    

def matrix_from_lanczos_coeffs(A, B):
    """ Return tridiagnonal (sparse) matrix built from
    Lanczos coefficient vectors `A` and `B` """
    
    N = len(A)
    diag = np.zeros((3, N))
    diag[0] = A
    diag[1, 1:] = B[1:]
    diag[2, :-1] = B[1:]

    offsets = np.array([0, 1, -1])
    H = sp.dia_matrix((diag, offsets), shape=(N, N))

    return H


def evaluate_continued_fractions(z, A, B):
    """ Evaluate the continued fraction expression for the spectral function 
    at frequencies `z` using the Lanczos recursion vectors `A` and `B`"""

    N = len(A)
    f = np.zeros_like(z)
    for i in range(N-1, 0, -1):
        f = B[i]**2 / (z - A[i] - f)

    f = 1. / (z - A[0] - f)
    
    return -f.imag / np.pi


def sparse_eigsh_emin_emax(H):
    from scipy.sparse.linalg import eigsh as sparse_eighsh
    Emin = sparse_eighsh(H, which='SA', return_eigenvectors=False).min()
    Emax = sparse_eighsh(H, which='LA', return_eigenvectors=False).max()
    return Emin, Emax


def chebyshev_shift_scale(Emin, Emax, eps=0.1):
    scale = (Emax - Emin)/(2 - eps)
    shift = (Emax + Emin)/2
    return shift, scale

    
def chebyshev_recursion(H, v, steps, shift=0., scale=1.):
    """ Returns the Chebyshev moments `mu_n`
    of the matrix `H` using the starting vector `v`
    and the given number of `steps` for the recursion. """

    def Hs(v):
        res = H * v
        res -= shift * v
        res /= scale
        return res
    
    v_0 = v
    v_1 = Hs(v_0)

    mu = np.zeros(2*steps)

    mu[0] = np.dot(v_0, v_0)
    mu[1] = np.dot(v_1, v_0)
    
    for n in range(1, steps):
        v_2 = 2*Hs(v_1) - v_0
        mu[2*n]   = 2 * np.dot(v_1, v_1) - mu[0]
        mu[2*n+1] = 2 * np.dot(v_2, v_1) - mu[1]
        v_0, v_1 = v_1, v_2

    return mu


def jackson_kernel(N):
    """ The optimal Jackson (gaussian) kernel of order `N` for the Chebyshev moments
    Eq. (71) in the KPM review [https://doi.org/10.1103/RevModPhys.78.275] """

    n = np.arange(N)
    theta = np.pi * n / (N + 1)
    k = (N - n + 1)*np.cos(theta) + np.sin(theta)/np.tan(np.pi/(N+1))
    k /= (N + 1)
    
    return k


def evaluate_chebyshev(w, mu_n, shift=0., scale=1.):
    """ Evaluation of the `mu_n` Chebyshev moment expansion for arbitrary frequencies. 
    (For faster evaluation on the Chebhshev collocation points use FFT.) """

    mu_n = mu_n.copy()
    mu_n[1:] *= 2
    dos = np.polynomial.chebyshev.chebval((w - shift)/scale, mu_n)
    dos /= np.sqrt(scale**2 - (w - shift)**2) * np.pi
    
    return dos


def chebyshev_gauss_quadrature(N):
    """ Chebyshev-Gauss quadrature points and weights.
    Jie Shen, Tao Tang, Li-Lian Wang, Spectral methods (2011) """
    i = np.arange(N+1)
    x_i = - np.cos((2*i + 1)*np.pi/(2*N + 2))
    w_i = np.pi / (N + 1) * np.sqrt(1 - x_i**2)
    return x_i, w_i


if __name__ == '__main__':

    N = 1000 # system size, number of sites in 1D chain

    recursion_steps = N // 4

    t  = 0.25  # nearest neighbour hopping
    tp = 0.20  # next-nearest neighbour hopping
    mu = 0.20  # chemical potential

    t, tp, mu = 1.0, 0.7, -0.5
    
    # -- Build sparse periodic tightbinding Hamiltonian
    
    offsets = np.array([0, 1, -1, N-1, -N+1, 2, -2, N-2, -N+2])
    
    diag = np.ones((len(offsets), N))
    diag[0] = -mu
    diag[1:5] *= -t
    diag[5:9] *= tp

    H = sp.dia_matrix((diag, offsets), shape=(N, N))

    # -- Determine spectrum range Emin Emax
    
    Emin, Emax = sparse_eigsh_emin_emax(H)

    eps = 0.5
    w_min, w_max = Emin - eps/4, Emax + eps/4
    w = np.linspace(w_min, w_max, num=recursion_steps*10)

    # -- Lanczos recursion

    v = np.zeros((N))
    v[0] = 1.

    A, B = lanczos_recursion(H, v, recursion_steps)

    eta = 1.e-2
    z = w + 1.j * eta
    dos_lanczos = evaluate_continued_fractions(z, A, B)

    # -- Chebyshev recursion

    shift, scale = chebyshev_shift_scale(Emin, Emax, eps=eps)
    mu_n = chebyshev_recursion(H, v, recursion_steps, shift, scale)
    gmu_n = mu_n * jackson_kernel(len(mu_n))
    dos_cheb = evaluate_chebyshev(w, gmu_n, shift, scale)

    # -- Occupied moments

    # -- Reference values using 2nd order trapezoidal method
    
    w_f = np.linspace(w_min, 0, num=10 * recursion_steps)
    dos_f = evaluate_chebyshev(w_f, gmu_n, shift, scale)

    M0_ref = np.trapz(dos_f, x=w_f)
    M1_ref = np.trapz(dos_f * w_f, x=w_f)
    M2_ref = np.trapz(dos_f * w_f**2, x=w_f)

    # -- Chebyshev-Gauss quadrature with rescaling
    
    x_i, q_i = chebyshev_gauss_quadrature(recursion_steps)
    w_i = 0.5*(x_i + 1) * w_min
    q_i *= -w_min / 2

    # -- Evaluate on quadrature nodes and integrate

    dos_i = evaluate_chebyshev(w_i, gmu_n, shift, scale)
    
    M0 = np.sum(dos_i * q_i)
    M1 = np.sum(dos_i * w_i * q_i)
    M2 = np.sum(dos_i * w_i**2 * q_i)

    print('--> Computed occupied moments')
    
    print(f'M0 (ref) = {M0_ref}')
    print(f'M0       = {M0}')

    print(f'M1 (ref) = {M1_ref}')
    print(f'M1       = {M1}')

    print(f'M2 (ref) = {M2_ref}')
    print(f'M2       = {M2}')
        
    # -- Visualization
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 8))
    subp = [3, 1, 1]
    
    plt.subplot(*subp); subp[-1] += 1
    plt.title(rf"1D chain, $t={t}$, $t'={tp}$, $\mu={mu}$")
    plt.plot(w, dos_lanczos, label='Lanczos cont.frac.', alpha=0.75)
    plt.plot(w, dos_cheb, label='Chebshev', zorder=-100)    
    plt.ylabel(r'DOS')
    plt.xlabel(r'$\omega$')
    plt.legend(loc='best')

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(A, label=r'$\alpha_n$')
    plt.plot(B, label=r'$\beta_n$')
    plt.ylabel('Lanczos coeffs.')
    plt.xlabel('Index $n$')
    plt.legend(loc='best')

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(np.abs(mu_n), '.-', label=r'$|\mu_n|$')
    plt.plot(np.abs(gmu_n), '.-', label=r'$|\mu_n \cdot g_n|$ (Jackson kernel)')
    plt.semilogy([], [])
    plt.ylabel('Chebyshev coeffs.')
    plt.xlabel('Index $n$')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('figure_1d_chain_Lanczos_vs_Chebyshev.pdf')
    plt.savefig('figure_1d_chain_Lanczos_vs_Chebyshev.svg')
    plt.show()
