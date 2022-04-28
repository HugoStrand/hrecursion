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


def chebyshev_recursion(H, v, steps):
    """ Returns the Chebyshev moments `mu_n`
    of the matrix `H` using the starting vector `v`
    and the given number of `steps` for the recursion. """

    v_0 = v
    v_1 = H*v_0

    mu = np.zeros(2*steps)

    mu[0] = np.dot(v_0, v_0)
    mu[1] = np.dot(v_1, v_0)
    
    for n in range(1, steps):
        v_2 = 2*H*v_1 - v_0
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


def evaluate_chebyshev(w, mu_n):
    """ Evaluation of the `mu_n` Chebyshev moment expansion for arbitrary frequencies. 
    (For faster evaluation on the Chebhshev collocation points use FFT.) """
    
    mu_n = mu_n.copy()
    mu_n[1:] *= 2
    dos = np.polynomial.chebyshev.chebval(w, mu_n)
    dos /= np.sqrt(1 - w**2) * np.pi
    
    return dos


if __name__ == '__main__':

    N = 400 # system size, number of sites in 1D chain

    recursion_steps = N // 4

    t  = 0.25   # nearest neighbour hopping
    tp = 0.2   # next-nearest neighbour hopping
    mu = 0.2 + -2*tp # chemical potential
    
    # -- Build sparse periodic tightbinding Hamiltonian
    
    offsets = np.array([0, 1, -1, N-1, -N+1, 2, -2, N-2, -N+2])
    
    diag = np.ones((len(offsets), N))
    diag[0] = mu
    diag[1:5] *= -t
    diag[5:9] *= tp

    H = sp.dia_matrix((diag, offsets), shape=(N, N))

    # -- Lanczos recursion

    v = np.zeros((N))
    v[0] = 1.

    A, B = lanczos_recursion(H, v, recursion_steps)

    eps = 1e-4
    w = np.linspace(-1+eps, 1-eps, num=4000)

    eta = 1.e-2
    z = w + 1.j * eta
    dos_lanczos = evaluate_continued_fractions(z, A, B)

    # -- Chebyshev recursion
    
    mu_n = chebyshev_recursion(H, v, recursion_steps)
    gmu_n = mu_n * jackson_kernel(len(mu_n))
    dos_cheb = evaluate_chebyshev(w, gmu_n)

    # -- Visualization
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 8))
    subp = [3, 1, 1]
    
    plt.subplot(*subp); subp[-1] += 1
    plt.title(rf"1D chain $t={t}$ $t'={tp}$")
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
    plt.plot(np.abs(mu_n), '.-', label=r'$\mu_n$')
    plt.plot(np.abs(gmu_n), '.-', label=r'$\mu_n \cdot g_n$ (Jackson kernel)')
    plt.semilogy([], [])
    plt.ylabel('Chebyshev coeffs.')
    plt.xlabel('Index $n$')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('figure_1d_chain_Lanczos_vs_Chebyshev.pdf')
    plt.show()
