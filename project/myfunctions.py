import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy import linalg as la

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def sigma_op(op1, j, N, op2=None, k=None):
    """Operator op1 on qubit j, optionally op2 on qubit k, identity on the rest."""
    ops = [I] * N
    ops[j] = op1
    if op2 is not None:
        ops[k] = op2
    return reduce(np.kron, ops)

def H_isingall(N, delta, g):
    """
    Transverse-field Ising Hamiltonian H/hbar (units of angular frequency).

    N     : number of qubits
    delta : array (N,), effective fields Delta_j^I
    g     : array (N, N), Ising couplings g_jk (zero where qubits don't couple)
    Returns a 2^N x 2^N matrix.
    """
    H = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        H -= delta[j] / 2 * sigma_op(sigma_z, j, N)
    for j in range(N):
        for k in range(j + 1, N):
            H += (g[j, k] + g[k, j]) * sigma_op(sigma_x, j, N) @ sigma_op(sigma_x, k, N)
    return H

def time_evolv(coffs, E, V, t):
    """Time evolution of the state given coefficients in the eigenbasis and eigenvalues."""
    psi_t= []
    for ti in t:
        w = coffs * np.exp(-1j * E * ti)
        psi_t.append(np.sum(w*V, axis=1))
    return np.array(psi_t)
def coffs(V, En):
    """Compute the coefficients of the initial state in the eigenbasis of H."""
    return V.conj().T @ En  # coefficients of the initial state in the eigenbasis of H, the overlap of the initial state with the eigenvectors of H.

def H_XY(N, delta, g):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        H -= delta[j] /2 * sigma_op(sigma_z, j, N)
    for j in range(N):
        for k in range(j + 1, N):
            H += 1/2 * (g[j, k]) * (sigma_op(sigma_x, j, N) @ sigma_op(sigma_x, k, N) + sigma_op(sigma_y, j, N) @ sigma_op(sigma_y, k, N))
    return H
def H_AXY(N, delta, g):
    return sigma_op(sigma_x,0,N)@H_XY(N, delta, g)@sigma_op(sigma_x,0,N)

def U_trotter(H, tau, n, N):
    """First-order Trotter round for 2 qubits, eq. (21): pairs (1,2) and (2,1)."""
    X1 = sigma_op(sigma_x, 0, N)
    X2 = sigma_op(sigma_x, 1, N)

    U_xy    = la.expm(-1j * H * tau)
    U_axy_1 = la.expm(-1j * (X1 @ H @ X1) * tau)   # A-XY, qubit 1 flipped
    U_axy_2 = la.expm(-1j * (X2 @ H @ X2) * tau)   # A-XY, qubit 2 flipped

    U_round = U_axy_2 @ U_xy @ U_axy_1 @ U_xy
    return np.linalg.matrix_power(U_round, n)