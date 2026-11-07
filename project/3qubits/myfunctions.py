import numpy as np
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
def H_ising(N, delta, g):
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

def H_XY_pair(N, j, k, delta_j, delta_k, g):
    """XY gate Hamiltonian on the pair (j, k) only."""
    H  = -delta_j/2 * sigma_op(sigma_z, j, N) 
    H += -delta_k/2 * sigma_op(sigma_z, k, N)
    H += g/2 * (sigma_op(sigma_x, j, N, sigma_x, k)
                + sigma_op(sigma_y, j, N, sigma_y, k))
    return H

def U_trotter_round(N, G, delta_I, tau):
    """One first-order Trotter round, eq. (21), for any coupling matrix G."""
    degree = np.count_nonzero(G, axis=1)          # d_j
    delta_gate = delta_I / (2 * degree)            # Δ_j1 = Δ_j^I / (2 d_j)
    U = np.eye(2**N, dtype=complex) # identity for the round
    for j in range(N):
        for k in range(N):
            if k == j or G[j, k] == 0:
                continue # skip if no coupling or same qubit
            H = H_XY_pair(N, j, k, delta_gate[j], delta_gate[k], G[j, k])
            Xj = sigma_op(sigma_x, j, N)
            U = la.expm(-1j * (Xj @ H @ Xj) * tau) @ la.expm(-1j * H * tau) @ U
    return U