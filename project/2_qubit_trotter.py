import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy import linalg as la
from myfunctions import sigma_op, sigma_x,sigma_y, sigma_z, coffs, H_XY, U_trotter

def U_trotter(H, tau, n, N):
    """First-order Trotter: (U_axy U_xy)^n for one XY + A-XY pair."""
    X1 = sigma_op(sigma_x, 0, N)
    U_xy  = la.expm(-1j * H * tau)
    U_axy = la.expm(-1j * (X1 @ H @ X1) * tau)
    return np.linalg.matrix_power(U_axy @ U_xy, n @ U_axy @ U_xy)
def H_XY(N, delta, g):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        H -= delta[j] /2 * sigma_op(sigma_z, j, N)
    for j in range(N):
        for k in range(j + 1, N):
            H += (g[j, k]) * (sigma_op(sigma_x, j, N) @ sigma_op(sigma_x, k, N) + sigma_op(sigma_y, j, N) @ sigma_op(sigma_y, k, N))
    return H
def H_AXY(N, delta, g):
    return sigma_op(sigma_x,0,N)@H_XY(N, delta, g)@sigma_op(sigma_x,0,N)
# System parameters for 2 qubits H__ising(units: rad/μs and μs)
N = 2
c = 2*np.pi * 3                          # coupling, 3 MHz
g = c * (np.ones((N, N)) - np.eye(N))    # all-to-all, zero diagonal
delta_j_values = np.array([0.1, 1, 10]) * c
t = np.linspace(0, 0.2, 1000)            # μs
# system parameters for 2 qubits trotter
tau = 0.2/20 # step size for trotterization 1 ns
n= 20 # number of steps for trotterization
g_t= g
delta_tj_values = 1/2*delta_j_values

# Computational basis states |00>, |01>, |10>, |11>
basis = np.eye(2**N, dtype=complex)
n_states = 2**N
E0 = basis[:, 0]  # initial state |00>
d = delta_tj_values[1]  # choose a specific delta value for the trotter simulation
H_xy = H_XY(N, d*np.ones(N), g_t)
U = U_trotter(H_xy, tau, n, N)
psi_t = U @ E0