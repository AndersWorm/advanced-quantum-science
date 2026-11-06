import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy import linalg as la
from myfunctions import sigma_op, sigma_x,sigma_y, sigma_z, coffs, H_XY, U_trotter
import os

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters for 2 qubits H__ising(units: rad/μs and μs)
N = 2
c = 2*np.pi * 3                          # coupling, 3 MHz
g = c * (np.ones((N, N)))    # all-to-all, zero diagonal
delta_j_values = np.array([0.1, 1, 10]) * c
t = np.linspace(0, 0.2, 1000)            # μs
# system parameters for 2 qubits trotter
n= 20 # number of steps for trotterization
tau = t[-1]/n # step size for trotterization 10 ns
g_t= g
delta_tj_values = 1/2*delta_j_values

# Computational basis states |00>, |01>, |10>, |11>
basis = np.eye(2**N, dtype=complex)
n_states = 2**N

for d in delta_j_values:
    H = H_XY(N, d*np.ones(N), g_t)
    E, V = la.eigh(H)
    ncols = 2
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5*nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(n_states):
        U_round = U_trotter(H, tau, 1, N)          # each round / timestep
        psi = basis[:, i].copy()
        psi_t = [psi]
        for step in range(n):
            psi = U_round @ psi
            psi_t.append(psi)
        psi_t = np.array(psi_t)                    # form (n+1, 2^N)
        ax = axes[i]
        for j in range(n_states):
            t_trotter = tau * np.arange(n + 1)
            ax.plot(t_trotter*1e3, np.abs(psi_t[:, j])**2, 'o-', label=f'|{j:0{N}b}>')
        ax.set_title(f'Initial state |{i:0{N}b}>')
        ax.grid()
    for ax in axes[-ncols:]:
        ax.set_xlabel('Time (ns)')
    for ax in axes[::ncols]:
        ax.set_ylabel(r'Population $|c_n(t)|^2$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    ratio = d / g[0, 1]
    fig.suptitle(f'Ising trotter model, all-to-all coupling, N = {N}, Δ/g = {ratio:.1f}, Trotter steps = {n} and step size = {tau*1e3:.2f} ns')
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    fig.savefig(os.path.join(fig_dir, f'Trotter_ising_N{N}_delta_over_g_{ratio:.1f}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)