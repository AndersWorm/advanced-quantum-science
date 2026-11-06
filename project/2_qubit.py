import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from myfunctions import H_isingall, time_evolv, coffs
import os

# Output folder next to this script
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters for 2 qubits (units: rad/μs and μs)
N = 2
c = 2*np.pi * 3                          # coupling, 3 MHz
g = c * (np.ones((N, N)) - np.eye(N))    # all-to-all, zero diagonal to make sure no self-coupling
delta_j_values = np.array([0.1, 1, 10]) * c
t = np.linspace(0, 0.2, 1000)            # μs

# Computational basis states |00>, |01>, |10>, |11>
basis = np.eye(2**N, dtype=complex)
n_states = 2**N

for d in delta_j_values:
    H = H_isingall(N, d*np.ones(N), g)
    E, V = la.eigh(H)
    coffs_list = [coffs(V, basis[:, i]) for i in range(n_states)]

    ncols = 2
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5*nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(n_states):
        psi_t = time_evolv(coffs_list[i], E, V, t)
        ax = axes[i]
        for j in range(n_states):
            ax.plot(t*1e3, np.abs(psi_t[:, j])**2, label=f'|{j:0{N}b}>')
        ax.set_title(f'Initial state |{i:0{N}b}>')
        ax.grid()

    for ax in axes[-ncols:]:
        ax.set_xlabel('Time (ns)')
    for ax in axes[::ncols]:
        ax.set_ylabel(r'Population $|c_n(t)|^2$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    ratio = d / g[0, 1]
    fig.suptitle(f'Ising model, all-to-all coupling, N = {N}, Δ/g = {ratio:.1f}')
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    fig.savefig(os.path.join(fig_dir, f'ising_N{N}_delta_over_g_{ratio:.1f}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)