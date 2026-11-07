import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import linalg as la
import os
from myfunctions import coffs, H_ising, time_evolv, U_trotter_round

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Parameters (rad/μs, μs)
N = 3
c = 2*np.pi * 3                                      # coupling, 3 MHz
d = 0.1*c                                                # Δ/g = 1
delta_I = d * np.ones(N)

n = 20                                               # Trotter rounds
tau = 0.010                                          # 10 ns
t_trotter = tau * np.arange(n + 1)
t_exact = np.linspace(0, n*tau, 1000)

g_all = c * (np.ones((N, N)) - np.eye(N))
g_nn = c * (np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))
topologies = {'all-to-all': g_all, 'nearest-neighbour': g_nn}

basis = np.eye(2**N, dtype=complex)
start_states = [0b000, 0b100]
threshold = 0.01

for topo_name, G in topologies.items():
    E, V = la.eigh(H_ising(N, delta_I, G))
    U_round = U_trotter_round(N, G, delta_I, tau)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, i0 in zip(axes, start_states):
        psi0 = basis[:, i0]

        # Exact
        P_ex = np.abs(time_evolv(coffs(V, psi0), E, V, t_exact))**2

        # Trotter
        psi = psi0.copy()
        psi_trot = [psi]
        for step in range(n):
            psi = U_round @ psi
            psi_trot.append(psi)
        P_trot = np.abs(np.array(psi_trot))**2

        for j in range(2**N):
            if P_ex[:, j].max() < threshold:
                continue
            ax.plot(t_exact*1e3, P_ex[:, j], '-', color=f'C{j}')
            ax.plot(t_trotter*1e3, P_trot[:, j], 'o', color=f'C{j}', markersize=4)

        ax.set_title(f'Initial state |{i0:0{N}b}>')
        ax.set_xlabel('Time (ns)')
        ax.grid()

    axes[0].set_ylabel(r'Population $|c_n(t)|^2$')
    state_handles = [Line2D([0], [0], color=f'C{j}', label=f'|{j:0{N}b}>')
                     for j in range(2**N)]
    style_handles = [Line2D([0], [0], color='k', ls='-', label='exact'),
                     Line2D([0], [0], color='k', ls='', marker='o', label='Trotter')]
    fig.legend(handles=state_handles + style_handles, loc='center right')

    fig.suptitle(f'Exact vs Trotter, {topo_name}, N = {N}, Δ/g = {d/c:.1f}, τ = {tau*1e3:.0f} ns')
    fig.tight_layout(rect=[0, 0, 0.85, 0.94])
    fig.savefig(os.path.join(fig_dir, f'exact_vs_trotter_N{N}_{topo_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)