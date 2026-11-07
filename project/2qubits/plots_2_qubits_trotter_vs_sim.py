import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import linalg as la
from myfunctions import coffs, H_XY, U_trotter, H_isingall, time_evolv
import os

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters (rad/μs, μs)
N = 2
c = 2*np.pi * 3                              # coupling, 3 MHz
g = c * (np.ones((N, N)) - np.eye(N))        # all-to-all, zero diagonal
g_t = g                                      # gate coupling, g^{12} = c
delta_j_values = np.array([0.1, 1, 10]) * c  # Δ/g = 0.1, 1, 10

# Trotter parameters
n = 20                                       # rounds
tau = 0.010                                  # 10 ns
t_trotter = tau * np.arange(n + 1)
t_exact = np.linspace(0, n*tau, 1000)

basis = np.eye(2**N, dtype=complex)
start_states = [0b00, 0b10, 0b11]            # one per number of excitations
threshold = 0.01

for d in delta_j_values:
    E, V = la.eigh(H_isingall(N, d*np.ones(N), g))
    H_gate = H_XY(N, (d/2)*np.ones(N), g_t)  # gate fields Δ_j1 = Δ^I / 2
    U_round = U_trotter(H_gate, tau, 1, N)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, i0 in zip(axes, start_states):
        psi0 = basis[:, i0]
        P_ex = np.abs(time_evolv(coffs(V, psi0), E, V, t_exact))**2

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

    ratio = d / c
    fig.suptitle(f'Exact vs Trotter, N = {N}, Δ/g = {ratio:.1f}, '
                 f'τ = {tau*1e3:.0f} ns, {n} rounds')
    fig.tight_layout(rect=[0, 0, 0.9, 0.93])
    fig.savefig(os.path.join(fig_dir, f'exact_vs_trotter_N{N}_delta_over_g_{ratio:.1f}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)