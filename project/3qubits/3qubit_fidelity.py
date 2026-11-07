import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import os
from myfunctions import coffs, H_ising, time_evolv, U_trotter_round

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Parameters (rad/μs, μs)
N = 3
c = 2*np.pi * 3                                      # coupling, 3 MHz
d = 1*c                                              # Δ/g = 1
delta_I = d * np.ones(N)

n = 50                                               # Trotter rounds
tau = 0.010                                          # 10 ns
t_trotter = tau * np.arange(n + 1)

g_all = c * (np.ones((N, N)) - np.eye(N))
g_nn = c * (np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))
topologies = {'all-to-all': g_all, 'nearest-neighbour': g_nn}

basis = np.eye(2**N, dtype=complex)
start_states = [0b000, 0b100]

fidelity = {}                                        # key: (topology, start state)
for topo_name, G in topologies.items():
    E, V = la.eigh(H_ising(N, delta_I, G))
    U_round = U_trotter_round(N, G, delta_I, tau)

    for i0 in start_states:
        psi0 = basis[:, i0]
        psi_exact = time_evolv(coffs(V, psi0), E, V, t_trotter)

        psi = psi0.copy()
        psi_trot = [psi]
        for step in range(n):
            psi = U_round @ psi
            psi_trot.append(psi)
        psi_trot = np.array(psi_trot)

        fidelity[(topo_name, i0)] = np.abs(np.sum(psi_exact.conj() * psi_trot, axis=1))**2

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, i0 in zip(axes, start_states):
    for topo_name in topologies:
        F = fidelity[(topo_name, i0)]
        ax.plot(t_trotter*1e3, F, 'o-', markersize=3,
                label=f'{topo_name} (mean {F.mean():.3f})')
    ax.set_title(f'Initial state |{i0:0{N}b}>')
    ax.set_xlabel('Time (ns)')
    ax.legend()
    ax.grid()

axes[0].set_ylabel(r'Fidelity $F(t)$')
fig.suptitle(f'Trotter vs exact, N = {N}, Δ/g = {d/c:.1f}, τ = {tau*1e3:.0f} ns')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(fig_dir, f'fidelity_vs_time_N{N}_topologies.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)