import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from myfunctions import coffs, H_XY, U_trotter, H_isingall, time_evolv
import os

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters (units: rad/μs and μs)
N = 2
c = 2*np.pi * 3                              # coupling, 3 MHz
g = c * (np.ones((N, N)) - np.eye(N))        # all-to-all, zero diagonal
g_t = g                                      # gate coupling, g^{12} = c
delta_j_values = np.array([0.1, 1, 10]) * c  # Δ/g = 0.1, 1, 10

# Trotter parameters
tau = 0.020                                  # fixed step size
n = 60                                       # number of rounds, T = n*tau
t_trotter = tau * np.arange(n + 1)

basis = np.eye(2**N, dtype=complex)
start_states = [0b00, 0b10]                  # |00>, |10>

fidelity = {}                                # key: (Δ, start state)
for d in delta_j_values:
    # Exact Hamiltonian and Trotter round depend only on Δ
    E, V = la.eigh(H_isingall(N, d*np.ones(N), g))
    H_gate = H_XY(N, (d/2)*np.ones(N), g_t)  # gate fields Δ_j1 = Δ^I / 2
    U_round = U_trotter(H_gate, tau, 1, N)

    for i0 in start_states:
        psi0 = basis[:, i0]
        psi_exact = time_evolv(coffs(V, psi0), E, V, t_trotter)

        psi = psi0.copy()
        psi_trot = [psi]
        for step in range(n):
            psi = U_round @ psi
            psi_trot.append(psi)
        psi_trot = np.array(psi_trot)

        fidelity[(d, i0)] = np.abs(np.sum(psi_exact.conj() * psi_trot, axis=1))**2

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, i0 in zip(axes, start_states):
    for k, d in enumerate(delta_j_values):
        F = fidelity[(d, i0)]
        color = f'C{k}'
        ax.plot(t_trotter*1e3, F, 'o-', markersize=3, color=color,
                label=f'Δ/g = {d/c:.1f}  (mean {F.mean():.3f})')
        ax.axhline(F.mean(), color=color, ls='--', lw=1)
    ax.set_title(f'Initial state |{i0:0{N}b}>')
    ax.set_xlabel('Time (ns)')
    ax.legend()
    ax.grid()

axes[0].set_ylabel(r'Fidelity $F(t)$')
fig.suptitle(f'Trotter vs exact, N = {N}, τ = {tau*1e3:.0f} ns')
fig.tight_layout(rect=[0, 0, 1, 0.94])

fig.savefig(os.path.join(fig_dir, f'fidelity_vs_time_N{N}_tau{tau*1e3:.0f}ns.png'),
            dpi=300, bbox_inches='tight')
plt.show()