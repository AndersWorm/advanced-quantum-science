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
t = 0.2 # total simulation time 200ns
# Trotter parameters
rounds = np.arange(4,21) # number of rounds, T = n*tau

basis = np.eye(2**N, dtype=complex)
start_states = [0b00, 0b10]                  # |00>, |10>

fidelity = {}                                # key: (Δ, start state, n)
for d in delta_j_values:
    E, V = la.eigh(H_isingall(N, d*np.ones(N), g))
    H_gate = H_XY(N, (d/2)*np.ones(N), g_t)  # gate fields Δ_j1 = Δ^I / 2

    for n in rounds:
        tau = t/n
        t_trotter = tau * np.arange(n + 1)
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

            F_t = np.abs(np.sum(psi_exact.conj() * psi_trot, axis=1))**2
            fidelity[(d, i0, n)] = np.mean(F_t)

# Plot: mean fidelity vs τ
taus = t / rounds * 1e3                      # ns, one per n
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, i0 in zip(axes, start_states):
    for k, d in enumerate(delta_j_values):
        F_avg = [fidelity[(d, i0, n)] for n in rounds]
        ax.plot(taus, F_avg, 'o-', markersize=4, color=f'C{k}',
                label=f'Δ/g = {d/c:.1f}')
    ax.set_title(f'Initial state |{i0:0{N}b}>')
    ax.set_xlabel('Trotter step size τ (ns)')
    ax.legend()
    ax.grid()

axes[0].set_ylabel(r'Time-averaged fidelity $\bar F$')
fig.suptitle(f'Trotter vs exact, N = {N}, T = {t*1e3:.0f} ns')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(fig_dir, f'fidelity_vs_tau_N{N}_T{t*1e3:.0f}ns.png'),
            dpi=300, bbox_inches='tight')
plt.show()