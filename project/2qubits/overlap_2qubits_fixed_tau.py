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
tau = 0.020                                  # fixed step size, 20 ns
n = 60                                      # number of rounds, T = n*tau = 2 μs
t_trotter = tau * np.arange(n + 1)

state = np.array([1, 0, 0, 0], dtype=complex)   # |00>

fidelity = {}                                # F(t) for each Δ
for d in delta_j_values:
    # Exact evolution on the Trotter times
    H_exact = H_isingall(N, d*np.ones(N), g)
    E, V = la.eigh(H_exact)
    psi_exact = time_evolv(coffs(V, state), E, V, t_trotter)

    # Trotter evolution, gate fields Δ_j1 = Δ^I / 2
    H_gate = H_XY(N, (d/2)*np.ones(N), g_t)
    U_round = U_trotter(H_gate, tau, 1, N)
    psi = state.copy()
    psi_trot = [psi]
    for step in range(n):
        psi = U_round @ psi
        psi_trot.append(psi)
    psi_trot = np.array(psi_trot)

    # Fidelity at each Trotter time
    fidelity[d] = np.abs(np.sum(psi_exact.conj() * psi_trot, axis=1))**2

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
for d in delta_j_values:
    ax.plot(t_trotter*1e3, fidelity[d], 'o-', markersize=3,
            label=f'Δ/g = {d/c:.1f}')

ax.set_xlabel('Time (ns)')
ax.set_ylabel(r'Fidelity $F(t)$')
ax.set_title(f'Trotter vs exact, N = {N}, |00>, τ = {tau*1e3:.0f} ns')
ax.legend()
ax.grid()

# Save the figure
fig.savefig(os.path.join(fig_dir, f'fidelity_vs_time_N{N}_tau{tau*1e3:.0f}ns_initial_state_|00>.png'),
            dpi=300, bbox_inches='tight')
plt.show()