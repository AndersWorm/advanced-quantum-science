import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from myfunctions import H_XY, U_trotter, H_isingall
import os

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters (rad/us, us)
N = 2
c = 2*np.pi * 3                              # coupling, 3 MHz
g = c * (np.ones((N, N)) - np.eye(N))        # zero diagonal
g_t = g                                      # gate coupling

# Trotter parameters
n = 20                                       # rounds
tau = 0.010                                  # 10 ns
t_rounds = tau * np.arange(1, n + 1)         # compare only after full rounds

ratios = np.logspace(-1, 2, 200)             # Delta/g from 0.1 to 10
basis = np.eye(2**N, dtype=complex)

F_avg = np.zeros_like(ratios)
F_per_state = np.zeros((len(ratios), 2**N))

for a, r in enumerate(ratios):
    d = r * c
    H_exact = H_isingall(N, d*np.ones(N), g)
    H_gate = H_XY(N, (d/2)*np.ones(N), g_t)  # gate fields Delta_j1 = Delta^I / 2
    U_round = U_trotter(H_gate, tau, 1, N)
    U_step_exact = la.expm(-1j * H_exact * tau)

    for i0 in range(2**N):
        psi_ex = basis[:, i0].copy()
        psi_tr = basis[:, i0].copy()
        F = []
        for step in range(n):
            psi_ex = U_step_exact @ psi_ex
            psi_tr = U_round @ psi_tr
            F.append(np.abs(np.vdot(psi_ex, psi_tr))**2)
        F_per_state[a, i0] = np.mean(F)
    F_avg[a] = F_per_state[a].mean()

fig, ax = plt.subplots(figsize=(7, 4.5))
for i0 in range(2**N):
    ax.plot(ratios, F_per_state[:, i0], lw=1, alpha=0.5, label=f'|{i0:0{N}b}>')
ax.plot(ratios, F_avg, 'k', lw=2.2, label='average')

# Where Delta * tau = 1 (BCH breaks down)
r_crit = 1 / (c * tau)
ax.axvline(r_crit, color='gray', ls='--', lw=1)
ax.text(r_crit*1.05, 0.05, r'$\Delta\tau = 1$', color='gray', transform=ax.get_xaxis_transform())

ax.set_xscale('log')
ax.set_xlabel(r'$\Delta / g$')
ax.set_ylabel('Time-averaged fidelity')
ax.set_title(f'Average fidelity vs Δ/g, N = {N}, τ = {tau*1e3:.0f} ns, {n} rounds')
ax.grid(which='both', alpha=0.4)
ax.legend(loc='lower left')
fig.tight_layout()
out = os.path.join(fig_dir, f'avg_fidelity_vs_delta_over_g_N{N}_tau{tau*1e3:.0f}ns.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print(out)
print('F_avg at Delta/g = 0.1, 1, 10:',
      np.interp([0.1, 1, 10], ratios, F_avg))