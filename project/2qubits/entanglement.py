import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from myfunctions import H_isingall, time_evolv, coffs
import os

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

def entanglement_entropy(psi, N):
    """Entropy of qubit 1 vs the rest, in bits."""
    M = psi.reshape(2, 2**(N-1))
    rho_A = M @ M.conj().T
    lam = np.linalg.eigvalsh(rho_A)
    lam = lam[lam > 1e-12]
    return -np.sum(lam * np.log2(lam))

# Parameters (rad/μs, μs)
N = 2
delta = 2*np.pi * 3                          # fixed field, 3 MHz
g_ratios = np.array([0.1, 0.25, 0.5, 1, 2])  # g/Δ
t = np.linspace(0, 0.3, 1500)

basis = np.eye(2**N, dtype=complex)
start_states = {'|00>': basis[:, 0], '|01>': basis[:, 1]}

psi0 = basis[:, 0]                        # |00>
r_fine = np.linspace(0.01, 1.5, 60)       # g/Δ

S_max_num = []
for r in r_fine:
    g = r * delta * (np.ones((N, N)) - np.eye(N))
    E, V = la.eigh(H_isingall(N, delta*np.ones(N), g))
    psi_t = time_evolv(coffs(V, psi0), E, V, t)
    S_max_num.append(max(entanglement_entropy(psi, N) for psi in psi_t))

# Analytic: P11_max = 4r^2 / (1 + 4r^2), entropy maximal at P = 1/2
A = 4*r_fine**2 / (1 + 4*r_fine**2)
p = np.minimum(A, 0.5)
S_max_ana = -p*np.log2(p) - (1-p)*np.log2(1-p)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(r_fine, S_max_num, 'o', markersize=4, label='numerical')
ax.plot(r_fine, S_max_ana, '-', label='analytic')
ax.axvline(0.5, color='gray', ls='--', label='g/Δ = 0.5')
ax.set_xlabel('g/Δ')
ax.set_ylabel(r'$S_{\max}$ (bits)')
ax.set_title(f'Maximum entanglement from |00>, N = {N}')
ax.legend()
ax.grid()

fig.savefig(os.path.join(fig_dir, f'max_entanglement_vs_g_N{N}.png'),
            dpi=300, bbox_inches='tight')
plt.show()