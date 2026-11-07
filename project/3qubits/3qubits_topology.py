import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import linalg as la
import os
from myfunctions import H_ising, time_evolv, coffs

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters (rad/μs, μs)
N = 3
c = 2*np.pi * 3                                      # coupling, 3 MHz
d = c                                                # Δ/g = 1
t = np.linspace(0, 0.2, 1000)

g_all = c * (np.ones((N, N)) - np.eye(N))            # all-to-all (= ring for N=3)
g_nn = c * (np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))   # open chain

topologies = {'all-to-all': (g_all, '-'),
              'nearest-neighbour': (g_nn, '--')}

basis = np.eye(2**N, dtype=complex)
start_states = [0b000, 0b100]                        # |000>, |100>
threshold = 0.01                                     # hide states that stay below this

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax, i0 in zip(axes, start_states):
    # Evolve under both topologies
    P = {}
    for name, (G, ls) in topologies.items():
        E, V = la.eigh(H_ising(N, d*np.ones(N), G))
        psi_t = time_evolv(coffs(V, basis[:, i0]), E, V, t)
        P[name] = np.abs(psi_t)**2

    # Plot only basis states that are populated in at least one topology
    for j in range(2**N):
        if max(P[name][:, j].max() for name in P) < threshold:
            continue
        for name, (G, ls) in topologies.items():
            ax.plot(t*1e3, P[name][:, j], ls, color=f'C{j}')

    ax.set_title(f'Initial state |{i0:0{N}b}>')
    ax.set_xlabel('Time (ns)')
    ax.grid()

axes[0].set_ylabel(r'Population $|c_n(t)|^2$')

# Legend: colours = basis states, line styles = topology
state_handles = [Line2D([0], [0], color=f'C{j}', label=f'|{j:0{N}b}>')
                 for j in range(2**N)]
topo_handles = [Line2D([0], [0], color='k', ls=ls, label=name)
                for name, (G, ls) in topologies.items()]
fig.legend(handles=state_handles + topo_handles, loc='center right')

fig.suptitle(f'Ising model, N = {N}, Δ/g = {d/c:.1f}: all-to-all vs nearest-neighbour')
fig.tight_layout(rect=[0, 0, 0.85, 0.94])
fig.savefig(os.path.join(fig_dir, f'ising_N{N}_topologies_compared.png'),
            dpi=300, bbox_inches='tight')
plt.show()