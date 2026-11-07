import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import linalg as la
import os
from myfunctions import H_ising, time_evolv, coffs

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Parameters (rad/μs, μs)
N = 3
c = 2*np.pi * 3                                      # coupling, 3 MHz
d = c                                                # Δ/g = 1
t = np.linspace(0, 0.2, 1000)

g_all = c * (np.ones((N, N)) - np.eye(N))            # all-to-all (= ring for N=3)
g_nn = c * (np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))   # open chain
topologies = {'all-to-all': (g_all, '-'),
              'nearest-neighbour': (g_nn, '--')}

basis = np.eye(2**N, dtype=complex)
start_states = [0b000, 0b100, 0b110, 0b111]
threshold = 0.01

# Eigenbasis per topology, computed once
eig = {name: la.eigh(H_ising(N, d*np.ones(N), G))
       for name, (G, ls) in topologies.items()}

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
axes = axes.flatten()

for ax, i0 in zip(axes, start_states):
    P = {}
    for name in topologies:
        E, V = eig[name]
        P[name] = np.abs(time_evolv(coffs(V, basis[:, i0]), E, V, t))**2

    for j in range(2**N):
        if max(P[name][:, j].max() for name in P) < threshold:
            continue
        for name, (G, ls) in topologies.items():
            ax.plot(t*1e3, P[name][:, j], ls, color=f'C{j}')

    ax.set_title(f'Initial state |{i0:0{N}b}>')
    ax.grid()

for ax in axes[2:]:
    ax.set_xlabel('Time (ns)')
for ax in axes[::2]:
    ax.set_ylabel(r'Population $|c_n(t)|^2$')

state_handles = [Line2D([0], [0], color=f'C{j}', label=f'|{j:0{N}b}>')
                 for j in range(2**N)]
topo_handles = [Line2D([0], [0], color='k', ls=ls, label=name)
                for name, (G, ls) in topologies.items()]
fig.legend(handles=state_handles + topo_handles, loc='center right')

fig.suptitle(f'All-to-all vs nearest-neighbour, N = {N}, Δ/g = {d/c:.1f}')
fig.tight_layout(rect=[0, 0, 0.85, 0.95])
fig.savefig(os.path.join(fig_dir, f'ising_N{N}_topologies_delta_over_g_{d/c:.1f}.png'),
            dpi=300, bbox_inches='tight')
plt.show()