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
d = c                                                # Δ/|g| = 1
t = np.linspace(0, 0.2, 1000)

g_all = c * (np.ones((N, N)) - np.eye(N))            # all-to-all (= ring for N=3)
g_nn = c * (np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))   # open chain
topologies = {'all-to-all': g_all, 'nearest-neighbour': g_nn}

signs = {+1: '-', -1: '--'}                          # sign of g -> line style
basis = np.eye(2**N, dtype=complex)
start_states = [0b000, 0b100]
threshold = 0.01

for topo_name, G in topologies.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, i0 in zip(axes, start_states):
        P = {}
        for s in signs:
            E, V = la.eigh(H_ising(N, d*np.ones(N), s*G))
            psi_t = time_evolv(coffs(V, basis[:, i0]), E, V, t)
            P[s] = np.abs(psi_t)**2

        for j in range(2**N):
            if max(P[s][:, j].max() for s in signs) < threshold:
                continue
            for s, ls in signs.items():
                ax.plot(t*1e3, P[s][:, j], ls, color=f'C{j}')

        ax.set_title(f'Initial state |{i0:0{N}b}>')
        ax.set_xlabel('Time (ns)')
        ax.grid()

    axes[0].set_ylabel(r'Population $|c_n(t)|^2$')

    state_handles = [Line2D([0], [0], color=f'C{j}', label=f'|{j:0{N}b}>')
                     for j in range(2**N)]
    sign_handles = [Line2D([0], [0], color='k', ls=ls, label=f'g {"> 0" if s > 0 else "< 0"}')
                    for s, ls in signs.items()]
    fig.legend(handles=state_handles + sign_handles, loc='center right')

    fig.suptitle(f'Sign of g, {topo_name}, N = {N}, Δ/|g| = {d/c:.1f}')
    fig.tight_layout(rect=[0, 0, 0.85, 0.94])
    fig.savefig(os.path.join(fig_dir, f'ising_N{N}_sign_of_g_{topo_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)