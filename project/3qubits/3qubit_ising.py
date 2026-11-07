import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import os
from myfunctions import H_ising, time_evolv, coffs

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters for 3 qubits (units: rad/μs and μs)
N = 3
c = 2*np.pi * 3                              # coupling, 3 MHz
g = c * (np.ones((N, N)) - np.eye(N))        # all-to-all, zero diagonal
delta_j_values = np.array([0.1, 1, 10]) * c
t = np.linspace(0, 0.2, 1000)                # μs

basis = np.eye(2**N, dtype=complex)
start_states = [0b000, 0b100, 0b110, 0b111]  # one per number of excitations
threshold = 0.01                             # hide states that stay below this

for d in delta_j_values:
    E, V = la.eigh(H_ising(N, d*np.ones(N), g))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, i0 in zip(axes, start_states):
        psi_t = time_evolv(coffs(V, basis[:, i0]), E, V, t)
        P = np.abs(psi_t)**2
        for j in range(2**N):
            if P[:, j].max() < threshold:
                continue
            ax.plot(t*1e3, P[:, j], color=f'C{j}', label=f'|{j:0{N}b}>')
        ax.set_title(f'Initial state |{i0:0{N}b}>')
        ax.grid()

    for ax in axes[2:]:
        ax.set_xlabel('Time (ns)')
    for ax in axes[::2]:
        ax.set_ylabel(r'Population $|c_n(t)|^2$')

    # Legend with all basis states, fixed colours
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=f'C{j}', label=f'|{j:0{N}b}>')
               for j in range(2**N)]
    fig.legend(handles=handles, loc='center right')

    ratio = d / c
    fig.suptitle(f'Ising model, all-to-all coupling, N = {N}, Δ/g = {ratio:.1f}')
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(os.path.join(fig_dir, f'ising_N{N}_all_to_all_delta_over_g_{ratio:.1f}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)