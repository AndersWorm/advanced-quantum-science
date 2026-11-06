import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from myfunctions import coffs, H_XY, U_trotter, H_isingall, time_evolv
import os

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# System parameters for 2 qubits H__ising(units: rad/μs and μs)
N = 2 # number of qubits
c = 2*np.pi * 3                          # coupling, 3 MHz
g = c * (np.ones((N, N)))    # all-to-all, zero diagonal
delta_j_values = np.array([0.1, 1, 10]) * c # so that delta/g = 0.1, 1, 10
t = np.linspace(0, 0.2, 1000)            # μs 0 to 200 ns, 1000 points
# system parameters for 2 qubits trotter
n_steps=np.arange(4, 21).astype(int) # number of steps for trotterization 10ns time step to 50ns time step
g_t= g # coupling for trotterization, same as exact
delta_tj_values = 1/2*delta_j_values # half the Ising field for trotterization

# Computational basis states |00>, |01>, |10>, |11>
ground_state = np.array([1, 0, 0, 0], dtype=complex) # |00>
overlap = {} # dictionary to store overlaps for each delta_j value and each number of Trotter steps
for n in n_steps:
    tau = t[-1]/n # step size for trotterization 
    t_trotter = tau * np.arange(n + 1)        # times after each Trotter round
    for d in delta_j_values:                  # d = Delta^I (Ising field)
        # Exact Ising model
        H_exact = H_isingall(N, d*np.ones(N), g)
        E, V = la.eigh(H_exact)

        # Trotter: gate parameters Delta_j1 = Delta^I / 2
        H_gate = H_XY(N, (d/2)*np.ones(N), g_t)

        U_round = U_trotter(H_gate, tau, 1, N)
        psi_exact = time_evolv(coffs(V,ground_state ), E, V, t_trotter) # array with psi_exact for each time step

        # Trotter evolution, one round at a time
        psi = ground_state.copy() # start with the ground state
        psi_trot = [psi] # at t=0, psi_trot is the initial state
        for step in range(n):
            psi = U_round @ psi # next state after one Trotter round
            psi_trot.append(psi) # append the state after each Trotter round
        psi_trot = np.array(psi_trot)

        #calculate overlap between exact and Trotter evolutions
        O = np.abs(np.sum(psi_exact.conj() * psi_trot, axis=1))**2
        overlap[(d, n)] = np.mean(O)

fig, ax = plt.subplots(figsize=(7, 4.5))

for d in delta_j_values:
    taus = np.array([t[-1]/n for n in n_steps]) * 1e3          # ns
    F_avg = np.array([overlap[(d, n)] for n in n_steps])
    ax.plot(taus, F_avg, 'o-', label=f'Δ/g = {d/c:.1f}')

ax.set_xlabel('Trotter step size τ (ns)')
ax.set_ylabel(r'Time-averaged fidelity $\bar F$')
ax.set_title(f'Trotter vs exact, N = {N}, initial state |00>')
ax.legend()
ax.grid()

T = t[-1]
fig.savefig(os.path.join(fig_dir, f'fidelity_vs_tau_N{N}_fixed_Ts.png'),
            dpi=300, bbox_inches='tight')
plt.show()