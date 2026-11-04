"""
Diagonalisation - 1D og 2D toy model i momentumrummet.

Python-oversaettelse af Diagonalisation-1.nb (Mathematica).

Ideen: Schroedingerligningen loeses som et matrix-egenvaerdiproblem i k-basis.
    H = T + V
    T[i,j] = k_i^2 * delta_ij          (diagonal - derfor arbejder vi i k-rummet)
    V[i,j] = 1 / (|k_i - k_j| + dk)    (koblingen mellem forskellige k'er)

Det lille "+dk" i naevneren er en regularisering, saa diagonalen ikke divergerer.

Kør:  python diagonalisation.py            (begge modeller)
      python diagonalisation.py --only 1d
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Plot-stil (svarer til ComputerModern/Frame->True-opsaetningen i notebooken)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.linewidth": 0.8,
    "figure.dpi": 110,
})

OUTDIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------
# Gitter
# ----------------------------------------------------------------------
def k_grid_1d(kmax, N0):
    """k-vaerdier fra -kmax til +kmax i skridt af dk = kmax/N0.

    Bygget paa HELTALSindeks i stedet for Mathematicas Table[i,{i,-kmax,kmax,dk}].
    Det undgaar at flydende tal-afrunding aendrer antallet af punkter.
    """
    dk = kmax / N0
    n = np.arange(-N0, N0 + 1)      # 2*N0+1 punkter, praecis
    return n * dk, dk


def k_grid_2d(kmax, N0):
    """Fladt gitter af (kx, ky)-par -- svarer til Flatten[Table[...], 1]."""
    k1, dk = k_grid_1d(kmax, N0)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    grid = np.column_stack([KX.ravel(), KY.ravel()])
    return grid, dk


# ----------------------------------------------------------------------
# Hamilton-matricer
# ----------------------------------------------------------------------
def hamiltonian_1d(k, dk):
    kinetic = np.diag(k ** 2) #sætter hbar^2/(2m) = 1
    potential = 1.0 / (np.abs(k[:, None] - k[None, :]) + dk) # k[:, None] er en søjlevektor med N rækker, k[None, :] er rækkevektor med N søjler. Resultatet er en NxN matrix med |k_i - k_j| i hver celle.
    return kinetic + potential


def hamiltonian_2d(grid, dk):
    # kinetisk: k.k paa diagonalen
    kinetic = np.diag(np.sum(grid ** 2, axis=1))
    # |k_i - k_j| for alle par (broadcast over sidste akse)
    diff = grid[:, None, :] - grid[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    potential = 1.0 / (dist + dk)
    return kinetic + potential


# ----------------------------------------------------------------------
# Diagonalisering
# ----------------------------------------------------------------------
def solve(H):
    """Egenvaerdier/-vektorer sorteret efter stigende energi (n=0 = grundtilstand).

    eigh bruges fordi H er reel og symmetrisk -- den garanterer reelle
    egenvaerdier og sorterer allerede stigende (svarer til Sort[...] i notebooken).
    Returnerer evecs[n] som den n'te egenvektor (raekker, ikke soejler).
    """
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs.T


def fix_phase(evecs, reference=0):
    """En egenvektor er kun bestemt op til et globalt fortegn.

    Notebooken ganger med Sign[evecs[[1]]] elementvis. Her goer vi det samme:
    fortegnsmoensteret fra grundtilstanden paatvinges alle tilstande, saa
    plottene ser konsistente ud.
    """
    s = np.sign(evecs[reference])
    s[s == 0] = 1.0
    return evecs * s


# ----------------------------------------------------------------------
# 1D
# ----------------------------------------------------------------------
def run_1d(N0=10, kmax=0.5, n_states=3, show=False): #juster her
    k, dk = k_grid_1d(kmax, N0)
    H = hamiltonian_1d(k, dk)
    evals, evecs = solve(H)
    psi = fix_phase(evecs)
    psi[0] = np.abs(evecs[0])          # grundtilstand plottes som |psi_0|

    print(f"[1D] gitter: {len(k)} punkter, dk = {dk:.4f}, H er {H.shape}")
    for n in range(4):
        print(f"     E_{n} = {evals[n]: .6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for n in range(n_states):
        ax1.plot(k, psi[n], "o-", ms=4, lw=1.2, label=rf"$\Psi_{n}$")
    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$\Psi_n(k)$")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for n in range(4):
        ax2.axhline(evals[n], lw=1.4, color=colors[n % len(colors)],
                    label=rf"$E_{n} = {evals[n]:.3f}$")
    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$E_n$")
    ax2.set_xticks([])
    ax2.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "toymodel_1d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"     gemt: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return k, evals, psi


# ----------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------
def run_2d(N0=18, kmax=0.5, n_states=5, show=False):  # her kan du justere antallet af tilstande, der plottes, k_mac og N
    grid, dk = k_grid_2d(kmax, N0)
    side = 2 * N0 + 1

    print(f"[2D] gitter: {side}x{side} = {len(grid)} punkter, dk = {dk:.4f}")
    print(f"     bygger og diagonaliserer {len(grid)}x{len(grid)} matrix ...")

    H = hamiltonian_2d(grid, dk)
    evals, evecs = solve(H)
    psi = fix_phase(evecs)
    psi[0] = np.abs(evecs[0])

    for n in range(4):
        print(f"     E_{n} = {evals[n]: .6f}")

    kx = grid[:, 0].reshape(side, side)
    ky = grid[:, 1].reshape(side, side)

    ncol = min(n_states, 5)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 3.5))
    axes = np.atleast_1d(axes)
    for n, ax in enumerate(axes):
        Z = psi[n].reshape(side, side)
        vmax = np.max(np.abs(Z))
        im = ax.pcolormesh(kx, ky, Z, cmap="RdYlBu_r", shading="auto",
                           vmin=-vmax if n > 0 else 0, vmax=vmax)
        ax.set_title(rf"$\Psi_{n}(\vec k)$")
        ax.set_xlabel(r"$k_x$")
        if n == 0:
            ax.set_ylabel(r"$k_y$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "toymodel_2d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"     gemt: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # energispektrum
    fig2, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(range(5), evals[:5], "o-", ms=6)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$E_n$")
    ax.set_xticks(range(5))
    ax.grid(alpha=0.25)
    fig2.tight_layout()
    path2 = os.path.join(OUTDIR, "spektrum_2d.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"     gemt: {path2}")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return grid, evals, psi


# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["1d", "2d"], help="koer kun den ene model")
    p.add_argument("--N0-1d", type=int, default=20)
    p.add_argument("--kmax-1d", type=float, default=1.0)
    p.add_argument("--N0-2d", type=int, default=18)
    p.add_argument("--kmax-2d", type=float, default=0.5)
    p.add_argument("--show", action="store_true", help="vis vinduer interaktivt")
    a = p.parse_args()

    if a.only != "2d":
        run_1d(N0=getattr(a, "N0_1d"), kmax=getattr(a, "kmax_1d"), show=a.show)
    if a.only != "1d":
        run_2d(N0=getattr(a, "N0_2d"), kmax=getattr(a, "kmax_2d"), show=a.show)
