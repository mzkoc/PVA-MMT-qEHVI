"""
Level-2 bridge (clay loading) Pareto front figure (ENGLISH labels).
qEHVI over clay loading: adhesion vs matrix mobility trade-off.
Run (mobo): python plot_L2_bridge_EN.py  ->  L2_bridge_pareto.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# qEHVI run data (bridge_level2.py output), sorted by n_chain
nchain = np.array([13, 20, 26, 34, 38, 42, 49])
adh    = np.array([258.0, 275.0, 296.5, 332.0, 375.0, 393.4, 429.7])  # mJ/m^2
msd    = np.array([17.80, 9.12, 5.10, 3.43, 2.28, 1.57, 0.65])         # A^2

# evaluation order (to show qEHVI exploration): init 4 then rounds
# init: 20,26,34,49 ; round1: 42 ; round2: 38 ; round3: 13
order_label = {20:"i", 26:"i", 34:"i", 49:"i", 42:"1", 38:"2", 13:"3"}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) objective space: adhesion vs mobility, Pareto front
ax[0].plot(msd, adh, "-", c="#2a78d6", alpha=0.4, zorder=1)
sc = ax[0].scatter(msd, adh, c=nchain, cmap="viridis", s=90, zorder=3,
                   edgecolors="k", linewidths=0.5)
for x, y, n in zip(msd, adh, nchain):
    ax[0].annotate(f"{n}", (x, y), textcoords="offset points",
                   xytext=(6, 5), fontsize=8)
cb = fig.colorbar(sc, ax=ax[0])
cb.set_label("PVA chains (clay loading)")
ax[0].set_xlabel("PVA MSD (A$^2$)  [matrix mobility / flexibility ->]")
ax[0].set_ylabel("Work of adhesion (mJ/m$^2$)  [adhesion ->]")
ax[0].set_title("(a) Pareto front: adhesion vs flexibility")
ax[0].grid(alpha=0.3)

# (b) both objectives vs design variable (clay loading)
ax2 = ax[1]
l1 = ax2.plot(nchain, adh, "o-", c="#2a78d6", label="Adhesion (mJ/m$^2$)")
ax2.set_xlabel("Number of PVA chains  (low = high clay loading ->)")
ax2.set_ylabel("Work of adhesion (mJ/m$^2$)", color="#2a78d6")
ax2.tick_params(axis="y", labelcolor="#2a78d6")
ax2.invert_xaxis()
ax2b = ax2.twinx()
l2 = ax2b.plot(nchain, msd, "s--", c="#c0392b", label="Mobility (MSD)")
ax2b.set_ylabel("PVA MSD (A$^2$)", color="#c0392b")
ax2b.tick_params(axis="y", labelcolor="#c0392b")
ax2.set_title("(b) Competing objectives vs clay loading")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("L2_bridge_pareto.png", dpi=300)
print("Saved: L2_bridge_pareto.png")
print(f"\nAdhesion range: {adh.min():.0f}-{adh.max():.0f} mJ/m^2 "
      f"({100*(adh.max()-adh.min())/adh.mean():.0f}% span)")
print(f"Mobility range: {msd.min():.2f}-{msd.max():.2f} A^2 "
      f"({msd.max()/msd.min():.0f}x span)")
print("All 7 points Pareto-optimal (monotonic trade-off).")
