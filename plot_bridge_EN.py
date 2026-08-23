"""
Bridge Level-1 results (ENGLISH labels for publication).
Run (mobo): python plot_bridge_EN.py  ->  bridge_level1_results.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# run data (from bridge_final2.txt)
T   = np.array([304.3, 326.8, 338.4, 358.6, 380.2, 392.7])
adh = np.array([1495.4, 1489.3, 1452.5, 1474.3, 1468.4, 1502.9])   # |adhesion| kcal/mol
msd = np.array([0.59,   0.72,   0.79,   0.83,   1.09,   1.29])       # MSD (A^2)
Tg  = 340.0

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

# panel 1: MSD vs T (mobility = Tg signature)
ax[0].scatter(T, msd, c="#2a78d6", s=60, zorder=3)
z = np.polyfit(T, msd, 1)
tt = np.linspace(T.min(), T.max(), 50)
ax[0].plot(tt, np.polyval(z, tt), "--", c="#c0392b", lw=1.4, label="linear trend")
ax[0].axvline(Tg, color="gray", ls=":", lw=1)
ax[0].annotate(f"Tg ~ {Tg:.0f} K", xy=(Tg, msd.max()*0.95), fontsize=10, color="gray")
ax[0].set_xlabel("Temperature (K)")
ax[0].set_ylabel("PVA MSD (A$^2$)  [mobility]")
ax[0].set_title("(a) Mobility increases with temperature")
ax[0].legend(frameon=False, fontsize=9)
ax[0].grid(alpha=0.3)

# panel 2: objective space (adhesion vs mobility), Pareto marked
ax[1].scatter(msd, adh, c="#2a78d6", s=60, zorder=3)
for xi, yi, ti in zip(msd, adh, T):
    ax[1].annotate(f"{ti:.0f}K", (xi, yi), textcoords="offset points",
                   xytext=(5, 4), fontsize=8, color="#333")
imax = np.argmax(msd)
ax[1].scatter([msd[imax]], [adh[imax]], s=180, facecolors="none",
              edgecolors="#c0392b", linewidths=2, label="Pareto-optimal", zorder=4)
ax[1].set_xlabel("PVA MSD (A$^2$)  [mobility ->]")
ax[1].set_ylabel("|Adhesion| (kcal/mol)  [adhesion ->]")
ax[1].set_title("(b) Adhesion-mobility objective space")
ax[1].legend(frameon=False, fontsize=9)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("bridge_level1_results.png", dpi=300)
print("Saved: bridge_level1_results.png")
