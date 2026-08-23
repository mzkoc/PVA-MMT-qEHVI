"""
Level-2 clay loading analysis (ENGLISH labels for publication).
Run (mobo): python plot_L2_clay_EN.py  ->  L2_clay_loading.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nchain = np.array([13, 25, 37, 50])
adh    = np.array([928.9, 1009.1, 1190.3, 1502.3])  # |adhesion| kcal/mol

AREA_A2 = 51.918 * 45.077
AREA_m2 = AREA_A2 * 1e-20
adh_J   = adh * 6.9477e-21
adh_mJm2 = adh_J / AREA_m2 * 1e3

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) adhesion vs chain number -- saturation
ax[0].scatter(nchain, adh, c="#2a78d6", s=70, zorder=3)
ax[0].plot(nchain, adh, "-", c="#2a78d6", alpha=0.4)
for i in range(len(nchain)-1):
    dslope = (adh[i+1]-adh[i])/(nchain[i+1]-nchain[i])
    ax[0].annotate(f"{dslope:.0f}/chain",
                   ((nchain[i]+nchain[i+1])/2, (adh[i]+adh[i+1])/2),
                   fontsize=8, color="#c0392b", ha="center")
ax[0].set_xlabel("Number of PVA chains  (low = high clay loading ->)")
ax[0].set_ylabel("|Adhesion| (kcal/mol)")
ax[0].set_title("(a) Adhesion saturation curve")
ax[0].grid(alpha=0.3)
ax[0].invert_xaxis()

# (b) per-area adhesion (mJ/m2)
ax[1].scatter(nchain, adh_mJm2, c="#27ae60", s=70, zorder=3)
ax[1].plot(nchain, adh_mJm2, "-", c="#27ae60", alpha=0.4)
for x, y in zip(nchain, adh_mJm2):
    ax[1].annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                   xytext=(6, 4), fontsize=9)
ax[1].set_xlabel("Number of PVA chains  (low = high clay loading ->)")
ax[1].set_ylabel("Work of adhesion (mJ/m$^2$)  [per clay area]")
ax[1].set_title("(b) Per-area work of adhesion")
ax[1].grid(alpha=0.3)
ax[1].invert_xaxis()

plt.tight_layout()
plt.savefig("L2_clay_loading.png", dpi=300)
print("Saved: L2_clay_loading.png")
