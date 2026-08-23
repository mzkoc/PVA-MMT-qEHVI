"""
Level-2 bridge Pareto front WITH ERROR BARS (ENGLISH).
Error bars from 3-seed statistics on n=13,25,49; single-run values for others.
Run (mobo): python plot_L2_bridge_errorbars_EN.py -> L2_bridge_pareto.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# qEHVI run points (single-run) sorted by n
nchain = np.array([13, 20, 26, 34, 38, 42, 49])
adh    = np.array([258.0, 275.0, 296.5, 332.0, 375.0, 393.4, 429.7])
msd    = np.array([17.80, 9.12, 5.10, 3.43, 2.28, 1.57, 0.65])

# 3-seed statistics (mean +/- std) for the three measured systems
eb_n    = {13: (268.5, 13.3, 12.92, 3.37),
           25: (314.2, 17.4, 6.97, 2.88),
           49: (436.3, 2.4, 0.75, 0.02)}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) objective space with error bars where available
ax[0].plot(msd, adh, "-", c="#2a78d6", alpha=0.35, zorder=1)
sc = ax[0].scatter(msd, adh, c=nchain, cmap="viridis", s=90, zorder=3,
                   edgecolors="k", linewidths=0.5)
# overlay error bars on measured systems
for n,(am,asd,mm,msd_) in eb_n.items():
    ax[0].errorbar(mm, am, xerr=msd_, yerr=asd, fmt="none",
                   ecolor="#444", elinewidth=1.2, capsize=3, zorder=4)
for x, y, n in zip(msd, adh, nchain):
    ax[0].annotate(f"{n}", (x, y), textcoords="offset points",
                   xytext=(6, 5), fontsize=8)
cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("PVA chains (clay loading)")
ax[0].set_xlabel("PVA MSD (A$^2$)  [matrix mobility / flexibility ->]")
ax[0].set_ylabel("Work of adhesion (mJ/m$^2$)  [adhesion ->]")
ax[0].set_title("(a) Pareto front: adhesion vs flexibility")
ax[0].grid(alpha=0.3)

# (b) objectives vs clay loading, with error bars on adhesion
ax2 = ax[1]
ax2.plot(nchain, adh, "o-", c="#2a78d6", label="Adhesion")
# error bars on adhesion for measured n
ns = sorted(eb_n); am = [eb_n[n][0] for n in ns]; asd = [eb_n[n][1] for n in ns]
ax2.errorbar(ns, am, yerr=asd, fmt="o", c="#2a78d6", capsize=3, elinewidth=1.2)
ax2.set_xlabel("Number of PVA chains  (low = high clay loading ->)")
ax2.set_ylabel("Work of adhesion (mJ/m$^2$)", color="#2a78d6")
ax2.tick_params(axis="y", labelcolor="#2a78d6")
ax2.invert_xaxis()
ax2b = ax2.twinx()
ax2b.plot(nchain, msd, "s--", c="#c0392b", label="Mobility (MSD)")
ns2 = sorted(eb_n); mm = [eb_n[n][2] for n in ns2]; ms = [eb_n[n][3] for n in ns2]
ax2b.errorbar(ns2, mm, yerr=ms, fmt="s", c="#c0392b", capsize=3, elinewidth=1.2)
ax2b.set_ylabel("PVA MSD (A$^2$)", color="#c0392b")
ax2b.tick_params(axis="y", labelcolor="#c0392b")
ax2.set_title("(b) Competing objectives vs clay loading")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("L2_bridge_pareto.png", dpi=300)
print("Saved: L2_bridge_pareto.png (with error bars)")
print("\nError bars (3 seeds): n=13: 268.5+/-13.3 | n=25: 314.2+/-17.4 | n=49: 436.3+/-2.4 mJ/m^2")
print("Std shrinks with PVA content: dense systems (n=49) are highly reproducible.")
