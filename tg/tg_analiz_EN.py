"""
Tg analysis (ENGLISH labels for publication).
Reads tg_data.txt (T density), bilinear fit -> Tg, saves tg_profile.png
Run (mobo, in ~/PVA_cgenff/tg): python tg_analiz_EN.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T, D = [], []
for line in open("tg_data.txt"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    p = line.split()
    if len(p) == 2:
        try:
            T.append(float(p[0])); D.append(float(p[1]))
        except ValueError:
            pass
T = np.array(T); D = np.array(D)

# drop first (450 K) step group as before -> cleaner break
nominal = np.arange(425, 249, -25)
Tavg, Davg = [], []
for Tn in nominal:
    mask = np.abs(T - Tn) < 8
    if mask.sum() > 5:
        idx = np.where(mask)[0]
        half = idx[len(idx)//2:]
        Tavg.append(T[half].mean()); Davg.append(D[half].mean())
Tavg = np.array(Tavg); Davg = np.array(Davg)

best = None
for k in range(2, len(Tavg)-2):
    p_lo = np.polyfit(Tavg[:k+1], Davg[:k+1], 1)
    p_hi = np.polyfit(Tavg[k:], Davg[k:], 1)
    res = ((np.polyval(p_lo, Tavg[:k+1]) - Davg[:k+1])**2).sum() \
        + ((np.polyval(p_hi, Tavg[k:]) - Davg[k:])**2).sum()
    if abs(p_lo[0]-p_hi[0]) > 1e-9:
        Tg = (p_hi[1]-p_lo[1])/(p_lo[0]-p_hi[0])
    else:
        continue
    if best is None or res < best[0]:
        best = (res, Tg, p_lo, p_hi, k)
_, Tg, p_lo, p_hi, k = best
print(f"Tg ~ {Tg:.0f} K")

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.scatter(Tavg, Davg, c="#2a78d6", s=45, zorder=3, label="MD data")
tt_lo = np.linspace(Tavg.min(), Tg, 20)
tt_hi = np.linspace(Tg, Tavg.max(), 20)
ax.plot(tt_lo, np.polyval(p_lo, tt_lo), "--", c="#c0392b", lw=1.5, label="glassy fit")
ax.plot(tt_hi, np.polyval(p_hi, tt_hi), "--", c="#27ae60", lw=1.5, label="rubbery fit")
ax.axvline(Tg, color="k", ls=":", lw=1)
ax.annotate(f"Tg ~ {Tg:.0f} K", xy=(Tg, Davg.mean()),
            xytext=(Tg+15, Davg.max()-0.01), fontsize=11)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Density (g/cm$^3$)")
ax.set_title("PVA glass transition temperature (MD)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig("tg_profile.png", dpi=300)
print("Saved: tg_profile.png")
