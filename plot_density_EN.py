"""
Interface density profile (ENGLISH labels for publication).
Reads dens_clay.txt and dens_pva.txt (LAMMPS ave/chunk output),
saves density_profile.png
Run (mobo, in ~/CLAY/interface): python plot_density_EN.py

NOTE: adjust the column indices (coord_col, dens_col) if your
ave/chunk files have a different layout. Typical LAMMPS ave/chunk:
  col1=chunk_id  col2=coord(z)  col3=Ncount  col4=density/mass
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(fn, coord_col=1, dens_col=3):
    z, d = [], []
    for line in open(fn):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        # skip timestep header lines (2 numbers)
        if len(p) < 4:
            continue
        try:
            z.append(float(p[coord_col])); d.append(float(p[dens_col]))
        except (ValueError, IndexError):
            pass
    return np.array(z), np.array(d)

zc, dc = load("dens_clay.txt")
zp, dp = load("dens_pva.txt")

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.fill_between(zc, dc, color="#c2a878", alpha=0.7, label="Clay (Na-MMT)")
ax.plot(zc, dc, color="#8a7040", lw=1)
ax.fill_between(zp, dp, color="#7fb3e8", alpha=0.6, label="PVA")
ax.plot(zp, dp, color="#2a6ab0", lw=1)
ax.set_xlabel("z axis (\u00c5) \u2014 normal to interface")
ax.set_ylabel("Mass density (g/cm$^3$)")
ax.set_title("Density profile across the interface")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("density_profile.png", dpi=300)
print("Saved: density_profile.png")
