"""
============================================================================
 HATA CUBUKLARI  --  adhezyon + MSD icin ortalama +/- standart sapma
============================================================================
 3 kil yuku sistemi (13, 25, 49 PVA zinciri) x 3 farkli rastgele seed
 = 9 kisa MD kosusu.  Her sistem icin adhezyon (mJ/m^2) ve MSD'nin
 ortalama +/- std'sini verir. Boylece Pareto cephesinin uc/orta
 noktalarina ve ana adhezyon degerine hata cubugu eklenebilir.

 Calistir (mobo, ~/CLAY/interface):
   nohup python error_bars.py > errorbars_out.txt 2>&1 &
   tail -f errorbars_out.txt
 Sure: ~9 x 5 dk = ~45 dk (sistemler kucuk, hizli).
============================================================================
"""
import re
import subprocess
import numpy as np
from pathlib import Path

WORKDIR = Path.home() / "CLAY" / "interface"
BASE = "interface_equil.data"
LAMMPS_CMD = "conda run -n md mpirun -np 10 lmp"

AREA_m2 = 51.918 * 45.077 * 1e-20
KCAL_TO_J = 6.9477e-21

PVA_FIRST_MOL = 85
PVA_LAST_MOL = 134

# hangi kil yukleri (PVA zincir sayisi) + kac seed
N_CHAINS = [13, 25, 49]
SEEDS = [12345, 67890, 24680]     # 3 farkli rastgele seed

TEMPLATE = """units           real
atom_style      full
boundary        p p f
pair_style      lj/charmmfsw/coul/long 10 12
pair_modify     mix arithmetic
bond_style      harmonic
angle_style     charmm
dihedral_style  charmmfsw
improper_style  harmonic
special_bonds   charmm
kspace_style    pppm 1e-6
kspace_modify   slab 3.0
processors      * * 1
read_data       BASEFILE
DELETE_BLOCK
group           clay type 1:16
group           pva  type 17:24
neighbor        2.0 bin
neigh_modify    delay 5 every 1 check yes
velocity        all create 300.0 SEED dist gaussian loop geom
fix             1 all nvt temp 300.0 300.0 100.0
thermo          2000
thermo_style    custom step temp pe
run             10000
reset_timestep  0
compute         msd pva msd
compute         adh clay group/group pva pair yes kspace yes
variable        adhv equal c_adh
fix             avea all ave/time 100 100 10000 v_adhv
thermo_style    custom step temp c_adh c_msd[4]
run             10000
variable        Aout equal f_avea
variable        Mout equal c_msd[4]
print           "RESULT ADH ${Aout} MSD ${Mout}"
"""


def run_one(n_chain, seed):
    n = int(n_chain)
    if n >= 50:
        delete_block = "# tum zincirler"
    else:
        first_del = PVA_FIRST_MOL + n
        delete_block = (f"group           kaldir molecule {first_del}:{PVA_LAST_MOL}\n"
                        f"delete_atoms    group kaldir bond yes mol yes")
    inp = (TEMPLATE.replace("BASEFILE", BASE)
                   .replace("DELETE_BLOCK", delete_block)
                   .replace("SEED", str(seed)))
    fname = WORKDIR / f"in_eb_{n}_{seed}.txt"
    fname.write_text(inp)
    res = subprocess.run(LAMMPS_CMD.split() + ["-in", fname.name],
                         cwd=str(WORKDIR), capture_output=True, text=True,
                         timeout=3600)
    out = res.stdout + res.stderr
    m = re.search(r"RESULT ADH\s+([\d.eE+-]+)\s+MSD\s+([\d.eE+-]+)", out)
    if not m:
        raise RuntimeError(f"okunamadi (n={n}, seed={seed}):\n{out[-1200:]}")
    adh_kcal = abs(float(m.group(1)))
    msd = float(m.group(2))
    adh_mJm2 = adh_kcal * KCAL_TO_J / AREA_m2 * 1e3
    return adh_mJm2, msd


def main():
    assert (WORKDIR / BASE).exists(), f"{BASE} yok"
    print("HATA CUBUKLARI: 3 sistem x 3 seed = 9 MD\n")
    results = {}
    for n in N_CHAINS:
        adhs, msds = [], []
        for seed in SEEDS:
            print(f"  [MD] n={n}, seed={seed} ...", flush=True)
            a, m = run_one(n, seed)
            adhs.append(a); msds.append(m)
            print(f"       adhezyon={a:.1f} mJ/m^2, MSD={m:.2f}", flush=True)
        adhs, msds = np.array(adhs), np.array(msds)
        results[n] = (adhs.mean(), adhs.std(ddof=1), msds.mean(), msds.std(ddof=1))
        print(f"  => n={n}: adhezyon {adhs.mean():.1f} +/- {adhs.std(ddof=1):.1f} mJ/m^2, "
              f"MSD {msds.mean():.2f} +/- {msds.std(ddof=1):.2f}\n", flush=True)

    print("=" * 55)
    print("OZET (ortalama +/- standart sapma, 3 seed):")
    print(" n_chain | adhezyon (mJ/m^2) |  MSD (A^2)")
    for n in N_CHAINS:
        am, asd, mm, msd_ = results[n]
        print(f"   {n:3d}   |  {am:6.1f} +/- {asd:4.1f}  | {mm:5.2f} +/- {msd_:4.2f}")
    print("\nBu degerleri makaleye hata cubugu olarak islenebilir.")


if __name__ == "__main__":
    main()
