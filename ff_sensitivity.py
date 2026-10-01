"""
============================================================================
 KUVVET ALANI YUK HASSASIYETI (FF charge sensitivity)  - Major 3 cevabi
============================================================================
 PVA atom yuklerini (tip 17-24) bir faktorle olcekle (0.9, 1.0, 1.1),
 3 kil yuku sisteminde (13, 25, 49) adhezyonu yeniden olc.
 Amac: Pareto sirasi / dengeli-tasarim bolgesi yuk modeline duyarli mi?

 Nötrluk korunur: PVA molekulu ~nötr, tum PVA yukleri ayni faktorle
 carpilinca k x (nötr) = nötr kalir. (Kil yukleri sabit tutulur.)

 Calistir (mobo, ~/CLAY/interface):
   cp .../ff_sensitivity.py .   # ONCE ls ile dogrula!
   nohup python ff_sensitivity.py > ffsens_out.txt 2>&1 &
   tail -f ffsens_out.txt
 Sure: 3 sistem x 3 olcek = 9 kisa MD x ~4 dk = ~35 dk
============================================================================
"""
import re, subprocess
from pathlib import Path
import numpy as np

WORKDIR = Path.home()/"CLAY"/"interface"
LAMMPS = "conda run -n md mpirun -np 10 lmp"
AREA_m2 = 51.918*45.077*1e-20
KCAL_J = 6.9477e-21

SYSTEMS = [13, 25, 50]
SCALES  = [0.9, 1.0, 1.1]     # PVA yuk olcekleri

TEMPLATE = """units real
atom_style full
boundary p p f
pair_style lj/charmmfsw/coul/long 10 12
pair_modify mix arithmetic
bond_style harmonic
angle_style charmm
dihedral_style charmmfsw
improper_style harmonic
special_bonds charmm
kspace_style pppm 1e-6
kspace_modify slab 3.0
read_data interface_L2_{SYS}.data
group clay type 1:16
group pva type 17:24
# --- PVA yuklerini olcekle (tip 17-24) ---
{SCALE_BLOCK}
neighbor 2.0 bin
neigh_modify delay 5 every 1 check yes
velocity all create 300.0 {SEED} dist gaussian loop geom
fix 1 all nvt temp 300.0 300.0 100.0
run 8000
reset_timestep 0
compute adh clay group/group pva pair yes kspace yes
variable adhv equal c_adh
fix avea all ave/time 100 100 10000 v_adhv
run 10000
variable Aout equal f_avea
print "FFSENS SYS={SYS} SCALE={SCALE} adh = ${{Aout}}"
"""

def scale_block(scale):
    if abs(scale-1.0) < 1e-9:
        return "# yuk olcekleme yok (referans)"
    # her PVA tipinin yukunu scale ile carp: set type T charge (q*scale)
    # ama mevcut yuku bilmemiz gerek -> LAMMPS'te 'set ... charge' mutlak deger ister
    # cozum: once grup yukunu olcekle -> 'set group pva charge/scale' yok, ama
    # 'set type T charge V' mutlak. Bunun yerine her atomun yukunu carpmak icin:
    # LAMMPS 'set' scale destekler mi? -> 'set group pva charge' hayir.
    # Alternatif: variable + set atom. En temizi: her tip icin bilinen yuku carp.
    charges = {17:0.09,18:0.09,19:0.09,20:0.42,21:0.14,22:-0.18,23:-0.27,24:-0.65}
    lines = []
    for t,q in charges.items():
        lines.append(f"set type {t} charge {q*scale:.5f}")
    return "\n".join(lines)

def run_one(sysn, scale, seed):
    inp = TEMPLATE.format(SYS=sysn, SCALE=scale, SEED=seed, SCALE_BLOCK=scale_block(scale))
    f = WORKDIR/f"_ffs_{sysn}_{int(scale*100)}.inp"
    f.write_text(inp)
    r = subprocess.run(LAMMPS.split()+["-in", f.name], cwd=str(WORKDIR),
                       capture_output=True, text=True, timeout=1800)
    m = re.search(r"FFSENS SYS=\d+ SCALE=[\d.]+ adh = ([\d.eE+-]+)", r.stdout+r.stderr)
    if not m:
        raise RuntimeError(f"okunamadi sys={sysn} scale={scale}:\n{(r.stdout+r.stderr)[-900:]}")
    adh_kcal = abs(float(m.group(1)))
    adh_mJm2 = adh_kcal*KCAL_J/AREA_m2*1e3
    return adh_mJm2

def main():
    res = {}   # (sys,scale) -> adh_mJm2
    for sc in SCALES:
        print(f"\n### PVA yuk olcegi = {sc} ###", flush=True)
        for sysn in SYSTEMS:
            a = run_one(sysn, sc, 4321)
            res[(sysn,sc)] = a
            print(f"  sys={sysn:2d} -> adhezyon = {a:.1f} mJ/m^2", flush=True)

    print("\n"+"="*56)
    print("YUK HASSASIYETI OZET (adhezyon, mJ/m^2):")
    print(" olcek |  13 zincir |  25 zincir |  49 zincir | siralama")
    for sc in SCALES:
        a13,a25,a49 = res[(13,sc)],res[(25,sc)],res[(49,sc)]
        order = "13<25<49" if a13<a25<a49 else "BOZULDU!"
        print(f"  {sc:.1f}  |  {a13:7.1f}   |  {a25:7.1f}   |  {a49:7.1f}   | {order}")
    print("\n-> Tum olceklerde siralama 13<25<49 korunuyorsa,")
    print("   Pareto cephesinin sekli yuk modeline duyarli DEGIL.")

if __name__ == "__main__":
    main()
