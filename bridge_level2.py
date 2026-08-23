"""
============================================================================
 KOPRU - Seviye 2:  qEHVI <-> GERCEK MD   (tasarim degiskeni = KIL YUKU)
============================================================================
 Tasarim degiskeni:  PVA zincir sayisi n_chain in [13, 50]
                      (dusuk n = yuksek kil yuku)
 Amaclar (ayni MD kosusundan, ikisi de MAKSIMIZE):
   1) arayuz adhezyonu |E_adh| (mJ/m^2)   -> kil yuku ile artar (doygunlukla)
   2) PVA matris mobilitesi (MSD, esneklik gostergesi) -> kil yuku ile azalir
 Bunlar CELISIR: yuksek kil yuku -> guclu yapisma + rijit/kirilgan matris;
                 dusuk kil yuku  -> zayif yapisma + esnek matris.
 -> qEHVI bu adhezyon-esneklik trade-off'unu gercek MD ile haritalar.

 run_md(n_chain):
   1) interface_equil.data'dan delete_atoms ile n_chain PVA zinciri birakir
   2) 300 K'de kisa NVT dengeler
   3) adhezyon (mJ/m^2) + PVA MSD olcer
   YENI SISTEM HER CAGRIDA URETILIR (Seviye 1'den farki bu).

 CALISTIR (mobo ortaminda, ~/CLAY/interface klasorunde):
   python bridge_level2.py
   (LAMMPS'i 'conda run -n md' ile cagirir.)
============================================================================
"""

import re
import random
import subprocess
from pathlib import Path

import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

tkwargs = {"dtype": torch.double, "device": "cpu"}

# ------------------------------------------------------------------ #
WORKDIR = Path.home() / "CLAY" / "interface"
BASE = "interface_equil.data"          # 50 zincirli dengelenmis referans
LAMMPS_CMD = "conda run -n md mpirun -np 10 lmp"

# kil alani (mJ/m^2 donusumu icin): 51.918 x 45.077 A^2
AREA_m2 = 51.918 * 45.077 * 1e-20
KCAL_TO_J = 6.9477e-21

BOUNDS = torch.tensor([[13.0], [50.0]], **tkwargs)   # PVA zincir sayisi araligi
N_INIT = 4
N_ROUNDS = 3
BATCH = 1

# PVA molekul ID'leri 85-134 (toplam 50 zincir). N birakmak icin (85+N):134 sil.
PVA_FIRST_MOL = 85
PVA_LAST_MOL = 134

# ------------------------------------------------------------------ #
#  LAMMPS: sistem uret + dengele + olc (tek girdi)                    #
# ------------------------------------------------------------------ #
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
variable        msdv equal c_msd[4]
fix             avea all ave/time 100 100 10000 v_adhv
thermo_style    custom step temp c_adh c_msd[4]
run             10000
variable        Aout equal f_avea
variable        Mout equal c_msd[4]
print           "RESULT NCHAIN ${Aout} ${Mout}"
"""


def run_md(n_chain):
    n = int(round(float(n_chain)))
    n = max(13, min(50, n))            # sinirla
    seed = random.randint(1, 99999)

    # silinecek molekul araligi: (85+n) .. 134   (n=50 ise silme yok)
    if n >= 50:
        delete_block = "# tum zincirler (silme yok)"
    else:
        first_del = PVA_FIRST_MOL + n
        delete_block = (f"group           kaldir molecule {first_del}:{PVA_LAST_MOL}\n"
                        f"delete_atoms    group kaldir bond yes mol yes")

    inp = (TEMPLATE.replace("BASEFILE", BASE)
                   .replace("DELETE_BLOCK", delete_block)
                   .replace("NCHAIN", str(n))
                   .replace("SEED", str(seed)))
    fname = WORKDIR / f"in_L2bridge_{n}.txt"
    fname.write_text(inp)

    print(f"   [MD] n_chain={n} kosuluyor (~5 dk) ...", flush=True)
    res = subprocess.run(LAMMPS_CMD.split() + ["-in", fname.name],
                         cwd=str(WORKDIR), capture_output=True, text=True,
                         timeout=3600)
    out = res.stdout + res.stderr
    m = re.search(r"RESULT\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", out)
    if not m:
        raise RuntimeError(f"MD okunamadi (n={n}). Son:\n{out[-1500:]}")
    adh_kcal, msd = abs(float(m.group(2))), float(m.group(3))
    # adhezyonu mJ/m^2'ye cevir (kil alani sabit -> adil karsilastirma)
    adh_mJm2 = adh_kcal * KCAL_TO_J / AREA_m2 * 1e3
    print(f"   [MD] n_chain={n} -> adhezyon={adh_mJm2:.1f} mJ/m^2, "
          f"MSD={msd:.2f} A^2", flush=True)
    return [adh_mJm2, msd]


# ------------------------------------------------------------------ #
def fit_models(X, Y):
    model = SingleTaskGP(X, Y, input_transform=Normalize(d=1),
                         outcome_transform=Standardize(m=2))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model

def hv(Y):
    ref = Y.min(dim=0).values - 0.1 * Y.std(dim=0)
    return DominatedPartitioning(ref_point=ref, Y=Y).compute_hypervolume().item()

def select_next(model, X, Y):
    ref = (Y.min(dim=0).values - 0.1 * Y.std(dim=0)).tolist()
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=ref, X_baseline=X, prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])))
    cand, _ = optimize_acqf(acqf, bounds=BOUNDS, q=BATCH,
                            num_restarts=10, raw_samples=128)
    return cand.detach()


def main():
    assert (WORKDIR / BASE).exists(), f"{BASE} yok: {WORKDIR}"
    sob = SobolEngine(dimension=1, scramble=True, seed=7)
    X = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * sob.draw(N_INIT).to(**tkwargs)
    print(f"Baslangic: {N_INIT} gercek MD (farkli kil yukleri)")
    Y = torch.stack([torch.tensor(run_md(x), **tkwargs) for x in X])
    print(f"Baslangic HV = {hv(Y):.3f}\n")

    for r in range(1, N_ROUNDS + 1):
        model = fit_models(X, Y)
        Xn = select_next(model, X, Y)
        Yn = torch.stack([torch.tensor(run_md(x), **tkwargs) for x in Xn])
        X = torch.cat([X, Xn]); Y = torch.cat([Y, Yn])
        print(f"Tur {r}: +{BATCH} MD (toplam {X.shape[0]}), HV = {hv(Y):.3f}\n")

    mask = is_non_dominated(Y)
    print("=== PARETO-OPTIMAL KIL YUKLERI ===")
    print(" n_chain | adhezyon(mJ/m^2)  MSD(mobilite)")
    order = torch.argsort(X.squeeze(-1))
    for x, y, nd in zip(X[order], Y[order], mask[order]):
        tag = " *" if nd else "  "
        print(f"   {int(round(float(x[0]))):3d}   |   {float(y[0]):7.1f}       "
              f"{float(y[1]):6.2f}{tag}")
    print("  (* = Pareto-optimal)")


if __name__ == "__main__":
    main()
