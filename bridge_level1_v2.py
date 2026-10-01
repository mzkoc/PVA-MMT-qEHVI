"""
============================================================================
 KOPRU - Seviye 1 (v2):  qEHVI <-> GERCEK MD   -- uzun olcum penceresi
============================================================================
 v1'e gore degisiklikler:
   - Sicaklik araligi [300, 400] K  (Tg ~330-350 K'yi CAPRAZLAR ->
     Tg altinda dusuk hareketlilik, ustunde keskin artis -> net trade-off)
   - Dengeleme 10 ps, MSD olcumu 20 ps  (v1'de 1 ps idi -> cok kisaydi)

 Amaclar (ayni MD kosusundan, ikisi de MAKSIMIZE):
   1) |E_adh|  (adhezyon gucu)     2) MSD (PVA hareketliligi)

 SURE UYARISI: her MD ~30000 adim (~15 dk, -np 10). Toplam 6 kosu ~ 1.5 saat.
   Daha hizli istersen N_ROUNDS'u 2 yap (5 kosu) ya da olcumu kisalt.

 CALISTIR (mobo ortaminda, ~/CLAY/interface klasorunde):
   python bridge_level1_v2.py
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
DATA = "interface_equil.data"
LAMMPS_CMD = "conda run -n md mpirun -np 10 lmp"
# Alternatif: "/home/mzahid/miniconda/envs/md/bin/mpirun -np 10 /home/mzahid/miniconda/envs/md/bin/lmp"

BOUNDS = torch.tensor([[300.0], [400.0]], **tkwargs)   # Tg'yi caprazlar
N_INIT = 3
N_ROUNDS = 3
BATCH = 1

# ------------------------------------------------------------------ #
#  uzun pencereli LAMMPS sablonu                                     #
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
read_data       DATAFILE
group           clay type 1:16
group           pva  type 17:24
neighbor        2.0 bin
neigh_modify    delay 5 every 1 check yes
velocity        all create TVAL SEED dist gaussian
fix             1 all nvt temp TVAL TVAL 100.0
run             10000
reset_timestep  0
compute         msd pva msd
compute         adh clay group/group pva pair yes kspace yes
variable        adhv equal c_adh
variable        msdv equal c_msd[4]
fix             avea all ave/time 100 200 20000 v_adhv
fix             avem all ave/time 100 200 20000 v_msdv
thermo          2000
thermo_style    custom step temp c_adh c_msd[4]
run             20000
variable        Aout equal f_avea
variable        Mout equal f_avem
print           "RESULT TVAL ${Aout} ${Mout}"
"""


def run_md(T):
    T = float(T)
    seed = random.randint(1, 99999)
    inp = (TEMPLATE.replace("DATAFILE", DATA)
                   .replace("TVAL", f"{T:.1f}")
                   .replace("SEED", str(seed)))
    fname = WORKDIR / f"in_bridge_{int(round(T))}.txt"
    fname.write_text(inp)
    print(f"   [MD] T={T:.1f} K kosuluyor (~15 dk) ...", flush=True)
    res = subprocess.run(LAMMPS_CMD.split() + ["-in", fname.name],
                         cwd=str(WORKDIR), capture_output=True, text=True,
                         timeout=5400)
    out = res.stdout + res.stderr
    m = re.search(r"RESULT\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", out)
    if not m:
        raise RuntimeError(f"MD okunamadi (T={T}). Son:\n{out[-1500:]}")
    adh, msd = float(m.group(2)), float(m.group(3))
    print(f"   [MD] T={T:.1f} K -> adhezyon={adh:.1f}, MSD={msd:.2f} A^2",
          flush=True)
    return [abs(adh), msd]


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
    assert (WORKDIR / DATA).exists(), f"{DATA} yok: {WORKDIR}"
    sob = SobolEngine(dimension=1, scramble=True, seed=1)
    X = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * sob.draw(N_INIT).to(**tkwargs)
    print(f"Baslangic: {N_INIT} gercek MD")
    Y = torch.stack([torch.tensor(run_md(x), **tkwargs) for x in X])
    print(f"Baslangic HV = {hv(Y):.3f}\n")

    for r in range(1, N_ROUNDS + 1):
        model = fit_models(X, Y)
        Xn = select_next(model, X, Y)
        Yn = torch.stack([torch.tensor(run_md(x), **tkwargs) for x in Xn])
        X = torch.cat([X, Xn]); Y = torch.cat([Y, Yn])
        print(f"Tur {r}: +{BATCH} MD (toplam {X.shape[0]}), HV = {hv(Y):.3f}\n")

    mask = is_non_dominated(Y)
    print("=== PARETO-OPTIMAL SICAKLIKLAR ===")
    print("   T[K]  | adhezyon_gucu   MSD (A^2)")
    order = torch.argsort(X.squeeze(-1))
    for x, y, nd in zip(X[order], Y[order], mask[order]):
        tag = " *" if nd else "  "
        print(f"  {float(x[0]):5.1f}  |  {float(y[0]):10.1f}   {float(y[1]):7.2f}{tag}")
    print("  (* = Pareto-optimal)")


if __name__ == "__main__":
    main()
