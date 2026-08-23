"""
============================================================================
 qEHVI vs Rastgele  --  GERCEK kil yuku MD verisi uzerinde
============================================================================
 Seviye 2 kopruden gelen 7 gercek MD noktasi (kil yuku -> adhezyon, MSD)
 bir yanit yuzeyi (response surface) tanimlar. qEHVI ve rastgele arama bu
 MD-turevli yuzey uzerinde cok-tohumla karsilastirilir.

 Bu, Fig.4'teki SENTETIK karsilastirmayi GERCEK-MD-tabanli hale getirir:
 optimizasyon, gercek PVA/MMT adhezyon-mobilite trade-off'u uzerinde test edilir.

 Calistir (mobo): python qehvi_vs_random_realMD.py
   -> qehvi_vs_random_realMD.png
============================================================================
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning)
from torch.quasirandom import SobolEngine

tk = {"dtype": torch.double}

# ---- GERCEK MD verisi (Seviye 2 kopru ciktisi) ----
n_real   = np.array([13, 20, 26, 34, 38, 42, 49], dtype=float)
adh_real = np.array([258.0, 275.0, 296.5, 332.0, 375.0, 393.4, 429.7])  # mJ/m^2
msd_real = np.array([17.80, 9.12, 5.10, 3.43, 2.28, 1.57, 0.65])        # A^2

# ---- MD-turevli yanit yuzeyi (duzgun enterpolasyon) ----
# adhezyon: kil yukuyle monoton artar (n azaldikca duser) -> polinom fit
# msd: n azaldikca ustel artar -> log-uzayda fit
padh = np.polyfit(n_real, adh_real, 2)
pmsd = np.polyfit(n_real, np.log(msd_real), 2)

def response(n):
    n = np.clip(n, 13, 49)
    a = np.polyval(padh, n)
    m = np.exp(np.polyval(pmsd, n))
    return a, m

BOUNDS = torch.tensor([[13.0], [49.0]], **tk)
REF = torch.tensor([250.0, 0.0], **tk)   # referans nokta (adh_min, msd_min alti)

def objective(X):
    n = X.squeeze(-1).numpy()
    a, m = response(n)
    return torch.tensor(np.stack([a, m], axis=-1), **tk)

def hv(Y):
    return DominatedPartitioning(ref_point=REF, Y=Y).compute_hypervolume().item()

def run_seed(seed, n_init=4, n_iter=8, method="qehvi"):
    torch.manual_seed(seed)
    sob = SobolEngine(1, scramble=True, seed=seed)
    X = BOUNDS[0] + (BOUNDS[1]-BOUNDS[0]) * sob.draw(n_init).to(**tk)
    Y = objective(X)
    hvs = [hv(Y)]
    for _ in range(n_iter):
        if method == "qehvi":
            model = SingleTaskGP(X, Y, input_transform=Normalize(d=1),
                                 outcome_transform=Standardize(m=2))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=REF.tolist(), X_baseline=X,
                prune_baseline=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])))
            cand, _ = optimize_acqf(acqf, bounds=BOUNDS, q=1,
                                    num_restarts=5, raw_samples=64)
            xn = cand.detach()
        else:
            xn = BOUNDS[0] + (BOUNDS[1]-BOUNDS[0]) * torch.rand(1, 1, **tk)
        X = torch.cat([X, xn]); Y = torch.cat([Y, objective(xn)])
        hvs.append(hv(Y))
    return np.array(hvs)

SEEDS = range(20)
print("qEHVI kosuluyor (20 tohum)...")
q = np.array([run_seed(s, method="qehvi") for s in SEEDS])
print("Rastgele kosuluyor (20 tohum)...")
r = np.array([run_seed(s, method="random") for s in SEEDS])
x = np.arange(q.shape[1]) + 4

fig, ax = plt.subplots(figsize=(7, 4.6))
for data, col, lab, mk, ls in [(q, "#2a78d6", "qEHVI", "o", "-"),
                                (r, "#888888", "Random", "s", "--")]:
    m = data.mean(0); sd = data.std(0)
    ax.plot(x, m, col, marker=mk, label=lab, lw=2, ls=ls)
    ax.fill_between(x, m-sd, m+sd, color=col, alpha=0.2)
ax.set_xlabel("Number of MD evaluations")
ax.set_ylabel("Hypervolume (adhesion x mobility)")
ax.set_title("qEHVI vs Random on the real MD-derived\nclay-loading response surface (20 seeds, mean +/- 1 sigma)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("qehvi_vs_random_realMD.png", dpi=300)
print("Saved: qehvi_vs_random_realMD.png")

# sayisal ozet
final_q, final_r = q[:, -1].mean(), r[:, -1].mean()
print(f"\nSon HV: qEHVI={final_q:.0f}, Random={final_r:.0f} "
      f"(qEHVI %{100*(final_q-final_r)/final_r:.0f} daha yuksek)")
# ne zaman ayrisiyorlar
for i in range(len(x)):
    if q[:, i].mean() - q[:, i].std() > r[:, i].mean() + r[:, i].std():
        print(f"Bantlar {x[i]}. degerlendirmede ayrisiyor.")
        break
