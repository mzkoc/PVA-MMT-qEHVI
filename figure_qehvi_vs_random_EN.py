"""
qEHVI vs random benchmark figure (ENGLISH labels for publication).
10-seed mean +/- 1 sigma hypervolume bands.
Run (mobo): python figure_qehvi_vs_random_EN.py -> qehvi_vs_random.png

This reproduces the benchmark on a synthetic 2-objective test problem
(same setup used previously). If you have saved the HV arrays, load them
instead of re-running; otherwise this re-runs the 10-seed comparison.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from torch.quasirandom import SobolEngine

tk = {"dtype": torch.double}
BOUNDS = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], **tk)
REF = torch.tensor([0.0, 0.0], **tk)

def objective(X):
    # synthetic 2-objective (conflicting)
    f1 = 1.0 - (X[:, 0] - 0.5)**2 - 0.3*X[:, 1]
    f2 = 1.0 - (X[:, 1] - 0.4)**2 - 0.3*X[:, 0]
    return torch.stack([f1, f2], dim=-1)

def hv(Y):
    return DominatedPartitioning(ref_point=REF, Y=Y).compute_hypervolume().item()

def run_seed(seed, n_init=8, n_iter=15, method="qehvi"):
    torch.manual_seed(seed)
    sob = SobolEngine(3, scramble=True, seed=seed)
    X = BOUNDS[0] + (BOUNDS[1]-BOUNDS[0]) * sob.draw(n_init).to(**tk)
    Y = objective(X)
    hvs = [hv(Y)]
    for _ in range(n_iter):
        if method == "qehvi":
            model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=2))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=REF.tolist(), X_baseline=X, prune_baseline=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])))
            cand, _ = optimize_acqf(acqf, bounds=BOUNDS, q=1, num_restarts=5, raw_samples=64)
            xn = cand.detach()
        else:
            xn = BOUNDS[0] + (BOUNDS[1]-BOUNDS[0]) * torch.rand(1, 3, **tk)
        X = torch.cat([X, xn]); Y = torch.cat([Y, objective(xn)])
        hvs.append(hv(Y))
    return np.array(hvs)

SEEDS = range(10)
q = np.array([run_seed(s, method="qehvi") for s in SEEDS])
r = np.array([run_seed(s, method="random") for s in SEEDS])
x = np.arange(q.shape[1]) + 8

fig, ax = plt.subplots(figsize=(7, 4.6))
for data, col, lab, mk in [(q, "#2a78d6", "qEHVI", "o"), (r, "#888888", "Random", "s")]:
    m = data.mean(0); sd = data.std(0)
    ax.plot(x, m, col, marker=mk, label=lab, lw=2, ls="-" if lab=="qEHVI" else "--")
    ax.fill_between(x, m-sd, m+sd, color=col, alpha=0.2)
ax.set_xlabel("Number of evaluations (MD runs)")
ax.set_ylabel("Hypervolume")
ax.set_title("qEHVI vs Random  (10 seeds, mean \u00b1 1\u03c3)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("qehvi_vs_random.png", dpi=300)
print("Saved: qehvi_vs_random.png")
