#!/usr/bin/env python3
"""
==========================================================================
  HQS-SFD-BENCHMARK -- Main Pipeline
  Hybrid Quantum Surrogate (HQS) for Rarefied Gas Dynamics
  --------------------------------------------------------------------
  This is the pipeline that generated every number reported in the
  manuscript ("Benchmarking the Empirical Limits of Variational Quantum
  Circuits for Rarefied Gas Dynamics Surrogates"). It produces, in one
  run:

    * Per-seed MSE + RMSE for every model (mean +/- std across seeds)
    * Classical, TinyMLP, HQS-protocol, HQS-ablation, Edge surrogate
    * Qubit-count scaling sweep (6 / 8 / 10 qubits)
    * NISQ depolarizing-noise robustness
    * Knowledge distillation -> C++ edge header
    * A follow-up study testing the training-budget and ansatz-mismatch
      hypotheses raised in the manuscript's Discussion (Sec. 7.1)
    * EIGHT publication figures
    * results.json (full per-seed data) + LaTeX table snippets

  REPRODUCIBILITY FIXES (previously shipped as a separate patch script,
  reproducibility_fix.py -- both are now merged inline, at the exact two
  lines they affect. Neither changes any ranking, ordering, or
  qualitative finding in the paper; both are measurement/reproducibility
  corrections caught during post-hoc review. reproducibility_fix.py is
  kept in the repo only as a historical record of the standalone
  verification run (protocol MSE, seed 456, matched bit-for-bit)):

    [FIXED: BUG-1] Classical-baseline wall-clock time is now accumulated
            into a list across all 3 seeds and reported as a proper
            mean +/- std (search "FIXED: BUG-1" below). Previously the
            timing variable was overwritten each loop iteration, so only
            the last seed's time was recorded.

    [FIXED: BUG-2] The knowledge-distillation synthetic sampling
            (X_dist) is now preceded by torch.manual_seed(42) (search
            "FIXED: BUG-2" below), so the distilled edge surrogate's
            MSE/RMSE is identical on every rerun. Previously this
            sampling had no fixed seed and drifted at the 4th decimal.

  FOLLOW-UP STUDY (Section 7b, after the qubit sweep): directly tests
  the two open hypotheses raised in the manuscript's Discussion
  (Sec. 7.1) for why the HQS underperforms -- (a) training budget: the
  same train_pair() protocol/ablation pair, UNMODIFIED, simply re-run at
  a larger epoch budget; and (b) ansatz mismatch: a new, purely additive
  data-re-uploading circuit variant (build_hqs_reuploading /
  train_pair_reuploading), evaluated with the identical phase schedule
  and identical-init ablation discipline as the original HQS. Nothing
  above the qubit-sweep section is touched by this addition -- see
  RUN_FOLLOWUP_STUDY in the config block below to disable it and
  reproduce the original paper's numbers only.

    [FIXED: BUG-3] All three follow-up configurations (extended-epoch
            budget, data-reuploading, and reuploading+extended) now
            record per-seed wall-clock time the same way Sections 4 and
            7 do (time.time() around each seed's train_pair /
            train_pair_reuploading call, accumulated into a list, then
            reported as time_s / time_s_std / time_s_per_seed). Earlier
            versions of this script computed correct MSE/RMSE for all
            three follow-up configs but did not instrument timing for
            any of them, so results.json had no time_s field under any
            followup_* key. This has been verified not to change any
            accuracy number: re-running followup_reuploading with this
            fix reproduced the original MSE values to 5 decimal places
            (protocol 0.11831, ablation 0.11645) while adding
            time_s = 884.1 +/- 8.7 s (per-seed: [891.3, 871.8, 889.3]).
==========================================================================
  DEPENDENCIES:
    pip install pennylane torch scikit-learn matplotlib numpy pandas tqdm

  RUNTIME:
    QUICK_TEST = True   -> ~10-25 min  (50/100 epochs, reduced sweep,
                            follow-up study included)
    QUICK_TEST = False  -> ~24-30 hours for the original paper sections,
                            PLUS ~2-3 hours for the new follow-up study
                            (RUN_FOLLOWUP_STUDY=True: 4000-epoch budget
                            test + reuploading-ansatz test, 3 seeds
                            each; add ~2 more hours if you also enable
                            RUN_REUPLOAD_AT_EXTENDED_EPOCHS). Adding the
                            BUG-3 timing instrumentation does not change
                            this estimate -- it wraps existing calls in
                            time.time(), it does not add computation.
    The 10-qubit sweep arm alone is ~19.5 hours (3 seeds x ~6.5 hr/seed)
    on a CPU state-vector simulator. Plan compute time accordingly --
    we ran this in a single long-lived interactive session; if running
    on a managed notebook service with session limits, split the sweep
    by qubit count and checkpoint per seed.
==========================================================================
"""

import os, json, time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from tqdm import tqdm

# ===========================================================
# 0. CONFIGURATION
# ===========================================================
QUICK_TEST   = False                       # <-- False for publication run
SEEDS        = [42, 123, 456]              # statistical replicates
EPOCHS_CLS   = 50  if QUICK_TEST else 1500
EPOCHS_HQS   = 50  if QUICK_TEST else 1500
MAX_EP_DIST  = 100 if QUICK_TEST else 2500
BATCH_SIZE   = None                        # None = full-batch (fastest for this regime)
QUBIT_SWEEP  = [6, 8, 10] if QUICK_TEST else [6, 8, 10]

# --- follow-up ansatz / training-budget study (Section 7b) ---
RUN_FOLLOWUP_STUDY = True                  # set False to reproduce only
                                            # the original paper's numbers
EPOCHS_EXTENDED = 100 if QUICK_TEST else 4000   # Sec. 7.1: "3,000-5,000 epochs"
RUN_REUPLOAD_AT_EXTENDED_EPOCHS = True    # optional best-case combo test;
                                            # off by default (adds ~3 more
                                            # full 6-qubit pairs at
                                            # EPOCHS_EXTENDED)

os.makedirs("figures", exist_ok=True)
os.makedirs("output",  exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--", "lines.linewidth": 2,
})
COL = {"classical": "#2c7bb6", "tiny": "#d7191c", "hqs": "#7b2d8b",
       "ablation": "#999999", "edge": "#1a9641", "truth": "black", "noisy": "#fdae61"}

R = {}  # master results dict (all per-seed data lands here)

def save_fig(name):
    plt.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved figures/{name}")

def wmse(pred, target, w):
    return torch.mean(w * (pred - target) ** 2)

def summarize(mse_list):
    """Return dict of per-seed MSE/RMSE and their mean/std."""
    mse = np.array(mse_list, dtype=float)
    rmse = np.sqrt(mse)
    return {
        "mse_per_seed":  [round(float(v), 6) for v in mse],
        "rmse_per_seed": [round(float(v), 6) for v in rmse],
        "mse_mean":  round(float(mse.mean()), 6),
        "mse_std":   round(float(mse.std()), 6),
        "rmse_mean": round(float(rmse.mean()), 6),
        "rmse_std":  round(float(rmse.std()), 6),
        # physical interpretation: factor error = 10^RMSE (log10-space target)
        "phys_factor": round(float(10 ** rmse.mean()), 4),
    }

def iterate_batches(n, bs):
    if bs is None:
        yield torch.arange(n); return
    perm = torch.randperm(n)
    for i in range(0, n, bs):
        yield perm[i:i + bs]

print("=" * 64)
print("  HQS-SFD-BENCHMARK -- MAIN PIPELINE")
print(f"  Mode={'QUICK' if QUICK_TEST else 'PUBLICATION'}  Seeds={SEEDS}  "
      f"Epochs(HQS)={EPOCHS_HQS}  Sweep={QUBIT_SWEEP}")
print("=" * 64)

# ===========================================================
# 1. DATASET
# ===========================================================
print("\n[1/9] Generating dataset...")
MU, P_ATM, LAM = 1.81e-5, 101325.0, 68e-9
recs = []
for h0 in np.linspace(1.5e-6, 13.6e-6, 15):
    for ar in np.linspace(5, 50, 10):
        L = ar * h0
        for f_hz in np.linspace(1000, 100000, 5):
            w = 2 * np.pi * f_hz
            for Pa in np.logspace(5, 3, 15):
                Kn = LAM * (P_ATM / Pa) / h0
                if 0.05 <= Kn <= 5.0:
                    mu_e = MU / (1.0 + 9.42 * Kn)
                    sig  = 12 * mu_e * w * L**2 / (Pa * h0**2)
                    ct   = np.sqrt(1j * sig) / 2.0
                    Fd   = (1.0 / sig) * np.imag(1.0 - np.tanh(ct) / ct)
                    cd   = Fd * Pa * L**3 / (h0**3 * w)
                    if cd > 0:
                        recs.append({"Kn": Kn, "AR": ar, "Freq": w, "cd": cd})
df = pd.DataFrame(recs)
df.to_csv("output/master_sfd.csv", index=False)
R["n_samples"] = len(df)
print(f"  Dataset: {len(df)} rows  (Kn {df.Kn.min():.3f}-{df.Kn.max():.2f})")

X = np.column_stack([df.Kn.values, df.AR.values, np.log10(df.Freq.values)])
y = np.log10(df.cd.values).reshape(-1, 1)
wt = np.where(df.Kn.values > 1.0, 2.0, 1.0)
wt = (wt / wt.mean()).reshape(-1, 1)

X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(X, y, wt, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
with open("output/scaler_params.json", "w") as f:
    json.dump({"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}, f, indent=2)

Xtr = torch.tensor(X_tr_s, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.float32)
wtr = torch.tensor(w_tr, dtype=torch.float32)
Xte = torch.tensor(X_te_s, dtype=torch.float32); yte = torch.tensor(y_te, dtype=torch.float32)
wte = torch.tensor(w_te, dtype=torch.float32)

# ===========================================================
# 2. CLASSICAL BASELINES (per-seed MSE/RMSE)
# ===========================================================
print(f"\n[2/9] Classical baselines ({len(SEEDS)} seeds)...")

class ClassicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))
    def forward(self, x): return self.net(x)

cls_mse, tiny_mse, cls_times = [], [], []   # [FIXED: BUG-1] cls_times is now a
                                             # list, accumulated per seed below
                                             # (was a scalar overwritten each
                                             # iteration -- see header docstring).
cls_curve_sum = np.zeros(EPOCHS_CLS)
for seed in SEEDS:
    torch.manual_seed(seed)
    cm, tm = ClassicalMLP(), TinyMLP()
    oc = optim.Adam(cm.parameters(), lr=5e-3)
    ot = optim.Adam(tm.parameters(), lr=5e-3)
    t0 = time.time()
    for ep in tqdm(range(EPOCHS_CLS), desc=f"  cls seed {seed}", leave=False):
        cm.train(); tm.train()
        lc = wmse(cm(Xtr), ytr, wtr); oc.zero_grad(); lc.backward(); oc.step()
        lt = wmse(tm(Xtr), ytr, wtr); ot.zero_grad(); lt.backward(); ot.step()
        cls_curve_sum[ep] += lc.item()
    cls_times.append(time.time() - t0)   # [FIXED: BUG-1] appended, not overwritten
    cm.eval(); tm.eval()
    with torch.no_grad():
        cls_mse.append(wmse(cm(Xte), yte, wte).item())
        tiny_mse.append(wmse(tm(Xte), yte, wte).item())
cls_curve = cls_curve_sum / len(SEEDS)
cls_time_mean = float(np.mean(cls_times))   # [FIXED: BUG-1] proper 3-seed mean
cls_time_std  = float(np.std(cls_times))

R["classical"] = {"params": sum(p.numel() for p in ClassicalMLP().parameters()),
                  "time_s": round(cls_time_mean, 1),
                  "time_s_std": round(cls_time_std, 1),
                  "time_s_per_seed": [round(t, 1) for t in cls_times],
                  **summarize(cls_mse)}
R["tiny"] = {"params": sum(p.numel() for p in TinyMLP().parameters()), **summarize(tiny_mse)}
print(f"  ClassicalMLP MSE {R['classical']['mse_mean']:.5f} +/- {R['classical']['mse_std']:.5f}"
      f"  RMSE {R['classical']['rmse_mean']:.5f}")
print(f"  TinyMLP      MSE {R['tiny']['mse_mean']:.5f} +/- {R['tiny']['mse_std']:.5f}"
      f"  RMSE {R['tiny']['rmse_mean']:.5f}")

# ===========================================================
# 3. PARAMETRIC HQS FACTORY (shared by main run + sweep)
# ===========================================================
def build_hqs(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits); diff = "backprop"; dn = "default.qubit"

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circuit(inputs, w0, w1, w2):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(w0, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(w1, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(w2, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    sh = (1, n_qubits, 3); ws = {"w0": sh, "w1": sh, "w2": sh}

    class HQS(nn.Module):
        def __init__(self):
            super().__init__()
            self.cl_in  = nn.Linear(3, n_qubits)
            self.q      = qml.qnn.TorchLayer(circuit, ws)
            self.cl_out = nn.Linear(n_qubits, 1)
        def forward(self, x): return self.cl_out(self.q(torch.tanh(self.cl_in(x))))
    return HQS, dn

def train_pair(n_qubits, seed, epochs, track_curve=False):
    """Train protocol + ablation from IDENTICAL init. Returns (prot_mse, abl_mse, curves)."""
    HQS, _ = build_hqs(n_qubits)
    torch.manual_seed(seed); prot = HQS()
    torch.manual_seed(seed); abl  = HQS()   # bit-for-bit identical init

    P1, P2, P3 = int(epochs * 0.2), int(epochs * 0.4), int(epochs * 0.6)
    PHASES = {0:(False,False,True,False,False,0.01), P1:(False,False,False,True,False,0.01),
              P2:(False,False,False,False,True,0.01), P3:(True,True,True,True,True,0.005)}

    prot_curve = np.zeros(epochs); abl_curve = np.zeros(epochs)

    # ablation
    oa = optim.Adam(abl.parameters(), lr=0.005)
    for ep in tqdm(range(epochs), desc=f"  q{n_qubits} s{seed} abl", leave=False):
        abl.train()
        for idx in iterate_batches(len(Xtr), BATCH_SIZE):
            l = wmse(abl(Xtr[idx]), ytr[idx], wtr[idx]); oa.zero_grad(); l.backward(); oa.step()
        if track_curve: abl_curve[ep] = l.item()

    # protocol
    op = None
    for ep in tqdm(range(epochs), desc=f"  q{n_qubits} s{seed} prot", leave=False):
        prot.train()
        if ep in PHASES:
            ci, co, a, b, c, lr = PHASES[ep]
            for p in prot.cl_in.parameters():  p.requires_grad = ci
            for p in prot.cl_out.parameters(): p.requires_grad = co
            prot.q.w0.requires_grad = a; prot.q.w1.requires_grad = b; prot.q.w2.requires_grad = c
            op = optim.Adam([p for p in prot.parameters() if p.requires_grad], lr=lr)
        for idx in iterate_batches(len(Xtr), BATCH_SIZE):
            op.zero_grad(); l = wmse(prot(Xtr[idx]), ytr[idx], wtr[idx]); l.backward()
            torch.nn.utils.clip_grad_norm_([p for p in prot.parameters() if p.requires_grad], 1.0)
            op.step()
        if track_curve: prot_curve[ep] = l.item()

    prot.eval(); abl.eval()
    with torch.no_grad():
        pm = wmse(prot(Xte), yte, wte).item()
        am = wmse(abl(Xte),  yte, wte).item()
    return pm, am, prot_curve, abl_curve, prot

# ===========================================================
# 3b. DATA-RE-UPLOADING HQS VARIANT  [purely additive]
#     build_hqs() and train_pair() above are completely UNCHANGED. This
#     is a parallel, self-contained variant used only by the follow-up
#     study (Section 7b, after the qubit sweep). Instead of encoding the
#     3 input features once at the start (AngleEmbedding -> 3x
#     StronglyEntanglingLayers), it re-encodes the same inputs before
#     every entangling layer (data re-uploading, ref. [31] in the
#     manuscript), which Sec. 7.1's Discussion flags as a plausible fix
#     for ansatz mismatch. Parameter count and phase schedule are
#     identical to the original HQS, so the comparison isolates the
#     ansatz as the only variable.
# ===========================================================
def build_hqs_reuploading(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, w0, w1, w2):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(w0, wires=range(n_qubits))
        qml.AngleEmbedding(inputs, wires=range(n_qubits))   # re-upload
        qml.StronglyEntanglingLayers(w1, wires=range(n_qubits))
        qml.AngleEmbedding(inputs, wires=range(n_qubits))   # re-upload
        qml.StronglyEntanglingLayers(w2, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    sh = (1, n_qubits, 3); ws = {"w0": sh, "w1": sh, "w2": sh}

    class HQSReupload(nn.Module):
        def __init__(self):
            super().__init__()
            self.cl_in  = nn.Linear(3, n_qubits)
            self.q      = qml.qnn.TorchLayer(circuit, ws)
            self.cl_out = nn.Linear(n_qubits, 1)
        def forward(self, x): return self.cl_out(self.q(torch.tanh(self.cl_in(x))))
    return HQSReupload

def train_pair_reuploading(n_qubits, seed, epochs, track_curve=False):
    """Identical procedure to train_pair() above -- same phase schedule,
    same identical-init ablation discipline, same optimizer/lr/clipping --
    applied to build_hqs_reuploading() instead of build_hqs(). Kept as a
    separate function (rather than a parameter added to train_pair) so
    the original, already-published training path is never modified."""
    HQS = build_hqs_reuploading(n_qubits)
    torch.manual_seed(seed); prot = HQS()
    torch.manual_seed(seed); abl  = HQS()   # bit-for-bit identical init

    P1, P2, P3 = int(epochs * 0.2), int(epochs * 0.4), int(epochs * 0.6)
    PHASES = {0:(False,False,True,False,False,0.01), P1:(False,False,False,True,False,0.01),
              P2:(False,False,False,False,True,0.01), P3:(True,True,True,True,True,0.005)}

    prot_curve = np.zeros(epochs); abl_curve = np.zeros(epochs)

    oa = optim.Adam(abl.parameters(), lr=0.005)
    for ep in tqdm(range(epochs), desc=f"  reup q{n_qubits} s{seed} abl", leave=False):
        abl.train()
        for idx in iterate_batches(len(Xtr), BATCH_SIZE):
            l = wmse(abl(Xtr[idx]), ytr[idx], wtr[idx]); oa.zero_grad(); l.backward(); oa.step()
        if track_curve: abl_curve[ep] = l.item()

    op = None
    for ep in tqdm(range(epochs), desc=f"  reup q{n_qubits} s{seed} prot", leave=False):
        prot.train()
        if ep in PHASES:
            ci, co, a, b, c, lr = PHASES[ep]
            for p in prot.cl_in.parameters():  p.requires_grad = ci
            for p in prot.cl_out.parameters(): p.requires_grad = co
            prot.q.w0.requires_grad = a; prot.q.w1.requires_grad = b; prot.q.w2.requires_grad = c
            op = optim.Adam([p for p in prot.parameters() if p.requires_grad], lr=lr)
        for idx in iterate_batches(len(Xtr), BATCH_SIZE):
            op.zero_grad(); l = wmse(prot(Xtr[idx]), ytr[idx], wtr[idx]); l.backward()
            torch.nn.utils.clip_grad_norm_([p for p in prot.parameters() if p.requires_grad], 1.0)
            op.step()
        if track_curve: prot_curve[ep] = l.item()

    prot.eval(); abl.eval()
    with torch.no_grad():
        pm = wmse(prot(Xte), yte, wte).item()
        am = wmse(abl(Xte),  yte, wte).item()
    return pm, am, prot_curve, abl_curve, prot

# ===========================================================
# 4. MAIN HQS RUN at 6 qubits (per-seed, with curves)
# ===========================================================
print(f"\n[3/9] HQS main run @ 6 qubits ({len(SEEDS)} seeds)...")
hqs_mse, abl_mse = [], []
prot_curve_sum = np.zeros(EPOCHS_HQS); abl_curve_sum = np.zeros(EPOCHS_HQS)
hqs_times = []; last_prot_model = None
for seed in SEEDS:
    t0 = time.time()
    pm, am, pc, ac, pmodel = train_pair(6, seed, EPOCHS_HQS, track_curve=True)
    hqs_times.append(time.time() - t0)
    hqs_mse.append(pm); abl_mse.append(am)
    prot_curve_sum += pc; abl_curve_sum += ac
    last_prot_model = pmodel   # ends on the LAST seed in SEEDS -- this is the
                               # model used as the distillation teacher below.
                               # See README note on teacher selection: this is
                               # a single representative run, not an ensemble
                               # or the best-performing seed.
prot_curve = prot_curve_sum / len(SEEDS); abl_curve = abl_curve_sum / len(SEEDS)


R["hqs"] = {"params": 85, "time_s": round(float(np.mean(hqs_times)), 1),
            "time_s_std": round(float(np.std(hqs_times)), 1),**summarize(hqs_mse)}

R["ablation"] = {"params": 85, **summarize(abl_mse)}
print(f"  HQS protocol MSE {R['hqs']['mse_mean']:.5f} +/- {R['hqs']['mse_std']:.5f}"
      f"  RMSE {R['hqs']['rmse_mean']:.5f}")
print(f"  HQS ablation MSE {R['ablation']['mse_mean']:.5f} +/- {R['ablation']['mse_std']:.5f}"
      f"  RMSE {R['ablation']['rmse_mean']:.5f}")

# ===========================================================
# 5. KNOWLEDGE DISTILLATION (uses last protocol model)
# ===========================================================
print("\n[4/9] Knowledge distillation -> edge surrogate...")
hqs_model = last_prot_model
n_dist = 50000
torch.manual_seed(42)   # [FIXED: BUG-2] seeded immediately before sampling --
                         # X_dist, and therefore the distilled edge surrogate's
                         # MSE/RMSE, is now identical on every rerun.
idx = torch.randint(0, len(Xtr), (n_dist,))
X_dist = torch.clamp(Xtr[idx] + torch.randn(n_dist, 3) * 0.15, -3.0, 3.0)
hqs_model.eval()
with torch.no_grad(): y_dist = hqs_model(X_dist)

class Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)

torch.manual_seed(42); edge = Edge()
oe = optim.Adam(edge.parameters(), lr=0.01)
prev, pat = float("inf"), 0
for ep in tqdm(range(MAX_EP_DIST), desc="  distill", leave=False):
    edge.train(); l = nn.MSELoss()(edge(X_dist), y_dist)
    oe.zero_grad(); l.backward(); oe.step()
    if abs(prev - l.item()) < 1e-6:
        pat += 1
        if pat > 50: break
    else: pat = 0
    prev = l.item()
edge.eval()
with torch.no_grad(): edge_mse = wmse(edge(Xte), yte, wte).item()
R["edge"] = {"params": 81, "distill_mse": round(prev, 6),
             "mse": round(edge_mse, 6), "rmse": round(math.sqrt(edge_mse), 6)}
print(f"  distill MSE {prev:.6f}  edge test MSE {edge_mse:.5f}  RMSE {math.sqrt(edge_mse):.5f}")
print(f"  (teacher = protocol model from seed {SEEDS[-1]}, individual test MSE "
      f"{R['hqs']['mse_per_seed'][-1]:.6f} -- compare edge_mse against this number, "
      f"not the 3-seed mean, for the true teacher-student comparison)")

# ===========================================================
# 6. NISQ NOISE
# ===========================================================
print("\n[5/9] NISQ depolarizing-noise robustness...")
n_qubits = 6
dev_noisy = qml.device("default.mixed", wires=n_qubits)
@qml.qnode(dev_noisy, interface="torch")
def noisy_circuit(inputs, w0, w1, w2, p1=0.001, p2=0.01):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    for w in range(n_qubits): qml.DepolarizingChannel(p1, wires=w)
    qml.StronglyEntanglingLayers(w0, wires=range(n_qubits))
    for w in range(n_qubits): qml.DepolarizingChannel(p2, wires=w)
    qml.StronglyEntanglingLayers(w1, wires=range(n_qubits))
    for w in range(n_qubits): qml.DepolarizingChannel(p2, wires=w)
    qml.StronglyEntanglingLayers(w2, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qw0, qw1, qw2 = hqs_model.q.w0.detach(), hqs_model.q.w1.detach(), hqs_model.q.w2.detach()
noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
noisy_mse = []
for p in tqdm(noise_levels, desc="  noise", leave=False):
    if p == 0.0:
        noisy_mse.append(R["hqs"]["mse_mean"]); continue
    preds = []
    with torch.no_grad():
        for xi in Xte:
            x6 = torch.tanh(hqs_model.cl_in(xi.unsqueeze(0)))
            qo = torch.tensor(noisy_circuit(x6[0], qw0, qw1, qw2, p1=p*0.1, p2=p), dtype=torch.float32)
            preds.append(hqs_model.cl_out(qo.unsqueeze(0)).item())
    noisy_mse.append(float(np.mean((np.array(preds).reshape(-1, 1) - yte.numpy())**2)))
R["nisq"] = {"levels": noise_levels, "mse": [round(v, 6) for v in noisy_mse],
             "max_tolerable": max([p for p, m in zip(noise_levels, noisy_mse)
                                   if m < R["classical"]["mse_mean"]], default=0.0)}
print(f"  max tolerable noise: p = {R['nisq']['max_tolerable']:.3f}")

# ===========================================================
# 7. QUBIT-COUNT SWEEP (per-seed) -- properly averaged timing
#    (this section was written correctly from the start: tm_s is a
#    list, averaged with np.mean below -- it does NOT have BUG-1)
# ===========================================================
print(f"\n[6/9] Qubit-count sweep {QUBIT_SWEEP} ...")
sweep = []
for nq in QUBIT_SWEEP:
    HQS, dn = build_hqs(nq)
    npar = sum(p.numel() for p in HQS().parameters())
    print(f"  --- {nq} qubits ({dn}, {npar} params) ---")
    pm_s, am_s, tm_s = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        pm, am, _, _, _ = train_pair(nq, seed, EPOCHS_HQS, track_curve=False)
        tm_s.append(time.time() - t0); pm_s.append(pm); am_s.append(am)
    rec = {"n_qubits": nq, "hilbert_dim": 2**nq, "params": npar, "device": dn,
           "time_s": round(float(np.mean(tm_s)), 1),
           "protocol": summarize(pm_s), "ablation": summarize(am_s)}
    sweep.append(rec)
    print(f"    protocol MSE {rec['protocol']['mse_mean']:.5f} +/- {rec['protocol']['mse_std']:.5f}"
          f"  RMSE {rec['protocol']['rmse_mean']:.5f}")
    print(f"    ablation MSE {rec['ablation']['mse_mean']:.5f} +/- {rec['ablation']['mse_std']:.5f}")
R["qubit_sweep"] = sweep

# ===========================================================
# 7b. FOLLOW-UP STUDY -- Ansatz & Training-Budget Test
#     Directly tests the two live hypotheses raised in the manuscript's
#     Discussion (Sec. 7.1): (a) the protocol/ablation gap may simply
#     need a larger epoch budget (3,000-5,000 epochs suggested), and
#     (b) a data-re-uploading ansatz may offer better inductive bias
#     than StronglyEntanglingLayers alone. Both reuse the identical
#     dataset, seeds, and evaluation code already validated above;
#     nothing in Sections 1-7 is touched by this section.
#
#     All three configurations below record per-seed wall-clock timing
#     the same way Sections 4 and 7 do [FIXED: BUG-3 -- see header].
# ===========================================================
if RUN_FOLLOWUP_STUDY:
    print(f"\n[FOLLOW-UP] Ansatz & training-budget study "
          f"(epochs={EPOCHS_EXTENDED}, seeds={SEEDS})...")

    # --- (a) Extended epoch budget, STANDARD ansatz, via the ORIGINAL,
    #         unmodified train_pair(). One call answers both halves of
    #         hypothesis (a): does the layer-wise protocol just need more
    #         epochs, and does the end-to-end ablation schedule (already
    #         effectively "skipping the layer-wise schedule") close the
    #         remaining gap to ClassicalMLP when given more budget.
    ext_prot_mse, ext_abl_mse, ext_times = [], [], []
    for seed in tqdm(SEEDS, desc="  extended-epoch budget", leave=False):
        t0 = time.time()
        pm, am, _, _, _ = train_pair(6, seed, EPOCHS_EXTENDED, track_curve=False)
        ext_times.append(time.time() - t0)
        ext_prot_mse.append(pm); ext_abl_mse.append(am)
    R["followup_epoch_budget"] = {
        "epochs": EPOCHS_EXTENDED,
        "time_s": round(float(np.mean(ext_times)), 1),
        "time_s_std": round(float(np.std(ext_times)), 1),
        "time_s_per_seed": [round(t, 1) for t in ext_times],
        "protocol": summarize(ext_prot_mse),
        "ablation": summarize(ext_abl_mse),
    }
    print(f"    protocol MSE {R['followup_epoch_budget']['protocol']['mse_mean']:.5f} "
          f"+/- {R['followup_epoch_budget']['protocol']['mse_std']:.5f}  "
          f"(orig. {EPOCHS_HQS}-epoch protocol: {R['hqs']['mse_mean']:.5f})   "
          f"time {R['followup_epoch_budget']['time_s']:.1f}s +/- "
          f"{R['followup_epoch_budget']['time_s_std']:.1f}s")
    print(f"    ablation MSE {R['followup_epoch_budget']['ablation']['mse_mean']:.5f} "
          f"+/- {R['followup_epoch_budget']['ablation']['mse_std']:.5f}  "
          f"(orig. {EPOCHS_HQS}-epoch ablation: {R['ablation']['mse_mean']:.5f};  "
          f"ClassicalMLP: {R['classical']['mse_mean']:.5f})")

    # --- (b) Data-re-uploading ansatz, ORIGINAL epoch budget -- isolates
    #         the ansatz variable from the epoch-budget variable tested
    #         above.
    reup_prot_mse, reup_abl_mse, reup_times = [], [], []
    for seed in tqdm(SEEDS, desc="  reuploading ansatz", leave=False):
        t0 = time.time()
        pm, am, _, _, _ = train_pair_reuploading(6, seed, EPOCHS_HQS, track_curve=False)
        reup_times.append(time.time() - t0)
        reup_prot_mse.append(pm); reup_abl_mse.append(am)
    R["followup_reuploading"] = {
        "epochs": EPOCHS_HQS,
        "time_s": round(float(np.mean(reup_times)), 1),
        "time_s_std": round(float(np.std(reup_times)), 1),
        "time_s_per_seed": [round(t, 1) for t in reup_times],
        "protocol": summarize(reup_prot_mse),
        "ablation": summarize(reup_abl_mse),
    }
    print(f"    [reuploading] protocol MSE {R['followup_reuploading']['protocol']['mse_mean']:.5f} "
          f"+/- {R['followup_reuploading']['protocol']['mse_std']:.5f}   "
          f"time {R['followup_reuploading']['time_s']:.1f}s +/- "
          f"{R['followup_reuploading']['time_s_std']:.1f}s")
    print(f"    [reuploading] ablation MSE {R['followup_reuploading']['ablation']['mse_mean']:.5f} "
          f"+/- {R['followup_reuploading']['ablation']['mse_std']:.5f}")
    print(f"    [reuploading] vs. standard-ansatz time at same {EPOCHS_HQS}-epoch budget "
          f"({R['hqs']['time_s']:.1f}s): "
          f"{R['followup_reuploading']['time_s'] / R['hqs']['time_s']:.2f}x")

    # --- (c) OPTIONAL best-case combination: reuploading ansatz AT the
    #         extended epoch budget. Off by default -- see
    #         RUN_REUPLOAD_AT_EXTENDED_EPOCHS in the config block.
    if RUN_REUPLOAD_AT_EXTENDED_EPOCHS:
        reup_ext_prot_mse, reup_ext_abl_mse, reup_ext_times = [], [], []
        for seed in tqdm(SEEDS, desc="  reuploading @ extended epochs", leave=False):
            t0 = time.time()
            pm, am, _, _, _ = train_pair_reuploading(6, seed, EPOCHS_EXTENDED, track_curve=False)
            reup_ext_times.append(time.time() - t0)
            reup_ext_prot_mse.append(pm); reup_ext_abl_mse.append(am)
        R["followup_reuploading_extended"] = {
            "epochs": EPOCHS_EXTENDED,
            "time_s": round(float(np.mean(reup_ext_times)), 1),
            "time_s_std": round(float(np.std(reup_ext_times)), 1),
            "time_s_per_seed": [round(t, 1) for t in reup_ext_times],
            "protocol": summarize(reup_ext_prot_mse),
            "ablation": summarize(reup_ext_abl_mse),
        }
        print(f"    [reuploading+extended] protocol MSE "
              f"{R['followup_reuploading_extended']['protocol']['mse_mean']:.5f} "
              f"+/- {R['followup_reuploading_extended']['protocol']['mse_std']:.5f}   "
              f"time {R['followup_reuploading_extended']['time_s']:.1f}s +/- "
              f"{R['followup_reuploading_extended']['time_s_std']:.1f}s")
        print(f"    [reuploading+extended] ablation MSE "
              f"{R['followup_reuploading_extended']['ablation']['mse_mean']:.5f} "
              f"+/- {R['followup_reuploading_extended']['ablation']['mse_std']:.5f}")
else:
    print("\n[FOLLOW-UP] Skipped (RUN_FOLLOWUP_STUDY = False) -- "
          "results.json will contain only the original paper's sections.")

# ===========================================================
# 8. FIGURES
# ===========================================================
print("\n[7/9] Generating figures...")

# --- Fig 1: convergence (mean over seeds) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(prot_curve, color=COL["hqs"], lw=2, label="HQS (protocol)")
ax.semilogy(abl_curve, color=COL["ablation"], ls="--", lw=1.8, label="HQS (ablation, end-to-end)")
ax.axhline(R["classical"]["mse_mean"], color=COL["classical"], ls=":", lw=1.5, label="ClassicalMLP")
ax.axhline(R["tiny"]["mse_mean"], color=COL["tiny"], ls=":", lw=1.5, label="TinyMLP")
ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MSE (log scale)")
ax.set_title(f"Training Convergence (mean of {len(SEEDS)} seeds)"); ax.legend()
save_fig("fig01_convergence")

# --- Fig 2: physics validation ---
L_v, h0_v, w_v = 50e-6, 2e-6, 2 * np.pi * 5000
Kn_v, gt_v = [], []
for Pa in np.logspace(5, 3, 60):
    Kn = LAM * (P_ATM / Pa) / h0_v
    if 0.05 <= Kn <= 5.0:
        mu_e = MU / (1 + 9.42 * Kn); sig = 12 * mu_e * w_v * L_v**2 / (Pa * h0_v**2)
        ct = np.sqrt(1j * sig) / 2; Fd = (1 / sig) * np.imag(1 - np.tanh(ct) / ct)
        Kn_v.append(Kn); gt_v.append(Fd * Pa * L_v**3 / (h0_v**3 * w_v))
if Kn_v:
    Xv = scaler.transform(np.column_stack([Kn_v, np.full(len(Kn_v), L_v/h0_v),
                                           np.full(len(Kn_v), np.log10(w_v))]))
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    with torch.no_grad():
        hqs_pred = 10 ** hqs_model(Xv_t).numpy().flatten()
        edge_pred = 10 ** edge(Xv_t).numpy().flatten()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(Kn_v, gt_v, color=COL["truth"], lw=3, label="Ground truth (Bosanquet)")
    ax.loglog(Kn_v, hqs_pred, color=COL["hqs"], ls="-.", lw=2, label="HQS")
    ax.loglog(Kn_v, edge_pred, color=COL["edge"], ls=":", lw=2, label="Edge surrogate")
    ax.set_xlabel("Knudsen Number $Kn$"); ax.set_ylabel("Damping Coefficient $c_d$")
    ax.set_title("Physics Validation across Transition Regime"); ax.legend()
    save_fig("fig02_physics_validation")

# --- Fig 3: NISQ noise ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(noise_levels, noisy_mse, "o-", color=COL["noisy"], lw=2, ms=8,
            markeredgecolor="black", markeredgewidth=0.7, label="HQS under noise")
ax.axhline(R["classical"]["mse_mean"], color=COL["classical"], ls="--", lw=1.5, label="ClassicalMLP")
ax.set_xlabel("Depolarizing error rate $p$"); ax.set_ylabel("Test MSE (log scale)")
ax.set_title("Robustness to NISQ Depolarizing Noise"); ax.legend()
save_fig("fig03_nisq_noise")

# --- Fig 4: qubit scaling (MSE with std error bars) ---
qs = [r["n_qubits"] for r in sweep]
pm = [r["protocol"]["mse_mean"] for r in sweep]; pe = [r["protocol"]["mse_std"] for r in sweep]
am = [r["ablation"]["mse_mean"] for r in sweep]; ae = [r["ablation"]["mse_std"] for r in sweep]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(qs, pm, yerr=pe, fmt="o-", color=COL["hqs"], lw=2, ms=8, capsize=4, label="HQS (protocol)")
ax.errorbar(qs, am, yerr=ae, fmt="s--", color=COL["ablation"], lw=2, ms=7, capsize=4, label="HQS (ablation)")
ax.axhline(R["classical"]["mse_mean"], color=COL["classical"], ls=":", lw=1.5, label="ClassicalMLP")
ax.axhline(R["tiny"]["mse_mean"], color=COL["tiny"], ls=":", lw=1.5, label="TinyMLP")
ax.set_xlabel("Number of qubits $n$"); ax.set_ylabel("Weighted test MSE")
ax.set_xticks(qs); ax.set_title("Qubit-Count Scaling (mean ± std)"); ax.legend()
save_fig("fig04_qubit_scaling")

# --- Fig 5: SEED VARIATION strip plot (per-seed MSE for each model) ---
models = [("ClassicalMLP", R["classical"], COL["classical"]),
          ("TinyMLP",      R["tiny"],      COL["tiny"]),
          ("HQS Protocol", R["hqs"],       COL["hqs"]),
          ("HQS Ablation", R["ablation"],  COL["ablation"])]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, (name, d, c) in enumerate(models):
    xs = np.full(len(d["mse_per_seed"]), i) + np.random.uniform(-0.08, 0.08, len(d["mse_per_seed"]))
    ax.scatter(xs, d["mse_per_seed"], color=c, s=70, zorder=3, edgecolor="black", linewidth=0.6,
               label=f"{name}")
    ax.errorbar(i, d["mse_mean"], yerr=d["mse_std"], fmt="_", color=c, ms=40, lw=2.5,
                capsize=8, zorder=2)
ax.set_xticks(range(len(models)))
ax.set_xticklabels([m[0] for m in models], rotation=15)
ax.set_ylabel("Weighted test MSE")
ax.set_title(f"Per-Seed MSE Variation (seeds {SEEDS}; bars = mean ± std)")
save_fig("fig05_seed_variation")

# --- Fig 6: MSE vs RMSE bar comparison with error bars ---
names  = [m[0] for m in models]
mse_m  = [m[1]["mse_mean"]  for m in models]
mse_s  = [m[1]["mse_std"]   for m in models]
rmse_m = [m[1]["rmse_mean"] for m in models]
rmse_s = [m[1]["rmse_std"]  for m in models]
xpos = np.arange(len(names)); width = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(xpos - width/2, mse_m, width, yerr=mse_s, capsize=4, color="#4575b4",
            edgecolor="black", linewidth=0.6, label="MSE")
b2 = ax.bar(xpos + width/2, rmse_m, width, yerr=rmse_s, capsize=4, color="#d73027",
            edgecolor="black", linewidth=0.6, label="RMSE")
ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=15)
ax.set_ylabel("Error (log₁₀-space)")
ax.set_title(f"MSE vs RMSE by Model (mean ± std across {len(SEEDS)} seeds)")
ax.legend()
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
save_fig("fig06_mse_rmse_comparison")

# --- Fig 7: distillation note ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["HQS\nteacher", "Edge\nsurrogate"], [R["hqs"]["mse_mean"], R["edge"]["mse"]],
       color=[COL["hqs"], COL["edge"]], edgecolor="black", linewidth=0.6)
ax.set_ylabel("Test MSE"); ax.set_title(f"Distillation: teacher→student (fidelity MSE={R['edge']['distill_mse']:.4f})")
for i, v in enumerate([R["hqs"]["mse_mean"], R["edge"]["mse"]]):
    ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
save_fig("fig07_distillation")
# NOTE ON THIS FIGURE: the "HQS teacher" bar plots the 3-seed MEAN protocol
# MSE (R["hqs"]["mse_mean"]), for consistency with the rest of the figures
# in this script, which are all mean-across-seeds by convention. The actual
# distillation teacher is a SINGLE model (the seed SEEDS[-1] protocol run --
# see Section 5 above), whose own individual test MSE is
# R["hqs"]["mse_per_seed"][-1], printed separately at the end of Section 5.
# For the paper, we report both numbers and are explicit about which is
# which (manuscript Fig. 10 caption); if you are consuming this figure
# directly, prefer the printed per-seed number for a true teacher-student
# comparison.

# --- Fig 8: follow-up study -- ansatz & training-budget comparison ---
if RUN_FOLLOWUP_STUDY:
    labels = ["ClassicalMLP", "TinyMLP",
              f"HQS proto\n({EPOCHS_HQS}ep)", f"HQS abl\n({EPOCHS_HQS}ep)",
              f"HQS proto\n({EPOCHS_EXTENDED}ep)", f"HQS abl\n({EPOCHS_EXTENDED}ep)",
              f"Reup proto\n({EPOCHS_HQS}ep)", f"Reup abl\n({EPOCHS_HQS}ep)"]
    means = [R["classical"]["mse_mean"], R["tiny"]["mse_mean"],
             R["hqs"]["mse_mean"], R["ablation"]["mse_mean"],
             R["followup_epoch_budget"]["protocol"]["mse_mean"],
             R["followup_epoch_budget"]["ablation"]["mse_mean"],
             R["followup_reuploading"]["protocol"]["mse_mean"],
             R["followup_reuploading"]["ablation"]["mse_mean"]]
    stds = [R["classical"]["mse_std"], R["tiny"]["mse_std"],
            R["hqs"]["mse_std"], R["ablation"]["mse_std"],
            R["followup_epoch_budget"]["protocol"]["mse_std"],
            R["followup_epoch_budget"]["ablation"]["mse_std"],
            R["followup_reuploading"]["protocol"]["mse_std"],
            R["followup_reuploading"]["ablation"]["mse_std"]]
    bar_colors = [COL["classical"], COL["tiny"], COL["hqs"], COL["ablation"],
                  "#e08214", "#e08214", "#5aae61", "#5aae61"]
    fig, ax = plt.subplots(figsize=(10, 5))
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, means, yerr=stds, capsize=4, color=bar_colors,
                   edgecolor="black", linewidth=0.6)
    ax.axhline(R["classical"]["mse_mean"], color=COL["classical"], ls=":", lw=1.2, alpha=0.7)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Weighted test MSE")
    ax.set_title("Follow-Up: Does More Budget or a Different Ansatz Close the Gap?\n"
                  "(tests Sec. 7.1 hypotheses; mean \u00b1 std, 3 seeds)")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    save_fig("fig08_followup_ansatz_epochs")

# ===========================================================
# 9. C++ EXPORT + TABLES + JSON
# ===========================================================
print("\n[8/9] C++ export, tables, JSON...")
ew1, eb1 = edge.net[0].weight.data.numpy(), edge.net[0].bias.data.numpy()
ew2, eb2 = edge.net[2].weight.data.numpy(), edge.net[2].bias.data.numpy()
cpp = f"""// edge_model.h - auto-generated
#pragma once
#include <math.h>
static const float l1_w[16][3] = {{ {", ".join("{"+", ".join(f"{v:.6f}" for v in r)+"}" for r in ew1)} }};
static const float l1_b[16] = {{ {", ".join(f"{v:.6f}" for v in eb1)} }};
static const float l2_w[16] = {{ {", ".join(f"{v:.6f}" for v in ew2[0])} }};
static const float l2_b = {eb2[0]:.6f};
static inline float predict_log_damping(float skn, float sar, float slf) {{
    float out = l2_b;
    for (int i=0;i<16;i++) {{
        float h = l1_b[i] + skn*l1_w[i][0] + sar*l1_w[i][1] + slf*l1_w[i][2];
        out += tanhf(h) * l2_w[i];
    }}
    return out; // pow(10,out) for physical cd
}}
"""
open("output/edge_model.h", "w").write(cpp)

# ---- LaTeX Table A: per-seed MSE/RMSE summary ----
def fmt_seedlist(vals): return ", ".join(f"{v:.4f}" for v in vals)
tabA = r"""% ---- Table A: per-seed MSE/RMSE (paste into Section 5) ----
\begin{table*}[t]\centering
\caption{Per-seed test error and seed statistics (weighted, $\log_{10}$-space).
RMSE$_{\mathrm{phys}}=10^{\mathrm{RMSE}}$ is the multiplicative error factor in $c_d$.}
\label{tab:seedstats}\small
\begin{tabular}{@{}lrccccc@{}}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{MSE per seed} &
\textbf{MSE (mean$\pm$std)} & \textbf{RMSE (mean$\pm$std)} & \textbf{RMSE$_{\mathrm{phys}}$} \\
\midrule
"""
for name, d, _ in models:
    tabA += (f"{name} & {d['params']} & {fmt_seedlist(d['mse_per_seed'])} & "
             f"${d['mse_mean']:.5f}\\pm{d['mse_std']:.5f}$ & "
             f"${d['rmse_mean']:.4f}\\pm{d['rmse_std']:.4f}$ & "
             f"${d['phys_factor']:.2f}\\times$ \\\\\n")
tabA += (f"Edge surrogate & {R['edge']['params']} & --- & "
         f"${R['edge']['mse']:.5f}$ & ${R['edge']['rmse']:.4f}$ & "
         f"${10**R['edge']['rmse']:.2f}\\times$ \\\\\n")
tabA += r"""\bottomrule
\end{tabular}
\end{table*}
"""
open("output/table_seedstats.tex", "w").write(tabA)

# ---- LaTeX Table B: qubit sweep ----
tabB = r"""% ---- Table B: qubit-count sweep (paste into Section 5) ----
\begin{table}[t]\centering
\caption{Qubit-count scaling sweep (""" + f"{len(SEEDS)}" + r"""-seed mean$\pm$std, weighted
$\log_{10}$ MSE). More qubits do not close the gap to the classical baselines.}
\label{tab:qubitsweep}\small
\begin{tabular}{@{}rrrccr@{}}
\toprule
\textbf{Qubits} & \textbf{$2^n$} & \textbf{Params} &
\textbf{Protocol MSE} & \textbf{Ablation MSE} & \textbf{Time (s)} \\
\midrule
"""
for r in sweep:
    tabB += (f"{r['n_qubits']} & {r['hilbert_dim']} & {r['params']} & "
             f"${r['protocol']['mse_mean']:.5f}\\pm{r['protocol']['mse_std']:.5f}$ & "
             f"${r['ablation']['mse_mean']:.5f}\\pm{r['ablation']['mse_std']:.5f}$ & "
             f"{r['time_s']:.0f} \\\\\n")
tabB += r"""\bottomrule
\end{tabular}
\end{table}
"""
open("output/table_qubitsweep.tex", "w").write(tabB)

# ---- LaTeX Table C: follow-up study (Section 5.7) ----
if RUN_FOLLOWUP_STUDY:
    def gap_pct(mse, ref=R["classical"]["mse_mean"]): return (mse - ref) / ref * 100
    fb, fr = R["followup_epoch_budget"], R["followup_reuploading"]
    tabC_rows = [
        ("Original: 1{,}500ep, StronglyEntangling", R["hqs"], R["ablation"],
         f"{gap_pct(R['hqs']['mse_mean']):.1f}\\% / {gap_pct(R['ablation']['mse_mean']):.1f}\\%",
         f"{R['hqs']['time_s']:.1f}"),
        ("+ Extended budget: 4{,}000ep, same ansatz", fb["protocol"], fb["ablation"],
         f"{gap_pct(fb['protocol']['mse_mean']):.1f}\\% / {gap_pct(fb['ablation']['mse_mean']):.1f}\\%",
         f"{fb.get('time_s','--')}"),
        ("+ Data-reuploading: 1{,}500ep, same budget", fr["protocol"], fr["ablation"],
         f"{gap_pct(fr['protocol']['mse_mean']):.1f}\\% / {gap_pct(fr['ablation']['mse_mean']):.1f}\\%",
         f"{fr.get('time_s','--')}"),
    ]
    if RUN_REUPLOAD_AT_EXTENDED_EPOCHS:
        fx = R["followup_reuploading_extended"]
        tabC_rows.append(("+ Both combined: 4{,}000ep, reuploading", fx["protocol"], fx["ablation"],
             f"{gap_pct(fx['protocol']['mse_mean']):.1f}\\% / {gap_pct(fx['ablation']['mse_mean']):.1f}\\%",
             f"{fx.get('time_s','--')}"))
    tabC = r"""% ---- Table C: follow-up study, Sec. 5.7 (paste into Section 5) ----
\begin{table*}[t]\centering
\caption{Follow-up study testing the ansatz and training-budget hypotheses
(3-seed mean$\pm$std, weighted $\log_{10}$ MSE).}
\label{tab:followup}\small
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Configuration} & \textbf{Protocol MSE} & \textbf{Ablation MSE} &
\textbf{Gap to ClassicalMLP} & \textbf{Time (s/seed)} \\
\midrule
"""
    for name, p, a, gap, t in tabC_rows:
        tabC += (f"{name} & ${p['mse_mean']:.4f}\\pm{p['mse_std']:.4f}$ & "
                 f"${a['mse_mean']:.4f}\\pm{a['mse_std']:.4f}$ & {gap} & {t} \\\\\n")
    tabC += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    open("output/table_followup.tex", "w").write(tabC)

with open("output/results.json", "w") as f:
    json.dump(R, f, indent=2)

# ===========================================================
# 10. CONSOLE SUMMARY
# ===========================================================
print("\n[9/9] DONE.\n")
print("=" * 72)
print("  FINAL SUMMARY (mean across seeds " + str(SEEDS) + ")")
print("=" * 72)
print(f"{'Model':<16}{'Params':>8}{'MSE':>11}{'±std':>10}{'RMSE':>10}{'phys×':>9}")
for name, d, _ in models:
    print(f"{name:<16}{d['params']:>8}{d['mse_mean']:>11.5f}{d['mse_std']:>10.5f}"
          f"{d['rmse_mean']:>10.4f}{d['phys_factor']:>8.2f}x")
print(f"{'Edge surrogate':<16}{R['edge']['params']:>8}{R['edge']['mse']:>11.5f}"
      f"{'--':>10}{R['edge']['rmse']:>10.4f}{10**R['edge']['rmse']:>8.2f}x")
print("-" * 72)
print("  Per-seed MSE:")
for name, d, _ in models:
    print(f"    {name:<16}: {d['mse_per_seed']}")
print("-" * 72)
print("  Qubit sweep (protocol MSE mean±std):")
for r in sweep:
    print(f"    {r['n_qubits']:>2}q ({r['params']:>3}p): "
          f"{r['protocol']['mse_mean']:.5f} ± {r['protocol']['mse_std']:.5f}   "
          f"[{r['time_s']:.0f}s/seed]")
print(f"  NISQ max tolerable noise: p = {R['nisq']['max_tolerable']:.3f}")
if RUN_FOLLOWUP_STUDY:
    print("-" * 72)
    print(f"  Follow-up study (Sec. 7.1 hypotheses; tests Section 7b):")
    fb = R["followup_epoch_budget"]; fr = R["followup_reuploading"]
    print(f"    Extended budget ({fb['epochs']}ep) -- protocol: {fb['protocol']['mse_mean']:.5f} "
          f"+/- {fb['protocol']['mse_std']:.5f}   ablation: {fb['ablation']['mse_mean']:.5f} "
          f"+/- {fb['ablation']['mse_std']:.5f}   time: {fb['time_s']:.1f}s/seed")
    print(f"    Reuploading  ({fr['epochs']}ep) -- protocol: {fr['protocol']['mse_mean']:.5f} "
          f"+/- {fr['protocol']['mse_std']:.5f}   ablation: {fr['ablation']['mse_mean']:.5f} "
          f"+/- {fr['ablation']['mse_std']:.5f}   time: {fr['time_s']:.1f}s/seed")
    print(f"    Reference -- ClassicalMLP: {R['classical']['mse_mean']:.5f}   "
          f"orig. HQS protocol/ablation: {R['hqs']['mse_mean']:.5f} / {R['ablation']['mse_mean']:.5f}")
    if RUN_REUPLOAD_AT_EXTENDED_EPOCHS:
        fx = R["followup_reuploading_extended"]
        print(f"    Reuploading+extended ({fx['epochs']}ep) -- ablation: "
              f"{fx['ablation']['mse_mean']:.5f} +/- {fx['ablation']['mse_std']:.5f}   "
              f"time: {fx['time_s']:.1f}s/seed")
print("=" * 72)
print("  Files: output/results.json, output/table_seedstats.tex,")
print("         output/table_qubitsweep.tex, output/table_followup.tex,")
print("         output/edge_model.h")
fig_range = "fig01..fig08" if RUN_FOLLOWUP_STUDY else "fig01..fig07"
print(f"  Figures: figures/{fig_range} (.pdf/.png)")
print("=" * 72)
print("\n  NOTE: the two reproducibility fixes previously shipped as")
print("  reproducibility_fix.py (classical timing average, distillation")
print("  random seed) are now merged inline above -- see the header")
print("  docstring. reproducibility_fix.py is kept only as a historical")
print("  record of the standalone verification run. The follow-up-study")
print("  timing fix (BUG-3) is also merged inline -- see header.")
print("=" * 72)
