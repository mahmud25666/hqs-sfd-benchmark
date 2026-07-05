# HQS-SFD-Benchmark

Code accompanying *"Benchmarking the Empirical Limits of Variational
Quantum Circuits for Rarefied Gas Dynamics Surrogates: A Rigorous
Multi-Seed Evaluation Against Classical Baselines."*

## What this is

A single, self-contained pipeline (`hqs_sfd_benchmark.py`) that generates
every number, table, and figure reported in the paper: the Bosanquet
squeeze-film-damping dataset, the classical baselines, the 6-qubit hybrid
quantum surrogate (layer-wise protocol and end-to-end ablation), the
8-/10-qubit scaling sweep, NISQ depolarizing-noise robustness, the
quantum-to-edge distillation pipeline, and the Section 5.7 follow-up
study testing the training-budget and ansatz-mismatch hypotheses.

## Quick start

```bash
pip install pennylane torch scikit-learn matplotlib numpy pandas tqdm
python3 hqs_sfd_benchmark.py
```

Set `QUICK_TEST = True` near the top of the file for a fast (~10–25 min)
sanity-check run at reduced epochs and a smaller distillation budget —
useful for confirming your environment works before committing to the
full run. Set it to `False` (the default) for the publication-accuracy
run.

**Runtime for the full run**: roughly 24–30 hours for the original paper
sections, plus 2–3 hours for the follow-up study (add ~2 more hours if
`RUN_REUPLOAD_AT_EXTENDED_EPOCHS = True`, which it is by default). The
10-qubit arm of the scaling sweep alone is ~19.5 hours on a CPU
state-vector simulator (3 seeds × ~6.5 hr/seed). If you're running on a
managed notebook service with session time limits, consider splitting
the qubit sweep by width and checkpointing per seed.

## Outputs

- `output/results.json` — full per-seed results for every experiment
- `output/table_seedstats.tex`, `output/table_qubitsweep.tex`,
  `output/table_followup.tex` — LaTeX table snippets, ready to paste
  into the manuscript
- `output/edge_model.h` — the distilled 81-parameter C++ inference
  header for microcontroller deployment
- `output/scaler_params.json` — the input feature scaler's mean/std,
  needed to reproduce the preprocessing in `edge_model.h`
- `figures/fig01`–`fig08` (`.pdf` and `.png`) — all publication figures

## Reproducibility notes

This script has gone through two rounds of post-hoc correction, both
documented inline (search the file for `FIXED: BUG-`):

1. **Classical-baseline timing** was originally a scalar overwritten
   each seed iteration (only the last seed's time survived); now a
   proper 3-seed mean ± std.
2. **Distillation sampling** originally had no fixed seed, so the
   distilled edge surrogate's MSE drifted at the 4th decimal between
   runs; now seeded immediately before sampling for bit-for-bit
   reproducibility.
3. **Follow-up study timing**: the three follow-up configurations
   (extended epoch budget, data-reuploading, reuploading+extended)
   originally had correct accuracy numbers but no wall-clock timing
   instrumentation. This has been added, matching the same
   `time.time()`-per-seed pattern used in the main HQS run and the
   qubit sweep. Verified not to change any accuracy number: re-running
   `followup_reuploading` with the fix reproduced the original MSE
   values to 5 decimal places while adding the previously-missing
   timing (884.1 ± 8.7 s).

Given fixed seeds, no dropout, and full-batch gradient descent
throughout (`BATCH_SIZE = None`), every number in this pipeline is
deterministic and should reproduce exactly on rerun, environment
permitting (PennyLane/PyTorch version and CPU backend can in principle
introduce floating-point-level nondeterminism; we have not observed
this affecting any reported conclusion).

## A note on the distillation teacher

The distillation teacher (`fig07_distillation` / manuscript Figure 10)
is a **single representative model** — the protocol run from the last
seed in `SEEDS` (seed 456) — not an ensemble or the best-performing
seed. The script prints this model's own individual test MSE
separately from the 3-seed mean at the end of Section 5, and the
`fig07_distillation` figure is annotated inline with a note on which
number is which. If you want to distill from a different seed or an
ensemble, `last_prot_model` in Section 4 is the place to change.

## Disabling the follow-up study

Set `RUN_FOLLOWUP_STUDY = False` to skip Section 7b entirely and
reproduce only the original (pre-follow-up) paper's numbers. Nothing
in Sections 1–7 depends on the follow-up study; it is purely additive.

## Citation

If you use this code, please cite the accompanying manuscript (details
in the paper's front matter / this repository's citation file).
