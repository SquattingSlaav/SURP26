# XOR evaluation: point-based vs. boundary-based scoring

## Problem

Professor feedback (2026-07-01): scoring the 15-param quantum networks on XOR by
checking only 4 discrete input points — `(0,0), (0,π/2), (π/2,0), (π/2,π/2)` — is
flawed. A model can classify those 4 exact points correctly without actually
implementing a genuine XOR-shaped decision boundary across the continuous input
space. Need to evaluate against a *region*, not isolated points.

## Investigation

Two files already existed from the prior 900-model random search:
- `results/survey_15param.npz` — full `(900, 20, 20)` expectation maps for all
  900 randomly-parameterized models, swept over α, β ∈ `linspace(0, π/2, 20)`.
- `results/xor_search.npz` — the old point-based search: MSE at the 4 corners
  only, per model, plus the "best" model by that metric (index 418).

No new circuit simulation was needed — the full 2D maps were already there, just
never evaluated as a region.

Defined a region-based ground truth: split the input plane into 4 quadrants at
α = π/4, β = π/4 (falls cleanly between grid indices 9 and 10, so no grid point
sits exactly on the split). Same-side quadrants → target 0, opposite-side
quadrants → target 1 (matches the discrete XOR truth table at the corners).
Scored all 900 models by region accuracy (fraction of the 400 grid cells
correctly classified via `expectation > 0.5`) and region MSE.

## Findings (2026-07-01)

- Model 418 (old point-based best, corner-MSE = 0.193) is *also* the best model
  by region accuracy — but that accuracy is only **64%** across all 400 grid
  cells. Barely better than chance for a binary label.
- Correlation between point-MSE ranking and region-accuracy across all 900
  models: **r ≈ 0.166** — essentially no relationship. Good point-scores don't
  predict a real decision boundary.
- Shot-noise fragility: model 418 classifies **4/4** corners correctly in the
  independent `xor_search.npz` simulation run, but only **3/4** against the
  `survey_15param.npz` grid — same model, same parameters, different 1024-shot
  sample, different classification near the 0.5 threshold. Point-based scoring
  isn't even reproducible run-to-run.
- Visually, the 0.5-expectation contour for the "best" model is a scattered,
  disconnected scribble — it does not track the true quadrant boundary at all,
  confirming none of the 900 random models learned a genuine XOR decision region.
- Only 4 of the 900 models exceed 60% region accuracy; 0 achieve perfect 4/4
  corner classification against the survey grid.

## Artifacts produced

- `src/Part_3_Quantum_Neuron/QuantumNeuronXORBoundary.py` — computes the region
  metric from existing data, re-ranks all 900 models, prints the comparison
  report above, and saves the figure below.
- `results/xor_boundary_search.npz` — `accuracy (900,)`, `mse (900,)`,
  `best_idx`, `best_params` under the new region-based metric.
- `figures/xor_boundary_comparison.png` — 3-panel figure: old point-based best
  model's 2D map + 0.5-contour + quadrant split, new region-based best model
  (same model in this case), and a scatter of point-MSE vs. region-accuracy
  across all 900 models showing the near-zero correlation.

## Status / next steps

- [x] Diagnose the point-based evaluation flaw with real data.
- [x] Define and implement a region/boundary-based scoring metric.
- [x] Re-rank all 900 existing models under the new metric; visualize the
      comparison.
- [ ] **Not started**: retrain `QuantumNeuralNetwork.py` against a region/boundary
      loss (e.g. averaging loss over multiple sampled grid points per quadrant)
      instead of the current 4-corner parameter-shift loss. Flagged as
      significantly more expensive (many more circuit evaluations per epoch) —
      needs its own pass.
