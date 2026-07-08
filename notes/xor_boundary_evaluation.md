# XOR evaluation: point-based vs. boundary-based scoring

## Problem

Shing Chi's feedback (2026-07-01): scoring the 15-param quantum networks on XOR by
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

## Retraining against a region/boundary loss (2026-07-03)

Replaced the 4-corner training loss (`QuantumNeuralNetwork.py`) with a genuine
region loss: 16 fixed interior points (4 per quadrant, cross of 2 "low"/2 "high"
representative angles per axis), targets set by the same quadrant-split rule as
above. Trained via the same parameter-shift + plain gradient descent scheme,
starting from the region-search best params (model 418).

- First attempt used the original `lr=0.01` (matching the old corner-training
  script) for 50 epochs — loss was completely flat/noisy (0.235 → 0.233, no real
  trend), and the resulting model only reached 65.5% region accuracy, barely
  above the untrained baseline (64%). Ran 300 epochs at the same `lr` to rule out
  "just needs more time" — still flat (min loss 0.218, no clear trend). Averaging
  gradients over 16 points spanning all 4 quadrants seems to produce much smaller
  net gradients than averaging over just 4 corners (competing directions from
  different quadrants partially cancel), so the same learning rate that worked
  for corner-only training was too small here.
- Note on runtime: original estimate (extrapolated from the earlier search/train
  session's per-eval timings) was ~35-50 minutes for this run. Actual measured
  rate in this session was far faster (~1.5-2s/epoch for 16 pts × 13 params × 2
  shifts), so the real cost of experimentation was minutes, not tens of minutes —
  worth remembering before assuming a script needs a long background run.
- Re-tuned to `lr=0.2`, confirmed a real monotonic loss decrease across 150
  epochs (chunked means: 0.230 → 0.220 → 0.214 → 0.211 → 0.209 → 0.206).
- **Result**: region-trained model reaches **86.5%** region accuracy on the full
  400-cell grid (region-MSE 0.206), vs. **78.0%** for the old corner-trained
  model (`xor_trained.npz`, re-evaluated on the same full grid for a fair
  comparison) and 64% for the untrained best-search model. **+8.5 percentage
  points** over corner-only training.
- Visually (`figures/xor_region_trained_comparison.png`), the region-trained
  model's 0.5-contour visibly tracks the quadrant split much more coherently
  than the old model's scattered scribble — a real, not just numerical,
  improvement in the learned decision boundary.

### Artifacts (this phase)

- `src/Part_3_Quantum_Neuron/QuantumNeuronXORRegionTrain.py` — region-loss
  training script (`lr=0.2`, `n_epochs=150`).
- `results/xor_region_trained.npz` — `params (13,)`, `loss_history (150,)`.
- `figures/xor_region_trained_comparison.png` — old vs. new full-grid maps +
  training loss curve.

## Replicating Shing Chi's MCMC architecture (2026-07-04)

Shing Chi sent a code snippet with a different architecture: three standalone
neurons (`qneuron5` ×2 → `qneuron3b`) composed *classically* — each neuron is
measured and its scalar output `M ∈ [0,1]` is fed as a rotation-angle input to
the next neuron's circuit, rather than one entangled multi-qubit circuit. The
search is a greedy single-index walk: 13 parameters stored as indices into a
40-entry angle table, one index nudged ±1 per trial, accepted only if the
mismatch count vs. `p_ref` (his domain-extended XOR target, scored with a
dynamic `(max+min)/2` threshold) strictly decreases.

Replicated in `src/Part_3_Quantum_Neuron/QuantumNeuronMCMC.py`, with
two explicitly flagged guesses (his snippet omits the class bodies):
`QNeuron3b`'s gate structure, and `angles40 = linspace(0, π/2, 40)`.

### Run 1 (failed silently, 2.2 hrs)

Seeded `results_old = 69` straight from his snippet — but that score belongs to
*his* circuit. Ours scores ~184–214 at the same warm-start indices, so no ±1
move could ever beat 69 and all 3,000 trials were rejected. Lesson: never carry
a warm-start *score* across implementations; re-evaluate it locally. Fixed by
computing the initial score from an actual evaluation.

### Run 2 (completed, 1.9 hrs) — improvement is a shot-noise artifact

- Nominal progress: 214 → 162 mismatches, 6 accepted moves (all in the first
  ~585 trials, nothing after).
- **Reality check**: re-scoring the final params 5 independent times gave
  [195, 199, 204, 177, 204] — mean ≈ 196, i.e. chance level (200/400 = coin
  flip). The "162" was a lucky 1024-shot draw that greedy strict-`<` acceptance
  locked in permanently. Note the initial score itself fluctuated 184 vs 214
  between runs at identical params — score noise is ±20-30.
- Root cause: the model's output range across the whole grid is only ~0.17–0.25.
  The dynamic threshold sits mid-band, so the thresholded mask is nearly a
  per-cell coin flip → mismatch count ~ Binomial(400, 0.5), std ≈ 10, and a
  greedy search on that just harvests noise minima.
- The compressed output range itself suggests our architecture guesses
  (`QNeuron3b` structure and/or the `angles40` range) differ from the
  Shing Chi's actual code in ways that matter — his run reached 69/400.

### Artifacts (this phase)

- `src/Part_3_Quantum_Neuron/QuantumNeuronMCMC.py`
- `results/mcmc_search.npz`, `figures/mcmc_search.png`

### Open questions for Shing Chi

- ~~Actual `qneuron3b` circuit definition~~ → resolved 2026-07-05 (see below).
- ~~Actual `angles40` contents~~ → he provided it: `[i*π/2/40 for i in range(40)]`.
- ~~How his run avoids the shot-noise ratchet~~ → moot; see below.

## Corrected replication from his qneuron5 reduction (2026-07-05)

Shing Chi provided `angles40` and his analytic `qneuron5` class — a classical
reduction of the 5-param circuit — with the instruction not to use the class
itself, but to make our circuit's **measured expectation equal his `self.M`**.
Matching his formula against exact statevector simulation pinned down every
convention we had guessed wrong:

1. **Inputs are probabilities, not angles.** `get_expectation(M1, M2)` takes
   `M ∈ [0,1]` (both the `phases20` grid values and upstream neuron outputs),
   encoded as `ry(2·arcsin(√M))` — the pure state with `P(1)=M` and coherence
   `√(M(1−M))`, exactly his `M1_01` term. Our first replication fed `M` as a
   raw rotation angle, which is why its output was a flat noise band.
2. **Circuit angles are 2× his parameters**: weights `ry(−2α)`, CRy pair
   `cry(2β)…cry(−2β)`, bias `ry(2δ)`.
3. **The phase gate is Qiskit `rz(π)`, not `rz(π/2)`.** His "Rz(π/2)" is
   evidently the `e^{iθσz}` convention (= Qiskit `rz(2θ)`). ⚠️ This touches
   the whole repo's "Rz(π/2) between the CRy pair" convention — every other
   script may be running with half the intended phase. **Confirm with him.**
4. **His (1,1) term `sin²(2β1−2β2+δ)` is unrealizable** by any circuit of this
   family: all target-qubit RYs commute, so the four control-branch angles
   must be additive, giving `sin²(2β1+2β2−δ)`. With that one substitution the
   circuit matches his formula to `~9e-16` over 300 random draws (branches
   (0,0), (1,0), (0,1) match his formula as written, exactly). Almost
   certainly an algebra slip in his class — **flag to him** with this evidence.
5. **`qneuron3b`** = the bare core of the 5-param neuron without the two
   input-weight rotations: `cry(2β1), cry(2β2), rz(π),` inverse pair, `ry(2δ)`
   — exactly 3 params, 2 probability inputs ("the 3 neuron behaves similarly").

Rewrote `QuantumNeuronMCMC.py` accordingly. The search now evaluates
with exact statevector expectations (greedy strict-`<` acceptance is
meaningless under ±20-30 shot noise, per the run-2 finding), with a final
1024-shot validation pass on the best model. Includes
`verify_against_reference_formula()` which prints the machine-precision match
and quantifies the (1,1) discrepancy at every run.

Immediate effects of the corrections: output dynamic range went from a flat
~0.17–0.25 to 0.165–0.898, per-trial cost from ~2.6 s to ~0.19 s (9.5 min per
3,000-trial search instead of 2.2 h), and the search accepts moves
deterministically from the very first trials.

### Search results (2026-07-05)

- Single greedy chain from Shing Chi's warm start: 205 → 141 mismatches,
  all 34 accepted moves in the first ~490 trials, then stuck — the model
  learned a smooth *diagonal* separator (a local minimum), not XOR. The ±1
  single-index greedy walk cannot cross the valley from "diagonal" to
  "checkerboard" without temporarily getting worse.
- **Random restarts fixed it.** 10 greedy chains (Shing Chi's warm start + 9
  random starts, early-stopped after 500 trials without an accepted move),
  18.3 min total. Final scores: [185, 142, 190, 142, **20**, 126, 127, 92,
  196, 211] — huge spread, confirming a very rugged landscape. The best chain
  (restart 4, random start) reached **20/400 mismatches (95% agreement)** with
  a clear XOR checkerboard in the expectation map — well past Shing Chi's
  69.
- **Caveat — the boundary is real but faint.** The best model's output range
  is only 0.215–0.324; the median cell sits 0.010 from the dynamic threshold,
  and 1024-shot sampling noise is ±0.014. Measured mismatch count vs shots:
  1024 → 97, 4096 → 69, 16384 → 44 (exact: 20). So the checkerboard exists in
  the true expectation but is not readable at 1024 shots — same
  shot-noise-fragility theme as the original point-based XOR scoring. Worth
  discussing whether score should reward *margin*, not just sign, if
  shot-efficient readout matters.

Artifacts: `results/mcmc_search.npz` (best params/array/history,
per-restart finals, shot check), `figures/mcmc_search.png` (best
model vs `p_ref` + per-restart progress curves).

## Circuit audit: 3-param and 2to1 (2026-07-05)

Prompted by comparing architectures against Shing Chi's snippet, audited
the older neuron circuits. The consistent family (confirmed by the user's
3-param circuit diagram) is: **3-param neuron = input-weight RY + CRy pair +
RY bias**; networks compose as first-layer neurons + a 3-param output neuron.
Under that definition the "15-param" network is really 13 params (5+5+3) —
which exactly matches Shing Chi's `qneuron5 + qneuron5 + qneuron3b`
structure — and the "9-param" 2to1 should be 3+3+3.

Two bugs found and fixed:

1. **`QuantumNeuronLayered.py` (2to1): `p[6]` was dead.** The output neuron
   was missing its weight rotation `ry(p[6], q1)` (present in
   `QuantumNeuron9D.py`, which even carries an "# add this line" comment for
   it). The survey's `p[6] = p[7]` "CRy pair must match" line was a no-op —
   pair symmetry was already guaranteed by `±p[7]`. Old `survey_2to1.npz` is
   therefore data for an 8-param family with a meaningless `p[6]` column.
   Fixed the circuit; survey re-run pending.
2. **`QuantumNeuronSurvey.py` (3-param): only 2 trainable params.** Missing
   the input weight `ry(w0, q0)`. Fixed to the true 3-param form and extended
   the survey to a 10×10×10 (w0, θ1, θ2) grid; old `survey_3param.npz` is
   2-param data. Re-run pending.

Neither circuit feeds the XOR pipeline or the MCMC replication, so
those results are unaffected. Both survey loops were also batched (one
`sim.run` per 400-circuit input grid) for ~10-15x speedup.

**Addendum (same day): extra target-qubit RYs.** A second audit pass (user
spotted it) found β preloads on the *target* qubits: `ry(beta, q1)` /
`ry(beta, q3)` in the 2to1 and `ry(beta, q1)` in the standalone 3-param
neuron. Per the authoritative 3-param diagram, targets start in |0⟩ — nothing
lands on them before the CRy pair. Removing the preloads exposed an input-
routing bug too: both 2to1 first-layer neurons were taking α, with β entering
only via the bogus preloads. Corrected topology: neuron 1 processes α,
neuron 2 processes β. The standalone 3-param neuron is single-input (α); its
survey now sweeps α × (w0, θ1, θ2) with results shape (10, 10, 10, 20).
CLAUDE.md's old "Neuron architectures"/"Survey conventions" sections described
the buggy circuits (they were written from that code) and were rewritten to
match the corrected family.

**Structural observation for Shing Chi**: in single-circuit *entangled*
composition, a hidden neuron's bias and the output neuron's weight on the same
qubit (e.g. `ry(p2, q1)` directly followed by `ry(p6, q1)` in the 2to1;
`ry(p4, q2)` / `ry(p10, q2)` in the 15-param) are consecutive RYs and merge
into one effective rotation — one redundant parameter direction per hidden
neuron (9-param has 8 effective, "15"(13)-param has 12). In his *measured*
feedforward composition the measurement between layers breaks the merge, so
the parameters are genuinely independent. A real structural difference between
the two composition styles.

Reference from Shing Chi (2026-07-05): his best model's expectation map
spans ~0.57–0.87 (margin ~0.30) over the θ1, θ2 ∈ [0,1] plane — much larger
margins than our statevector-search winner (0.22–0.32), i.e. far more robust
to shot readout. Useful benchmark for the fully shot-based search.

## Figures for Shing Chi (2026-07-05)

Presentation order, one talking point each
(generated by `src/Part_3_Quantum_Neuron/QuantumNeuronFigures.py`
plus earlier scripts):

1. `figures/xor_boundary_comparison.png` — your point-based feedback was
   right: the "best" of 900 models by corner-MSE only gets 64% of the region
   right, and point-MSE is uncorrelated (r≈0.166) with region accuracy.
2. `figures/xor_region_trained_comparison.png` — retraining against a
   16-point region loss instead of the 4 corners: 78% → 86.5% full-grid
   region accuracy.
3. `figures/formula_verification.png` — our circuit's measured
   expectation equals your analytic `self.M` to ~1e-15... once the (1,1)
   term is `sin²(2β1+2β2−δ)`. As written, `sin²(2β1−2β2+δ)` can't come from
   any CRy/phase circuit (branch angles must be additive). *Question: also
   confirm the Rz convention — matching requires Qiskit `rz(π)`, i.e. your
   "Rz(π/2)" in the `e^{iθσz}` convention; our other scripts use Qiskit
   `rz(π/2)`.*
4. `figures/mcmc_search.png` — your architecture + greedy search
   with 10 random restarts: best model 20/400 mismatches (95%), a real XOR
   checkerboard. Single chains get stuck (final scores 20–211 depending on
   start).
5. `figures/mcmc_shot_budget.png` — the caveat: the checkerboard
   margin (median 0.010) is below the 1024-shot noise floor (±0.014), so the
   measured score is ~97 at 1024 shots and still ~45 at 16384. *Discussion
   point: should the score reward margin, not just sign?*

## Status / next steps

- [x] Diagnose the point-based evaluation flaw with real data.
- [x] Define and implement a region/boundary-based scoring metric.
- [x] Re-rank all 900 existing models under the new metric; visualize the
      comparison.
- [x] Retrain against a region/boundary loss instead of the 4-corner loss —
      confirmed real improvement (78% → 86.5% region accuracy).
- [ ] **Not started**: the region-training loss curve was still trending down at
      epoch 150 (not clearly converged) — could likely push accuracy higher with
      more epochs at `lr=0.2`, or try learning-rate scheduling. Cheap to explore
      further given actual per-epoch cost is ~2s, not tens of seconds.
- [x] Replicate Shing Chi's classical-feedforward MCMC architecture —
      first attempt was flat noise due to wrong conventions; corrected
      2026-07-05 from his analytic qneuron5 reduction (machine-precision match,
      exact statevector search + shot check).
- [ ] Ask Shing Chi: (a) confirm the Rz convention — his formula implies
      Qiskit `rz(π)`, the repo has been using `rz(π/2)` everywhere; (b) his
      qneuron5 (1,1) term looks like an algebra slip — circuit-realizable form
      is `sin²(2β1+2β2−δ)`.
