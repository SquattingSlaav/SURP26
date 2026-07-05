# Figures for the professor meeting: (1) verification that the circuit's
# measured expectation equals his analytic qneuron5 self.M (and where his
# (1,1) term departs), (2) shot-budget readout cost of the best XOR model
# found by the restart search. The professor's formula is used only as a
# verification reference here — the model itself is always the circuit.
import numpy as np
import matplotlib.pyplot as plt
import os

from QuantumNeuronProfessorMCMC import (
    QNeuron5, professor_M, update_neuron_params, evaluate_array, score,
    phases20, SHOTS,
)

os.makedirs("figures", exist_ok=True)


def fig_formula_verification(n_draws=300, seed=0):
    rng = np.random.default_rng(seed)
    circ_vals, prof_written, prof_additive = [], [], []
    for _ in range(n_draws):
        M1, M2 = rng.uniform(0, 1, 2)
        w = rng.uniform(0, np.pi / 2, 5)
        qn = QNeuron5(w)
        circ_vals.append(qn.get_expectation(M1, M2))
        prof_written.append(professor_M(M1, M2, *w))
        prof_additive.append(professor_M(M1, M2, *w, additive_11=True))
    circ_vals = np.array(circ_vals)
    prof_written = np.array(prof_written)
    prof_additive = np.array(prof_additive)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for ax, prof, title in (
        (axes[0], prof_written,
         "Professor's formula as written"),
        (axes[1], prof_additive,
         "With additive (1,1) term  sin²(2β1+2β2−δ)"),
    ):
        ax.scatter(prof, circ_vals, s=14, alpha=0.6, color='steelblue')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)
        max_err = np.max(np.abs(prof - circ_vals))
        ax.set_xlabel('analytic  self.M')
        ax.set_ylabel('circuit expectation (exact statevector)')
        ax.set_title(f'{title}\nmax |ΔM| = {max_err:.2e}  ({n_draws} random draws)')
        ax.set_aspect('equal')

    # branch coefficients vs delta: controls forced to |a>|b>, input weights 0
    b1, b2 = rng.uniform(0, np.pi / 2, 2)
    deltas = np.linspace(0, np.pi / 2, 100)
    prof_branch = {
        (0, 0): lambda d: np.sin(d)**2,
        (1, 0): lambda d: np.sin(2 * b1 - d)**2,
        (0, 1): lambda d: np.sin(2 * b2 - d)**2,
        (1, 1): lambda d: np.sin(2 * b1 - 2 * b2 + d)**2,
    }
    colors = {(0, 0): 'tab:blue', (1, 0): 'tab:green',
              (0, 1): 'tab:orange', (1, 1): 'tab:red'}
    ax = axes[2]
    for (a, b), f in prof_branch.items():
        circ_curve = [QNeuron5([0, 0, b1, b2, d]).get_expectation(a, b)
                      for d in deltas]
        ax.plot(deltas, circ_curve, color=colors[(a, b)], linewidth=2,
                label=f'circuit, controls |{a}{b}⟩')
        ax.plot(deltas, f(deltas), color=colors[(a, b)], linestyle='--',
                linewidth=1.5, label=f'formula ({a},{b}) term')
    ax.set_xlabel('δ')
    ax.set_ylabel('P(1) of target qubit')
    ax.set_title(f'Control-branch coefficients (β1={b1:.2f}, β2={b2:.2f})\n'
                 'solid = circuit, dashed = his formula; only (1,1) separates')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/professor_formula_verification.png', dpi=150,
                bbox_inches='tight')
    print("Saved → figures/professor_formula_verification.png")


def fig_shot_budget(budgets=(256, 1024, 4096, 16384), n_repeats=3):
    d = np.load('results/professor_mcmc_search.npz')
    best_params = [int(v) for v in d['params_old']]
    exact_array = d['array']
    exact_score = int(d['results_old'])
    update_neuron_params(best_params)

    scores = {}
    for b in budgets:
        scores[b] = []
        for r in range(n_repeats):
            sa, _, _ = evaluate_array(shots=b)
            scores[b].append(score(sa))
        print(f"shots={b}: scores {scores[b]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    means = [np.mean(scores[b]) for b in budgets]
    spread = [(np.mean(scores[b]) - min(scores[b]),
               max(scores[b]) - np.mean(scores[b])) for b in budgets]
    yerr = np.array(spread).T
    ax = axes[0]
    ax.errorbar(budgets, means, yerr=yerr, marker='o', color='steelblue',
                capsize=4, linewidth=2, label='measured (min–max of repeats)')
    ax.axhline(200, color='grey', linestyle=':', label='chance (200/400)')
    ax.axhline(exact_score, color='seagreen', linestyle='--',
               label=f'exact expectation ({exact_score}/400)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(list(budgets))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('shots per grid cell')
    ax.set_ylabel('mismatch count vs p_ref')
    ax.set_title('Readout cost: the XOR boundary is real but faint')
    ax.legend()

    thr = (exact_array.max() + exact_array.min()) / 2
    margins = np.abs(exact_array - thr).ravel()
    p_mid = thr
    ax = axes[1]
    ax.hist(margins, bins=40, color='steelblue', alpha=0.8)
    for b, ls in zip(budgets, (':', '--', '-.', '-')):
        sigma = np.sqrt(p_mid * (1 - p_mid) / b)
        ax.axvline(sigma, color='crimson', linestyle=ls, linewidth=1.5,
                   label=f'1σ shot noise @ {b} shots ({sigma:.4f})')
    ax.set_xlabel('|expectation − threshold|  (cell margin)')
    ax.set_ylabel('grid cells')
    ax.set_title(f'Cell margins of the best model (median '
                 f'{np.median(margins):.4f})')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/professor_mcmc_shot_budget.png', dpi=150,
                bbox_inches='tight')
    print("Saved → figures/professor_mcmc_shot_budget.png")


def survey_style_map(array, fname):
    # single-panel expectation map in the survey house style, matching the
    # professor's best-model figure: autoscaled RdBu, plain theta axes
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(phases20, phases20, array, shading='auto', cmap='RdBu')
    fig.colorbar(im, ax=ax, label='expectation')
    ax.set_xlabel('θ1')
    ax.set_ylabel('θ2')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {fname}")


def fig_survey_style_maps():
    d = np.load('results/professor_mcmc_search.npz')
    survey_style_map(d['array'], 'figures/professor_mcmc_best_map.png')

    ds = np.load('results/professor_mcmc_shotbased.npz')
    survey_style_map(ds['hi_grids'].mean(axis=0),
                     'figures/professor_mcmc_shotbased_map.png')


def fig_best_model_measured(hi_shots=16384, n_repeats=3):
    d = np.load('results/professor_mcmc_search.npz')
    update_neuron_params([int(v) for v in d['params_old']])

    grids, scores = [], []
    for _ in range(n_repeats):
        arr, _, _ = evaluate_array(shots=hi_shots)
        grids.append(arr)
        scores.append(score(arr))
    print(f"best model, {hi_shots}-shot measured scores: {scores} "
          f"(exact: {int(d['results_old'])})")

    np.savez('results/professor_mcmc_best_measured.npz',
             grids=np.array(grids), scores=scores, hi_shots=hi_shots,
             params=d['params_old'])
    survey_style_map(np.mean(grids, axis=0),
                     'figures/professor_mcmc_best_map_measured.png')


if __name__ == "__main__":
    fig_formula_verification()
    fig_shot_budget()
    fig_survey_style_maps()
    fig_best_model_measured()
    print("Done.")
