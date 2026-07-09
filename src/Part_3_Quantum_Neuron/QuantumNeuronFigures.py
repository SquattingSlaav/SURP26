# Figures for the results-review meeting: shot-budget readout cost of the
# best XOR model found by the restart search, run on the single entangled
# 13-param circuit (figures/circuit_15param.png).
import numpy as np
import matplotlib.pyplot as plt
import os

from QuantumNeuronMCMC import (
    update_neuron_params, network_array, score, phases20, SHOTS,
)

os.makedirs("figures", exist_ok=True)


def fig_shot_budget(budgets=(256, 1024, 4096, 16384), n_repeats=3):
    d = np.load('results/mcmc_search.npz')
    best_params = [int(v) for v in d['params_old']]
    exact_array = d['array']
    exact_score = int(d['results_old'])
    update_neuron_params(best_params)

    scores = {}
    for b in budgets:
        scores[b] = []
        for r in range(n_repeats):
            sa = network_array(shots=b)
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
    plt.savefig('figures/mcmc_shot_budget.png', dpi=150,
                bbox_inches='tight')
    print("Saved → figures/mcmc_shot_budget.png")


def survey_style_map(array, fname):
    # single-panel expectation map in the survey house style, matching the
    # reference best-model figure: autoscaled RdBu, plain theta axes
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
    d = np.load('results/mcmc_search.npz')
    survey_style_map(d['array'], 'figures/mcmc_best_map.png')

    ds = np.load('results/mcmc_shotbased.npz')
    survey_style_map(ds['hi_grids'].mean(axis=0),
                     'figures/mcmc_shotbased_map.png')


def fig_best_model_measured(hi_shots=16384, n_repeats=3):
    d = np.load('results/mcmc_search.npz')
    update_neuron_params([int(v) for v in d['params_old']])

    grids, scores = [], []
    for _ in range(n_repeats):
        arr = network_array(shots=hi_shots)
        grids.append(arr)
        scores.append(score(arr))
    print(f"best model, {hi_shots}-shot measured scores: {scores} "
          f"(exact: {int(d['results_old'])})")

    np.savez('results/mcmc_best_measured.npz',
             grids=np.array(grids), scores=scores, hi_shots=hi_shots,
             params=d['params_old'])
    survey_style_map(np.mean(grids, axis=0),
                     'figures/mcmc_best_map_measured.png')


if __name__ == "__main__":
    fig_shot_budget()
    fig_survey_style_maps()
    fig_best_model_measured()
    print("Done.")
