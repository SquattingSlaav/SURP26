from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time

from QuantumNeuron15Param import build_15param_network

SHOTS = 1024
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sim = AerSimulator()

angles40 = [i * np.pi / 2 / 40 for i in range(40)]  # param value grid
phases20 = list(np.linspace(0, np.pi / 2, 20))  # input angles alpha, beta

# domain-extended XOR ground truth: True where (i,j) fall in the *same* half
# (both < 10 or both >= 10), False in the *different*-half quadrants
p_ref = np.array(
    [[True] * 10 + [False] * 10 for _ in range(10)] +
    [[False] * 10 + [True] * 10 for _ in range(10)]
)

class QNeuron13:
    # The entangled 13-param network, built by the single function
    # build_15param_network (matches figures/circuit_15param.png): two
    # 5-param neurons (q0,q1->q2 and q3,q4->q5) feed a 3-param output neuron
    # (q2,q5->q5), all in one circuit with one measurement at the end.
    def __init__(self, param_values):
        self.alpha = Parameter('alpha')
        self.beta = Parameter('beta')
        self.p = [Parameter(f'p{i}') for i in range(13)]
        self.param_values = list(param_values)

        self.qc = build_15param_network(self.alpha, self.beta, self.p, measure=False)
        self.qc_meas = build_15param_network(self.alpha, self.beta, self.p, measure=True)

    def set_params(self, param_values):
        self.param_values = list(param_values)

    def _bindings(self, alpha, beta):
        bindings = {self.alpha: alpha, self.beta: beta}
        for i, p in enumerate(self.p):
            bindings[p] = self.param_values[i]
        return bindings

    def bind_meas(self, alpha, beta):
        return self.qc_meas.assign_parameters(self._bindings(alpha, beta))

    def get_expectation(self, alpha, beta, shots=None):
        if shots is None:
            bound = self.qc.assign_parameters(self._bindings(alpha, beta))
            return Statevector.from_instruction(bound).probabilities([5])[1]
        counts = sim.run(self.bind_meas(alpha, beta), shots=shots).result().get_counts()
        return counts.get('1', 0) / shots


net = QNeuron13([angles40[i] for i in
                 [1, 7, 10, 26, 6, 35, 17, 12, 6, 37, 36, 26, 1]])


def update_neuron_params(params):
    net.set_params([angles40[i] for i in params])


def batched_expectations(input_pairs, shots):
    # one sim.run call for a whole list of (alpha, beta) inputs — the same
    # per-circuit measurement records as looping get_expectation, minus the
    # per-job overhead
    circuits = [net.bind_meas(a, b) for a, b in input_pairs]
    result = sim.run(circuits, shots=shots).result()
    return np.array([result.get_counts(i).get('1', 0) / shots
                     for i in range(len(circuits))])


def network_array(shots=None):
    n = len(phases20)
    if shots is not None:
        pairs = [(a, b) for a in phases20 for b in phases20]
        return batched_expectations(pairs, shots).reshape(n, n)
    array = np.zeros((n, n))
    for i, a in enumerate(phases20):
        for j, b in enumerate(phases20):
            array[i, j] = net.get_expectation(a, b)
    return array


def score(array):
    threshold = (np.max(array) + np.min(array)) / 2
    mask = array > threshold
    return int(np.sum(np.logical_xor(mask, p_ref)))


def run_mcmc(n_trials, params_old, verbose_every=500, patience=None, shots=None):
    update_neuron_params(params_old)
    best_array = network_array(shots=shots)
    results_old = score(best_array)
    start_score = results_old
    print(f"Initial score at warm-start params: {results_old}/{p_ref.size} mismatches",
          flush=True)
    history = [results_old]
    last_accept = 0

    t_start = time.time()
    for trial in range(n_trials):
        if verbose_every and trial % verbose_every == 0:
            print(f"Now running: {trial}  (elapsed {time.time()-t_start:.1f}s)", flush=True)

        idx_choice = random.randint(0, 12)
        sgn_choice = random.randint(0, 1)

        params = [v for v in params_old]
        params[idx_choice] = int(np.clip(params[idx_choice] + (1 if sgn_choice else -1), 0, 39))

        update_neuron_params(params)
        array = network_array(shots=shots)
        results = score(array)

        if results < results_old:
            print(f"good {trial} {idx_choice} {sgn_choice} {results} {results_old} {params}",
                  flush=True)
            params_old = [v for v in params]
            results_old = results
            best_array = array
            last_accept = trial
        else:
            update_neuron_params(params_old)  # revert

        history.append(results_old)

        if patience is not None and trial - last_accept >= patience:
            print(f"No accepted move in {patience} trials — stopping at trial {trial}",
                  flush=True)
            break

    elapsed = time.time() - t_start
    return params_old, results_old, start_score, best_array, history, elapsed


def run_restarts(n_restarts=10, n_trials=3000, patience=500, shots=None):
    # restart 0 is the fixed warm start; the rest are random index vectors
    starts = [[1, 7, 10, 26, 6, 35, 17, 12, 6, 37, 36, 26, 1]]
    starts += [[random.randint(0, 39) for _ in range(13)]
               for _ in range(n_restarts - 1)]

    best = None
    histories, finals = [], []
    t_start = time.time()
    for k, start in enumerate(starts):
        print(f"\n=== Restart {k}/{n_restarts - 1}: start={start} ===", flush=True)
        params, res, s0, arr, hist, el = run_mcmc(
            n_trials, start, verbose_every=0, patience=patience, shots=shots)
        print(f"Restart {k}: {s0} -> {res} in {el:.0f}s ({len(hist)-1} trials)",
              flush=True)
        histories.append(hist)
        finals.append(res)
        if best is None or res < best['score']:
            best = {'score': res, 'params': params, 'array': arr, 'restart': k}

    total = time.time() - t_start
    return best, histories, finals, total


def plot_result(best_array, results_old, shot_score, histories, best_restart, fname=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].pcolormesh(phases20, phases20, best_array, shading='auto', cmap='RdBu')
    fig.colorbar(im0, ax=axes[0], label='expectation')
    axes[0].set_xlabel('θ1 (input probability)')
    axes[0].set_ylabel('θ0 (input probability)')
    axes[0].set_title(f'Best model, restart {best_restart} (mismatches={results_old}'
                      f'/{p_ref.size}, {SHOTS}-shot check={shot_score})')

    axes[1].pcolormesh(phases20, phases20, p_ref, shading='auto', cmap='RdBu')
    axes[1].set_xlabel('θ1 (input probability)')
    axes[1].set_ylabel('θ0 (input probability)')
    axes[1].set_title('p_ref (domain-extended XOR target)')

    for k, hist in enumerate(histories):
        lw, alpha = (2.5, 1.0) if k == best_restart else (1.2, 0.6)
        axes[2].plot(hist, linewidth=lw, alpha=alpha,
                     label=f'restart {k}' + (' (best)' if k == best_restart else ''))
    axes[2].set_xlabel('Trial')
    axes[2].set_ylabel('Mismatch count')
    axes[2].set_title('Greedy search progress per restart (exact scores)')
    axes[2].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fname = fname or 'figures/mcmc_search.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved → {fname}")


def plot_shotbased(mean_hi_grid, hi_shots, hi_scores, locked_in, honest_scores,
                   exact_score, histories, best_restart, fname=None):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    im0 = axes[0].pcolormesh(phases20, phases20, mean_hi_grid, shading='auto',
                             cmap='RdBu')
    fig.colorbar(im0, ax=axes[0], label='measured P(1)')
    axes[0].set_xlabel('θ1 (input probability)')
    axes[0].set_ylabel('θ0 (input probability)')
    axes[0].set_title(f'Best model, mean of {len(hi_scores)}× {hi_shots}-shot '
                      f'measured grids (scores {hi_scores})')

    for k, hist in enumerate(histories):
        lw, alpha = (2.5, 1.0) if k == best_restart else (1.2, 0.6)
        axes[1].plot(hist, linewidth=lw, alpha=alpha,
                     label=f'restart {k}' + (' (best)' if k == best_restart else ''))
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Mismatch count')
    axes[1].set_title(f'Greedy search progress per restart ({SHOTS}-shot scores)')
    axes[1].legend(fontsize=8, ncol=2)

    ax = axes[2]
    x = np.random.default_rng(0).uniform(-0.08, 0.08, len(honest_scores))
    ax.scatter(x, honest_scores, s=60, color='steelblue', zorder=5,
               label=f'independent {SHOTS}-shot re-measurements')
    ax.axhline(locked_in, color='crimson', linestyle='--',
               label=f'locked-in search score ({locked_in})')
    ax.axhline(exact_score, color='seagreen', linestyle=':',
               label=f'exact expectation score ({exact_score})')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel('Mismatch count')
    ax.set_title('Locked-in vs honest re-measured score\n'
                 '(greedy acceptance keeps lucky draws)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    fname = fname or 'figures/mcmc_shotbased.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved → {fname}")


if __name__ == "__main__":
    # search itself runs on the exact statevector (shots=None), matching
    # QuantumNeuronMH.py and the CLAUDE.md convention -- greedy accept/reject
    # must not be driven by shot noise, since results_old is never
    # re-measured once accepted and would otherwise lock in lucky draws
    best, histories, finals, total = run_restarts(n_restarts=5, n_trials=3000,
                                                  patience=500, shots=None)

    print(f"\nAll restarts done in {total:.1f}s ({total/60:.2f} min)")
    print(f"Final scores per restart: {finals}")
    print(f"Best (exact statevector): {best['score']}/{p_ref.size} mismatches "
          f"(restart {best['restart']})")
    print(f"Best params (indices): {best['params']}")

    np.savez("results/mcmc_search.npz",
             params_old=best['params'], results_old=best['score'], array=best['array'],
             finals=finals, best_restart=best['restart'], phases20=phases20,
             history=np.array(histories[best['restart']]))

    update_neuron_params(best['params'])

    print(f"\nHonest re-measurement: 5 independent {SHOTS}-shot evaluations...")
    honest_scores = []
    for _ in range(5):
        arr = network_array(shots=SHOTS)
        honest_scores.append(score(arr))
    print(f"{SHOTS}-shot re-measured scores: {honest_scores} "
          f"(exact-statevector search score {best['score']})")

    # shot_score is a genuine SHOTS-measured value (not the exact search score)
    plot_result(best['array'], best['score'], honest_scores[0], histories, best['restart'])

    HI_SHOTS = 16384
    print(f"\nDefinitive measured grids: 3× at {HI_SHOTS} shots...")
    hi_grids, hi_scores = [], []
    for _ in range(3):
        arr = network_array(shots=HI_SHOTS)
        hi_grids.append(arr)
        hi_scores.append(score(arr))
    print(f"{HI_SHOTS}-shot scores: {hi_scores}")

    # best['array']/best['score'] already are the exact-statevector result of
    # these params (the search ran with shots=None), so reuse them directly
    exact_array, exact_score = best['array'], best['score']
    print(f"Exact-expectation score of the same params (search score, for reference): "
          f"{exact_score}")

    np.savez("results/mcmc_shotbased.npz",
             params_old=best['params'], locked_in_score=best['score'],
             honest_scores=honest_scores,
             hi_shots=HI_SHOTS, hi_grids=np.array(hi_grids), hi_scores=hi_scores,
             exact_score=exact_score,
             exact_array=exact_array,
             finals=finals, best_restart=best['restart'],
             search_array=best['array'], phases20=phases20,
             history=np.array(histories[best['restart']]))

    plot_shotbased(np.mean(hi_grids, axis=0), HI_SHOTS, hi_scores,
                   best['score'], honest_scores, exact_score,
                   histories, best['restart'])
    print("Done.")
