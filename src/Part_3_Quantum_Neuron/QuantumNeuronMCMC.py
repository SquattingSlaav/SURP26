from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time

SHOTS = 1024
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sim = AerSimulator()

angles40 = [i * np.pi / 2 / 40 for i in range(40)]  # exact definition, matches the reference spec
phases20 = [i * 0.05 for i in range(20)]  # grid of input *probabilities* M in [0, 0.95]

# domain-extended XOR ground truth: True where (i,j) fall in the *same* half
# (both < 10 or both >= 10), False in the *different*-half quadrants
p_ref = np.array(
    [[True] * 10 + [False] * 10 for _ in range(10)] +
    [[False] * 10 + [True] * 10 for _ in range(10)]
)


# Circuit angles here are 2x reference_M's parameters (weights ry(-2*alpha),
# CRy pair cry(2*beta)...cry(-2*beta), bias ry(2*delta)), and the phase gate
# is rz(pi) rather than this repo's usual rz(pi/2) — reference_M's "Rz(pi/2)"
# uses the e^{i*theta*sigma_z} convention, i.e. Qiskit rz(2*theta). Both are
# deliberate deviations needed to match reference_M to ~1e-16 (see
# verify_against_reference_formula and notes/xor_boundary_evaluation.md).
# The one exception is reference_M's (1,1) term, sin^2(2b1-2b2+d): no
# CRy/phase-gate circuit can realize it since target-qubit RYs commute, so
# branch angles are additive — the circuit gives sin^2(2b1+2b2-d) instead.

def input_angle(M):
    return 2 * np.arcsin(np.sqrt(np.clip(M, 0.0, 1.0)))


class QNeuron5:
    # 2 probability inputs -> 1 output; 5 trainable weights (alpha1, alpha2,
    # beta1, beta2, delta), each drawn from angles40. Circuit built once with
    # Qiskit Parameters and rebound per call.
    def __init__(self, param_values):
        self.p = [Parameter(f'p{i}') for i in range(5)]
        self.m0 = Parameter('m0')
        self.m1 = Parameter('m1')
        self.param_values = list(param_values)
        self.M = None

        qc = QuantumCircuit(3)
        qc.ry(self.m0, 0)
        qc.ry(-2 * self.p[0], 0)
        qc.ry(self.m1, 1)
        qc.ry(-2 * self.p[1], 1)
        qc.cry(2 * self.p[2], 0, 2)
        qc.cry(2 * self.p[3], 1, 2)
        qc.rz(np.pi, 2)
        qc.cry(-2 * self.p[3], 1, 2)
        qc.cry(-2 * self.p[2], 0, 2)
        qc.ry(2 * self.p[4], 2)
        self.qc = qc

        mqc = qc.copy()
        cr = ClassicalRegister(1, 'cr')
        mqc.add_register(cr)
        mqc.measure(2, cr[0])
        self.qc_meas = mqc

    def set_params(self, param_values):
        self.param_values = list(param_values)

    def _bindings(self, M1, M2):
        bindings = {self.m0: input_angle(M1), self.m1: input_angle(M2)}
        for i, p in enumerate(self.p):
            bindings[p] = self.param_values[i]
        return bindings

    def bind_meas(self, M1, M2):
        return self.qc_meas.assign_parameters(self._bindings(M1, M2))

    def get_expectation(self, M1, M2, shots=None):
        if shots is None:
            bound = self.qc.assign_parameters(self._bindings(M1, M2))
            self.M = Statevector.from_instruction(bound).probabilities([2])[1]
        else:
            counts = sim.run(self.bind_meas(M1, M2), shots=shots).result().get_counts()
            self.M = counts.get('1', 0) / shots
        return self.M


class QNeuron3b:
    # 2 probability inputs -> 1 output; 3 trainable weights (beta1, beta2,
    # delta). The bare core of QNeuron5 without the two input-weight
    # rotations, matching "the 3 neuron behaves similarly": 3 params, 2 inputs.
    def __init__(self, param_values):
        self.p = [Parameter(f'p{i}') for i in range(3)]
        self.m0 = Parameter('m0')
        self.m1 = Parameter('m1')
        self.param_values = list(param_values)
        self.M = None

        qc = QuantumCircuit(3)
        qc.ry(self.m0, 0)
        qc.ry(self.m1, 1)
        qc.cry(2 * self.p[0], 0, 2)
        qc.cry(2 * self.p[1], 1, 2)
        qc.rz(np.pi, 2)
        qc.cry(-2 * self.p[1], 1, 2)
        qc.cry(-2 * self.p[0], 0, 2)
        qc.ry(2 * self.p[2], 2)
        self.qc = qc

        mqc = qc.copy()
        cr = ClassicalRegister(1, 'cr')
        mqc.add_register(cr)
        mqc.measure(2, cr[0])
        self.qc_meas = mqc

    def set_params(self, param_values):
        self.param_values = list(param_values)

    def _bindings(self, M1, M2):
        bindings = {self.m0: input_angle(M1), self.m1: input_angle(M2)}
        for i, p in enumerate(self.p):
            bindings[p] = self.param_values[i]
        return bindings

    def bind_meas(self, M1, M2):
        return self.qc_meas.assign_parameters(self._bindings(M1, M2))

    def get_expectation(self, M1, M2, shots=None):
        if shots is None:
            bound = self.qc.assign_parameters(self._bindings(M1, M2))
            self.M = Statevector.from_instruction(bound).probabilities([2])[1]
        else:
            counts = sim.run(self.bind_meas(M1, M2), shots=shots).result().get_counts()
            self.M = counts.get('1', 0) / shots
        return self.M


def reference_M(M1, M2, a1, a2, b1, b2, d, additive_11=False):
    # the reference analytic qneuron5.get_expectation, verbatim; with
    # additive_11=True the (1,1) term is replaced by the circuit-realizable
    # additive form sin^2(2b1 + 2b2 - d)
    M1_00 = 1.0 - M1
    M1_01 = np.sqrt(M1 * (1.0 - M1))
    M2_00 = 1.0 - M2
    M2_01 = np.sqrt(M2 * (1.0 - M2))

    rho00A = M1_00 * np.cos(a1)**2 + M1 * np.sin(a1)**2 + M1_01 * np.sin(2.0 * a1)
    rho11A = M1_00 * np.sin(a1)**2 + M1 * np.cos(a1)**2 - M1_01 * np.sin(2.0 * a1)
    rho00B = M2_00 * np.cos(a2)**2 + M2 * np.sin(a2)**2 + M2_01 * np.sin(2.0 * a2)
    rho11B = M2_00 * np.sin(a2)**2 + M2 * np.cos(a2)**2 - M2_01 * np.sin(2.0 * a2)

    t11 = np.sin(2 * b1 + 2 * b2 - d)**2 if additive_11 else np.sin(2 * b1 - 2 * b2 + d)**2
    return (rho00A * rho00B * np.sin(d)**2 +
            rho11A * rho00B * np.sin(2 * b1 - d)**2 +
            rho00A * rho11B * np.sin(2 * b2 - d)**2 +
            rho11A * rho11B * t11)


def verify_against_reference_formula(n_draws=300, seed=0):
    # the stated key requirement: the circuit's measurement
    # expectation must equal the reference analytic self.M
    rng = np.random.default_rng(seed)
    errs_as_written, errs_additive = [], []
    for _ in range(n_draws):
        M1, M2 = rng.uniform(0, 1, 2)
        w = rng.uniform(0, np.pi / 2, 5)
        qn = QNeuron5(w)
        circ = qn.get_expectation(M1, M2)
        errs_as_written.append(abs(circ - reference_M(M1, M2, *w)))
        errs_additive.append(abs(circ - reference_M(M1, M2, *w, additive_11=True)))

    print("=== Verification: circuit expectation vs reference self.M ===")
    print(f"vs the reference formula as written:  max|dM| = {max(errs_as_written):.2e}  "
          f"mean = {np.mean(errs_as_written):.2e}")
    print(f"vs formula with additive (1,1):   max|dM| = {max(errs_additive):.2e}  "
          f"mean = {np.mean(errs_additive):.2e}")
    print("(the residual against the as-written formula is confined to the reference "
          "(1,1) term,\n which no CRy/phase-gate circuit can realize — see notes)\n")
    return max(errs_additive)


qn5_0 = QNeuron5([angles40[i] for i in [1, 7, 10, 26, 6]])
qn5_1 = QNeuron5([angles40[i] for i in [35, 17, 12, 6, 37]])
qn3_0 = QNeuron3b([angles40[i] for i in [36, 26, 1]])


def update_neuron_params(params):
    qn5_0.set_params([angles40[i] for i in params[0:5]])
    qn5_1.set_params([angles40[i] for i in params[5:10]])
    qn3_0.set_params([angles40[i] for i in params[10:13]])


def batched_expectations(qn, input_pairs, shots):
    # one sim.run call for a whole list of (M1, M2) inputs — the same
    # per-circuit measurement records as looping get_expectation, minus the
    # per-job overhead
    circuits = [qn.bind_meas(M1, M2) for M1, M2 in input_pairs]
    result = sim.run(circuits, shots=shots).result()
    return np.array([result.get_counts(i).get('1', 0) / shots
                     for i in range(len(circuits))])


def neuron_grid(qn, shots=None):
    n = len(phases20)
    if shots is not None:
        pairs = [(t0, t1) for t0 in phases20 for t1 in phases20]
        return batched_expectations(qn, pairs, shots).reshape(n, n)
    g = np.zeros((n, n))
    for i, theta0 in enumerate(phases20):
        for j, theta1 in enumerate(phases20):
            g[i, j] = qn.get_expectation(theta0, theta1)
    return g


def evaluate_array(g0=None, g1=None, shots=None):
    # pass a cached layer-1 grid as g0/g1 to skip recomputing that neuron
    if g0 is None:
        g0 = neuron_grid(qn5_0, shots=shots)
    if g1 is None:
        g1 = neuron_grid(qn5_1, shots=shots)
    n = len(phases20)
    if shots is not None:
        pairs = list(zip(g0.ravel(), g1.ravel()))
        array = batched_expectations(qn3_0, pairs, shots).reshape(n, n)
    else:
        array = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                array[i, j] = qn3_0.get_expectation(g0[i, j], g1[i, j])
    return array, g0, g1


def score(array):
    threshold = (np.max(array) + np.min(array)) / 2
    mask = array > threshold
    return int(np.sum(np.logical_xor(mask, p_ref)))


def run_mcmc(n_trials, params_old, verbose_every=500, patience=None, shots=None):
    update_neuron_params(params_old)
    best_array, g0_old, g1_old = evaluate_array(shots=shots)
    results_old = score(best_array)
    start_score = results_old
    print(f"Initial score at warm-start params: {results_old} mismatches", flush=True)
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
        # only one neuron's params changed — reuse the other layer-1 grid(s)
        g0 = None if idx_choice < 5 else g0_old
        g1 = None if 5 <= idx_choice < 10 else g1_old
        array, g0, g1 = evaluate_array(g0, g1, shots=shots)
        results = score(array)

        if results < results_old:
            print(f"good {trial} {idx_choice} {sgn_choice} {results} {results_old} {params}",
                  flush=True)
            params_old = [v for v in params]
            results_old = results
            best_array = array
            g0_old, g1_old = g0, g1
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
    axes[0].set_title(f'Best model, restart {best_restart} '
                      f'(exact mismatches={results_old}, {SHOTS}-shot check={shot_score})')

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
    verify_against_reference_formula()

    # every search score below comes from sampled measurement records
    best, histories, finals, total = run_restarts(n_restarts=5, n_trials=3000,
                                                  patience=500, shots=SHOTS)

    print(f"\nAll restarts done in {total:.1f}s ({total/60:.2f} min)")
    print(f"Final scores per restart: {finals}")
    print(f"Best (locked-in): {best['score']} mismatches (restart {best['restart']})")
    print(f"Best params (indices): {best['params']}")

    # QuantumNeuronFigures.py reads these back in
    np.savez("results/mcmc_search.npz",
             params_old=best['params'], results_old=best['score'], array=best['array'],
             finals=finals, best_restart=best['restart'], phases20=phases20,
             history=np.array(histories[best['restart']]))
    plot_result(best['array'], best['score'], best['score'], histories, best['restart'])

    update_neuron_params(best['params'])

    print(f"\nHonest re-measurement: 5 independent {SHOTS}-shot evaluations...")
    honest_scores = []
    for _ in range(5):
        arr, _, _ = evaluate_array(shots=SHOTS)
        honest_scores.append(score(arr))
    print(f"{SHOTS}-shot re-measured scores: {honest_scores} "
          f"(search locked in {best['score']})")

    HI_SHOTS = 16384
    print(f"\nDefinitive measured grids: 3× at {HI_SHOTS} shots...")
    hi_grids, hi_scores = [], []
    for _ in range(3):
        arr, _, _ = evaluate_array(shots=HI_SHOTS)
        hi_grids.append(arr)
        hi_scores.append(score(arr))
    print(f"{HI_SHOTS}-shot scores: {hi_scores}")

    exact_array, _, _ = evaluate_array()
    exact_score = score(exact_array)
    print(f"Exact-expectation score of the same params (for reference): {exact_score}")

    np.savez("results/mcmc_shotbased.npz",
             params_old=best['params'], locked_in_score=best['score'],
             honest_scores=honest_scores, hi_shots=HI_SHOTS,
             hi_grids=np.array(hi_grids), hi_scores=hi_scores,
             exact_score=exact_score, exact_array=exact_array,
             finals=finals, best_restart=best['restart'],
             search_array=best['array'], phases20=phases20,
             history=np.array(histories[best['restart']]))

    plot_shotbased(np.mean(hi_grids, axis=0), HI_SHOTS, hi_scores,
                   best['score'], honest_scores, exact_score,
                   histories, best['restart'])
    print("Done.")
