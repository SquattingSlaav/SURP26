import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time

from QuantumNeuronMCMC import (
    update_neuron_params, evaluate_array, score, phases20, SHOTS,
    plot_result, plot_shotbased,
)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

WARM_START = [1, 7, 10, 26, 6, 35, 17, 12, 6, 37, 36, 26, 1]
T0_LADDER_FULL = [5, 8, 12, 18, 25, 35, 50, 70]
T_MIN = 0.5
POLISH_TRIALS = 500
STEP_CHOICES = [-3, -2, -1, 1, 2, 3]


def reflect(val, lo=0, hi=39):
    if val < lo:
        val = 2 * lo - val
    if val > hi:
        val = 2 * hi - val
    return int(np.clip(val, lo, hi))


def propose(params):
    idx = random.randint(0, 12)
    delta = random.choice(STEP_CHOICES)
    new_params = list(params)
    new_params[idx] = reflect(params[idx] + delta)
    return new_params, idx


def temperature(trial, n_trials, T0, T_min=T_MIN, polish_trials=POLISH_TRIALS):
    main_trials = max(1, n_trials - polish_trials)
    if trial >= main_trials:
        return 0.0
    frac = trial / main_trials
    return T0 * (T_min / T0) ** frac


def accept(new_score, old_score, T):
    if new_score <= old_score:
        return True
    if T <= 1e-9:
        return False
    return random.random() < np.exp(-(new_score - old_score) / T)


# greedy search from this warm start got permanently stuck at 141 mismatches
# (notes/xor_boundary_evaluation.md, "Search results 2026-07-05") — reference
# line for whether annealed MH can escape that trap.
DOCUMENTED_GREEDY_TRAP_SCORE = 141


def make_chain_configs(n_chains, ladder=T0_LADDER_FULL, seed=42):
    # chain 0: warm start at lowest T (near-greedy); chain 1: same warm start
    # at highest T (A/B test of escaping the greedy trap); rest: random starts
    starts = [WARM_START]
    T0s = [ladder[0]]
    if n_chains > 1:
        starts.append(WARM_START)
        T0s.append(ladder[-1])

    remaining = n_chains - len(starts)
    if remaining > 0:
        middle = ladder[1:-1] if len(ladder) > 2 else ladder
        idxs = np.linspace(0, len(middle) - 1, remaining).round().astype(int)
        rng = random.Random(seed)
        for i in idxs:
            T0s.append(middle[int(i)])
            starts.append([rng.randint(0, 39) for _ in range(13)])

    return starts[:n_chains], T0s[:n_chains]


def run_mh_chain(chain_id, start_params, T0, n_trials, snapshot_every=50):
    update_neuron_params(start_params)
    array, g0, g1 = evaluate_array(shots=None)
    cur_score = score(array)
    cur_params = list(start_params)
    best_score = cur_score
    best_params = list(cur_params)
    history = [cur_score]
    snapshots = [(0, list(cur_params), cur_score, array.copy())]

    t_start = time.time()
    for trial in range(n_trials):
        T = temperature(trial, n_trials, T0)
        proposal, idx = propose(cur_params)
        update_neuron_params(proposal)
        g0_arg = None if idx < 5 else g0
        g1_arg = None if 5 <= idx < 10 else g1
        arr, g0n, g1n = evaluate_array(g0_arg, g1_arg, shots=None)
        new_score = score(arr)

        if accept(new_score, cur_score, T):
            cur_params, cur_score = proposal, new_score
            array, g0, g1 = arr, g0n, g1n
            if new_score < best_score:
                best_score, best_params = new_score, list(proposal)
        else:
            update_neuron_params(cur_params)

        history.append(cur_score)
        if (trial + 1) % snapshot_every == 0:
            snapshots.append((trial + 1, list(cur_params), cur_score, array.copy()))

    elapsed = time.time() - t_start
    print(f"Chain {chain_id} (T0={T0}): {history[0]} -> {cur_score} "
          f"(best {best_score}) in {elapsed:.0f}s ({n_trials} trials)", flush=True)
    return {
        "chain_id": chain_id, "T0": T0, "final_params": cur_params, "final_score": cur_score,
        "best_params": best_params, "best_score": best_score, "history": np.array(history),
        "snapshots": snapshots, "elapsed": elapsed,
    }


def run_chains(n_chains, n_trials, snapshot_every=50):
    starts, T0s = make_chain_configs(n_chains)
    if n_chains > 1:
        print(f"Chains 0 and 1 both start from the fixed warm start "
              f"(T0={T0s[0]} vs T0={T0s[1]}) -- same-start, different-temperature "
              f"A/B test against the documented greedy trap "
              f"({DOCUMENTED_GREEDY_TRAP_SCORE} mismatches).")

    results = []
    t_start = time.time()
    for i, (start, T0) in enumerate(zip(starts, T0s)):
        print(f"\n=== Chain {i}/{n_chains-1}: T0={T0} start={start} ===", flush=True)
        results.append(run_mh_chain(i, start, T0, n_trials, snapshot_every))
    print(f"\nAll {n_chains} chains done in {time.time()-t_start:.0f}s "
          f"({(time.time()-t_start)/60:.1f} min)")
    return results


def build_snapshot_pool(results):
    arrays, params, scores, chain_ids, trials = [], [], [], [], []
    for res in results:
        for trial, p, s, arr in res["snapshots"]:
            arrays.append(arr)
            params.append(p)
            scores.append(s)
            chain_ids.append(res["chain_id"])
            trials.append(trial)
    return (np.array(arrays), np.array(params, dtype=int), np.array(scores),
            np.array(chain_ids), np.array(trials))


def honest_validation(best_params):
    update_neuron_params(best_params)
    print(f"\nHonest re-measurement: 5 independent {SHOTS}-shot evaluations...")
    honest_scores = []
    for _ in range(5):
        arr, _, _ = evaluate_array(shots=SHOTS)
        honest_scores.append(score(arr))
    print(f"{SHOTS}-shot re-measured scores: {honest_scores}")

    HI_SHOTS = 16384
    print(f"\nDefinitive measured grids: 3x at {HI_SHOTS} shots...")
    hi_grids, hi_scores = [], []
    for _ in range(3):
        arr, _, _ = evaluate_array(shots=HI_SHOTS)
        hi_grids.append(arr)
        hi_scores.append(score(arr))
    print(f"{HI_SHOTS}-shot scores: {hi_scores}")

    exact_array, _, _ = evaluate_array()
    exact_score = score(exact_array)
    print(f"Exact-expectation score of the winning params: {exact_score}")

    return honest_scores, HI_SHOTS, hi_grids, hi_scores, exact_array, exact_score


def plot_trap_escape(chain0_history, chain0_T0, chain1_history, chain1_T0,
                      fname="figures/mh_vs_greedy.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(chain0_history, color='grey', linewidth=1.2,
            label=f'same warm start, near-greedy (T0={chain0_T0})')
    ax.plot(chain1_history, color='steelblue', linewidth=1.5,
            label=f'same warm start, real MH (T0={chain1_T0})')
    ax.axhline(DOCUMENTED_GREEDY_TRAP_SCORE, color='crimson', linestyle='--',
               label=f'documented greedy trap ({DOCUMENTED_GREEDY_TRAP_SCORE} mismatches)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Mismatch count (current state)')
    ax.set_title('Same warm start, different temperature:\n'
                 'does real MH escape where greedy got stuck?')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved -> {fname}")


if __name__ == "__main__":
    N_CHAINS = 5
    N_TRIALS = 5000
    SNAPSHOT_EVERY = 50

    print(f"Fast validation pass: {N_CHAINS} chains x {N_TRIALS} trials "
          f"(~0.19s/trial exact-statevector -> ~{N_CHAINS*N_TRIALS*0.19/60:.0f} min estimated)")

    results = run_chains(N_CHAINS, N_TRIALS, SNAPSHOT_EVERY)

    best = min(results, key=lambda r: r["best_score"])
    best_chain_id = best["chain_id"]
    print(f"\nBest chain: {best_chain_id} (T0={best['T0']}), "
          f"best_score={best['best_score']}, best_params={best['best_params']}")

    chain_histories = np.array([r["history"] for r in results])
    np.savez("results/mh_chains.npz",
             chain_histories=chain_histories,
             chain_T0=np.array([r["T0"] for r in results]),
             chain_best_scores=np.array([r["best_score"] for r in results]),
             chain_final_params=np.array([r["final_params"] for r in results]),
             best_chain_id=best_chain_id, best_params=best["best_params"],
             best_score=best["best_score"])

    arrays, params, scores, chain_ids, trials = build_snapshot_pool(results)
    np.savez("results/mh_snapshots.npz",
             snapshot_arrays=arrays, snapshot_params=params, snapshot_scores=scores,
             snapshot_chain_id=chain_ids, snapshot_trial=trials,
             chain_T0=np.array([r["T0"] for r in results]))
    print(f"Snapshot pool: {len(scores)} entries saved -> results/mh_snapshots.npz")

    if len(results) > 1:
        print(f"\nSame-start temperature comparison: chain 0 (T0={results[0]['T0']}) "
              f"ended at {results[0]['final_score']} (best {results[0]['best_score']}); "
              f"chain 1 (T0={results[1]['T0']}) ended at {results[1]['final_score']} "
              f"(best {results[1]['best_score']}); documented greedy trap = "
              f"{DOCUMENTED_GREEDY_TRAP_SCORE}")
        plot_trap_escape(results[0]["history"], results[0]["T0"],
                          results[1]["history"], results[1]["T0"])

    honest_scores, hi_shots, hi_grids, hi_scores, exact_array, exact_score = honest_validation(
        best["best_params"])

    np.savez("results/mh_shotbased.npz",
             params_old=best["best_params"], locked_in_score=best["best_score"],
             honest_scores=honest_scores, hi_shots=hi_shots,
             hi_grids=np.array(hi_grids), hi_scores=hi_scores,
             exact_score=exact_score, exact_array=exact_array,
             finals=[r["final_score"] for r in results], best_restart=best_chain_id,
             search_array=exact_array, phases20=phases20,
             history=chain_histories[best_chain_id])

    plot_result(exact_array, best["best_score"], hi_scores[0], chain_histories, best_chain_id,
                fname="figures/mh_search.png")
    plot_shotbased(np.mean(hi_grids, axis=0), hi_shots, hi_scores, best["best_score"],
                   honest_scores, exact_score, chain_histories, best_chain_id,
                   fname="figures/mh_shotbased.png")
    print("Done.")
