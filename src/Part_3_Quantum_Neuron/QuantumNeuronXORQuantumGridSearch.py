import numpy as np
import matplotlib.pyplot as plt
import os
import time

from QuantumNeuronXORRegionTrain import (
    build_15param_network, region_target, region_training_points, region_stats,
    sim, SHOTS, plot_panel,
)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

SHIFT = np.pi / 2
LRS = [0.05, 0.1, 0.2, 0.3, 0.5]
RANK_EPOCHS = 300
EXTENDED_EPOCHS = 1200
GRID_EVAL_EVERY = 5


def batched_expectations(circuits, shots):
    # one sim.run call for a whole list of circuits, instead of one call each
    result = sim.run(circuits, shots=shots).result()
    return np.array([result.get_counts(i).get('1', 0) / shots for i in range(len(circuits))])


def train_step_batched(p, region_pts, lr, shots=SHOTS):
    # base-point circuits (for loss) + both parameter-shift directions for
    # every (point, param) pair, all submitted in a single sim.run job
    n_pts = len(region_pts)
    n_p = len(p)
    circuits = [build_15param_network(a, b, p) for a, b, _ in region_pts]
    for a, b, _ in region_pts:
        for i in range(n_p):
            p_plus = p.copy(); p_plus[i] += SHIFT
            p_minus = p.copy(); p_minus[i] -= SHIFT
            circuits.append(build_15param_network(a, b, p_plus))
            circuits.append(build_15param_network(a, b, p_minus))

    vals = batched_expectations(circuits, shots)
    outputs = vals[:n_pts]
    shift_vals = vals[n_pts:].reshape(n_pts, n_p, 2)
    gradients = (shift_vals[:, :, 0] - shift_vals[:, :, 1]) / 2

    targets = np.array([t for _, _, t in region_pts])
    loss = np.mean((outputs - targets) ** 2)
    grad = np.mean(2 * (outputs - targets)[:, None] * gradients, axis=0)
    p_new = p - lr * grad
    return p_new, loss


def evaluate_full_grid_batched(p, input_vals, shots=SHOTS):
    circuits = [build_15param_network(a, b, p) for a in input_vals for b in input_vals]
    vals = batched_expectations(circuits, shots)
    n = len(input_vals)
    return vals.reshape(n, n)


def first_crossing(epoch_idx, acc_vals, threshold):
    acc_vals = np.array(acc_vals)
    hit = np.where(acc_vals >= threshold)[0]
    if len(hit) == 0:
        return -1
    return int(epoch_idx[hit[0]])


def time_pilot(region_pts, input_vals, n_epochs=20):
    print(f"Timing pilot: {n_epochs} training epochs + 1 full-grid eval, batched...")
    p = np.random.default_rng(0).uniform(0, np.pi / 2, 13)
    t0 = time.time()
    for _ in range(n_epochs):
        p, loss = train_step_batched(p, region_pts, lr=0.2)
    t_train = time.time() - t0
    print(f"  {n_epochs} epochs: {t_train:.1f}s -> {t_train/n_epochs:.2f} s/epoch")

    t0 = time.time()
    evaluate_full_grid_batched(p, input_vals)
    t_grid = time.time() - t0
    print(f"  1 full-grid (400-cell) eval: {t_grid:.1f}s")
    return t_train / n_epochs, t_grid


def run_ranking_phase(input_vals, target, region_pts, warm_start):
    print(f"\nRanking phase: {len(LRS)} lrs x 3 restarts, {RANK_EPOCHS} epochs each...")
    rng = np.random.default_rng(0)
    restarts = [warm_start, rng.uniform(0, np.pi / 2, 13), rng.uniform(0, np.pi / 2, 13)]

    combos = [(lr, r) for lr in LRS for r in range(len(restarts))]
    ranking_acc = np.zeros(len(combos))
    final_params = [None] * len(combos)

    t_start = time.time()
    for ci, (lr, r) in enumerate(combos):
        p = restarts[r].copy()
        for _ in range(RANK_EPOCHS):
            p, loss = train_step_batched(p, region_pts, lr)
        grid = evaluate_full_grid_batched(p, input_vals)
        acc, _ = region_stats(grid, target)
        ranking_acc[ci] = acc
        final_params[ci] = p
        print(f"[{ci+1}/{len(combos)}] lr={lr} restart={r} region_acc={acc:.3f} "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

    best_idx = int(np.argmax(ranking_acc))
    best_lr, best_restart = combos[best_idx]
    print(f"\nRanking-phase winner: lr={best_lr} restart={best_restart} "
          f"acc={ranking_acc[best_idx]:.3f}")

    np.savez("results/xor_quantum_grid_search.npz",
             lrs=np.array([lr for lr, _ in combos]),
             restart_ids=np.array([r for _, r in combos]),
             ranking_phase_acc=ranking_acc,
             best_combo_idx=best_idx, best_lr=best_lr, best_restart=best_restart)

    return best_lr, final_params[best_idx].copy()


def run_extended_training(p_start, best_lr, input_vals, target, region_pts,
                           extra_epochs=EXTENDED_EPOCHS, eval_every=GRID_EVAL_EVERY):
    print(f"\nExtended training: continuing winning combo (lr={best_lr}) for "
          f"{extra_epochs} more epochs, full-grid eval every {eval_every} epochs...")
    p = p_start.copy()
    epoch_idx = []
    acc_hist = []
    loss_hist = np.zeros(extra_epochs)

    t_start = time.time()
    for epoch in range(extra_epochs):
        p, loss = train_step_batched(p, region_pts, best_lr)
        loss_hist[epoch] = loss
        if epoch % eval_every == 0 or epoch == extra_epochs - 1:
            grid = evaluate_full_grid_batched(p, input_vals)
            acc, _ = region_stats(grid, target)
            epoch_idx.append(epoch + RANK_EPOCHS)  # continue the epoch count from ranking phase
            acc_hist.append(acc)
            if epoch % 100 == 0:
                print(f"  epoch {epoch+RANK_EPOCHS}: region_acc={acc:.3f} "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)

    epoch_idx = np.array(epoch_idx)
    acc_hist = np.array(acc_hist)
    final_grid = evaluate_full_grid_batched(p, input_vals)
    final_acc, final_mse = region_stats(final_grid, target)

    e90 = first_crossing(epoch_idx, acc_hist, 0.90)
    e95 = first_crossing(epoch_idx, acc_hist, 0.95)
    reached_90 = e90 >= 0
    reached_95 = e95 >= 0

    plateaued = (acc_hist[-1] - acc_hist[max(0, len(acc_hist) - 40)]) < 0.005
    if not reached_95:
        status = "plateaus" if plateaued else "still improving"
        print(f"Quantum network does NOT reach 95% within {RANK_EPOCHS + extra_epochs} epochs "
              f"({status}); final accuracy {final_acc*100:.1f}%. Reporting honestly.")

    np.savez("results/xor_quantum_best_extended.npz",
             params=p, loss_history=loss_hist, region_acc_history=acc_hist,
             acc_epoch_idx=epoch_idx, final_grid=final_grid,
             final_region_acc=final_acc, final_region_mse=final_mse,
             epoch_90=e90, epoch_95=e95, reached_90=reached_90, reached_95=reached_95,
             best_lr=best_lr, param_count=len(p),
             eval_every=eval_every, rank_epochs=RANK_EPOCHS)

    print(f"Final region accuracy: {final_acc*100:.1f}%  MSE: {final_mse:.4f}")
    print(f"epoch_90={e90}  epoch_95={e95}  (±{eval_every}-epoch resolution)")

    return p, epoch_idx, acc_hist, loss_hist, final_grid, final_acc, final_mse, e90, e95


def plot_ranking(combos_lrs, combos_restarts, ranking_acc):
    lrs = sorted(set(combos_lrs))
    restarts = sorted(set(combos_restarts))
    heat = np.zeros((len(restarts), len(lrs)))
    for lr, r, acc in zip(combos_lrs, combos_restarts, ranking_acc):
        heat[restarts.index(r), lrs.index(lr)] = acc

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat, cmap='RdBu', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(lrs))); ax.set_xticklabels([str(lr) for lr in lrs])
    ax.set_yticks(range(len(restarts))); ax.set_yticklabels([f'restart {r}' for r in restarts])
    ax.set_xlabel('learning rate')
    ax.set_ylabel('restart')
    ax.set_title(f'Pure-quantum grid search: region accuracy at epoch {RANK_EPOCHS}')
    fig.colorbar(im, ax=ax, label='region accuracy')
    plt.tight_layout()
    plt.savefig('figures/xor_quantum_grid_search.png', dpi=150, bbox_inches='tight')
    print("Saved -> figures/xor_quantum_grid_search.png")


def plot_extended(epoch_idx, acc_hist, loss_hist, final_grid, input_vals, final_acc, e90, e95):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = plot_panel(axes[0], final_grid, input_vals,
                      f'Pure quantum (13-param), region accuracy {final_acc*100:.1f}%')
    fig.colorbar(im0, ax=axes[0], label='expectation', shrink=0.8)

    full_epochs = np.arange(RANK_EPOCHS, RANK_EPOCHS + len(loss_hist))
    axes[1].plot(full_epochs, loss_hist, color='steelblue')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Region MSE loss (16 training points)')
    axes[1].set_title('Extended training loss')

    axes[2].plot(epoch_idx, np.array(acc_hist) * 100, color='darkorange', marker='o', markersize=3)
    axes[2].axhline(90, color='grey', linestyle='--', label='90%')
    axes[2].axhline(95, color='grey', linestyle=':', label='95%')
    if e90 >= 0:
        axes[2].axvline(e90, color='grey', linestyle='--', alpha=0.5)
    if e95 >= 0:
        axes[2].axvline(e95, color='grey', linestyle=':', alpha=0.5)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Region accuracy (%)')
    axes[2].set_title(f'Region accuracy vs. epoch (±{GRID_EVAL_EVERY}-epoch resolution)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('figures/xor_quantum_best_extended.png', dpi=150, bbox_inches='tight')
    print("Saved -> figures/xor_quantum_best_extended.png")


if __name__ == "__main__":
    input_vals = np.linspace(0, np.pi / 2, 20)
    target = region_target(input_vals)
    region_pts = region_training_points(input_vals)
    warm_start = np.load("results/xor_boundary_search.npz")["best_params"].astype(float)

    t_epoch, t_grid = time_pilot(region_pts, input_vals, n_epochs=20)
    n_rank_combos = len(LRS) * 3
    est_rank = n_rank_combos * RANK_EPOCHS * t_epoch + n_rank_combos * t_grid
    est_ext = EXTENDED_EPOCHS * t_epoch + (EXTENDED_EPOCHS // GRID_EVAL_EVERY) * t_grid
    print(f"Estimated ranking phase: {est_rank/60:.1f} min  "
          f"| estimated extended phase: {est_ext/60:.1f} min  "
          f"| total ~{(est_rank+est_ext)/60:.1f} min\n")

    best_lr, p_after_rank = run_ranking_phase(input_vals, target, region_pts, warm_start)
    d = np.load("results/xor_quantum_grid_search.npz")
    plot_ranking(d["lrs"], d["restart_ids"], d["ranking_phase_acc"])

    (p, epoch_idx, acc_hist, loss_hist, final_grid, final_acc, final_mse,
     e90, e95) = run_extended_training(p_after_rank, best_lr, input_vals, target, region_pts)
    plot_extended(epoch_idx, acc_hist, loss_hist, final_grid, input_vals, final_acc, e90, e95)
    print("Done.")
