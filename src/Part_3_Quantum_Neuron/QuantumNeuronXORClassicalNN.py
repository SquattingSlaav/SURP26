import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time

from QuantumNeuronXORRegionTrain import (
    region_target, region_training_points, region_stats, plot_panel,
)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

torch.set_num_threads(1)

# Search axes:
#   - architecture: 1 or 2 hidden layers, several widths
#   - preprocessing: raw normalized (alpha,beta) vs. a sin/cos feature map
#     (mirrors the trigonometric structure of the quantum circuits)
#   - batch training: full-batch GD (parity with the quantum parameter-shift
#     script) vs. shuffled mini-batches
ARCHITECTURES = [(4,), (8,), (16,), (32,), (4, 4), (8, 8), (16, 16), (8, 4), (16, 8), (32, 16)]
LRS = [0.05, 0.1, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2]
FEATURE_MODES = ['raw', 'sincos']
BATCH_MODES = ['full', 'mini4']
N_EPOCHS = 3000
FEATURE_DIMS = {'raw': 2, 'sincos': 4}


def build_mlp(hidden_sizes, in_dim):
    layers = []
    d = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        d = h
    layers.append(nn.Linear(d, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def featurize(alpha, beta, mode):
    # alpha, beta: 1D numpy arrays of raw angles in [0, pi/2]
    if mode == 'raw':
        return np.stack([alpha / (np.pi / 2), beta / (np.pi / 2)], axis=1)
    if mode == 'sincos':
        return np.stack([np.sin(alpha), np.cos(alpha), np.sin(beta), np.cos(beta)], axis=1)
    raise ValueError(mode)


def grid_features(input_vals, mode):
    a, b = np.meshgrid(input_vals, input_vals, indexing='ij')
    feats = featurize(a.ravel(), b.ravel(), mode)
    return torch.tensor(feats, dtype=torch.float32)


def eval_grid(model, grid_x, n):
    with torch.no_grad():
        out = model(grid_x).numpy().reshape(n, n)
    return out


def first_crossing(acc_history, threshold):
    idx = np.argmax(np.array(acc_history) >= threshold)
    if acc_history[idx] >= threshold:
        return int(idx)
    return -1


def train_one(hidden_sizes, lr, seed, input_vals, target, region_pts,
              feature_mode='raw', batch_mode='full', n_epochs=N_EPOCHS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_mlp(hidden_sizes, FEATURE_DIMS[feature_mode])

    alphas = np.array([a for a, _, _ in region_pts])
    betas = np.array([b for _, b, _ in region_pts])
    targets = np.array([t for _, _, t in region_pts])
    X_full = torch.tensor(featurize(alphas, betas, feature_mode), dtype=torch.float32)
    y_full = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)
    n_pts = len(region_pts)

    n = len(input_vals)
    grid_x = grid_features(input_vals, feature_mode)
    loss_fn = nn.MSELoss()

    acc_history = np.zeros(n_epochs)
    loss_history = np.zeros(n_epochs)

    batch_size = n_pts if batch_mode == 'full' else 4

    for epoch in range(n_epochs):
        order = rng.permutation(n_pts) if batch_mode != 'full' else np.arange(n_pts)
        for start in range(0, n_pts, batch_size):
            idx = order[start:start + batch_size]
            Xb, yb = X_full[idx], y_full[idx]
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad

        with torch.no_grad():
            full_loss = loss_fn(model(X_full), y_full).item()
        grid = eval_grid(model, grid_x, n)
        acc, _ = region_stats(grid, target)
        acc_history[epoch] = acc
        loss_history[epoch] = full_loss

    return model, acc_history, loss_history


def screen_preprocessing_batching(input_vals, target, region_pts,
                                   screen_arch=(8,), screen_lr=0.2, n_epochs=500):
    print("Stage A: screening preprocessing x batch-training combos "
          f"(arch={screen_arch}, lr={screen_lr}, {n_epochs} epochs, {len(SEEDS)} seeds)...")
    results = {}
    for fmode in FEATURE_MODES:
        for bmode in BATCH_MODES:
            accs = []
            for seed in SEEDS:
                _, acc_hist, _ = train_one(screen_arch, screen_lr, seed, input_vals, target,
                                            region_pts, feature_mode=fmode, batch_mode=bmode,
                                            n_epochs=n_epochs)
                accs.append(acc_hist[-1])
            results[(fmode, bmode)] = np.mean(accs)
            print(f"  feature={fmode:7s} batch={bmode:6s} mean_final_acc={np.mean(accs):.3f}")

    best_combo = max(results, key=results.get)
    print(f"Stage A winner: feature={best_combo[0]}, batch={best_combo[1]} "
          f"(mean_acc={results[best_combo]:.3f})\n")
    return best_combo, results


def run_grid_search(feature_mode, batch_mode, input_vals, target, region_pts):
    print(f"Stage B: architecture x learning-rate grid search "
          f"(feature={feature_mode}, batch={batch_mode})...")
    combos = [(arch, lr) for arch in ARCHITECTURES for lr in LRS]
    final_acc = np.zeros((len(combos), len(SEEDS)))
    epoch_90 = np.zeros((len(combos), len(SEEDS)), dtype=int)
    epoch_95 = np.zeros((len(combos), len(SEEDS)), dtype=int)

    t_start = time.time()
    for ci, (arch, lr) in enumerate(combos):
        for si, seed in enumerate(SEEDS):
            _, acc_hist, _ = train_one(arch, lr, seed, input_vals, target, region_pts,
                                        feature_mode=feature_mode, batch_mode=batch_mode,
                                        n_epochs=800)
            final_acc[ci, si] = acc_hist[-1]
            epoch_90[ci, si] = first_crossing(acc_hist, 0.90)
            epoch_95[ci, si] = first_crossing(acc_hist, 0.95)
        print(f"[{ci+1}/{len(combos)}] arch={arch} lr={lr} "
              f"mean_acc={final_acc[ci].mean():.3f} elapsed={time.time()-t_start:.1f}s", flush=True)

    mean_acc = final_acc.mean(axis=1)
    best_idx = int(np.argmax(mean_acc))
    # tie-break within 1pt toward smaller architecture (fewer params)
    tol = 0.01
    candidates = [i for i in range(len(combos)) if mean_acc[i] >= mean_acc[best_idx] - tol]
    candidates.sort(key=lambda i: sum(combos[i][0]))
    best_idx = candidates[0]
    best_arch, best_lr = combos[best_idx]

    print(f"\nStage B winner: arch={best_arch} lr={best_lr} mean_acc={mean_acc[best_idx]:.3f}")

    depths = np.array([len(a) for a, _ in combos])
    arch_arr = np.array([str(a) for a, _ in combos])
    lr_arr = np.array([lr for _, lr in combos])

    np.savez("results/xor_classical_grid_search.npz",
             archs=arch_arr, lrs=lr_arr, depths=depths,
             final_region_acc=final_acc, epoch_90=epoch_90, epoch_95=epoch_95,
             best_combo_idx=best_idx, best_arch=str(best_arch), best_lr=best_lr,
             feature_mode=feature_mode, batch_mode=batch_mode)

    return best_arch, best_lr, combos, mean_acc


def train_best_full(best_arch, best_lr, feature_mode, batch_mode, input_vals, target, region_pts):
    print(f"\nStage C: training winning combo (arch={best_arch}, lr={best_lr}, "
          f"feature={feature_mode}, batch={batch_mode}) for {N_EPOCHS} epochs...")
    model, acc_hist, loss_hist = train_one(best_arch, best_lr, seed=0,
                                            input_vals=input_vals, target=target,
                                            region_pts=region_pts, feature_mode=feature_mode,
                                            batch_mode=batch_mode, n_epochs=N_EPOCHS)
    n = len(input_vals)
    grid_x = grid_features(input_vals, feature_mode)
    final_grid = eval_grid(model, grid_x, n)
    final_acc, final_mse = region_stats(final_grid, target)

    e90 = first_crossing(acc_hist, 0.90)
    e95 = first_crossing(acc_hist, 0.95)
    reached_90 = e90 >= 0
    reached_95 = e95 >= 0

    plateaued = (acc_hist[-1] - acc_hist[max(0, N_EPOCHS - 200)]) < 0.005
    if not reached_95:
        status = "plateaus" if plateaued else "still improving"
        print(f"Classical NN does NOT reach 95% within {N_EPOCHS} epochs ({status}); "
              f"final accuracy {final_acc*100:.1f}%. Reporting honestly.")

    torch.save(model.state_dict(), "results/xor_classical_best_model.pt")
    np.savez("results/xor_classical_best.npz",
             region_acc_history=acc_hist, loss_history=loss_hist,
             final_grid=final_grid, final_region_acc=final_acc, final_region_mse=final_mse,
             epoch_90=e90, epoch_95=e95, reached_90=reached_90, reached_95=reached_95,
             param_count=param_count(model), arch=str(best_arch), lr=best_lr,
             feature_mode=feature_mode, batch_mode=batch_mode)

    print(f"Final region accuracy: {final_acc*100:.1f}%  MSE: {final_mse:.4f}")
    print(f"epoch_90={e90}  epoch_95={e95}")

    return model, acc_hist, loss_hist, final_grid, final_acc, final_mse, e90, e95


def plot_grid_search(combos, mean_acc, feature_mode, batch_mode):
    archs = sorted(set(a for a, _ in combos), key=lambda a: (len(a), a))
    lrs = sorted(set(lr for _, lr in combos))
    heat = np.zeros((len(archs), len(lrs)))
    for i, (arch, lr) in enumerate(combos):
        heat[archs.index(arch), lrs.index(lr)] = mean_acc[i]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heat, cmap='RdBu', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([str(lr) for lr in lrs])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([str(a) for a in archs])
    ax.set_xlabel('learning rate')
    ax.set_ylabel('architecture (hidden layer sizes)')
    ax.set_title(f'Classical MLP grid search (feature={feature_mode}, batch={batch_mode})\n'
                 'mean final region accuracy (800 epochs, 3 seeds)')
    fig.colorbar(im, ax=ax, label='region accuracy')
    plt.tight_layout()
    plt.savefig('figures/xor_classical_grid_search.png', dpi=150, bbox_inches='tight')
    print("Saved -> figures/xor_classical_grid_search.png")


def plot_best_comparison(final_grid, input_vals, final_acc, loss_history, acc_history):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = plot_panel(axes[0], final_grid, input_vals,
                      f'Classical NN (region accuracy {final_acc*100:.1f}%)')
    fig.colorbar(im0, ax=axes[0], label='output', shrink=0.8)

    axes[1].plot(loss_history, color='steelblue')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE loss (16 training points)')
    axes[1].set_title('Classical NN training loss')

    axes[2].plot(acc_history * 100, color='darkorange')
    axes[2].axhline(90, color='grey', linestyle='--', label='90%')
    axes[2].axhline(95, color='grey', linestyle=':', label='95%')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Region accuracy (%)')
    axes[2].set_title('Classical NN region accuracy vs. epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('figures/xor_classical_best_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved -> figures/xor_classical_best_comparison.png")


if __name__ == "__main__":
    input_vals = np.linspace(0, np.pi / 2, 20)
    target = region_target(input_vals)
    region_pts = region_training_points(input_vals)

    t0 = time.time()
    (feature_mode, batch_mode), _ = screen_preprocessing_batching(input_vals, target, region_pts)

    best_arch, best_lr, combos, mean_acc = run_grid_search(
        feature_mode, batch_mode, input_vals, target, region_pts)
    plot_grid_search(combos, mean_acc, feature_mode, batch_mode)

    (model, acc_hist, loss_hist, final_grid, final_acc, final_mse,
     e90, e95) = train_best_full(best_arch, best_lr, feature_mode, batch_mode,
                                  input_vals, target, region_pts)
    plot_best_comparison(final_grid, input_vals, final_acc, loss_hist, acc_hist)
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("Done.")
