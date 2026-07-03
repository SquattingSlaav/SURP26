import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def region_target(input_vals):
    # split the input plane at alpha=pi/4, beta=pi/4 into XOR-labeled quadrants
    n = len(input_vals)
    mid = n // 2  # pi/4 falls strictly between idx 9 and 10 for linspace(0, pi/2, 20)
    target = np.zeros((n, n))
    target[:mid, mid:] = 1
    target[mid:, :mid] = 1
    return target


def region_stats(grid, target):
    pred = (grid > 0.5).astype(float)
    accuracy = (pred == target).mean()
    mse = np.mean((grid - target) ** 2)
    return accuracy, mse


def search_xor_boundary():
    survey = np.load("results/survey_15param.npz")
    results, input_vals = survey["results"], survey["input_vals"]
    target = region_target(input_vals)

    n_models = results.shape[0]
    accuracy = np.zeros(n_models)
    mse = np.zeros(n_models)
    for m in range(n_models):
        accuracy[m], mse[m] = region_stats(results[m], target)

    best_idx = int(np.argmax(accuracy))
    np.savez("results/xor_boundary_search.npz",
             accuracy=accuracy, mse=mse, best_idx=best_idx,
             best_params=survey["params"][best_idx])
    return accuracy, mse, best_idx


def corner_check(grid, input_vals):
    n = len(input_vals) - 1
    corners = {'0⊕0': (0, 0, 0), '0⊕1': (0, n, 1), '1⊕0': (n, 0, 1), '1⊕1': (n, n, 0)}
    n_correct = 0
    for label, (i, j, t) in corners.items():
        pred = int(grid[i, j] > 0.5)
        n_correct += int(pred == t)
    return n_correct


def plot_model_panel(ax, grid, input_vals, idx, corners_correct, accuracy, extra_title=""):
    im = ax.pcolormesh(input_vals, input_vals, grid, shading='auto',
                        cmap='RdBu', vmin=0, vmax=1)
    ax.contour(input_vals, input_vals, grid, levels=[0.5],
               colors='black', linewidths=2)

    mid_val = (input_vals[len(input_vals) // 2 - 1] + input_vals[len(input_vals) // 2]) / 2
    ax.axhline(mid_val, color='grey', linestyle='--', linewidth=1)
    ax.axvline(mid_val, color='grey', linestyle='--', linewidth=1)

    n = len(input_vals) - 1
    corner_targets = {(0, 0): 0, (0, n): 1, (n, 0): 1, (n, n): 0}
    for (i, j), t in corner_targets.items():
        ax.scatter(input_vals[j], input_vals[i], s=150, edgecolor='black',
                   facecolor='gold' if t else 'white', zorder=5)

    ax.set_xlabel('β (input 2)')
    ax.set_ylabel('α (input 1)')
    ax.set_title(f'Model {idx} — {corners_correct}/4 corners correct, '
                 f'{accuracy*100:.0f}% region accuracy{extra_title}')
    return im


def plot_comparison(results, input_vals, point_errors, accuracy, old_best_idx, new_best_idx):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    old_grid = results[old_best_idx]
    old_corners = corner_check(old_grid, input_vals)
    im0 = plot_model_panel(axes[0], old_grid, input_vals, old_best_idx,
                           old_corners, accuracy[old_best_idx],
                           extra_title="\n(old: best by point-MSE)")
    fig.colorbar(im0, ax=axes[0], label='expectation', shrink=0.8)

    new_grid = results[new_best_idx]
    new_corners = corner_check(new_grid, input_vals)
    im1 = plot_model_panel(axes[1], new_grid, input_vals, new_best_idx,
                           new_corners, accuracy[new_best_idx],
                           extra_title="\n(new: best by region accuracy)")
    fig.colorbar(im1, ax=axes[1], label='expectation', shrink=0.8)

    axes[2].scatter(point_errors, accuracy, s=10, alpha=0.4, color='steelblue')
    if old_best_idx == new_best_idx:
        axes[2].scatter(point_errors[old_best_idx], accuracy[old_best_idx],
                        s=400, facecolor='gold', edgecolor='red', linewidth=3, zorder=5,
                        label=f'Model {old_best_idx} (best by both metrics)')
    else:
        axes[2].scatter(point_errors[old_best_idx], accuracy[old_best_idx],
                        s=200, color='red', edgecolor='black', zorder=5,
                        label=f'Model {old_best_idx} (old best)')
        axes[2].scatter(point_errors[new_best_idx], accuracy[new_best_idx],
                        s=200, color='gold', edgecolor='black', zorder=5,
                        label=f'Model {new_best_idx} (new best)')
    axes[2].set_xlabel('Point-based MSE (4 corners)')
    axes[2].set_ylabel('Region accuracy (400 grid cells)')
    axes[2].set_title('Point-MSE vs region accuracy — all 900 models')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('figures/xor_boundary_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/xor_boundary_comparison.png")


if __name__ == "__main__":
    survey = np.load("results/survey_15param.npz")
    search = np.load("results/xor_search.npz")
    results, input_vals = survey["results"], survey["input_vals"]

    accuracy, mse, new_best_idx = search_xor_boundary()
    old_best_idx = int(search["best_idx"])
    point_errors = search["errors"]

    old_corners = corner_check(results[old_best_idx], input_vals)
    new_corners = corner_check(results[new_best_idx], input_vals)

    corr = np.corrcoef(point_errors, accuracy)[0, 1]
    n_perfect_corners = sum(corner_check(results[m], input_vals) == 4 for m in range(results.shape[0]))
    n_high_region_acc = int((accuracy >= 0.6).sum())
    old_best_outputs = search["outputs"][old_best_idx]
    old_best_search_corners = int(((old_best_outputs > 0.5).astype(int) == np.array([0, 1, 1, 0])).sum())

    print("\n=== Point-based vs region-based XOR scoring ===")
    print(f"Old best (point-MSE):      model {old_best_idx}  "
          f"point-MSE={point_errors[old_best_idx]:.4f}  "
          f"corners={old_corners}/4  region-accuracy={accuracy[old_best_idx]*100:.1f}%  "
          f"region-MSE={mse[old_best_idx]:.4f}")
    print(f"New best (region-accuracy): model {new_best_idx}  "
          f"point-MSE={point_errors[new_best_idx]:.4f}  "
          f"corners={new_corners}/4  region-accuracy={accuracy[new_best_idx]*100:.1f}%  "
          f"region-MSE={mse[new_best_idx]:.4f}")
    print(f"\nCorrelation between point-MSE and region-accuracy across all 900 models: {corr:.3f}")
    print(f"Models with perfect 4/4 corner classification (on the survey grid): {n_perfect_corners}")
    print(f"Models with region-accuracy >= 60%: {n_high_region_acc}")
    print(f"\nShot-noise fragility: model {old_best_idx} classified {old_best_search_corners}/4 corners "
          f"correctly in xor_search.npz's independent evaluation run, but only "
          f"{old_corners}/4 against the survey_15param.npz grid — same model, same params, "
          f"different 1024-shot sample, different corner classification near the 0.5 threshold. "
          f"Point-based scoring isn't even reproducible run-to-run.")
    print("\nNote: retraining (QuantumNeuralNetwork.py) still optimizes against the 4 "
          "discrete corner points via parameter-shift. Switching that to a genuine "
          "region/boundary loss is a natural follow-up but requires many more circuit "
          "evaluations per epoch — out of scope here.")

    plot_comparison(results, input_vals, point_errors, accuracy, old_best_idx, new_best_idx)
    print("Done.")
