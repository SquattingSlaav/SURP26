from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import os
import time

SHOTS = 1024
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sim = AerSimulator()


def build_15param_network(alpha, beta, p):
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    q2 = QuantumRegister(1, 'qreg_2')
    q3 = QuantumRegister(1, 'qreg_3')
    q4 = QuantumRegister(1, 'qreg_4')
    q5 = QuantumRegister(1, 'qreg_5')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, q2, q3, q4, q5, cr)

    qc.ry(alpha, q0)
    qc.ry(p[0],  q0)
    qc.ry(beta,  q1)
    qc.ry(p[1],  q1)
    qc.cry( p[2], q0, q2)
    qc.cry( p[3], q1, q2)
    qc.rz(np.pi / 2, q2)
    qc.cry(-p[3], q1, q2)
    qc.cry(-p[2], q0, q2)
    qc.ry(p[4], q2)

    qc.ry(alpha, q3)
    qc.ry(p[5],  q3)
    qc.ry(beta,  q4)
    qc.ry(p[6],  q4)
    qc.cry( p[7], q3, q5)
    qc.cry( p[8], q4, q5)
    qc.rz(np.pi / 2, q5)
    qc.cry(-p[8], q4, q5)
    qc.cry(-p[7], q3, q5)
    qc.ry(p[9], q5)

    qc.ry(p[10], q2)
    qc.cry( p[11], q2, q5)
    qc.rz(np.pi / 2, q5)
    qc.cry(-p[11], q2, q5)
    qc.ry(p[12], q5)

    qc.measure(q5, cr)
    return qc


def run_circuit(qc):
    counts = sim.run(qc, shots=SHOTS).result().get_counts()
    return counts.get('1', 0) / SHOTS


def expectation(p, alpha, beta):
    return run_circuit(build_15param_network(alpha, beta, p))


def parameter_shift(p, alpha, beta, idx):
    shift = np.pi / 2
    p_plus  = p.copy(); p_plus[idx]  += shift
    p_minus = p.copy(); p_minus[idx] -= shift
    return (expectation(p_plus, alpha, beta) - expectation(p_minus, alpha, beta)) / 2


def compute_gradients(p, alpha, beta):
    grads = np.zeros(len(p))
    for i in range(len(p)):
        grads[i] = parameter_shift(p, alpha, beta, i)
    return grads


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


def region_training_points(input_vals):
    # 2 low (<pi/4) + 2 high (>pi/4) representative angles per axis
    low = [input_vals[2], input_vals[7]]
    high = [input_vals[12], input_vals[17]]
    axis_vals = low + high

    points = []
    for a in axis_vals:
        for b in axis_vals:
            a_low = a in low
            b_low = b in low
            target = 0.0 if a_low == b_low else 1.0
            points.append((a, b, target))
    return points  # 16 points, 4 per quadrant


def train_xor_region(n_epochs=150, lr=0.2):
    survey = np.load("results/survey_15param.npz")
    boundary_search = np.load("results/xor_boundary_search.npz")
    p = boundary_search["best_params"].copy().astype(float)

    region_pts = region_training_points(survey["input_vals"])
    targets = np.array([pt[2] for pt in region_pts])
    n_pts = len(region_pts)

    loss_history = []
    t_start = time.time()
    for epoch in range(n_epochs):
        outputs = np.zeros(n_pts)
        gradients = np.zeros((n_pts, len(p)))

        for k, (alpha, beta, target) in enumerate(region_pts):
            outputs[k] = expectation(p, alpha, beta)
            gradients[k] = compute_gradients(p, alpha, beta)

        loss = np.mean((outputs - targets) ** 2)
        loss_history.append(loss)

        grad = np.zeros(len(p))
        for k in range(n_pts):
            grad += 2 * (outputs[k] - targets[k]) * gradients[k] / n_pts
        p -= lr * grad

        elapsed = time.time() - t_start
        print(f"Epoch {epoch+1}/{n_epochs} — region loss: {loss:.4f} — {elapsed:.1f}s")

    np.savez("results/xor_region_trained.npz", params=p, loss_history=loss_history)
    return p, loss_history


def evaluate_full_grid(p, input_vals):
    n = len(input_vals)
    grid = np.zeros((n, n))
    for i, alpha in enumerate(input_vals):
        for j, beta in enumerate(input_vals):
            grid[i, j] = expectation(p, alpha, beta)
    return grid


def plot_panel(ax, grid, input_vals, title):
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
    ax.set_title(title)
    return im


def plot_comparison(old_grid, new_grid, input_vals, old_acc, new_acc, loss_history):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = plot_panel(axes[0], old_grid, input_vals,
                      f'Old: corner-trained (region accuracy {old_acc*100:.0f}%)')
    fig.colorbar(im0, ax=axes[0], label='expectation', shrink=0.8)

    im1 = plot_panel(axes[1], new_grid, input_vals,
                      f'New: region-trained (region accuracy {new_acc*100:.0f}%)')
    fig.colorbar(im1, ax=axes[1], label='expectation', shrink=0.8)

    axes[2].plot(loss_history, color='steelblue')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Region MSE loss (16 training points)')
    axes[2].set_title('Region-loss training curve')

    plt.tight_layout()
    plt.savefig('figures/xor_region_trained_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/xor_region_trained_comparison.png")


if __name__ == "__main__":
    survey = np.load("results/survey_15param.npz")
    input_vals = survey["input_vals"]
    target = region_target(input_vals)

    old_trained = np.load("results/xor_trained.npz")
    old_p = old_trained["params"]
    print("Evaluating old corner-trained model on the full 400-point grid...")
    old_grid = evaluate_full_grid(old_p, input_vals)
    old_acc, old_mse = region_stats(old_grid, target)

    print("\nTraining against the 16-point region loss...")
    p, loss_history = train_xor_region(n_epochs=150, lr=0.2)

    print("\nEvaluating new region-trained model on the full 400-point grid...")
    new_grid = evaluate_full_grid(p, input_vals)
    new_acc, new_mse = region_stats(new_grid, target)

    print("\n=== Corner-trained vs region-trained: full-grid region accuracy ===")
    print(f"Old (corner-trained): region-accuracy={old_acc*100:.1f}%  region-MSE={old_mse:.4f}")
    print(f"New (region-trained): region-accuracy={new_acc*100:.1f}%  region-MSE={new_mse:.4f}")
    if new_acc > old_acc:
        print(f"Region training improved the generalized decision boundary "
              f"by {(new_acc - old_acc)*100:.1f} percentage points.")
    else:
        print(f"Region training did NOT improve region accuracy "
              f"({(new_acc - old_acc)*100:.1f} percentage points) — report this honestly.")

    plot_comparison(old_grid, new_grid, input_vals, old_acc, new_acc, loss_history)
    print("Done.")
