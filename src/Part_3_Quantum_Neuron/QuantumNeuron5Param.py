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

def build_5param_neuron(alpha, beta, p):
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    q2 = QuantumRegister(1, 'qreg_2')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, q2, cr)

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

    qc.measure(q2, cr)
    return qc

def run_circuit(qc):
    counts = sim.run(qc, shots=SHOTS).result().get_counts()
    return counts.get('1', 0) / SHOTS

def survey_5param(n_models=400):
    input_vals = np.linspace(0, np.pi / 2, 20)
    param_pool = np.linspace(0, np.pi / 2, 40)

    # results[model, alpha_idx, beta_idx]
    results = np.zeros((n_models, len(input_vals), len(input_vals)))
    params  = np.zeros((n_models, 5))

    t_start = time.time()
    for m in range(n_models):
        p = np.random.choice(param_pool, size=5)
        params[m] = p

        for i, alpha in enumerate(input_vals):
            for j, beta in enumerate(input_vals):
                qc = build_5param_neuron(alpha, beta, p)
                results[m, i, j] = run_circuit(qc)

        if (m + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"Model {m+1}/{n_models} — {elapsed:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    np.savez("results/survey_5param.npz", results=results,
             input_vals=input_vals, params=params)
    return results, input_vals, params

def plot_5param(results, input_vals):
    n_models = results.shape[0]
    ncols = 20
    nrows = 20
    fig, axes = plt.subplots(nrows, ncols, figsize=(40, 40))
    axes = axes.ravel()

    for m in range(n_models):
        im = axes[m].pcolormesh(input_vals, input_vals, results[m],
                                shading='auto', cmap='RdBu')
        axes[m].set_xticks([])
        axes[m].set_yticks([])

    fig.colorbar(im, ax=axes[:n_models], label='expectation', shrink=0.3)
    fig.suptitle('5-parameter neuron — 400 models (random params)', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/survey_5param.png', dpi=100, bbox_inches='tight')
    print("Saved → figures/survey_5param.png")

if __name__ == "__main__":
    results, input_vals, params = survey_5param(n_models=400)
    data = np.load("results/survey_5param.npz")
    plot_5param(data["results"], data["input_vals"])
    build_5param_neuron(np.pi/3, np.pi/6,
                        [0, np.pi/4, np.pi/3, 2*np.pi/5, 2*np.pi/3]
                        ).draw("mpl").savefig(
        "figures/circuit_5param.png", dpi=150, bbox_inches="tight")
    print("Done.")
