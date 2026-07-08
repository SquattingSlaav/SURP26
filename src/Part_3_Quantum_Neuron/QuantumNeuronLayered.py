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

def build_2to1_network(alpha, beta, p):
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    q2 = QuantumRegister(1, 'qreg_2')
    q3 = QuantumRegister(1, 'qreg_3')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, q2, q3, cr)

    # neuron 1: q0 -> q1, input alpha (target q1 starts in |0>)
    qc.ry(alpha, q0)
    qc.ry(p[0],  q0)
    qc.cry( p[1], q0, q1)
    qc.rz(np.pi / 2, q1)
    qc.cry(-p[1], q0, q1)
    qc.ry(p[2], q1)

    # neuron 2: q2 -> q3, input beta
    qc.ry(beta,  q2)
    qc.ry(p[3],  q2)
    qc.cry( p[4], q2, q3)
    qc.rz(np.pi / 2, q3)
    qc.cry(-p[4], q2, q3)
    qc.ry(p[5], q3)

    # neuron 3: q1 -> q3 (3-param output neuron: weight, CRy pair, bias —
    # same structure as the 15-param network's output neuron)
    qc.ry(p[6], q1)
    qc.cry( p[7], q1, q3)
    qc.rz(np.pi / 2, q3)
    qc.cry(-p[7], q1, q3)
    qc.ry(p[8], q3)

    qc.measure(q3, cr)
    return qc

def run_circuit(qc):
    counts = sim.run(qc, shots=SHOTS).result().get_counts()
    return counts.get('1', 0) / SHOTS

def survey_2to1(n_models=400):
    input_vals  = np.linspace(0, np.pi / 2, 20)
    param_pool  = np.linspace(0, np.pi / 2, 40)

    # results[model, alpha_idx, beta_idx]
    results = np.zeros((n_models, len(input_vals), len(input_vals)))
    params  = np.zeros((n_models, 9))

    t_start = time.time()
    for m in range(n_models):
        p = np.random.choice(param_pool, size=9)
        params[m] = p

        circuits = [build_2to1_network(alpha, beta, p)
                    for alpha in input_vals for beta in input_vals]
        res = sim.run(circuits, shots=SHOTS).result()
        grid = [res.get_counts(k).get('1', 0) / SHOTS
                for k in range(len(circuits))]
        results[m] = np.reshape(grid, (len(input_vals), len(input_vals)))

        if (m + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"Model {m+1}/{n_models} — {elapsed:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    np.savez("results/survey_2to1.npz", results=results,
             input_vals=input_vals, params=params)
    return results, input_vals, params

def plot_2to1(results, input_vals):
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
    fig.suptitle('2→1 network — 400 models (random params)', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/survey_2to1.png', dpi=100, bbox_inches='tight')
    print("Saved → figures/survey_2to1.png")

if __name__ == "__main__":
    results, input_vals, params = survey_2to1(n_models=400)
    data = np.load("results/survey_2to1.npz")
    plot_2to1(data["results"], data["input_vals"])
    build_2to1_network(np.pi/4, np.pi/4,
                       np.random.choice(np.linspace(0, np.pi/2, 40), size=9)
                       ).draw("mpl").savefig(
        "figures/circuit_2to1.png", dpi=150, bbox_inches="tight")
    print("Done.")
