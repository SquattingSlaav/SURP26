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

    # neuron 1: q0, q1 -> q2 (params 0-4)
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

    # neuron 2: q3, q4 -> q5 (params 5-9)
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

    # neuron 3: q2 -> q5 (params 10-14)
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

def survey_15param(n_models=900):
    input_vals = np.linspace(0, np.pi / 2, 20)
    param_pool = np.linspace(0, np.pi / 2, 40)

    results = np.zeros((n_models, len(input_vals), len(input_vals)))
    params  = np.zeros((n_models, 13))

    t_start = time.time()
    for m in range(n_models):
        p = np.random.choice(param_pool, size=13)
        params[m] = p

        for i, alpha in enumerate(input_vals):
            for j, beta in enumerate(input_vals):
                qc = build_15param_network(alpha, beta, p)
                results[m, i, j] = run_circuit(qc)

        if (m + 1) % 30 == 0:
            elapsed = time.time() - t_start
            print(f"Model {m+1}/{n_models} — {elapsed:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s ({elapsed/3600:.2f} hr)")
    np.savez("results/survey_15param.npz", results=results,
             input_vals=input_vals, params=params)
    return results, input_vals, params

def plot_15param(results, input_vals):
    n_models = results.shape[0]
    ncols = 30
    nrows = 30
    fig, axes = plt.subplots(nrows, ncols, figsize=(60, 60))
    axes = axes.ravel()

    for m in range(n_models):
        im = axes[m].pcolormesh(input_vals, input_vals, results[m],
                                shading='auto', cmap='RdBu')
        axes[m].set_xticks([])
        axes[m].set_yticks([])

    fig.colorbar(im, ax=axes[:n_models], label='expectation', shrink=0.3)
    fig.suptitle('15-parameter network — 900 models (random params)', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/survey_15param.png', dpi=100, bbox_inches='tight')
    print("Saved → figures/survey_15param.png")

if __name__ == "__main__":
    results, input_vals, params = survey_15param(n_models=900)
    data = np.load("results/survey_15param.npz")
    plot_15param(data["results"], data["input_vals"])
    build_15param_network(np.pi/3, np.pi/6,
                          np.random.choice(np.linspace(0, np.pi/2, 40), size=15)
                          ).draw("mpl").savefig(
        "figures/circuit_15param.png", dpi=150, bbox_inches="tight")
    print("Done.")
