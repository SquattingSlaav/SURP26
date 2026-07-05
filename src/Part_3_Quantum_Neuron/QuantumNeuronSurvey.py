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

def build_3param_neuron(alpha, w0, theta1, theta2):
    # single input alpha; 3 trainable params: w0 (input weight),
    # theta1 (CRy pair), theta2 (bias); target q1 starts in |0>
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, cr)

    qc.ry(alpha, q0)
    qc.ry(w0, q0)
    qc.cry(theta1, q0, q1)
    qc.rz(np.pi / 2, q1)
    qc.cry(-theta1, q0, q1)
    qc.ry(theta2, q1)
    qc.measure(q1, cr)
    return qc

def run_circuit(qc):
    counts = sim.run(qc, shots=SHOTS).result().get_counts()
    return counts.get('1', 0) / SHOTS

def survey_3param():
    input_vals = np.linspace(0, np.pi / 2, 20)
    w0_vals    = np.linspace(0, np.pi / 2, 10)
    t1_vals    = np.linspace(0, np.pi / 2, 10)
    t2_vals    = np.linspace(0, np.pi,     10)

    # results[w0_idx, t1_idx, t2_idx, alpha_idx]
    results = np.zeros((10, 10, 10, len(input_vals)))

    t_start = time.time()
    for i, t1 in enumerate(t1_vals):
        for j, t2 in enumerate(t2_vals):
            circuits = [build_3param_neuron(alpha, w0, t1, t2)
                        for w0 in w0_vals for alpha in input_vals]
            res = sim.run(circuits, shots=SHOTS).result()
            vals = [res.get_counts(k).get('1', 0) / SHOTS
                    for k in range(len(circuits))]
            results[:, i, j] = np.reshape(
                vals, (len(w0_vals), len(input_vals)))
        print(f"t1={t1:.2f} done — {time.time()-t_start:.1f}s")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    np.savez("results/survey_3param.npz", results=results,
             input_vals=input_vals, w0_vals=w0_vals,
             t1_vals=t1_vals, t2_vals=t2_vals)
    return results, input_vals, w0_vals, t1_vals, t2_vals

def plot_3param(results, input_vals, w0_vals, t1_vals, t2_vals):
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    for i, t1 in enumerate(t1_vals):
        for j, t2 in enumerate(t2_vals):
            ax = axes[i, j]
            # panel: alpha (x) vs w0 (y)
            im = ax.pcolormesh(input_vals, w0_vals, results[:, i, j],
                               shading='auto', cmap='RdBu')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'θ1={t1:.2f}', fontsize=6)
            if i == 9:
                ax.set_xlabel(f'θ2={t2:.2f}', fontsize=6)

    fig.colorbar(im, ax=axes, label='expectation', shrink=0.5)
    fig.suptitle('3-parameter neuron — 10×10 (θ1, θ2) grid; '
                 'each panel: α (x) × w0 (y)', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/survey_3param.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/survey_3param.png")

if __name__ == "__main__":
    results, input_vals, w0_vals, t1_vals, t2_vals = survey_3param()
    data = np.load("results/survey_3param.npz")
    plot_3param(data["results"], data["input_vals"], data["w0_vals"],
                data["t1_vals"], data["t2_vals"])
    build_3param_neuron(np.pi/2, np.pi/2, np.pi/4, np.pi/6).draw("mpl").savefig(
        "figures/circuit_3param.png", dpi=150, bbox_inches="tight")
    print("Done.")
