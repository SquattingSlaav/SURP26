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

# XOR truth table mapped to angles
# 0 -> 0, 1 -> pi/2
XOR_INPUTS = [
    (0,        0,        0),  # 0 XOR 0 = 0
    (0,        np.pi/2,  1),  # 0 XOR 1 = 1
    (np.pi/2,  0,        1),  # 1 XOR 0 = 1
    (np.pi/2,  np.pi/2,  0),  # 1 XOR 1 = 0
]

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

    # neuron 3: q2 -> q5 (params 10-12)
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

def evaluate_xor(p):
    p = np.atleast_1d(p)  # add this line
    outputs = []
    for alpha, beta, target in XOR_INPUTS:
        qc  = build_15param_network(alpha, beta, p)
        exp = run_circuit(qc)
        outputs.append(exp)
    return np.array(outputs)

def xor_error(outputs):
    targets = np.array([x[2] for x in XOR_INPUTS], dtype=float)
    return np.mean((outputs - targets) ** 2)

def search_xor():
    data   = np.load("results/survey_15param.npz")
    params = data["params"]
    n_models = params.shape[0]

    errors  = np.zeros(n_models)
    outputs = np.zeros((n_models, 4))

    t_start = time.time()
    for m in range(n_models):
        outputs[m] = evaluate_xor(params[m])
        errors[m]  = xor_error(outputs[m])

        if (m + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"Model {m+1}/{n_models} — best error so far: {errors[:m+1].min():.4f} — {elapsed:.1f}s")

    best_idx = np.argmin(errors)
    print(f"\nBest model: {best_idx} — error: {errors[best_idx]:.4f}")
    print(f"Outputs: {outputs[best_idx]}")
    print(f"Params:  {params[best_idx]}")

    np.savez("results/xor_search.npz", errors=errors, outputs=outputs,
             best_idx=best_idx, best_params=params[best_idx])
    return errors, outputs, best_idx, params[best_idx]

def plot_xor(errors, outputs, best_idx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(errors, alpha=0.5, color='steelblue', linewidth=0.8)
    ax1.axvline(best_idx, color='red', linewidth=1, label=f'Best: model {best_idx}')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MSE')
    ax1.set_title('XOR error across models')
    ax1.legend()

    labels = ['0⊕0', '0⊕1', '1⊕0', '1⊕1']
    targets = [0, 1, 1, 0]
    x = np.arange(4)
    ax2.bar(x - 0.2, targets,          0.4, label='Target',  color='steelblue')
    ax2.bar(x + 0.2, outputs[best_idx], 0.4, label='Output', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Expectation')
    ax2.set_title(f'Best model output vs XOR target')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('figures/xor_search.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/xor_search.png")

if __name__ == "__main__":
    errors, outputs, best_idx, best_params = search_xor()
    data = np.load("results/xor_search.npz")
    plot_xor(data["errors"], data["outputs"], int(data["best_idx"]))
    print("Done.")
