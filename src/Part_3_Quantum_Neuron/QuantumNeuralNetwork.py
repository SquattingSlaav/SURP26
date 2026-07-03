import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import os
import time

SHOTS = 1024
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

sim = AerSimulator()

XOR_INPUTS = [
    (0,        0,        0),
    (0,        np.pi/2,  1),
    (np.pi/2,  0,        1),
    (np.pi/2,  np.pi/2,  0),
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
    qc = build_15param_network(alpha, beta, p)
    return run_circuit(qc)

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

def train_xor(n_epochs=100, lr=0.01):
    # load best params from step 5 as starting point
    data       = np.load("results/xor_search.npz")
    p          = data["best_params"].copy().astype(float)
    targets    = np.array([x[2] for x in XOR_INPUTS], dtype=float)

    loss_history = []

    t_start = time.time()
    for epoch in range(n_epochs):
        outputs   = np.zeros(4)
        gradients = np.zeros((4, len(p)))

        for k, (alpha, beta, target) in enumerate(XOR_INPUTS):
            outputs[k]      = expectation(p, alpha, beta)
            gradients[k]    = compute_gradients(p, alpha, beta)

        loss = np.mean((outputs - targets) ** 2)
        loss_history.append(loss)

        # MSE gradient
        grad = np.zeros(len(p))
        for k in range(4):
            grad += 2 * (outputs[k] - targets[k]) * gradients[k] / 4

        p -= lr * grad

        elapsed = time.time() - t_start
        print(f"Epoch {epoch+1}/{n_epochs} — loss: {loss:.4f} — {elapsed:.1f}s")

    np.savez("results/xor_trained.npz", params=p, loss_history=loss_history)
    return p, loss_history

def plot_training(loss_history, p):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(loss_history, color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training loss')

    # final outputs
    outputs = []
    for alpha, beta, target in XOR_INPUTS:
        outputs.append(expectation(p, alpha, beta))

    labels  = ['0⊕0', '0⊕1', '1⊕0', '1⊕1']
    targets = [0, 1, 1, 0]
    x = np.arange(4)
    ax2.bar(x - 0.2, targets, 0.4, label='Target',  color='steelblue')
    ax2.bar(x + 0.2, outputs, 0.4, label='Output', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Expectation')
    ax2.set_title('Trained model output vs XOR target')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('figures/xor_trained.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/xor_trained.png")

if __name__ == "__main__":
    p, loss_history = train_xor(n_epochs=100, lr=0.01)
    data = np.load("results/xor_trained.npz")
    plot_training(data["loss_history"], data["params"])
    print("Done.")
