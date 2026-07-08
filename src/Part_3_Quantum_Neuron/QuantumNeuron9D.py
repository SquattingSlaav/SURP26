from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
import numpy as np
import matplotlib.pyplot as plt
import os
import time

SHOTS = 1024
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 9 parameters: 3 per neuron (initial Ry, RZR theta, bias)
params = ParameterVector('θ', 9)

def build_circuit():
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    q2 = QuantumRegister(1, 'qreg_2')
    q3 = QuantumRegister(1, 'qreg_3')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, q2, q3, cr)

    # neuron 1: q0 -> q1 (params 0,1,2)
    qc.ry(params[0], q0)
    qc.cry(2 * params[1], q0, q1)
    qc.rz(np.pi / 2, q1)
    qc.cry(-2 * params[1], q0, q1)
    qc.ry(params[2], q1)

    # neuron 2: q2 -> q3 (params 3,4,5)
    qc.ry(params[3], q2)
    qc.cry(2 * params[4], q2, q3)
    qc.rz(np.pi / 2, q3)
    qc.cry(-2 * params[4], q2, q3)
    qc.ry(params[5], q3)

    # neuron 3: q1 -> q3 (params 6,7,8)
    qc.ry(params[6], q1)
    qc.cry(2 * params[7], q1, q3)
    qc.rz(np.pi / 2, q3)
    qc.cry(-2 * params[7], q1, q3)
    qc.ry(params[8], q3)

    qc.measure(q3, cr)
    return qc

def run_circuit(compiled, sim, p_vals):
    bound  = compiled.assign_parameters({params[i]: p_vals[i] for i in range(9)})
    counts = sim.run(bound, shots=SHOTS).result().get_counts()
    return counts.get('1', 0) / SHOTS

def mcmc_sweep(n_samples=10000, step_size=0.3):
    sim      = AerSimulator()
    qc       = build_circuit()
    compiled = transpile(qc, sim, optimization_level=0)

    chain  = np.zeros((n_samples, 9))
    values = np.zeros(n_samples)

    current = np.random.uniform(0, np.pi, 9)
    current_val = run_circuit(compiled, sim, current)

    t_start = time.time()
    accepted = 0
    for i in range(n_samples):
        proposal = current + np.random.normal(0, step_size, 9)
        proposal = np.clip(proposal, 0, np.pi)

        proposal_val = run_circuit(compiled, sim, proposal)

        # uniform target (just exploring), so acceptance is a coin flip, not exp(-dE/T)
        if np.random.rand() < 0.5:
            current     = proposal
            current_val = proposal_val
            accepted   += 1

        chain[i]  = current
        values[i] = current_val

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f"Sample {i+1}/{n_samples} — elapsed {elapsed:.1f}s — acceptance {accepted/(i+1):.2f}")

    elapsed = time.time() - t_start
    print(f"MCMC done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    np.savez("results/mcmc.npz", chain=chain, values=values)
    return chain, values

def plot_pairwise(chain, values):
    n_params = chain.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(16, 16))
    fig.suptitle('MCMC pairwise parameter projections', fontsize=14)

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                ax.hist(chain[:, i], bins=30, color='gray')
                ax.set_xlabel(f'θ_{i}')
            else:
                sc = ax.scatter(chain[:, j], chain[:, i],
                                c=values, cmap='RdBu_r',
                                s=1, alpha=0.3, vmin=0, vmax=1)
                ax.set_xlabel(f'θ_{j}')
                ax.set_ylabel(f'θ_{i}')
            ax.tick_params(labelsize=6)

    fig.colorbar(sc, ax=axes, label='expectation', shrink=0.5)
    plt.tight_layout()
    plt.savefig('figures/mcmc_pairwise.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/mcmc_pairwise.png")

if __name__ == "__main__":
    chain, values = mcmc_sweep(n_samples=10000, step_size=0.3)
    data = np.load("results/mcmc.npz")
    plot_pairwise(data["chain"], data["values"])
    print("Done.")
