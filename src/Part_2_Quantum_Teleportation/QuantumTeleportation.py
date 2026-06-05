from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

alice1 = QuantumRegister(1, "alice1")
alice2 = QuantumRegister(1, "alice2")
bob1 = QuantumRegister(1, "bob1")
c0 = ClassicalRegister(1, "c0")
c1 = ClassicalRegister(1, "c1")
c2 = ClassicalRegister(1, "c2")

def build_circuit():
    qc = QuantumCircuit(alice1, alice2, bob1, c0, c1, c2)
    theta = 2 * np.arccos(1/np.sqrt(3))
    qc.ry(theta, alice1)
    qc.h(bob1)
    qc.cx(bob1, alice2)
    qc.cx(alice1, alice2)
    qc.h(alice1)
    qc.measure(alice1, c0)
    qc.measure(alice2, c1)
    with qc.if_test((c1, 1)):
        qc.x(bob1)
    with qc.if_test((c0, 1)):
        qc.z(bob1)
    qc.measure(bob1, c2)
    return qc
    qc.draw("mpl").savefig("figures/teleportation_circuit.png", dpi=150, bbox_inches="tight")

def bob_stats(counts):
    zero = sum(v for k, v in counts.items() if k[0] == '0')
    one  = sum(v for k, v in counts.items() if k[0] == '1')
    total = zero + one
    return zero/total, one/total

def filter_bob(counts):
    bob_counts = {'0': 0, '1': 0}
    for k, v in counts.items():
        bob_counts[k[0]] += v
    return bob_counts

def run(noise=False, shots=1024, error_rate=0.01):
    qc = build_circuit()
    if noise:
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ['h', 'ry'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(error_rate, 2), ['cx'])
        sim = AerSimulator(noise_model=noise_model)
    else:
        sim = AerSimulator()
    compiled = transpile(qc, sim, optimization_level=0)
    result = sim.run(compiled, shots=shots).result()
    return result.get_counts()

print(f"Expected: |0⟩: {1/3:.3f}, |1⟩: {2/3:.3f}\n")

# Part 1: vary shot counts
print("=== Shot count dependence ===")
for shots in [512, 1024, 2048, 4096]:
    counts = run(noise=False, shots=shots)
    z, o = bob_stats(counts)
    print(f"shots={shots:5d} | |0⟩: {z:.3f}, |1⟩: {o:.3f}")

# Part 2: multiple trials at 2048 shots clean
print("\n=== Multiple trials at 2048 shots (clean) ===")
trials = 20
clean_results = [bob_stats(run(noise=False, shots=2048))[1] for _ in range(trials)]
print(f"|1⟩ mean: {np.mean(clean_results):.3f}, std dev: {np.std(clean_results):.3f}")

# Part 3: multiple trials at 2048 shots noisy
print("\n=== Multiple trials at 2048 shots (noisy) ===")
noisy_results = [bob_stats(run(noise=True, shots=2048))[1] for _ in range(trials)]
print(f"|1⟩ mean: {np.mean(noisy_results):.3f}, std dev: {np.std(noisy_results):.3f}")

# Part 4: vary error rates for both clean and noisy
print("\n=== Error rate dependence ===")
error_rates = [0.01, 0.05, 0.10, 0.20]
clean_means, clean_stds = [], []
noisy_means, noisy_stds = [], []

for rate in error_rates:
    c_trials = [bob_stats(run(noise=False, shots=2048))[1] for _ in range(trials)]
    n_trials = [bob_stats(run(noise=True, shots=2048, error_rate=rate))[1] for _ in range(trials)]
    clean_means.append(np.mean(c_trials))
    clean_stds.append(np.std(c_trials))
    noisy_means.append(np.mean(n_trials))
    noisy_stds.append(np.std(n_trials))
    print(f"error={rate:.2f} | clean: {clean_means[-1]:.3f} noisy: {noisy_means[-1]:.3f}")

# Plot error rate dependence
plt.figure()
plt.errorbar(error_rates, noisy_means, yerr=noisy_stds, fmt='o-', capsize=5, label='Noisy')
plt.errorbar(error_rates, clean_means, yerr=clean_stds, fmt='s-', capsize=5, label='Clean')
plt.axhline(y=2/3, color='r', linestyle='--', label='Expected')
plt.xlabel('Depolarizing Error Rate')
plt.ylabel('|1⟩ Probability')
plt.title('Effect of Noise on Quantum Teleportation')
plt.legend()
plt.savefig('figures/noise_effect.png', dpi=150, bbox_inches='tight')

# Histogram of clean vs noisy counts at 1% error
clean_counts = filter_bob(run(noise=False, shots=2048))
noisy_counts = filter_bob(run(noise=True, shots=2048, error_rate=0.01))
fig = plot_histogram([clean_counts, noisy_counts], legend=['Clean', 'Noisy'],
                     title="Bob's Qubit State: Clean vs Noisy",
                     bar_labels=True)
fig.savefig('figures/histogram.png', dpi=150, bbox_inches='tight')

print("\nPlots saved to figures/")
