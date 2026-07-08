from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import DensityMatrix, entropy as vn_entropy, state_fidelity, Statevector
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results",  exist_ok=True)

alice1 = QuantumRegister(1, "alice1")
alice2 = QuantumRegister(1, "alice2")
bob1 = QuantumRegister(1, "bob1")
c0 = ClassicalRegister(1, "c0")
c1 = ClassicalRegister(1, "c1")
c2 = ClassicalRegister(1, "c2")

# Alice's input state: ry(theta)|0>, theta = 2*arccos(1/sqrt(3))
# Gives |psi> = (1/sqrt(3))|0> + sqrt(2/3)|1>, i.e. P(0)=1/3, P(1)=2/3
THETA = 2 * np.arccos(1 / np.sqrt(3))
ALICE_INPUT = Statevector([1 / np.sqrt(3), np.sqrt(2 / 3)])

def build_circuit(density_matrix=False):
    """
    density_matrix=False: measure Bob's qubit (normal simulation).
    density_matrix=True:  save Bob's reduced density matrix instead (for entropy/fidelity).
    """
    qc = QuantumCircuit(alice1, alice2, bob1, c0, c1, c2)
    qc.ry(THETA, alice1)
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
    if density_matrix:
        qc.save_density_matrix(qubits=[bob1[0]], label="bob_dm")
    else:
        qc.measure(bob1, c2)
    return qc

def bob_stats(counts):
    zero  = sum(v for k, v in counts.items() if k[0] == '0')
    one   = sum(v for k, v in counts.items() if k[0] == '1')
    total = zero + one
    return zero / total, one / total

def filter_bob(counts):
    bob_counts = {'0': 0, '1': 0}
    for k, v in counts.items():
        bob_counts[k[0]] += v
    return bob_counts

def run(noise=False, shots=1024, error_rate=0.01, density_matrix=False):
    """
    density_matrix=False: returns counts dict (normal simulation).
    density_matrix=True:  returns Bob's DensityMatrix (for entropy/fidelity).
    """
    qc = build_circuit(density_matrix=density_matrix)
    sim_kwargs = {}
    if density_matrix:
        sim_kwargs["method"] = "density_matrix"
    if noise:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ['h', 'ry'])
        nm.add_all_qubit_quantum_error(depolarizing_error(error_rate, 2), ['cx'])
        sim_kwargs["noise_model"] = nm
    sim = AerSimulator(**sim_kwargs)
    compiled = transpile(qc, sim, optimization_level=0)
    result = sim.run(compiled, shots=shots).result()
    if density_matrix:
        return DensityMatrix(result.data()["bob_dm"])
    return result.get_counts()

print(f"Expected: |0⟩: {1/3:.3f}, |1⟩: {2/3:.3f}\n")

# ── Part 1: Shot count dependence ─────────────────────────────────────────────
print("=== Shot count dependence ===")
shot_counts_list = [512, 1024, 2048, 4096]
shot_p1_vals = []
for shots in shot_counts_list:
    counts = run(noise=False, shots=shots)
    z, o = bob_stats(counts)
    shot_p1_vals.append(o)
    print(f"shots={shots:5d} | |0⟩: {z:.3f}, |1⟩: {o:.3f}")

np.savez("results/shot_dependence.npz",
         shots=shot_counts_list,
         p1=shot_p1_vals)

# ── Part 2: Multiple trials at 2048 shots (clean) ─────────────────────────────
print("\n=== Multiple trials at 2048 shots (clean) ===")
trials = 20
clean_results = [bob_stats(run(noise=False, shots=2048))[1] for _ in range(trials)]
print(f"|1⟩ mean: {np.mean(clean_results):.3f}, std dev: {np.std(clean_results):.3f}")

np.savez("results/clean_trials_2048.npz", p1_trials=clean_results)

# ── Part 3: Multiple trials at 2048 shots (noisy) ─────────────────────────────
print("\n=== Multiple trials at 2048 shots (noisy) ===")
noisy_results = [bob_stats(run(noise=True, shots=2048))[1] for _ in range(trials)]
print(f"|1⟩ mean: {np.mean(noisy_results):.3f}, std dev: {np.std(noisy_results):.3f}")

np.savez("results/noisy_trials_2048.npz", p1_trials=noisy_results)

# ── Part 4: Error rate dependence ─────────────────────────────────────────────
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

np.savez("results/error_rate_dependence.npz",
         error_rates=error_rates,
         clean_means=clean_means, clean_stds=clean_stds,
         noisy_means=noisy_means, noisy_stds=noisy_stds)

plt.figure()
plt.errorbar(error_rates, noisy_means, yerr=noisy_stds, fmt='o-', capsize=5, label='Noisy')
plt.errorbar(error_rates, clean_means, yerr=clean_stds, fmt='s-', capsize=5, label='Clean')
plt.axhline(y=2/3, color='r', linestyle='--', label='Expected')
plt.xlabel('Depolarizing Error Rate')
plt.ylabel('|1⟩ Probability')
plt.title('Effect of Noise on Quantum Teleportation')
plt.legend()
plt.savefig('figures/noise_effect.png', dpi=150, bbox_inches='tight')

clean_counts = filter_bob(run(noise=False, shots=2048))
noisy_counts = filter_bob(run(noise=True,  shots=2048, error_rate=0.01))
fig = plot_histogram([clean_counts, noisy_counts], legend=['Clean', 'Noisy'],
                     title="Bob's Qubit State: Clean vs Noisy",
                     bar_labels=True)
fig.savefig('figures/histogram.png', dpi=150, bbox_inches='tight')

build_circuit().draw("mpl").savefig("figures/teleportation_circuit.png", dpi=150, bbox_inches="tight")  # default density_matrix=False
plt.close('all')

# ── Part 5: Von Neumann Entropy and Fidelity vs Error Rate ────────────────────
# Von Neumann entropy S = -Tr(rho log2 rho) on Bob's reduced density matrix.
#   S = 0  →  pure state, no information lost
#   S = 1  →  maximally mixed qubit, total information loss
#
# Fidelity F = <psi|rho_bob|psi> where |psi> is Alice's input state.
#   F = 1  →  Bob's state perfectly matches Alice's input
#   F = 0.5 → Bob has no information about the input (random guess baseline)

print("\n=== Von Neumann Entropy and Fidelity vs Depolarizing Error Rate ===")
print(f"{'Error rate':>12} | {'S (bits)':>10} | {'Fidelity':>10}")
print("-" * 40)

# Include error_rate=0 as noiseless baseline
dm_error_rates = [0.0] + error_rates
entropy_values  = []
fidelity_values = []

for rate in dm_error_rates:
    dm = run(noise=(rate > 0), error_rate=rate, density_matrix=True, shots=4096)
    S  = float(vn_entropy(dm, base=2))
    F  = float(state_fidelity(ALICE_INPUT, dm))
    entropy_values.append(S)
    fidelity_values.append(F)
    label = "clean" if rate == 0.0 else f"{rate:.2f}"
    print(f"{label:>12} | {S:>10.4f} | {F:>10.4f}")

np.savez("results/entropy_fidelity.npz",
         error_rates=dm_error_rates,
         entropy=entropy_values,
         fidelity=fidelity_values)

# Plot entropy and fidelity on the same axes with dual y-axis
fig, ax1 = plt.subplots(figsize=(7, 4))

color_S = 'steelblue'
color_F = 'darkorange'

ax1.set_xlabel('Depolarizing Error Rate')
ax1.set_ylabel('Von Neumann Entropy  S  [bits]', color=color_S)
ax1.plot(dm_error_rates, entropy_values, 'o-', color=color_S, label='Entropy S')
ax1.axhline(1.0, color=color_S, linestyle=':', linewidth=0.8, label='Max mixed (S=1)')
ax1.tick_params(axis='y', labelcolor=color_S)
ax1.set_ylim(-0.05, 1.1)

ax2 = ax1.twinx()
ax2.set_ylabel("Fidelity  F = ⟨ψ|ρ_bob|ψ⟩", color=color_F)
ax2.plot(dm_error_rates, fidelity_values, 's--', color=color_F, label='Fidelity F')
ax2.axhline(0.5, color=color_F, linestyle=':', linewidth=0.8, label='Random baseline (F=0.5)')
ax2.tick_params(axis='y', labelcolor=color_F)
ax2.set_ylim(0.45, 1.05)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

plt.title("Information Loss vs Noise\n(Bob's qubit, Quantum Teleportation)")
fig.tight_layout()
fig.savefig('figures/entropy_fidelity_vs_noise.png', dpi=150, bbox_inches='tight')
plt.close('all')

print("\nAll numerical results saved to results/")
print("All plots saved to figures/")
