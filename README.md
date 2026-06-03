# Quantum SURP 2026

Exploring Quantum Machine Learning: Simulating a Quantum Neuron with Qiskit  
**Mentor:** Shing Chi Leung | **Student:** Joseph Scholl

## Structure

```
src/                      # All quantum circuit python files
figures/                  # Saved plots (matplotlib)
data/                     # Training data
notes/                    # Reading notes, scratch thoughts
```

## Setup

```bash
# Clone and enter
git clone <https://www.github.com/SquattingSlaav/SURP>
cd SURP26

# Create env and install deps (uv handles everything)
uv sync

# Run anything
uv run python src/part2.py
```

## Saving figures

Plots are saved to `figures/` rather than shown inline:

```python
import matplotlib.pyplot as plt
plt.savefig("figures/teleportation_accuracy.png", dpi=150, bbox_inches="tight")
```
