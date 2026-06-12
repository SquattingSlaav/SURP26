import qiskit
import qiskit_aer
import numpy as np
import matplotlib as mpl
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

in = QuantumRegister(1, "input")
anc = QuantumRegister(1, "ancilla")
out = QuantumRegister(1, "output")


