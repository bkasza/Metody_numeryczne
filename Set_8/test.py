#%%
from qiskit import (ClassicalRegister, QuantumRegister, QuantumCircuit, transpile)
# %%
q = QuantumRegister(2, name = 'qubits')
c = ClassicalRegister(2, name = 'bits')
circuit = QuantumCircuit(q, c)
# %%
from qiskit_aer import Aer
# %%
