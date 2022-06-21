from typing import List

from qiskit import QuantumCircuit
from qiskit import QuantumRegister

def optimized_bitwise_not(x: QuantumRegister, y: QuantumRegister,constants_x: List[int], constants_y: List[int],
                          circuit: QuantumCircuit) -> QuantumRegister:

    assert(y.size == x.size)

    for (index, value) in enumerate(constants_x):
        assert(constants_y[index] == 0)
        if value != -1:
            if value != 1:
                continue
            circuit.x(y[index])
        else:
            circuit.cx(x[index], y[index])
    return y