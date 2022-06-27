from typing import List, Any
from qiskit import QuantumCircuit
from copy import deepcopy

from qiskit.circuit.library import MCXGate

CHECKPOINT_TYPE = 0
GATE_TYPE = 1

NOT = "X"
CX = "CX"
MCX = "MCX"


class StackElement:
    element_type: int
    gate_name: str
    controls: List[Any]
    target: Any

    def __init__(self, element_type, gate_name, controls, target):
        self.element_type = element_type
        self.gate_name = gate_name
        self.operands = controls
        self.target = target

    def apply_element(self, circuit: QuantumCircuit):
        if type == CHECKPOINT_TYPE:
            return False
        else:
            if self.gate_name == NOT:
                circuit.x(self.target)
            if self.gate_name == CX:
                assert(len(self.operands) == 1)
                circuit.cx(self.operands[0], self.target)
            elif self.gate_name == MCX:
                control_qubits = deepcopy(self.controls)
                gate = MCXGate(len(self.controls))
                control_qubits.append(self.target)
                circuit.append(gate, control_qubits)
            else:
                raise Exception(f"Invalid stack element with gate {self.gate_name}")