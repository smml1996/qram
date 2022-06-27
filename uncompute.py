from typing import List, Any
from qiskit import QuantumCircuit
from copy import deepcopy
from collections import deque

from qiskit.circuit.library import MCXGate

CHECKPOINT_TYPE = 0
GATE_TYPE = 1

X = "X"
CX = "CX"
MCX = "MCX"
CCX = "CCX"


class Element:
    element_type: int
    gate_name: str
    controls: List[Any]
    target: Any

    def __init__(self, element_type, gate_name, controls, target):
        self.element_type = element_type
        self.gate_name = gate_name
        self.operands = controls
        self.target = target

    def apply(self, circuit: QuantumCircuit):
        if type == CHECKPOINT_TYPE:
            return False
        else:
            if self.gate_name == X:
                assert(len(self.operands) == 0)
                circuit.x(self.target)
            elif self.gate_name == CX:
                assert(len(self.operands) == 1)
                circuit.cx(self.operands[0], self.target)
            elif self.gate_name == CCX:
                assert(len(self.operands) == 2)
                circuit.ccx(self.operands[0], self.operands[1], self.target)
            elif self.gate_name == MCX:
                control_qubits = deepcopy(self.controls)
                gate = MCXGate(len(self.controls))
                control_qubits.append(self.target)
                circuit.append(gate, control_qubits)
            else:
                raise Exception(f"Invalid stack element with gate {self.gate_name}")


class Queue:
    data_structure: deque
    size: int
    def __init__(self):
        self.data_structure = deque()
        self.size = 0

    def push(self, element: Element):
        self.push(element)
        self.size += 1

    def is_empty(self):
        return self.size == 0

    def pop(self):
        if self.size == 0:
            return None
        self.size -=1
        return self.data_structure.popleft()


class Stack:
    data_structure: deque
    size: int

    def __init__(self):
        self.data_structure = deque()
        self.size = 0

    def push(self, element: Element):
        self.push(element)
        self.size += 1

    def is_empty(self):
        return self.size == 0

    def pop(self):
        if self.size == 0:
            return None
        self.size -= 1
        return self.data_structure.pop()

