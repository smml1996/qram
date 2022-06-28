from typing import Optional
from utils import *
from qiskit import QuantumRegister, QuantumCircuit


class QWord:
    """
    This class is a general abstraction of a piece of memory. This class group qubits to represent
    a variable/constant and change its value as required.
    """
    size_in_bits: int
    name: str
    actual: Optional
    previous: Optional
    uncomputation_space: Optional
    is_actual_constant: List[int]
    is_previous_constant: List[int]

    def __init__(self, name, size_in_bits: int = 64):

        self.actual = None
        self.previous = None
        self.uncomputation_space = None
        self.size_in_bits = size_in_bits
        self.name = name

    def __repr__(self):
        return self.name

    def create_state(self, circuit: QuantumCircuit, set_previous=False) -> QuantumRegister:

        if set_previous:
            self.is_previous_constant = [0 for _ in range(self.size_in_bits)]
            self.previous = QuantumRegister(self.size_in_bits, name=self.name)
            circuit.add_register(self.previous)
            return self.previous
        else:
            self.actual = QuantumRegister(self.size_in_bits, name=self.name)
            self.is_actual_constant = [0 for _ in range(self.size_in_bits)]
            circuit.add_register(self.actual)
            return self.actual

    def create_uncomputation_qubits(self, circuit: QuantumCircuit) -> QuantumRegister:
        self.uncomputation_space = QuantumRegister(self.size_in_bits, name="unc-"+self.name)
        circuit.add_register(self.uncomputation_space)
        return self.uncomputation_space

    def is_actual_bit_constant(self, bit_index) -> bool:
        return self.is_actual_constant[bit_index] != -1

    def is_previous_bit_constant(self, bit_index) -> bool:
        return self.is_previous_constant[bit_index] != -1

    def force_current_state(self, current_state, are_constants):
        self.actual = current_state
        self.is_actual_constant = are_constants

    @staticmethod
    def are_all_constants(values):
        for i in values:
            if i == -1:
                return False
        return True




