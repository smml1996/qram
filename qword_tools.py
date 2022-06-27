from typing import List
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qword import QWord


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
            circuit.x(y[index])
    return y

def optimized_is_equal(bitset1: QuantumRegister, bitset2: QuantumRegister, result_qword: QWord, constants1: List[int],
                       constants2: List[int], circuit: QuantumCircuit, ancillas: QuantumRegister) -> QuantumRegister:
        assert(bitset1.size == bitset2.size)
        assert(ancillas.size == bitset1.size)

        assert(result_qword.size_in_bits == 1)
        assert(result_qword.is_actual_constant[0]==0)

        dummy_logic_one = ancillas[ancillas.size-1]

        control_qubits = []

        for i in range(bitset1.size):
            if constants1[i] != -1:
                if constants2[i] != -1:
                    if constants1[i] != constants2[i]:
                        return result_qword.actual
                    else:
                        # the elements at this position are equal
                        pass
                else:
                    # bit of the second operand is not constant
                    control_qubits.append(ancillas[i])
                    circuit.cx(bitset2[i], ancillas[i])
                    if constants1[i] == 0:
                        # bit of second operand must be 0
                        circuit.x(ancillas[i])
                    else:
                        # bit of second operand must be 1
                        pass
            elif constants2[i] != -1:
                control_qubits.append(ancillas[i])
                circuit.cx(bitset1[i], ancillas[i])
                if constants2[i] == 0:
                    circuit.x(ancillas[i])
                else:
                    pass
            else:
                # there are no constants
                control_qubits.append(ancillas[i])
                circuit.cx(dummy_logic_one, ancillas[i])
                circuit.cx(bitset1[i], ancillas[i])
                circuit.cx(bitset2[i], ancillas[i])

        if len(control_qubits) > 0:
            gate = MCXGate(len(control_qubits))
            control_qubits.append(result_qword.actual[0])
            circuit.append(gate, control_qubits)
            result_qword.is_actual_constant[0] = -1
        else:
            result_qword.is_actual_constant[0] = 1
            circuit.x(result_qword.actual[0])
        return result_qword.actual


def optimized_bitwise_and(bitset1: QuantumRegister, bitset2: QuantumRegister, result_qword: QWord, constants1: List[int],
                       constants2: List[int], circuit: QuantumCircuit) -> QuantumRegister:

    assert(bitset1.size == bitset2.size)
    assert(bitset1.size == result_qword.size_in_bits)
    for i in range(bitset1.size):
        assert(result_qword.is_actual_constant[i] == 0)
        if bitset1[i] != -1:
            if bitset2[i] != -1:
                if bitset1[i] == bitset2[i] and bitset1[i] == 1:
                    result_qword.is_actual_constant[i] = 1
                    circuit.x(result_qword.actual[i])
            else:
                if bitset1 == 0:
                    pass
                else:
                    circuit.cx(bitset2[i], result_qword.actual[i])
                    result_qword.is_actual_constant[i] = -1

        elif bitset2[i] != -1:
            if bitset2[i] == 0:
                pass
            else:
                circuit.cx(bitset1[i], result_qword.actual[i])
                result_qword.is_actual_constant[i] = -1
        else:
            # there are no constants
            circuit.ccx(bitset1[i], bitset2[i], result_qword.actual[i])
            result_qword.is_actual_constant[i] = -1
    return result_qword.actual

