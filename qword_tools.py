from typing import List, Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qword import QWord
from uncompute import *
from copy import deepcopy


def optimized_bitwise_not(x: QuantumRegister, y: QuantumRegister,constants_x: List[int], constants_y: List[int],
                          circuit: QuantumCircuit, stack: Stack) -> QuantumRegister:
    assert(y.size == x.size)

    for (index, value) in enumerate(constants_x):
        assert(constants_y[index] == 0)
        if value != -1:
            if value != 1:
                continue
            circuit.x(y[index])
            stack.push(Element(GATE_TYPE, X, [], y[index]))
        else:
            circuit.cx(x[index], y[index])
            circuit.x(y[index])

            stack.push(Element(GATE_TYPE, CX, [x[index]], y[index]))
            stack.push(Element(GATE_TYPE, X, [], y[index]))
    return y

def optimized_is_equal(bitset1: QuantumRegister, bitset2: QuantumRegister, result_qword: QWord, constants1: List[int],
                       constants2: List[int], circuit: QuantumCircuit, ancillas: QuantumRegister, stack: Stack, n:int) -> QuantumRegister:
        assert(bitset1.size == bitset2.size)
        assert(ancillas.size == bitset1.size)

        result_bits, result_constants = result_qword[n]
        assert(result_qword.size_in_bits == 1)
        assert(result_constants[0]==0)

        dummy_logic_one = ancillas[ancillas.size-1]

        control_qubits = []

        for i in range(bitset1.size):
            if constants1[i] != -1:
                if constants2[i] != -1:
                    if constants1[i] != constants2[i]:
                        return result_bits
                    else:
                        # the elements at this position are equal
                        pass
                else:
                    # bit of the second operand is not constant
                    control_qubits.append(ancillas[i])

                    circuit.cx(bitset2[i], ancillas[i])
                    stack.push(Element(GATE_TYPE, CX, [bitset2[i]], ancillas[i]))

                    if constants1[i] == 0:
                        # bit of second operand must be 0
                        circuit.x(ancillas[i])
                        stack.push(Element(GATE_TYPE, X, [], ancillas[i]))
                    else:
                        # bit of second operand must be 1
                        pass
            elif constants2[i] != -1:
                control_qubits.append(ancillas[i])
                circuit.cx(bitset1[i], ancillas[i])
                stack.push(Element(GATE_TYPE, CX, [bitset1[i]], ancillas[i]))
                if constants2[i] == 0:
                    circuit.x(ancillas[i])
                    stack.push(Element(GATE_TYPE, X, [], ancillas[i]))
                else:
                    pass
            else:
                # there are no constants
                control_qubits.append(ancillas[i])
                circuit.cx(dummy_logic_one, ancillas[i])
                circuit.cx(bitset1[i], ancillas[i])
                circuit.cx(bitset2[i], ancillas[i])

                stack.push(Element(GATE_TYPE, CX, [dummy_logic_one], ancillas[i]))
                stack.push(Element(GATE_TYPE, CX, [bitset1[i]], ancillas[i]))
                stack.push(Element(GATE_TYPE, CX, [bitset2[i]], ancillas[i]))


        if len(control_qubits) > 0:
            gate = MCXGate(len(control_qubits))
            stack.push(Element(GATE_TYPE, MCX, control_qubits, result_bits[0]))
            control_qubits.append(result_bits[0])
            circuit.append(gate, control_qubits)
            result_constants[0] = -1
        else:
            result_constants[0] = 1
            circuit.x(result_bits[0])
            stack.push(Element(GATE_TYPE, X, [], result_bits[0]))
        return result_bits


def optimized_bitwise_and(bitset1: QuantumRegister, bitset2: QuantumRegister, result_qword: QWord, constants1: List[int],
                       constants2: List[int], circuit: QuantumCircuit, stack: Stack, n:int) -> QuantumRegister:

    assert(bitset1.size == bitset2.size)
    assert(bitset1.size == result_qword.size_in_bits)
    result_bits, result_constants = result_qword[n]
    for i in range(bitset1.size):
        assert(result_constants[i] == 0)
        if constants1[i] != -1:
            if constants2[i] != -1:
                if constants1[i] == constants2[i] and constants1[i] == 1:
                    result_constants[i] = 1
                    circuit.x(result_bits[i])
                    stack.push(Element(GATE_TYPE, X, [], result_bits[i]))
            else:
                if constants1 == 0:
                    pass
                else:
                    circuit.cx(bitset2[i], result_bits[i])
                    stack.push(Element(GATE_TYPE, CX, [bitset2[i]], result_bits[i]))
                    result_constants[i] = -1

        elif constants2[i] != -1:
            if constants2[i] == 0:
                pass
            else:
                circuit.cx(bitset1[i], result_bits[i])
                stack.push(Element(GATE_TYPE, CX, [bitset1[i]], result_bits[i]))
                result_constants[i] = -1
        else:
            # there are no constants
            circuit.ccx(bitset1[i], bitset2[i], result_bits[i])
            stack.push(Element(GATE_TYPE, CCX, [bitset1[i], bitset2[i]], result_bits[i]))
            result_constants[i] = -1
    return result_qword[n]


def add_one_bitset(bitset: QuantumRegister, constants: List[int], result_qword: QWord, circuit: QuantumCircuit,
                   stack: Stack, current_n: int):
    carry = 0
    sort = len(constants)

    result_qubits, result_constants = result_qword[current_n]
    for (index, qubit) in enumerate(bitset):
        if carry is None or constants[index] == -1 or result_constants[index] == -1:
            result_constants[index] = -1
            carry = None
        else:
            result_constants[index] = (constants[index] + result_constants[index] + carry) % 2
            carry = (constants[index] + result_constants[index] + carry) > 2
        for i in range(index, len(result_qubits)):
            # add control qubits
            qubits = result_qubits[index:(sort - 1 - i)]
            qubits.append(qubit)

            gate = MCXGate(len(qubits))
            stack.push(Element(GATE_TYPE, MCX, qubits, result_qubits[sort - 1 - i]))

            # add target qubit
            qubits.append(result_qubits[sort - 1 - i])
            circuit.append(gate, qubits)

def optimized_get_twos_complement(bitset: QuantumRegister, circuit: QuantumCircuit, global_stack) -> Stack:
    stack = Stack()

    for bit in bitset:
        circuit.x(bit)
        global_stack.push(Element(GATE_TYPE, X, [], bit))

    for i in range(len(bitset)-1):
        qubits = bitset[:len(bitset)-i-1]
        gate = MCXGate(len(qubits))
        stack.push(Element(GATE_TYPE, MCX, qubits, bitset[len(bitset)-i-1]))
        global_stack.push(Element(GATE_TYPE, MCX, qubits, bitset[len(bitset) - i - 1]))
        qubits.append(bitset[len(bitset)-i-1])
        circuit.append(gate, qubits)
    circuit.x(bitset[0])
    stack.push(Element(GATE_TYPE, X, [], bitset[0]))
    global_stack.push(Element(GATE_TYPE, X, [], bitset[0]))
    return stack



def optimized_bitwise_add(bitset1: QuantumRegister, bitset2: QuantumRegister, result_qword: QWord, constants1: List[int],
                          constants2: List[int], circuit: QuantumCircuit, stack: Stack, current_n: int):

    assert(len(bitset1) == len(bitset2))

    add_one_bitset(bitset1, constants1, result_qword, circuit, stack, current_n)
    add_one_bitset(bitset2, constants2, result_qword, circuit, stack, current_n)

def optimized_unsigned_ult(bits1: QuantumRegister, bits2: QuantumRegister, result_qword: QWord,
                           consts1: List[int], consts2: List[int], circuit: QuantumCircuit, ancillas, stack: Stack, n: int):
    result_bits, result_constants = result_qword[n]
    bitset1 = bits1[:]
    bitset1.append(ancillas[0])
    bitset2 = bits2[:]
    bitset2.append(ancillas[1])

    constants1 = deepcopy(consts1)
    constants1.append(0)
    constants2 = deepcopy(consts2)
    constants2.append(0)

    local_stack = optimized_get_twos_complement(bitset2, circuit, stack)

    bits_result_addition = ancillas[2:]
    bits_result_addition.append(result_bits[0])
    assert(len(result_bits) == 1)
    addition_result = QuantumRegister(bits=bits_result_addition)
    addition_qword = QWord(len(bits_result_addition),"ult_add")
    addition_qword.append_state(addition_result, [0 for _ in bits_result_addition], n)


    optimized_bitwise_add(bitset1, bitset2, addition_qword, constants1, constants2, circuit, stack, n)

    while not local_stack.is_empty():
        element = local_stack.pop()
        element.apply()
        stack.push(element)


def optimized_unsigned_ugt(bits1: QuantumRegister, bits2: QuantumRegister, result_qword: QWord,
                           consts1: List[int], consts2: List[int], circuit: QuantumCircuit, ancillas, stack: Stack, n:int):
    optimized_unsigned_ult(bits2, bits1, result_qword,consts2, consts1, circuit, ancillas, stack, n)

def logic_or(controls, result_bit, circuit, stack):
    assert(len(controls)>0)
    for bit in controls:
        circuit.x(bit)
        stack.push(Element(GATE_TYPE, X, [], bit))

    gate = MCXGate(len(controls))
    stack.push(Element(GATE_TYPE, MCX, controls, result_bit))
    controls.append(result_bit)
    circuit.append(gate, controls)

    for bit in controls:
        circuit.x(bit)
        stack.push(Element(GATE_TYPE, X, [], bit))

def optimized_unsigned_ulte(bits1: QuantumRegister, bits2: QuantumRegister, result_qword: QWord,
                           consts1: List[int], consts2: List[int], circuit: QuantumCircuit, ancillas, stack: Stack, n: int):

    assert(len(bits1) == len(bits2))
    assert(len(consts1) == len(bits2))
    result_bits, result_constants = result_qword[n]
    assert(len(result_bits) == 1)
    ancillas_lte = ancillas[:2 + len(bits1)]
    ancillas_eq = ancillas[2 + len(bits1):-2]

    lte_result_qubit = ancillas[len(ancillas)-1]
    eq_result_qubit = ancillas[len(ancillas)-2]
    register_lte = QuantumRegister(bits=lte_result_qubit)
    register_eq = QuantumRegister(bits=eq_result_qubit)
    qword_lte = QWord(1)
    qword_lte.append_state(register_lte, [0], n)
    qword_eq = QWord(1)
    qword_eq.append_state(register_eq, [0], n)

    optimized_unsigned_ult(bits1, bits2, qword_lte, consts1, consts2, circuit, ancillas_lte, stack, n)

    optimized_is_equal(bits1, bits2, qword_eq, consts1,consts2, circuit, ancillas_eq, stack, n)

    logic_or([qword_lte[n][0][0], qword_eq[n][0][0]], result_bits[0], circuit, stack)



def optimized_unsigned_ugte(bits1: QuantumRegister, bits2: QuantumRegister, result_qword: QWord,
                           consts1: List[int], consts2: List[int], circuit: QuantumCircuit, ancillas, stack: Stack, n:int):
    optimized_unsigned_ulte(bits2, bits1, result_qword, consts2, consts1, circuit, ancillas, stack, n)


def multiply_word_by_bit(bits, bit, const_bits, consts_bit, result_word, circuit, shift, stack) -> (Stack, List[int]):
    local_stack = Stack()
    consts_ancillas = []
    i = 0
    while len(consts_ancillas) <= len(bits):
        if i < shift:
            consts_ancillas.append(0)
        else:
            circuit.ccx(bits[i], bit, result_word[i])
            local_stack.push(Element(GATE_TYPE, CCX,  [bits[i], bit], result_word[i]))
            stack.push(Element(GATE_TYPE, CCX, [bits[i], bit], result_word[i]))
        if consts_bit != -1:
            if consts_bit == 0:
                consts_ancillas.append(0)
            else:
                consts_ancillas.append(const_bits[i])
        else:
            if const_bits[i] == 0:
                consts_ancillas.append(0)
            else:
                consts_ancillas.append(-1)
    return local_stack, consts_ancillas

def optimized_mul(bits1: QuantumRegister, bits2: QuantumRegister, result_qword: QWord,
                           consts1: List[int], consts2: List[int], circuit: QuantumCircuit, ancillas, stack: Stack, n: int):

    assert(len(ancillas) == len(result_qword[n]))
    for (index,bit) in enumerate(bits1):
        local_stack, consts_ancillas = multiply_word_by_bit(bits2, bit, ancillas, consts2, consts1[index], circuit, index, stack)

        add_one_bitset(ancillas, consts_ancillas, result_qword, circuit, local_stack, n)
        # make ancillas |0> again
        while not local_stack.is_empty():
            element = local_stack.pop()
            element.apply(circuit)
            stack.push(element)
