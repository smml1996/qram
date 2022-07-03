import sys
from utils import *
from instructions import Instruction
from settings import *
from qiskit import QuantumRegister
from math import sqrt
from uncompute import *

n = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]
generate_with_grover = int(sys.argv[4])

current_settings = get_btor2_settings(input_file)
Instruction.all_instructions = read_file(input_file, modify_memory_sort=False, setting=current_settings)

Instruction.ZERO_CONST = QuantumRegister(1, name="zeroq")
Instruction.ONE_CONST = QuantumRegister(1, name="oneq")
Instruction.with_grover = generate_with_grover
# Instruction.circuit.add_register(Instruction.ZERO_CONST)
# Instruction.circuit.add_register(Instruction.ONE_CONST)
# Instruction.circuit.x(Instruction.ONE_CONST)
# Instruction.initialize_memory_addresses()
for i in range(1, n+1):
    Instruction.current_n = i

    for instruction in Instruction.all_instructions.values():
        if instruction[1] == INIT and i == 1:
            Instruction(instruction).execute()
        elif instruction[1] == NEXT or instruction[1] == BAD:
            Instruction(instruction).execute()

Instruction.or_bad_states()


if generate_with_grover:

    input_qubits = Instruction.get_input_qubits()

    # uncompute
    apply_and_reverse_stack(Instruction.global_stack, Instruction.circuit)

    apply_amplitude_amplification(input_qubits, Instruction.circuit)

    iterations = int(sqrt(2**len(input_qubits)))

    for i in range(iterations):
        # compute
        apply_and_reverse_stack(Instruction.global_stack, Instruction.circuit)
        #uncompute
        apply_and_reverse_stack(Instruction.global_stack, Instruction.circuit)

        apply_amplitude_amplification(input_qubits, Instruction.circuit)

    Instruction.circuit.measure_active()

Instruction.circuit.qasm(filename=output_file,formatted=True)
