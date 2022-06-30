import sys
from utils import *
from instructions import Instruction
from settings import *
from qiskit import QuantumRegister

n = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

current_settings = get_btor2_settings(input_file)
Instruction.all_instructions = read_file(input_file, modify_memory_sort=False, setting=current_settings)

Instruction.ZERO_CONST = QuantumRegister(1, name="zeroq")
Instruction.ONE_CONST = QuantumRegister(1, name="oneq")
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

# uncompute
# while not Instruction.global_stack.is_empty():
#     Instruction.global_stack.pop().apply(Instruction.circuit)

# TODO: logic-OR bad states
Instruction.circuit.qasm(filename=output_file)
