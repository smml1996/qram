import sys
from utils import *
from instructions import Instruction
from settings import *
from uncompute import *

n = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

current_settings = get_btor2_settings(input_file)
Instruction.all_instructions = read_file(input_file, modify_memory_sort=True, setting=current_settings)

for i in range(1, n+1):
    Instruction.current_n = i

    for (key, instruction) in Instruction.all_instructions.values():
        if instruction[1] == INIT and i == 1:
            Instruction(instruction).execute()
        elif instruction[1] == NEXT or instruction[1] == BAD:
            Instruction(instruction).execute()

# uncompute

Instruction.or_bad_states()

while not Instruction.global_stack.is_empty():
    Instruction.global_stack.pop().apply()

# TODO: logic-OR bad states
Instruction.circuit.qasm(filename=output_file)
