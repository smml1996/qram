import sys
from utils import *
from instructions import Instruction
from settings import *

n = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

current_settings = get_btor2_settings(input_file)
Instruction.all_instructions = read_file(input_file, modify_memory_sort=True, setting=current_settings)

for i in range(n):

    # clear nids set of nids created in the previous time-step
    Instruction.created_nids_in_timestep.clear()

    for instruction in Instruction.all_instructions.values():
        if instruction[1] == INIT and i == 1:
            Instruction(instruction).execute()
        elif instruction[1] == NEXT:
            Instruction(instruction).execute()
        elif instruction[1] == BAD:
            Instruction(instruction).execute()

    # TODO: GARBAGE COlLECTOR algorithm

# TODO: logic-OR bad states
Instruction.circuit.qasm(filename=output_file)
