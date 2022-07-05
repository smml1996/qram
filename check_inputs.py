import sys
from utils import *
from instructions import Instruction
from settings import *
from uncompute import *

n = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]
generate_with_grover = int(sys.argv[4])

current_settings = get_btor2_settings(input_file)
Instruction.all_instructions = read_file(input_file, modify_memory_sort=False, setting=current_settings)
Instruction.with_grover = 0

for i in range(1, n+1):
    Instruction.current_n = i

    for instruction in Instruction.all_instructions.values():
        if instruction[1] == INIT and i == 1:
            Instruction(instruction).execute()
        elif instruction[1] == NEXT or instruction[1] == BAD:
            Instruction(instruction).execute()

result_bad_states = Instruction.or_bad_states()
assert(len(result_bad_states) == 1)

print("# inputs:", len(Instruction.input_nids))

circuit_queue = get_circuit_queue(Instruction.global_stack)

def are_all_controls_true(values, controls):
    for c in controls:
        if values[c] == 0:
            return False
    return True

def check_input(value):
    global circuit_queue
    # we only set the value of the first input the other ones are set to |0>
    qubit_values = dict()
    assert(len(Instruction.input_nids[0]) == 8)
    for qubit in Instruction.input_nids[0]:
        qubit_values[qubit] = value % 2
        value = value // 2

    element: Element = circuit_queue.pop()
    assert(element.element_type != CHECKPOINT_TYPE)

    while element.element_type != CHECKPOINT_TYPE:
        # do something with element
        element: Element = circuit_queue.pop()
        for o in element.operands:
            if o not in qubit_values.keys():
                qubit_values[o] = 0

        assert (element.target is not None)

        if element.target not in qubit_values.keys():
            qubit_values[element.target] = 0

        flip_target = True
        if element.gate_name == X:
            assert(len(element.operands) == 0)

        else:
            assert((element.gate_name == CX and len(element.operands) ==1) or
                   (element.gate_name == CCX and len(element.operands) == 2) or
                    element.gate_name == MCX)
            flip_target = are_all_controls_true(qubit_values, element.operands)

        if flip_target:
            qubit_values[element.target] = (qubit_values[element.target] + 1) % 2

        element: Element = circuit_queue.pop()
    assert element.element_type == CHECKPOINT_TYPE
    circuit_queue.push(element)
    return qubit_values[result_bad_states[0]]



for i in range(256):
    print(f"input: {i} -> {check_input(i)}")