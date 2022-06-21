import sys
from utils import *
from qasm_gen import *
from instructions import Instruction

n = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

for i in range(n):
    # clear nids set of nids created in the previous timestep
    Instruction.created_nids_in_timestep.clear()