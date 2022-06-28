from qword import *
from qword_tools import *
from uncompute import *
from settings import *


class Instruction:
    # BEGIN static attributes
    all_instructions: Dict[int, List[str]] = {}  # maps <sid> -> (list of tokens of the instruction)
    bad_states: List[int] = []
    bad_states_to_line_no: Dict[int, int] = {}
    inputs: List[QWord] = []
    circuit = QuantumCircuit()
    created_nids: Dict[int, QWord] = dict()
    created_nids_in_timestep: set = set()
    ancillas: Dict[int, QuantumRegister] = dict()

    # uncompute variables
    global_stack: Stack = Stack()
    current_stack: Stack = Stack()
    current_queue: Queue = Queue()

    # END static attributes

    def __init__(self, instruction: List[str]):

        self.register_name = f"{instruction[0]}{instruction[1]}"
        self.instruction = instruction
        self.base_class = None

        if len(instruction) < 2:
            # each instruction has at least 2 elements. Always.
            raise Exception(f'error parsing instruction: {" ".join(instruction)}')
        self.name = instruction[1]
        try:
            self.id = int(instruction[0])
        except IndexError:
            raise Exception(f'error parsing instruction: {" ".join(instruction)}')

    def set_instruction(self):

        if self.name == SORT:
            self.base_class = Sort(self.instruction)
        elif self.name == STATE:
            self.base_class = State(self.instruction)
        elif self.name == CONSTD:
            self.base_class = State(self.instruction)
        elif self.name == ZERO or self.name == ONE:
            self.base_class = State(self.instruction)
        elif self.name == INPUT:
            self.base_class = Input(self.instruction)
        elif self.name == INIT:
            self.base_class = Init(self.instruction)
        elif self.name == NEXT:
            self.base_class = Next(self.instruction)
        elif self.name == ADD or self.name == SUB:
            self.base_class = Add(self.instruction)
        elif self.name == INC or self.name == DEC:
            self.base_class = Add(self.instruction)
        elif self.name == MUL:
            self.base_class = Mul(self.instruction)
        elif self.name == ITE:
            self.base_class = Ite(self.instruction)
        elif self.name == UEXT:
            self.base_class = Uext(self.instruction)
        elif self.name == AND:
            self.base_class = And(self.instruction)
        elif self.name == NOT:
            self.base_class = Not(self.instruction)
        elif self.name == EQ:
            self.base_class = Eq(self.instruction)
        elif self.name == ULT:
            self.base_class = Ult(self.instruction)
        elif self.name == ULTE:
            self.base_class = Ulte(self.instruction)
        elif self.name == UGT:
            self.base_class = Ugt(self.instruction)
        elif self.name == UGTE:
            self.base_class = Ugte(self.instruction)
        elif self.name == UDIV:
            self.base_class = Udiv(self.instruction)
        elif self.name == UREM:
            self.base_class = Urem(self.instruction)
        elif self.name == BAD:
            self.base_class = Bad(self.instruction)
        elif self.name == NEQ:
            self.base_class = Neq(self.instruction)
        elif self.name == SLICE:
            self.base_class = Slice(self.instruction)
        else:
            raise Exception(f"Not valid instruction: {self}")

    def get_instruction_at_index(self, index: int) -> List[str]:
        return self.all_instructions[abs(int(self.instruction[index]))]

    def get_last_qubitset(self, name: str, qword: QWord) -> (QuantumRegister, List[int]):
        if name in [STATE, INPUT]:
            return qword.previous, qword.is_previous_constant

        if name in [CONSTD, ONE, ZERO]:
            return qword.actual, qword.is_actual_constant

        if name in [NEXT, SORT, INIT]:
            raise Exception(f"Cannot determine prev. state for instruction {self.instruction}")

        return qword.actual, qword.is_actual_constant

    def get_data_2_operands(self, register_size, ancilla_size):
        operand1 = Instruction(self.get_instruction_at_index(3))
        operand1_qword = operand1.execute()

        operand2 = Instruction(self.get_instruction_at_index(4))
        operand2_qword = operand2.execute()

        bitset1, constants1 = self.get_last_qubitset(operand1.name, operand1_qword)
        bitset2, constants2 = self.get_last_qubitset(operand2.name, operand2_qword)

        if self.id not in Instruction.created_nids.keys():
            Instruction.created_nids[self.id] = QWord(self.register_name, register_size)
            if ancilla_size > 0:
                ancillas = QuantumRegister(ancilla_size)
                Instruction.ancillas[self.id] = ancillas
                Instruction.circuit.add_register(ancillas)

        return bitset1, constants1, bitset2, constants2

    @property
    def specific_subclass(self) -> object:
        if self.base_class is None:
            self.set_instruction()
        return self.base_class

    def get_sort(self) -> int:
        return Sort(self.all_instructions[int(self.instruction[2])]).execute().size_in_bits

    def execute(self) -> Optional[QWord]:
        return self.specific_subclass.execute()

    @staticmethod
    def clean_static_variables():
        Instruction.all_instructions = {}
        Instruction.bad_states = []
        Instruction.bad_states_to_line_no = {}
        Instruction.input_nids = []
        Instruction.circuit = QuantumCircuit()
        Instruction.created_nids = dict()
        Instruction.ancillas = dict()
        Instruction.created_nids_in_timestep = set()

        # uncompute variables
        Instruction.global_stack = Stack()
        Instruction.current_stack = Stack()
        Instruction.current_queue = Queue()

    @staticmethod
    def or_bad_states():
        raise Exception("missing implementation")


class Init(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self):
        operand1 = Instruction(self.get_instruction_at_index(3))
        qword1 = operand1.execute()

        operand2 = Instruction(self.get_instruction_at_index(4))
        qword2 = operand2.execute()

        for (index, value) in enumerate(qword2.is_actual_constant):
            assert(value != -1)
            assert(qword1.is_actual_constant[index] == 0)
            if value == 1:
                Instruction.circuit.x(qword1.actual[index])
                Instruction.current_stack.push(Element(GATE_TYPE, X, [], qword1.actual[index]))
                qword1.is_actual_constant[index] = 1
        return qword1


class Input(Instruction):

    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        qword = QWord(self.register_name, sort)
        qword.create_state(Instruction.circuit, set_previous=True)
        Instruction.circuit.h(qword.previous)
        for i in range(sort):
            qword.is_previous_constant[i] = -1
        Instruction.inputs.append(qword)
        Instruction.created_nids[self.id] = qword
        return qword


class Sort(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort_name = self.instruction[2]
        bit_count = int(self.instruction[3])
        if sort_name == "array":
            raise Exception("Not implemented")
        elif sort_name != "bitvec":
            raise Exception(f"not valid instruction: {self}")

        return QWord(bit_count)


class State(Instruction):
    # TODO: Custom names

    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        if self.id in Instruction.created_nids.keys():
            sort = self.get_sort()
            qword = QWord(self.register_name, sort)
            qword.create_state(Instruction.circuit, set_previous=True)

            # returns a vector full of zeros, we use this to initialize memory with zeros
            bit_representation = get_bit_repr_of_number(0, qword.size_in_bits)
            Instruction.created_nids[self.id] = qword

            if self.instruction[1] == CONSTD:
                # returns a vector in which index 0 is LSB, and it represent the value of this constant value
                bit_representation = get_bit_repr_of_number(int(self.instruction[3]), qword.size_in_bits)

            if self.instruction[1] == ZERO:
                # returns a vector full of zeros, used to initialize this constant
                bit_representation = get_bit_repr_of_number(0, qword.size_in_bits)

            if self.instruction[1] == ONE:
                # first element of this vector represents a 1. Used to initialize some qubits that represent this value
                bit_representation = get_bit_repr_of_number(1, qword.size_in_bits)

            if self.instruction[1] in [CONSTD, ZERO, ONE]:
                # if flag is turn on or we are dealing with constants then we initialize this state/constant
                for (index, value) in enumerate(bit_representation):
                    assert(value == 1 or value == 0)
                    if value == 1:
                        assert(qword.is_previous_constant[index] == 0)
                        Instruction.circuit.x(qword.previous[index])
                        Instruction.current_stack.push(Element(GATE_TYPE, X, [], qword.previous[index]))
                        qword.is_previous_constant[index] = 1
                qword.force_current_state(qword.previous, qword.is_previous_constant)
            else:
                qword.create_uncomputation_qubits(Instruction.circuit)
        return Instruction.created_nids[self.id]


class Next(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        previous_state_qword = Instruction(self.get_instruction_at_index(3)).execute()
        future_state = Instruction(self.get_instruction_at_index(4))
        future_state_qword = future_state.execute()
        if previous_state_qword and future_state_qword:
            assert(previous_state_qword.actual is not None)
            assert (previous_state_qword.uncomputation_space is not None)

            for q in previous_state_qword.is_actual_constant:
                assert (q == 0)
            future_bits, _ = self.get_last_qubitset(future_state.name, future_state_qword)
            for (index, bit) in enumerate(future_bits):
                Instruction.circuit.cx(bit, previous_state_qword.actual)
                # TODO: add to STACK
        else:
            # if for some reason one, evaluating one of the operands returns None we throw an error
            raise Exception(f"not valid transition: {self}")
        return None


# Arithmetic operations
class Add(Instruction):
    def __init__(self, instruction):
        super(Add, self).__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        if self.id not in Instruction.created_nids.keys():
            Instruction.created_nids[self.id] = QWord(self.register_name, sort)
            Instruction.created_nids[self.id].create_state(Instruction.circuit)
            if self.instruction[1] == INC or self.instruction[1] == DEC:
                if self.instruction == INC:
                    bit_ = get_bit_repr_of_number(1, sort)
                else:
                    # is decrementing by 1
                    bit_ = get_bit_repr_of_number(-1, sort)

                Instruction.ancillas[self.id] = QuantumRegister(sort, name=self.register_name)
                Instruction.circuit.add_register(Instruction.ancillas[self.id])
                for (index, b) in enumerate(bit_):
                    if b == 1:
                        Instruction.circuit.x(Instruction.ancillas[self.id][index])
                    else:
                        assert(b == 0)

        if self.id not in Instruction.created_nids_in_timestep:
            local_stack = None
            operand1 = Instruction(self.get_instruction_at_index(3))
            qword1 = operand1.execute()
            qubit_set1, constants1 = self.get_last_qubitset(operand1.name, qword1)
            if self.instruction[1] == ADD:
                operand2 = Instruction(self.get_instruction_at_index(4))
                qword2 = operand2.execute()
                qubit_set2, constants2 = self.get_last_qubitset(operand2.name, qword2)
            elif self.instruction[1] == SUB:
                operand2 = Instruction(self.get_instruction_at_index(4))
                qword2 = operand2.execute()
                qubit_set2, constants2 = self.get_last_qubitset(operand2.name, qword2)
                local_stack = optimized_get_twos_complement(qubit_set2, Instruction.circuit)
            else:
                qubit_set2 = Instruction.ancillas[self.id]
                if self.instruction == INC:
                    constants2 = get_bit_repr_of_number(1, sort)
                else:
                    # is decrementing by 1
                    constants2 = get_bit_repr_of_number(-1, sort)


            assert len(qubit_set1) == len(qubit_set2)
            assert len(qubit_set1) == sort
            optimized_bitwise_add(qubit_set1, qubit_set2, Instruction.created_nids[self.id], constants1, constants2,
                                  Instruction.circuit, Instruction.current_stack)
            Instruction.created_nids_in_timestep.add(self.id)
            if local_stack is not None:
                # uncomputes twos complement
                while not local_stack.is_empty():
                    local_stack.pop().apply(Instruction.circuit)
        return Instruction.created_nids[self.id]


class Mul(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, sort)
        result_qword = Instruction.created_nids[self.id]
        assert(result_qword.size_in_bits == sort)

        if self.id not in self.created_nids_in_timestep:
            pass
        return result_qword

class Ite(Instruction):
    # TODO: Correct Optimization
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        if self.id not in Instruction.created_nids.keys():
            Instruction.created_nids[self.id] = QWord(self.register_name, sort)
        if self.id not in Instruction.created_nids_in_timestep:
            condition = Instruction(self.get_instruction_at_index(3))

            qword_condition = condition.execute()
            assert qword_condition.size_in_bits == 1

            if self.instruction[3][0] == '-':
                true_part = Instruction(self.get_instruction_at_index(5))
                false_part = Instruction(self.get_instruction_at_index(4))
            else:
                true_part = Instruction(self.get_instruction_at_index(4))
                false_part = Instruction(self.get_instruction_at_index(5))

            qubit_condition, is_condition_constant = self.get_last_qubitset(condition.name, qword_condition)

            assert(len(is_condition_constant) == 1)
            assert(len(qubit_condition) == 1)

            result_qword = Instruction.created_nids[self.id]
            if is_condition_constant[0] != -1:
                if is_condition_constant == 1:
                    true_part_qword = true_part.execute()
                    bitset, constants = self.get_last_qubitset(true_part.name, true_part_qword)
                else:
                    false_part_qword = false_part.execute()
                    bitset, constants = self.get_last_qubitset(false_part.name, false_part_qword)
                for (index, value) in enumerate(constants):
                    Instruction.circuit.cx(bitset[index], result_qword.actual[index])
                    result_qword.is_actual_constant[index] = value
                    Instruction.current_stack.push(Element(GATE_TYPE, CX, [bitset[index]], result_qword.actual[index]))
            else:
                true_part_qword = true_part.execute()
                true_part_bits, constants_true_part = self.get_last_qubitset(true_part.name, true_part_qword)

                false_part_qword = false_part.execute()
                false_part_bits, constants_false_part = self.get_last_qubitset(false_part.name, false_part_qword)

                for (index, result_bit) in enumerate(result_qword.actual):
                    Instruction.circuit.ccx(qubit_condition[0], true_part_bits[index], result_bit)
                    Instruction.current_stack.push(Element(GATE_TYPE, CCX, [qubit_condition[0], true_part_bits[index]],
                                                           result_bit))

                Instruction.circuit.x(qubit_condition[0])
                Instruction.current_stack.push(Element(GATE_TYPE, X, [], qubit_condition[0]))
                for (index, result_bit) in enumerate(result_qword.actual):
                    Instruction.circuit.ccx(qubit_condition[0], false_part_bits[index], result_bit)
                    Instruction.current_stack.push(Element(GATE_TYPE, CCX, [qubit_condition[0], false_part_bits[index]],
                                                           result_bit))
                Instruction.circuit.x(qubit_condition[0])
                Instruction.current_stack.push(Element(GATE_TYPE, X, [], qubit_condition[0]))

                # constant propagation
                for (index,(const_true_part, const_false_part)) in enumerate(zip(constants_true_part,
                                                                                 constants_false_part)):
                    if const_true_part == const_false_part:
                        if const_true_part == -1:
                            result_qword.is_actual_constant[index] = -1
                        else:
                            result_qword.is_actual_constant[index] = const_true_part
                    else:
                        result_qword.is_actual_constant[index] =  -1


        return Instruction.created_nids[self.id]


class Uext(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        sort = self.get_sort()
        previous = Instruction(self.get_instruction_at_index(3))
        previous_qword = previous.execute()
        previous_qubits, constants_op = self.get_last_qubitset(previous.name, previous_qword)
        constants = [0 for _ in range(sort)]

        ext_value = int(self.instruction[4])

        assert sort == ext_value + previous_qword.size_in_bits
        if self.id not in Instruction.created_nids.keys():
            dummy_register = QuantumRegister(ext_value)
            Instruction.ancillas[self.id] = dummy_register

        result_qubits = []

        for (i,qubit) in enumerate(previous_qubits):
            result_qubits.append(qubit)
            constants[i] = constants_op[i]

        for qubit in Instruction.ancillas[self.id]:
            result_qubits.append(qubit)

        result = QWord(self.register_name, sort)
        result.force_current_state(result_qubits, constants)

        return result


class And(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, 0)
        result_qword = Instruction.created_nids[self.id]
        assert(result_qword.size_in_bits == sort)
        if self.id not in Instruction.created_nids_in_timestep:
            optimized_bitwise_and(bitset1, bitset2, result_qword, constants1, constants2, Instruction.circuit,
                                  Instruction.current_stack)
        return result_qword


class Not(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        if self.id not in Instruction.created_nids.keys():
            Instruction.created_nids[self.id] = QWord(self.register_name, sort)
            Instruction.created_nids[self.id].create_state(Instruction.circuit)

        if self.id not in Instruction.created_nids_in_timestep:
            operand1 = Instruction(self.get_instruction_at_index(3))
            operand1_qword = operand1.execute()
            x, constants = self.get_last_qubitset(operand1.name, operand1_qword)
            optimized_bitwise_not(x, Instruction.created_nids[self.id].actual, constants,
                                  Instruction.created_nids[self.id].is_actual_constant,
                                  Instruction.circuit, Instruction.current_stack)
        return Instruction.created_nids[self.id]


class Eq(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, sort+1)
        result_qword = Instruction.created_nids[self.id]
        ancillas = Instruction.ancillas[self.id]
        if self.id not in Instruction.created_nids_in_timestep:
            Instruction.circuit.x(ancillas[ancillas.size - 1])
            optimized_is_equal(bitset1, bitset2, result_qword, constants1, constants2, Instruction.circuit, ancillas,
                               Instruction.current_stack)
        return result_qword


class Neq(Instruction):

    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, sort+1)
        result_qword = Instruction.created_nids[self.id]
        ancillas = Instruction.ancillas[self.id]

        if self.id not in Instruction.created_nids_in_timestep:
            Instruction.circuit.x(ancillas[ancillas.size - 1])
            optimized_is_equal(bitset1, bitset2, result_qword, constants1, constants2, Instruction.circuit, ancillas,
                               Instruction.current_stack)
            assert(result_qword.size_in_bits == 1)
            Instruction.circuit.x(result_qword.actual[0])
            Instruction.current_stack.push(Element(GATE_TYPE, X, [], result_qword.actual[0]))
            if result_qword.is_actual_constant[0] != -1:
                result_qword.is_actual_constant[0] += 1
                result_qword.is_actual_constant[0] = int(result_qword.is_actual_constant[0] % 2)
        return result_qword


class Ult(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, 2 + sort-1)
        result_qword = Instruction.created_nids[self.id]
        assert (result_qword.size_in_bits == self.get_sort())
        if self.id not in Instruction.created_nids_in_timestep:
            optimized_unsigned_ult(bitset1, bitset2, result_qword, constants1, constants2, Instruction.circuit,
                                  Instruction.ancillas[self.id], Instruction.current_stack)
        return result_qword


class Ulte(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, 2 + sort-1 + 2 + sort+1)
        result_qword = Instruction.created_nids[self.id]
        assert (result_qword.size_in_bits == self.get_sort())
        if self.id not in Instruction.created_nids_in_timestep:
            optimized_unsigned_ulte(bitset1, bitset2, result_qword, constants1, constants2, Instruction.circuit,
                                   Instruction.current_stack)
        return result_qword


class Ugt(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, 2 + sort-1)
        result_qword = Instruction.created_nids[self.id]
        assert (result_qword.size_in_bits == self.get_sort())
        if self.id not in Instruction.created_nids_in_timestep:
            optimized_unsigned_ugt(bitset1, bitset2, result_qword, constants1, constants2, Instruction.circuit,
                                   Instruction.current_stack)
        return result_qword


class Ugte(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        sort = self.get_sort()
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands(sort, 2 + sort -1 + 2 + sort + 1)
        result_qword = Instruction.created_nids[self.id]
        assert (result_qword.size_in_bits == self.get_sort())
        if self.id not in Instruction.created_nids_in_timestep:
            optimized_unsigned_ugte(bitset1, bitset2, result_qword, constants1, constants2, Instruction.circuit,
                                   Instruction.current_stack)
        return result_qword


class Udiv(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        if self.id not in Instruction.created_nids_in_timestep:
            operand1 = Instruction(self.get_instruction_at_index(3))
            operand1_qword = operand1.execute()

            operand2 = Instruction(self.get_instruction_at_index(4))
            operand2_qword = operand2.execute()

            bitset1, are_constants1 = self.get_last_qubitset(operand1.name, operand1_qword)
            bitset2, are_constants2 = self.get_last_qubitset(operand2.name, operand2_qword)
            assert len(bitset1) == len(bitset2)
            if QWord.are_all_constants(are_constants1) and QWord.are_all_constants(are_constants2):
                bitset1_in_decimal = get_decimal_representation(are_constants1)
                bitset2_in_decimal = get_decimal_representation(are_constants2)
                result_in_decimal = bitset1_in_decimal // bitset2_in_decimal

                result_in_binary = get_bit_repr_of_number(result_in_decimal, len(bitset1))
                if self.id not in Instruction.created_nids.keys():
                    Instruction.created_nids[self.id] = QWord(self.register_name, self.get_sort())
                    Instruction.created_nids[self.id].create_state(Instruction.circuit)

                for (index, res) in enumerate(result_in_binary):
                    assert (Instruction.created_nids[self.id].is_actual_constant[index] == 0)
                    Instruction.circuit.x(Instruction.created_nids[self.id].actual[index])
                    Instruction.current_stack.push(Element(GATE_TYPE, X, [], Instruction.created_nids[self.id].actual[index]))
                    Instruction.created_nids[self.id].is_actual_constant[index] = 1

            else:
                raise Exception("non constant operands on UDIV not implemented")
        return Instruction.created_nids[self.id]


class Urem(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        if self.id not in Instruction.created_nids_in_timestep:
            operand1 = Instruction(self.get_instruction_at_index(3))
            operand1_qword = operand1.execute()

            operand2 = Instruction(self.get_instruction_at_index(4))
            operand2_qword = operand2.execute()

            bitset1, are_constants1 = self.get_last_qubitset(operand1.name, operand1_qword)
            bitset2, are_constants2 = self.get_last_qubitset(operand2.name, operand2_qword)
            assert len(bitset1) == len(bitset2)
            if QWord.are_all_constants(are_constants1) and QWord.are_all_constants(are_constants2):
                bitset1_in_decimal = get_decimal_representation(are_constants1)
                bitset2_in_decimal = get_decimal_representation(are_constants2)
                result_in_decimal = bitset1_in_decimal % bitset2_in_decimal

                result_in_binary = get_bit_repr_of_number(result_in_decimal, len(bitset1))
                if self.id not in Instruction.created_nids.keys():
                    Instruction.created_nids[self.id] = QWord(self.register_name, self.get_sort())
                    Instruction.created_nids[self.id].create_state(Instruction.circuit)

                for (index, res) in enumerate(result_in_binary):
                    assert(Instruction.created_nids[self.id].is_actual_constant[index] == 0)
                    Instruction.circuit.x(Instruction.created_nids[self.id].actual[index])
                    Instruction.current_stack.push(Element(GATE_TYPE, X, [], Instruction.created_nids[self.id].actual[index]))
                    Instruction.created_nids[self.id].is_actual_constant[index] = 1
            else:
                raise Exception("non constant operands on UREM not implemented")
        return Instruction.created_nids[self.id]


class Bad(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        bad_state = Instruction(self.get_instruction_at_index(2))
        bad_state_qword = bad_state.execute()
        bad_state_qubits, are_constants = self.get_last_qubitset(bad_state.name, bad_state_qword)
        assert bad_state_qword.size_in_bits == 1
        # Instruction.qubits_to_fix[bad_state_qubits[0]] = 1  # make the bad state happen
        qword_result = QWord(1)
        qword_result.force_current_state(bad_state_qword, are_constants)
        assert(len(are_constants) == 1)
        if are_constants == -1:
            # only care if this bad state does not evaluates no a concrete value
            Instruction.bad_states.append(bad_state_qubits[0])
            Instruction.bad_states_to_line_no[bad_state_qubits[0]] = self.id
        else:
            print("bad state value found:", self.register_name, "->", are_constants[0])
        return qword_result


class Slice(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()

        operand = Instruction(self.get_instruction_at_index(3))
        qword = operand.execute()
        qubitset, are_constants = self.get_last_qubitset(operand.name, qword)

        bottom = int(self.instruction[5])
        top = int(self.instruction[4])

        result_qubits = qubitset[bottom:top+1]

        assert len(result_qubits) == (top - bottom) + 1
        assert(len(result_qubits) == sort)
        result_qword = QWord(self.register_name, sort)
        result_qword.force_current_state(result_qubits, are_constants[bottom:top+1])

        return result_qword
