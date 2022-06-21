from qword import *
from qword_tools import *


class Instruction:
    # BEGIN static attributes
    all_instructions: Dict[int, List[str]] = {}  # maps <sid> -> (list of tokens of the instruction)
    bad_states: List[int] = []
    bad_states_to_line_no: Dict[int, int] = {}
    inputs: List[QWord] = []
    circuit = QuantumCircuit()
    created_nids: Dict[int, QWord] = dict()
    ancillas: Dict[int, QuantumRegister] = dict()
    # model settings

    # END static attributes

    def __init__(self, instruction: List[str]):

        self.register_name= f"{instruction[0]}{instruction[1]}"
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

    def get_data_2_operands(self):
        operand1 = Instruction(self.get_instruction_at_index(3))
        operand1_qword = operand1.execute()

        operand2 = Instruction(self.get_instruction_at_index(4))
        operand2_qword = operand2.execute()

        bitset1, constants1 = self.get_last_qubitset(operand1.name, operand1_qword)
        bitset2, constants2 = self.get_last_qubitset(operand2.name, operand2_qword)

        if self.id not in self.created_nids.keys():
            self.created_nids[self.id] = QWord(self.register_name, 1)
            sort = self.get_sort()
            ancillas = QuantumRegister(sort + 1)
            self.ancillas[self.id] = ancillas
            self.circuit.add_register(ancillas)
            self.circuit.x(ancillas[ancillas.size - 1])

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
                self.circuit.x(qword1.actual[index])
                qword1.is_actual_constant[index] = 1
        return qword1


class Input(Instruction):

    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        qword = QWord(self.register_name, sort)
        qword.create_state(self.circuit, set_previous=True)
        self.circuit.h(qword.previous)
        for i in range(sort):
            qword.is_previous_constant[i] = -1
        self.inputs.append(qword)
        self.created_nids[self.id] = qword
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
        if self.id in self.created_nids.keys():
            sort = self.get_sort()
            qword = QWord(self.register_name, sort)
            qword.create_state(self.circuit, set_previous=True)

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
                        self.circuit.x(qword.previous[index])
                        qword.is_previous_constant[index] = 1
                qword.force_current_state(qword.previous, qword.is_previous_constant)
        return Instruction.created_nids[self.id]


class Next(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


# Arithmetic operations
class Add(Instruction):

    # TODO: optimization
    def __init__(self, instruction):
        super(Add, self).__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class Mul(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        raise Exception("missing implementation")


class Ite(Instruction):
    # TODO: Correct Optimization
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class Uext(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class And(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class Not(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        sort = self.get_sort()
        if self.id not in self.created_nids.keys():
            self.created_nids[self.id] = QWord(self.register_name, sort)
            self.created_nids[self.id].create_state(self.circuit)

        operand1 = Instruction(self.get_instruction_at_index(3))
        operand1_qword = operand1.execute()
        x, constants = self.get_last_qubitset(operand1.name, operand1_qword)
        optimized_bitwise_not(x, self.created_nids[self.id].actual, constants,
                              self.created_nids[self.id].is_actual_constant,
                              self.circuit)
        return self.created_nids[self.id]


class Eq(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands()
        result_qword = self.created_nids[self.id]
        optimized_is_equal(bitset1, bitset2, result_qword, constants1, constants2, self.circuit, self.ancillas[self.id])
        return result_qword





class Neq(Instruction):

    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> QWord:
        bitset1, constants1, bitset2, constants2 = self.get_data_2_operands()
        result_qword = self.created_nids[self.id]
        optimized_is_equal(bitset1, bitset2, result_qword, constants1, constants2, self.circuit, self.ancillas[self.id])
        assert(result_qword.size_in_bits == 1)
        self.circuit.x(result_qword.actual[0])
        if result_qword.is_actual_constant[0] != -1:
            result_qword.is_actual_constant[0]+=1
            result_qword.is_actual_constant[0] = int(result_qword.is_actual_constant[0] % 2)
        return result_qword


class Ult(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class Ulte(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class Ugt(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class Ugte(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
        raise Exception("missing implementation")


class Udiv(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:
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
            if self.id not in self.created_nids.keys():
                self.created_nids[self.id] = QWord(self.register_name, self.get_sort())
                self.created_nids[self.id].create_state(self.circuit)

            for (index, res) in enumerate(result_in_binary):
                assert (self.created_nids[self.id].is_actual_constant[index] == 0)
                self.circuit.x(self.created_nids[self.id].actual[index])
                self.created_nids[self.id].is_actual_constant[index] = 1
            return self.created_nids[self.id]
        else:
            raise Exception("non constant operands on UREM not implemented")


class Urem(Instruction):
    def __init__(self, instruction):
        super().__init__(instruction)

    def execute(self) -> Optional[QWord]:

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
            if self.id not in self.created_nids.keys():
                self.created_nids[self.id] = QWord(self.register_name, self.get_sort())
                self.created_nids[self.id].create_state(self.circuit)

            for (index, res) in enumerate(result_in_binary):
                assert(self.created_nids[self.id].is_actual_constant[index] == 0)
                self.circuit.x(self.created_nids[self.id].actual[index])
                self.created_nids[self.id].is_actual_constant[index] = 1
            return self.created_nids[self.id]
        else:
            raise Exception("non constant operands on UREM not implemented")


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
