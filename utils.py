from typing import List, Dict


def read_file(filename: str, modify_memory_sort: bool = False, setting: Dict[str, int] = None):
    """
    updates the static variable Instruction.all_instructions
    :param setting:
    :param modify_memory_sort:
    :param filename: btor2 file
    :return:
    """
    file = open(filename, 'r')
    result = {}
    for line in file.readlines():

        comment_index = line.find(";")
        if comment_index > -1:
            cleaned_line = line[:comment_index]
        else:
            cleaned_line = line

        temp = cleaned_line.lower().split()
        if len(temp) > 0:

            if int(temp[0]) == 3 or int(temp[0]) == 5 and modify_memory_sort:
                # this is memory sort. We need to modify this so it matches with our definition of memory
                memory_size = setting['word_size'] * (setting['size_datasegment'] + setting['size_heap']
                                                      + setting['size_stack'])
                temp = [temp[0], "sort", "bitvec", str(memory_size)]
                print("sort memory modified to be bitvector of size: ", memory_size)

            result[int(temp[0])] = temp
    file.close()
    return result


def get_bit_repr_of_number(number: int, size_in_bits: int = 64) -> List[int]:
    if number > (2 ** size_in_bits - 1):
        raise Exception(f"{number} cannot be represented with {size_in_bits} bits")
    s = bin(number & int("1" * size_in_bits, 2))[2:]
    str_repr = ("{0:0>%s}" % size_in_bits).format(s)

    bits = []

    # index 0 contains the least significant bit
    for c in str_repr[::-1]:
        bits.append(int(c))

    return bits


def get_decimal_representation(binary_repr: List[int]) -> int:
    result = 0

    for (index, value) in enumerate(binary_repr):
        result += value * (2 ** index)

    return result


def bit_level_sum(bits1: List[int], bits2: List[int]) -> List[int]:
    """
    LSB is at index 0
    :param bits1:
    :param bits2:
    :return:
    """

    carry = 0
    result = []
    for (bit1, bit2) in zip(bits1, bits2):
        current = (bit1 + bit2 + carry) % 2
        carry = (bit1 + bit2 + carry) > 1
        result.append(current)

    if carry == 1:
        print(f"WARNING {carry}: overflow at tools.bit_level_sum")
    return result
