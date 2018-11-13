#!/usr/bin/python
from collections import defaultdict, Counter
import argparse
import sys
from llvmcpy.llvm import *
import math
import matplotlib.pylab as plt
import helper
from multiprocessing import Process, Queue, cpu_count
import dill as pickle
from time import time
from operator import itemgetter

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def compute_csv(args):
    # Queue(s) for the process, one for input data to the process, the other for the output data to the main process
    q_in = Queue()
    q_out = Queue()
    try:
        cpus = cpu_count() - 1
    except NotImplementedError:
        cpus = 2   # arbitrary default
    buffer_1 = create_memory_buffer_with_contents_of_file(args[0])
    buffer_2 = create_memory_buffer_with_contents_of_file(args[1])
    context = get_global_context()
    module_1 = context.parse_ir(buffer_1)
    module_2 = context.parse_ir(buffer_2)
    global list_opcodes
    list_opcodes = get_opcodes([module_1, module_2])
    global helper_names
    helper_names = get_helper_names(module_1, module_2)
    header = list(flatten(['function1', 'function2', 'match', '#bb_ratio', '#bb_diff', '#instr_ratio', '#instr_diff', 'byte_dim_ratio', 'byte_dim_diff', '#instructions_ratio', '#instructions_diff', 'load_dim_ratio', 'load_dim_diff', '#loads_ratio', '#loads_diff', 'store_dim_ratio', 'store_dim_diff', '#lstores_ratio', '#stores_diff', '#indirect_calls_ratio', '#indirect_calls_diff', '#function_calls_ratio', '#function_calls_diff']))
    header.extend(list(flatten([[str(elem) + "_ratio", str(elem) + "_diff"] for elem in helper_names])))
    header.extend(list(flatten([[str(elem) + "_ratio", str(elem) + "_diff"] for elem in list_opcodes])))

    #pool = Pool(cpus) 
    functions_list = [get_names, cmp_name, cmp_dimension_llvm_bb, cmp_dimension_llvm_instr, cmp_byte_dimension_num_instr, cmp_load_store_instructions, cmp_indirect_calls, cmp_revamb_function_calls, cmp_helper_calls, cmp_instruction_opcodes]
    # "enumerate" takes the list and returns a tuple composed by (index_of_element, element)
    [q_in.put((i, pickle.dumps(x))) for i, x in enumerate(functions_list)]
    [q_in.put((-1, -1)) for _ in xrange(cpus)]

    tuples_space = [(fun1, fun2) for fun1 in module_1.iter_functions() for fun2 in module_2.iter_functions() if "bb." in fun1.name and "bb." in fun2.name]
    # Starting the process
    rows = []
    proc = [Process(target=run, args=(tuples_space, q_in, q_out, i)) for i in xrange(cpus)]

    for p in proc:
        p.daemon = True
        p.start()

    for i in xrange(len(functions_list)):
        r = q_out.get()
        rows.append(r)
    [p.join() for p in proc]
    rows = [elem[1] for elem in sorted(rows, key=itemgetter(0))]
    rows = [list(flatten(elem)) for elem in zip(*rows)]
    #rows = [list(flatten(map(lambda x: x(fun1, fun2), functions_list))) for fun1 in module_1.iter_functions() for fun2 in module_2.iter_functions() if "bb." in fun1.name and "bb." in fun2.name]
    '''
    rows = [list(flatten([get_names(fun1, fun2), compare_name(fun1, fun2), cmp_dimension_llvm_bb(fun1, fun2), cmp_dimension_llvm_instr(fun1, fun2), cmp_byte_dimension_num_instr(fun1, fun2), cmp_load_instructions(fun1, fun2), cmp_store_instructions(fun1, fun2), cmp_indirect_calls(fun1, fun2), cmp_revamb_function_calls(fun1, fun2), cmp_num_function_calls(fun1, fun2), cmp_helper_calls(fun1, fun2), cmp_instruction_opcodes(fun1, fun2)])) for fun1 in module_1.iter_functions() for fun2 in module_2.iter_functions() if "bb." in fun1.name and "bb." in fun2.name]
    '''
    filename = args[2]
    helper.write(rows, header, filename=filename)

def run(tuples_space, q_in, q_out, i):
    while True:
        # getting the data from the queue in
        (identifier, target) = q_in.get()
        # if target = -1
        if target == -1:
            break
        result = map(lambda x: pickle.loads(target)(x[0], x[1]), tuples_space)
        q_out.put((identifier, result))

def get_names(fun1, fun2):
    return [fun1.name, fun2.name]

def cmp_name(fun1, fun2):
    '''
    Compare fun1 name with fun2 name
    '''
    return fun1.name == fun2.name

def cmp_dimension_llvm_bb(fun1, fun2):
    '''
    Compute the sum and difference between the number of llvm instructions of fun1 and fun2
    '''
    bb1 = fun1.count_basic_blocks()
    bb2 = fun2.count_basic_blocks()
    return [(bb1 + bb2)/2, bb1-bb2]

def cmp_dimension_llvm_instr(fun1, fun2):
    count1 = 0
    for bb in fun1.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            count1 += 1
    count2 = 0
    for bb in fun2.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            count2 += 1

    return [(count1 + count2)/2, count1 - count2]

def get_opcodes(array_module):
    opcode_set = set()
    for module in array_module:
        for function in module.iter_functions():
            for bb in function.iter_basic_blocks():
                for instruction in bb.iter_instructions():
                    opcode_set.add(instruction.instruction_opcode)
    return sorted(list(opcode_set))

def get_opcode_dictionary(function):
    opcode_dictionary = defaultdict(lambda:0)

    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            opcode_dictionary[instruction.instruction_opcode] += 1
    return opcode_dictionary

def cmp_instruction_opcodes(fun1, fun2):

    fun1_opcode_dictionary = get_opcode_dictionary(fun1)
    fun2_opcode_dictionary = get_opcode_dictionary(fun2)

    return [[(fun1_opcode_dictionary[elem] + fun2_opcode_dictionary[elem])/2, fun1_opcode_dictionary[elem] - fun2_opcode_dictionary[elem]] for elem in list_opcodes]


def get_byte_dimension(function):
    sum_dimension = 0
    num_instructions = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if instruction.get_num_operands() >= 4:
                    if instruction.get_operand(instruction.get_num_operands() - 1).get_name() == "newpc":
                        sum_dimension += helper.get_value_from_operand(instruction.get_operand(1))
                        num_instructions += 1
    return sum_dimension, num_instructions

def cmp_byte_dimension_num_instr(fun1, fun2):
    fun1_dim, fun1_num_instr = get_byte_dimension(fun1)
    fun2_dim, fun2_num_instr = get_byte_dimension(fun2)
    return [(fun1_dim + fun2_dim)/2, fun1_dim - fun2_dim, (fun1_num_instr + fun2_num_instr)/2, fun1_num_instr - fun2_num_instr]

def get_loads_stores(function):
    load_inst = 0
    store_inst = 0
    len_load_inst = 0
    len_store_inst = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_load_inst() != None and instruction.get_operand(0).is_a_global_variable() == None:
                load_inst += instruction.type_of().get_int_type_width()
                len_load_inst += 1
            if instruction.is_a_store_inst() != None and instruction.get_operand(1).is_a_global_variable() == None:
                store_inst += instruction.type_of().get_int_type_width()
                len_store_inst += 1
    return load_inst / 8.0, len_load_inst, store_inst / 8.0, len_store_inst

def cmp_load_store_instructions(fun1, fun2):
    fun1_loads_dim, fun1_loads_count, fun1_store_dim, fun1_store_count = get_loads_stores(fun1)
    fun2_loads_dim, fun2_loads_count, fun2_store_dim, fun2_store_count = get_loads_stores(fun2)
    return [(fun1_loads_dim + fun2_loads_dim)/2, fun1_loads_dim - fun2_loads_dim, (fun1_loads_count + fun2_loads_count)/2, fun1_loads_count - fun2_loads_count, (fun1_store_dim + fun2_store_dim)/2, fun1_store_dim - fun2_store_dim, (fun1_store_count + fun2_store_count)/2, fun1_store_count - fun2_store_count]

def get_helper_names(module1, module2):
    helper_names = []
    for function1 in module1.iter_functions():
        if "helper_" in function1.get_name():
            helper_names.append(function1.get_name())

    for function2 in module2.iter_functions():
        if "helper_" in function2.get_name():
            helper_names.append(function2.get_name())
    return set(helper_names)

def get_helper_calls(function, helper_set):
    helper_names = [instruction.get_operand(instruction.get_num_operands() - 1).get_name() for bb in function.iter_basic_blocks() for instruction in bb.iter_instructions() if instruction.is_a_call_inst() != None]
    return Counter(helper_names)

def get_revamb_function_calls(function):

    revamb_calls = []
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if "bb." in  instruction.get_operand(instruction.get_num_operands()-1).get_name():
                    revamb_calls.append(instruction)

    return len(revamb_calls)


def cmp_revamb_function_calls(fun1, fun2):

    fun1_revamb_calls = get_revamb_function_calls(fun1)
    fun2_revamb_calls = get_revamb_function_calls(fun2)

    return [(fun1_revamb_calls + fun2_revamb_calls)/2, fun1_revamb_calls - fun2_revamb_calls]


def cmp_helper_calls(fun1, fun2):
    fun1_helper_calls = get_helper_calls(fun1, helper_names)
    fun2_helper_calls = get_helper_calls(fun2, helper_names)
    return [[(fun1_helper_calls[elem] + fun2_helper_calls[elem])/2, fun1_helper_calls[elem] - fun2_helper_calls[elem]] for elem in helper_names]

def get_indirect_calls(function):
    dispatcher_calls = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if instruction.get_operand(instruction.get_num_operands()-1).get_name() == "function_dispatcher":
                    dispatcher_calls += 1
    return dispatcher_calls

def cmp_indirect_calls(fun1, fun2):
    fun1_indirect_calls = get_indirect_calls(fun1)
    fun2_indirect_calls = get_indirect_calls(fun2)

    return [(fun1_indirect_calls + fun2_indirect_calls)/2, fun1_indirect_calls - fun2_indirect_calls]

def get_num_function_calls(function):
    count = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if "function_call" in instruction.print_value_to_string():
                    count += 1
    return count

def cmp_num_function_calls(fun1, fun2):
    fun1_function_calls = get_num_function_calls(fun1)
    fun2_function_calls = get_num_function_calls(fun2)

    return [(fun1_function_calls + fun2_function_calls)/2, fun1_function_calls - fun2_function_calls]



function_map = {
    #'graphic': show_graphic,
    #'opcodes': show_sim_opcode,
    'csv': compute_csv
}
parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('filename', nargs=3)
args= parser.parse_args()
function = function_map[args.command]
start_time = time()
function(args.filename)
end_time = time()
total_min = (end_time - start_time)/60 
total_sec = (end_time - start_time) - total_min * 60
print "\n Total time:", total_min, "min", total_sec, "sec" 
