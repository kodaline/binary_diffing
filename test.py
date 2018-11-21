#!/usr/bin/python

"""@package binarydiff
A tool for binary analysis based on rev.ng.

Binarydiff is an academic project aimed to analyze and compare binaries compiled with rev.ng. The following script is used to generate a csv file containing the data analyzed later with the rattle library for R.
"""
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
import random

def flatten(container):
    """Flat elements of a list.

    This function is used to flat arbitrarily nested lists into a single list.
    
    @type container: list
    @param container: The list to flatten.

    @rtype: list
    @return: Returns the flattened list.
    """
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def compute_csv(args):
    """Compute csv rows.

    This function computes the elements of the csv file to analyze. First of all it prepares the environment using llvmcpy, a python library with bindings for LLVM auto-generated from the LLVM-C API. Then it calls all the functions that must be computed in order to fill the rows of the analysis. The process of computation is multicored in order to optimize the computation process. 
    
    @type args: dict
    @param args: The number of arguments the script is taking. They are: the type of computation to execute (csv), the two files compiled with revamb to compare, the name of the output file where to write. 
    """
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
    header = list(flatten(['function1', 'function2', 'match', '#bb_mean', '#bb_diff', '#instr_mean', '#instr_diff', 'byte_size_mean', 'byte_size_diff', '#instructions_mean', '#instructions_diff', 'load_size_mean', 'load_size_diff', '#loads_mean', '#loads_diff', 'store_size_mean', 'store_size_diff', '#stores_mean', '#stores_diff', '#indirect_calls_mean', '#indirect_calls_diff', '#function_calls_mean', '#function_calls_diff']))
    header.extend(list(flatten([[str(elem) + "_mean", str(elem) + "_diff"] for elem in helper_names])))
    header.extend(list(flatten([[str(elem) + "_mean", str(elem) + "_diff"] for elem in list_opcodes])))

    #pool = Pool(cpus) 
    functions_list = [get_names, cmp_name, cmp_size_llvm_bb, cmp_size_llvm_instr, cmp_byte_size_num_instr, cmp_load_store_instructions, cmp_indirect_calls, cmp_revamb_function_calls, cmp_helper_calls, cmp_instruction_opcodes]
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
    
    filename = args[2]
    helper.write(rows, header, filename=filename)
    rewrite(rows, header, filename+"_shorter")

def rewrite(rows, header, filename):
    """Rewrite the csv in a smaller one, better comparable.

    This function is used to re-write the csv file taking all the rows with "match" column equal to True, and one row with "match" column equal to "False" for each different rev.ng generated function. 
    
    @type rows: list
    @param rows: The rows to write into the csv file.
    
    @type header: list
    @param header: The header of the csv, that is the name of each metric computed.
    
    @type filename: string
    @param filename: The name of the csv where to write the rows.
    """
    new_rows = []
    name = rows[0][0]
    old_name = name
    false_rows = []
    flag = True
    for row in rows:
        name = row[0]
        match = row[2]
        if name == old_name:
            if match == True:
                new_rows.append(row)
                flag = False
            else:
                if flag:
                    false_rows.append(row)
        else:
            flag = True
            if len(false_rows) > 0:
                new_rows.append(random.choice(false_rows))
            false_rows = []
            old_name = name 
    helper.write(new_rows, header, filename=filename)


def run(tuples_space, q_in, q_out, i):
    """Multiprocessing the computation.

    This function is used to create the queue of data that each process then pickles in order to compute the required metric and preparing it to be written then in the csv output file.

    @type tuples_space:
    @param tuples_space:
    
    @type q_in:
    @param q_in:
    
    @type q_out:
    @param q_out:
    """
    while True:
        # getting the data from the queue in
        (identifier, target) = q_in.get()
        # if target = -1
        if target == -1:
            break
        result = map(lambda x: pickle.loads(target)(x[0], x[1]), tuples_space)
        q_out.put((identifier, result))

def get_names(fun1, fun2):
    """Get the names of the two functions to compare.

    This function gets the name of the two functions that are compared, the first one taken from the first file given as input to the script, the second one taken from the second file. 
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns the list of names of the two given functions.
    """
    return [fun1.name, fun2.name]

def cmp_name(fun1, fun2):
    """Compare the names of the two functions.

    This function is used to compare the names of the two functions, populating the "match" column of the csv file, that is the ground truth. The result is "True" if the name is the same, "False" otherwise.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: Bool
    @return: Returns a boolean result obtained by comparing the string names of the two functions.
    """
    return fun1.name == fun2.name

def cmp_size_llvm_bb(fun1, fun2):
    """Sum and difference between the number of llvm basic blocks.

    This function is used to compute the mean and difference between the number of llvm basic blocks for the two functions in input.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns the result of the compared values between the number of llvm basic blocks, in this order: mean, difference.
    """
    bb1 = fun1.count_basic_blocks()
    bb2 = fun2.count_basic_blocks()
    return [(bb1 + bb2)/2, bb1-bb2]

def cmp_size_llvm_instr(fun1, fun2):
    """Sum and difference between the number of llvm instructions.

    This function is used to compute the mean and difference between the number of llvm instructions for the two functions in input.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns the result of the compared values between the number of llvm instructions for the two given functions.
    """
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
    """Get the opcodes for a given module.

    This function is used to get all the opcodes present in the modules of the two files in input.
    
    @type array_module: list
    @param array_module: Both modules obtained from the input files.
    
    @rtype: list
    @return: Returns the list of the set of opcodes in the two modules, sorted.
    """
    opcode_set = set()
    for module in array_module:
        for function in module.iter_functions():
            for bb in function.iter_basic_blocks():
                for instruction in bb.iter_instructions():
                    opcode_set.add(instruction.instruction_opcode)
    return sorted(list(opcode_set))

def get_opcode_dictionary(function):
    """Create a dictionary of the opcodes present in the modules.

    This function is used to create a dictionary of the various opcodes in a given function. 
    
    @type function: llvmcpyimpl.Value
    @param function: The function from where the opcodes are pulled out. 
    
    @rtype: dict
    @return: Returns the dictionary with the opcodes of the given function.
    """
    opcode_dictionary = defaultdict(lambda:0)

    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            opcode_dictionary[instruction.instruction_opcode] += 1
    return opcode_dictionary

def cmp_instruction_opcodes(fun1, fun2):
    """Compare the opcodes of the given functions.

    This function is used to compare the various opcodes used in the two given functions.  Both mean and difference of the values are returned.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns the list of compared values of the various opcodes used in the two given functions.
    """

    fun1_opcode_dictionary = get_opcode_dictionary(fun1)
    fun2_opcode_dictionary = get_opcode_dictionary(fun2)

    return [[(fun1_opcode_dictionary[elem] + fun2_opcode_dictionary[elem])/2, fun1_opcode_dictionary[elem] - fun2_opcode_dictionary[elem]] for elem in list_opcodes]


def get_byte_size(function):
    """Byte size of instructions and number of instructions.

    This function is used to get the number of instructions present in a given function and the size in bytes of the instructions in the function.
    
    @type function: llvmcpyimpl.Value
    @param function: The function which instructions are analyzed.
    
    @rtype: int, int
    @return: Returns the number of instructions counted using "newpc" and the byte size of the instructions. 
    """
    sum_size = 0
    num_instructions = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if instruction.get_num_operands() >= 4:
                    if instruction.get_operand(instruction.get_num_operands() - 1).get_name() == "newpc":
                        sum_size += helper.get_value_from_operand(instruction.get_operand(1))
                        num_instructions += 1
    return sum_size, num_instructions

def cmp_byte_size_num_instr(fun1, fun2):
    """Compare number of instructions and byte size of the instructions.

    This function compares the number of instructions present in the given functions and the byte size of the instructions of each function.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1 The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2 The function taken from the second input file.
    
    @rtype: list
    @return: Returns the compared values of number of instructions and size in bytes of the instructions for the two given functions, in this order: size mean, size difference, number of instructions mean, number of instructions difference.
    """
    fun1_size, fun1_num_instr = get_byte_size(fun1)
    fun2_size, fun2_num_instr = get_byte_size(fun2)
    return [(fun1_size + fun2_size)/2, fun1_size - fun2_size, (fun1_num_instr + fun2_num_instr)/2, fun1_num_instr - fun2_num_instr]

def get_loads_stores(function):
    """Get byte size of load and store operations.

    This function is used to get the size in bytes of the load and store operations for a given function.
    
    @type function: llvmcpyimpl.Value
    @param function The function to be analyzed.
    
    @rtype: int, int, int, int
    @return: Returns the size in bytes of the load/store operations and the number of load/store operations executed.
    """
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
    """Compare load and store operations.

    This function is used to compare the load and store operations for the given functions in terms of size in bytes, and number of.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun1: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns a list of the compared size of load and store operations of the given functions and the number of load/store operations, in this order: load size mean, load size difference, loads mean count, loads difference count, store size mean, store size difference, store count mean, store count difference. 
    """
    fun1_loads_size, fun1_loads_count, fun1_store_size, fun1_store_count = get_loads_stores(fun1)
    fun2_loads_size, fun2_loads_count, fun2_store_size, fun2_store_count = get_loads_stores(fun2)
    return [(fun1_loads_size + fun2_loads_size)/2, fun1_loads_size - fun2_loads_size, (fun1_loads_count + fun2_loads_count)/2, fun1_loads_count - fun2_loads_count, (fun1_store_size + fun2_store_size)/2, fun1_store_size - fun2_store_size, (fun1_store_count + fun2_store_count)/2, fun1_store_count - fun2_store_count]

def get_helper_names(module1, module2):
    """Get the names of all the helpers present in the modules.

    This function is used to get the names of all the helpers that are present in the two modules. It is used to give then the name of the columns where the number of calls for each helper is computed and written.
    
    @type module1: llvmcpyimpl.Module
    @param module1: The module for the first file in input.
    
    @type module2: llvmcpyimpl.Module 
    @param module2: The module for the second file in input. 
    
    @rtype: set
    @return: Returns the set of helper names of both modules. 
    """
    helper_names = []
    for function1 in module1.iter_functions():
        if "helper_" in function1.get_name():
            helper_names.append(function1.get_name())

    for function2 in module2.iter_functions():
        if "helper_" in function2.get_name():
            helper_names.append(function2.get_name())
    return set(helper_names)

def get_helper_calls(function, helper_set):
    """Get the calls to all helper functions.

    This function is used to get all the helper calls in a given function with the corresponding value of calls for each call.
    
    @type function: llvmcpyimpl.Value
    @param functioni: The function to be analyzed.
    
    @type helper_set: list 
    @param helper_set: The set of all the helpers of the two modules. 
    
    @rtype: dict
    @return: Returns a dictionary of the helpers present in the given function and a counter of the times it is called. The key type is string, the value type is int. 

    """
    helper_names = [instruction.get_operand(instruction.get_num_operands() - 1).get_name() for bb in function.iter_basic_blocks() for instruction in bb.iter_instructions() if instruction.is_a_call_inst() != None]
    return Counter(helper_names)

def get_revamb_function_calls(function):
    """Get the calls to revamb generated functions.

    This function is used to get the calls to the functions generated by revamb, usually their name start with "bb.".
    
    @type function: llvmcpyimpl.Value
    @param function: The function to be analyzed.
    
    @rtype: int
    @return: Returns the length of the calls to revamb generated functions.
    """
    revamb_calls = []
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if "bb." in  instruction.get_operand(instruction.get_num_operands()-1).get_name():
                    revamb_calls.append(instruction)

    return len(revamb_calls)


def cmp_revamb_function_calls(fun1, fun2):
    """Compare the calls to revamb functions.

    This function is used to compare the number of calls to functions  generated by revamb. 
    
    @type fun1: llvmcpyimpl.Value 
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns the compared values of calls identified by "function_call", in this order: mean, difference. 
    """

    fun1_revamb_calls = get_revamb_function_calls(fun1)
    fun2_revamb_calls = get_revamb_function_calls(fun2)

    return [(fun1_revamb_calls + fun2_revamb_calls)/2, fun1_revamb_calls - fun2_revamb_calls]


def cmp_helper_calls(fun1, fun2):
    """Compare the calls to helper functions.

    This function is used to compare the number of calls to helper functions.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1 The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2 The function taken from the second input file.
    
    @rtype: list
    @return: Returns a list of the compared values of calls to helper functions for the two given functions, in this order: mean, difference.
    """
    fun1_helper_calls = get_helper_calls(fun1, helper_names)
    fun2_helper_calls = get_helper_calls(fun2, helper_names)
    return [[(fun1_helper_calls[elem] + fun2_helper_calls[elem])/2, fun1_helper_calls[elem] - fun2_helper_calls[elem]] for elem in helper_names]

def get_indirect_calls(function):
    """Get the indirect calls, those to "function_dispatcher".

    This function is used to get the number of indirect calls, that is to function_dispatcher.
    
    @type function: llvmcpyimpl.Value
    @param function: The function to be analyzed.

    @rtype: int
    @return: Returns the count of calls to "function_dispatcher".
    """
    dispatcher_calls = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if instruction.get_operand(instruction.get_num_operands()-1).get_name() == "function_dispatcher":
                    dispatcher_calls += 1
    return dispatcher_calls

def cmp_indirect_calls(fun1, fun2):
    """Compare the indirect calls, those to "function_dispatcher".

    This function is used to compare the number of indirect calls.
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns the list of the compared values between the indirect calls to "function_dispatcher"  in this order: mean value, difference.
    """
    fun1_indirect_calls = get_indirect_calls(fun1)
    fun2_indirect_calls = get_indirect_calls(fun2)

    return [(fun1_indirect_calls + fun2_indirect_calls)/2, fun1_indirect_calls - fun2_indirect_calls]

def get_num_function_calls(function):
    """Get the number of function calls.

    This function is used to get the number of function calls in a given function, that is the calls identified by "function_call".
    
    @type function: llvmcpyimpl.Value 
    @param function: The function to be analyzed.

    @rtype: int
    @return: Returns the count of the calls identified by "function_call".args 
    """
    count = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_call_inst() != None:
                if "function_call" in instruction.print_value_to_string():
                    count += 1
    return count

def cmp_num_function_calls(fun1, fun2):
    """Compare the number of function calls made by the two functions.

    This function is used to compare the number of function calls made by the two compared functions. 
    
    @type fun1: llvmcpyimpl.Value
    @param fun1: The function taken from the first input file.
    
    @type fun2: llvmcpyimpl.Value
    @param fun2: The function taken from the second input file.
    
    @rtype: list
    @return: Returns the list of the compared values between the calls to "function_call" in this order: mean value, difference. 
    """
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
