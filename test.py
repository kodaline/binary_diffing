#!/usr/bin/python
from collections import defaultdict
import argparse
import sys
from llvmcpy.llvm import *
import math
import matplotlib.pylab as plt
import helper

def show_graphic(args):
    for filename in args:
        opcode_dictionary = defaultdict(lambda:0)
        
        buffer = create_memory_buffer_with_contents_of_file(filename)
        context = get_global_context()
        module = context.parse_ir(buffer)
        for function in module.iter_functions():
            for bb in function.iter_basic_blocks():
                for instruction in bb.iter_instructions():
                    opcode_dictionary[instruction.instruction_opcode] += 1
                    
        
        
        lists = sorted(opcode_dictionary.items()) # sorted by key, return a list of tuples
        
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        
        plt.plot(x, y, label=filename)
        plt.legend(loc='best')
    plt.show()


def show_sim_opcode(args):
    buffer_1 = create_memory_buffer_with_contents_of_file(args[0])
    buffer_2 = create_memory_buffer_with_contents_of_file(args[1])
    context = get_global_context()
    module_1 = context.parse_ir(buffer_1)
    module_2 = context.parse_ir(buffer_2)
    l_bb_1 = [bb for fun_1 in module_1.iter_functions() for bb in fun_1.iter_basic_blocks()]
    l_bb_2 = [bb for fun_2 in module_2.iter_functions() for bb in fun_2.iter_basic_blocks()]
    l_bb_1_len = len(l_bb_1)
    l_bb_2_len = len(l_bb_2)
    if len(l_bb_1) > len(l_bb_2):
        first = l_bb_2
        last = l_bb_1
    else:
        first = l_bb_1
        last = l_bb_2

    one_shot_match = 0
    drift_match = 0
    never_matched = 0
    found = False
    i = 0
    while(True):
        if not found:
            i += 1
        if (i >= len(first) or i >= len(last)):
            break
        j = i
        found = False
        while (j < len(last)):
            opcode_1 = [inst.instruction_opcode for inst in first[i].iter_instructions()]
            opcode_2 = [inst.instruction_opcode for inst in last[j].iter_instructions()]
            if opcode_1 == opcode_2:
                if i == j:
                    #print i , "and" , j , "match!"
                    first.remove(first[i])
                    last.remove(last[j])
                    one_shot_match += 1
                else:
                    #print "...found with" , j
                    first.remove(first[i])
                    last.remove(last[j])
                    drift_match += 1
                found = True
                break
            else:
                if i == j:
                    pass
                    #print i , "and" , j , "does NOT match! Starting to drift until match..."
                j += 1
        if not found:
            never_matched += 1
            #print "...match never found"
    print "#BB for module 1: ", l_bb_1_len, "\n#BB for module 2: ", l_bb_2_len
    print "Match at first attempt: ", one_shot_match, "\nMatch after some drift: ", drift_match, "\nNever matched: ", never_matched
    print "Remaining unmatched blocks:\n", first, "\n", last

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def compute_csv(args):
    
    buffer_1 = create_memory_buffer_with_contents_of_file(args[0])
    buffer_2 = create_memory_buffer_with_contents_of_file(args[1])
    context = get_global_context()
    module_1 = context.parse_ir(buffer_1)
    module_2 = context.parse_ir(buffer_2)
    list_opcodes = get_opcodes([module_1, module_2])
    
    helper_names = get_helper_names(module_1, module_2)
    
    header = list(flatten(['function1', 'function2', 'match', '#bb_sum', '#bb_diff', '#instr_sum', '#instr_diff', 'byte_dim_sum', 'byte_dim_diff', '#instructions_sum', '#instructions_diff', 'load_dim_sum', 'load_dim_diff', '#loads_sum', '#loads_diff', 'store_dim_sum', 'store_dim_diff', '#lstores_sum', '#stores_diff', '#indirect_calls_sum', '#indirect_calls_diff', '#revamb_function_calls_sum', '#revamb_function_calls_diff', '#function_calls_sum', '#function_calls_diff']))
    header.extend(list(flatten([[str(elem) + "_sum", str(elem) + "_diff"] for elem in helper_names])))
    
    header.extend(list(flatten([[str(elem) + "_sum", str(elem) + "_diff"] for elem in list_opcodes])))
    
    rows = [list(flatten([fun1.name, fun2.name, compare_name(fun1, fun2), cmp_dimension_llvm_bb(fun1, fun2), cmp_dimension_llvm_instr(fun1, fun2), cmp_byte_dimension_num_instr(fun1, fun2), cmp_load_instructions(fun1, fun2), cmp_store_instructions(fun1, fun2), cmp_indirect_calls(fun1, fun2), cmp_revamb_function_calls(fun1, fun2), cmp_num_function_calls(fun1, fun2), cmp_helper_calls(fun1, fun2,helper_names), cmp_instruction_opcodes(fun1, fun2, list_opcodes)])) for fun1 in module_1.iter_functions() for fun2 in module_2.iter_functions() if "bb." in fun1.name and "bb." in fun2.name]
    
    helper.write(rows, header)            

def compare_name(fun1, fun2):
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
    return [bb1 + bb2, math.fabs(bb1-bb2)]
    
def cmp_dimension_llvm_instr(fun1, fun2):
    count1 = 0
    for bb in fun1.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            count1 += 1
    count2 = 0
    for bb in fun2.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            count2 += 1

    return [count1 + count2, math.fabs(count1 - count2)]

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

def cmp_instruction_opcodes(fun1, fun2, opcode_list):

    fun1_opcode_dictionary = get_opcode_dictionary(fun1)
    fun2_opcode_dictionary = get_opcode_dictionary(fun2)

    return [[fun1_opcode_dictionary[elem] + fun2_opcode_dictionary[elem], fun1_opcode_dictionary[elem] - fun2_opcode_dictionary[elem]] for elem in opcode_list]


def get_byte_dimension(function):
    sum_dimension = 0
    num_instructions = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if "newpc" in instruction.print_value_to_string():
                sum_dimension += helper.get_value_from_operand(instruction.get_operand(1))
                num_instructions += 1
    return sum_dimension, num_instructions

def cmp_byte_dimension_num_instr(fun1, fun2):
    fun1_dim, fun1_num_instr = get_byte_dimension(fun1)
    fun2_dim, fun2_num_instr = get_byte_dimension(fun2)
    return [fun1_dim + fun2_dim, fun1_dim - fun2_dim, fun1_num_instr + fun2_num_instr, fun1_num_instr - fun2_num_instr]

def get_loads(function):
    load_inst = []
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_load_inst() != None and instruction.get_operand(0).is_a_global_variable() == None:
                load_inst.append(instruction.type_of().get_int_type_width())
    return sum(load_inst), len(load_inst)

def cmp_load_instructions(fun1, fun2):
    
    fun1_loads_dim, fun1_loads_count = get_loads(fun1) 
    fun2_loads_dim, fun2_loads_count = get_loads(fun2)
    
    return [fun1_loads_dim + fun2_loads_dim, fun1_loads_dim - fun2_loads_dim, fun1_loads_count + fun2_loads_count, fun1_loads_count - fun2_loads_count]
   
def get_stores(function):
    store_inst = []
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if instruction.is_a_store_inst() != None and instruction.get_operand(1).is_a_global_variable() == None:
                store_inst.append(instruction.get_operand(0).type_of().get_int_type_width())
    return sum(store_inst), len(store_inst)

def cmp_store_instructions(fun1, fun2):

    fun1_store_dim, fun1_store_count = get_stores(fun1) 
    fun2_store_dim, fun2_store_count = get_stores(fun2) 
    
    return [fun1_store_dim + fun2_store_dim, fun1_store_dim - fun2_store_dim, fun1_store_count + fun2_store_count, fun1_store_count - fun2_store_count]

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
    helper_calls = defaultdict(lambda:0)

    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            for elem in helper_set:
                if elem in instruction.print_value_to_string():
                    helper_calls[elem] += 1
    return helper_calls

def get_revamb_function_calls(function):

    revamb_calls = []
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if "bb." in instruction.print_value_to_string() and instruction.is_a_call_inst() != None:
                revamb_calls.append(instruction)

    return len(revamb_calls)


def cmp_revamb_function_calls(fun1, fun2):

    fun1_revamb_calls = get_revamb_function_calls(fun1)
    fun2_revamb_calls = get_revamb_function_calls(fun2)

    return [fun1_revamb_calls + fun2_revamb_calls, fun1_revamb_calls - fun2_revamb_calls]


def cmp_helper_calls(fun1, fun2, helper_set): 
               
    fun1_helper_calls = get_helper_calls(fun1, helper_set)
    fun2_helper_calls = get_helper_calls(fun2, helper_set)
    
    return [[fun1_helper_calls[elem] + fun2_helper_calls[elem], fun1_helper_calls[elem] - fun2_helper_calls[elem]] for elem in helper_set]

def get_indirect_calls(function):
    
    dispatcher_calls = []
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if "function_dispatcher" in instruction.print_value_to_string():
                dispatcher_calls.append(instruction)
    return len(dispatcher_calls)

def cmp_indirect_calls(fun1, fun2):
    
    fun1_indirect_calls = get_indirect_calls(fun1)
    fun2_indirect_calls = get_indirect_calls(fun2)

    return [fun1_indirect_calls + fun2_indirect_calls, fun1_indirect_calls - fun2_indirect_calls]

def get_num_function_calls(function):
    count = 0
    for bb in function.iter_basic_blocks():
        for instruction in bb.iter_instructions():
            if "function_call" in instruction.print_value_to_string() and instruction.is_a_call_inst() != None:
                count += 1    
    return count 

def cmp_num_function_calls(fun1, fun2):
    
    fun1_function_calls = get_num_function_calls(fun1)
    fun2_function_calls = get_num_function_calls(fun2)

    return [fun1_function_calls + fun2_function_calls, fun1_function_calls - fun2_function_calls]



function_map = { 
    'graphic': show_graphic,
    'opcodes': show_sim_opcode,
    'csv': compute_csv
}
parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('filename', nargs=2)
args= parser.parse_args()
function = function_map[args.command]
function(args.filename)
