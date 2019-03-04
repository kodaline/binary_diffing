import csv

def write(content, header, filename='metrics', delimiter_char=','):
    """ Write the result on the file.

    
    @param filename: The name of the file to create, 'metrics'.
    @param content: The content to save.
    @header: The header of the csv file.
    @param delimiter_char: The separator char for fields (csv).
    
    @return: None.
    """
    fp = open('data/' + filename + '.csv', 'w', 0)
    writer = csv.writer(fp, delimiter=delimiter_char, quoting=csv.QUOTE_NONE)
    writer.writerow(header)
    writer.writerows(content)
    fp.close()

def get_value_from_operand(operand):
    """ Recovering a value from a given operand.
    
    @param operand: The operand from which the value is recovered
    @return: Returns the value of the operand as a string. 
    """
    return int(operand.print_value_to_string().split(' ')[1])

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
