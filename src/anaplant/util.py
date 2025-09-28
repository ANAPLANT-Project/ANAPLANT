def str_to_float(data: str, decimal_comma: bool = False):
    """
    Convert a string to a float, optionally treating a comma as a decimal 
    and vice versa using german floating point notation
    """
    if decimal_comma:
        return decimal_comma_str_to_float(data)
    else:
        return float(data)
    
def decimal_comma_str_to_float(data: str):
    out_str = ''
    for char in data:
        if char == ',':
            out_str += '.'
        elif char == '.':
            out_str += ','
        else:
            out_str += char
    return float(out_str)