import struct

def xor_float_operator(sequence):
    result = [struct.unpack('I',struct.pack('f', sequence[0]))[0]]

    for i in range(1, len(sequence)):
        a_bytes = struct.pack('f', sequence[i])
        b_bytes = struct.pack('f', sequence[i - 1])

        result_bytes = bytes(a ^ b for a, b in zip(a_bytes, b_bytes))
        result_int = struct.unpack('I', result_bytes)[0]
        
        result.append(result_int)
        
    return result
