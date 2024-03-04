import struct


def xor_float_operator(sequence):
    result = [struct.unpack('I', struct.pack('f', sequence[0]))[0]]

    for i in range(1, len(sequence)):
        a_bytes = struct.pack('f', sequence[i])
        b_bytes = struct.pack('f', sequence[i - 1])

        result_bytes = bytes(a ^ b for a, b in zip(a_bytes, b_bytes))
        result_int = struct.unpack('I', result_bytes)[0]

        result.append(result_int)

    return result


def xor_float_decode(sequence):
    result = [struct.unpack('f', struct.pack('I', sequence[0]))[0]]

    for i in range(1, len(sequence)):
        prev_float_bytes = struct.pack('f', result[i - 1])
        xor_result_bytes = struct.pack('I', sequence[i])

        decoded_bytes = bytes(b ^ xor_byte for b, xor_byte in zip(
            prev_float_bytes, xor_result_bytes))
        decoded_float = struct.unpack('f', decoded_bytes)[0]

        result.append(decoded_float)

    return result
