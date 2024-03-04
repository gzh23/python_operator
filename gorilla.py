import struct


def compress_float_sequence(data):
    compressed_data = bytearray()
    prev_value = 0.0
    for value in data:
        diff = value - prev_value
        diff_bytes = struct.pack('<f', diff)
        diff_len = len(diff_bytes) * 8
        compressed_data += struct.pack('<I', diff_len)
        compressed_data += diff_bytes
        prev_value = value
    return compressed_data


def decompress_float_sequence(compressed_data):
    decompressed_data = []
    offset = 0
    prev_value = 0.0
    while offset < len(compressed_data):
        diff_len = struct.unpack('<I', compressed_data[offset:offset + 4])[0]
        offset += 4
        if offset + diff_len // 8 <= len(compressed_data):
            diff_val = struct.unpack(
                '<f', compressed_data[offset:offset + diff_len // 8])[0]
            offset += diff_len // 8
            prev_value += diff_val
            decompressed_data.append(prev_value)
        else:
            break
    return decompressed_data


if __name__ == "__main__":
    original_data = [float(x) for x in input("请输入待压缩数据（空格分隔的浮点数序列）：").split()]
    compressed_data = compress_float_sequence(original_data)
    decompressed_data = decompress_float_sequence(compressed_data)
    print("原始数据：", original_data)
    print("gorilla压缩结果：", compressed_data)
    print("解压缩结果：", decompressed_data)
