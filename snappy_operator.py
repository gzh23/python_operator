import snappy


def snappy_encode(data_bytes):
    if not data_bytes:
        return b""
    return snappy.compress(data_bytes)


def snappy_decode(encoded_bytes):
    if not encoded_bytes:
        return b""
    return snappy.decompress(encoded_bytes)


if __name__ == "__main__":
    original_data = b"Example data that needs to be compressed using snappy."
    print("原始数据：", original_data)

    snappy_encoded_data = snappy_encode(original_data)
    print("Snappy编码结果：", snappy_encoded_data)

    snappy_decoded_data = snappy_decode(snappy_encoded_data)
    print("Snappy解码结果：", snappy_decoded_data)
