import lz4.frame


def lz4_encode(data_bytes):
    if not data_bytes:
        return b""
    return lz4.frame.compress(data_bytes)


def lz4_decode(encoded_bytes):
    if not encoded_bytes:
        return b""
    return lz4.frame.decompress(encoded_bytes)


if __name__ == "__main__":
    original_data = b"Example data that needs to be compressed using lz4."
    print("原始数据：", original_data)

    lz4_encoded_data = lz4_encode(original_data)
    print("LZ4编码结果：", lz4_encoded_data)

    lz4_decoded_data = lz4_decode(lz4_encoded_data)
    print("LZ4解码结果：", lz4_decoded_data)
