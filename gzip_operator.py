import gzip


def gzip_encode(data_bytes):
    if not data_bytes:
        return b""
    return gzip.compress(data_bytes)


def gzip_decode(encoded_bytes):
    if not encoded_bytes:
        return b""
    return gzip.decompress(encoded_bytes)


if __name__ == "__main__":
    original_data = b"Example data that needs to be compressed using gzip."
    print("原始数据：", original_data)

    encoded_data = gzip_encode(original_data)
    print("GZIP编码结果：", encoded_data)

    decoded_data = gzip_decode(encoded_data)
    print("GZIP解码结果：", decoded_data)
