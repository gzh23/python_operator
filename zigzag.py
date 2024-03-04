def int_to_zigzag(n):
    return (n <<1) ^ (n >>31)

def int_to_int(n):
    return (n >>1) ^ -(n &1)

def zigzag_encode(data):
    return [int_to_zigzag(num) for num in data]

def zigzag_decode(data):
    return [int_to_int(num) for num in data]

if __name__ == "__main__":
    original_data = [int(x) for x in input("请输入待压缩数据（空格分隔的整数序列）：").split()]
    encoded_data = zigzag_encode(original_data)
    decoded_data = zigzag_decode(encoded_data)
    print("原始数据：", original_data)
    print("zigzag压缩结果：", encoded_data)
    print("解压缩结果：", decoded_data)
