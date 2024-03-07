def encode_rle(input_list):
    # if not input_string:
    #     return []
    # input_list = list(map(int, input_string.split()))  # 将输入的字符串解析为整数列表
    # if not input_list:
    #     return []
    count_list = []
    value_list = []
    current_digit = input_list[0]
    count = 1
    for digit in input_list[1:]:
        if digit == current_digit:
            count += 1
        else:
            count_list.append(count)
            value_list.append(current_digit)
            # encoded_result.append(f'[{count}]{current_digit}')
            current_digit = digit
            count = 1
    # 处理最后一个元素
    count_list.append(count)
    value_list.append(current_digit)
    # encoded_result.append(f'[{count}]{current_digit}')
    return count_list+value_list


def decode_rle(encoded_list):
    decoded_result = ""
    for item in encoded_list:
        count_str, digit = item[1:].split(']')
        count = int(count_str)
        decoded_result += count * str(digit) + ' '  # 添加空格分隔
    return decoded_result.rstrip()  # 移除末尾多余的空格


if __name__ == "__main__":
    # original_data = [int(x) for x in input("请输入待压缩数据（空格分隔的整数序列）：").split()]
    original_data = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    print("原始数据：", original_data)
    encoded_result = encode_rle(original_data)
    print("RLE压缩结果：", encoded_result)
    decoded_result = decode_rle(encoded_result)
    print("解压缩结果：", decoded_result)
