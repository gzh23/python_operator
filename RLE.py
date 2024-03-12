def encode_rle(input_list):
    if input_list == []:
        return [], []
    value_list = [input_list[0]]
    count_list = [1]

    for i in range(1, len(input_list)):
        if input_list[i] == input_list[i-1]:
            count_list[-1] += 1
        else:
            value_list.append(input_list[i])
            count_list.append(1)
    return value_list, count_list


def decode_rle(value_list, count_list):
    if value_list == []:
        return []
    decoded_result = []
    for i in range(len(value_list)):
        decoded_result += count_list[i] * [value_list[i]]
    return decoded_result


if __name__ == "__main__":
    original_data = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    print("原始数据：", original_data)
    encoded_value, encoded_count = encode_rle(original_data)
    print("RLE压缩结果：", encoded_value, encoded_count)
    decoded_result = decode_rle(encoded_value, encoded_count)
    print("解压缩结果：", decoded_result)
