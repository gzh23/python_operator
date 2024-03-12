def delta_operator(sequence):
    transformed_sequence = []

    # 分段为8个8个的
    segmented_sequence = [sequence[i:i+8] for i in range(0, len(sequence), 8)]

    for segment in segmented_sequence:
        transformed_segment = [segment[0]]  # 第一个数不变
        for i in range(1, len(segment)):
            # 其与数字变为与前一个数字的差值
            difference = segment[i] - segment[i-1]
            transformed_segment.append(difference)

        transformed_sequence.extend(transformed_segment)

    return transformed_sequence


def delta_decode(sequence):
    result = []
    segmented_sequence = [sequence[i:i+8] for i in range(0, len(sequence), 8)]

    for segment in segmented_sequence:
        transformed_segment = [segment[0]]
        for i in range(1, len(segment)):
            difference = segment[i] + transformed_segment[i-1]
            transformed_segment.append(difference)

        result.extend(transformed_segment)

    return result


if __name__ == "__main__":
    original_sequence = [1, 5, 9, 15, 22, 30, 39, 49, 2, 7, 14,
                         23, 33, 44, 56, 70, 3, 9, 16, 25, 35, 46, 58, 71, 5, 10]
    transformed_sequence = delta_operator(original_sequence)

    print("原始序列:", original_sequence)
    print("变换后的序列:", transformed_sequence)

    decoded_sequence = delta_decode(transformed_sequence)
    print("解码后的序列:", decoded_sequence)
