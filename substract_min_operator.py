def substract_min_operator(sequence):
    transformed_sequence = []

    # 分段为8个8个的
    segmented_sequence = [sequence[i:i+8] for i in range(0, len(sequence), 8)]

    for segment in segmented_sequence:
        min_value = min(segment)
        transformed_segment = [elem - min_value for elem in segment]

        transformed_sequence.extend(transformed_segment)

    return transformed_sequence

# original_sequence = [1, 5, 9, 15, 22, 30, 39, 49, 2, 7, 14, 23, 33, 44, 56, 70, 3, 9, 16, 25, 35, 46, 58, 71, 5, 10]
# transformed_sequence = substract_min_operator(original_sequence)

# print("原始序列:", original_sequence)
# print("变换后的序列:", transformed_sequence)