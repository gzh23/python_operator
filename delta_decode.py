def delta_decode(sequence):
    result = [sequence[0]]
    for i in range(1, len(sequence)):
        result.append(result[i - 1] + sequence[i])
    return result
