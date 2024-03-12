# Fibonacci encoding and decoding
# Fibonacci number to be used
N = 30
fib = [0 for i in range(N)]

# Stores values in fib and returns index of

# the largest fibonacci number smaller than n.


def largestFiboLessOrEqual(n):
    fib[0] = 1  # Fib[0] stores 2nd Fibonacci No.
    fib[1] = 2  # Fib[1] stores 3rd Fibonacci No.
    i = 2
    while fib[i - 1] <= n:
        fib[i] = fib[i - 1] + fib[i - 2]
        i += 1
    return (i - 2)


def fibonacciEncoding(n):
    index = largestFiboLessOrEqual(n)
    # allocate memory for codeword
    codeword = ['a' for i in range(index + 2)]
    i = index
    while (n):
        # Mark usage of Fibonacci f (1 bit)
        # print(fib[i])
        codeword[i] = '1'
        # Subtract f from n
        n = n - fib[i]
        # Move to Fibonacci just smaller than f
        i = i - 1
        # Mark all Fibonacci > n as not used (0 bit),
        # progress backwards
        while (i >= 0 and fib[i] > n):
            codeword[i] = '0'
            i = i - 1
    codeword[index + 1] = '1'
    return "".join(codeword)


def fibonacciEncodings(values):
    return [fibonacciEncoding(value) for value in values]


def fibonacciDecoding(s):
    n = len(s)
    fib = [0] * n
    fib[0], fib[1] = 1, 2
    for i in range(2, n):
        fib[i] = fib[i-1] + fib[i-2]
    res = 0
    for i in range(n-1):
        if s[i] == '1':
            res += fib[i]
    return res


def fibonacciDecodings(values):
    return [fibonacciDecoding(value) for value in values]


if __name__ == "__main__":
    input_list = [1, 2, 4, 5, 6]
    encoded_result = fibonacciEncodings(input_list)
    print(encoded_result)
    decoded_result = fibonacciDecodings(encoded_result)
    print(decoded_result)
