import numpy as np

def firstL(vector):
    tot = 0
    for i in vector:
        tot += abs(i)
    return tot

def secondL(vector):
    tot = 0
    for i in vector:
        tot += i ** 2
    return tot ** 0.5

def infinityN(vector):
    return max(abs(i) for i in vector)

def pNorm(vector, p):
    return sum(abs(i) ** p for i in vector) ** (1 / p)

def convert_string_to_list(input_string):
    return [float(num) for num in input_string.split()]

def get_vector():
    while True:
        _input = input("Enter a space-separated vector: ").strip()
        if not _input:
            print("No input detected. Please enter a valid vector.")
            continue
        try:
            vector = convert_string_to_list(_input)
            return vector
        except ValueError:
            print("Invalid input. Please enter numeric values only, separated by spaces.")

def main():
    vector = get_vector()
    print("Valid vector: {}".format(vector))
    print("First norm (L1): {}".format(firstL(vector)))
    print("Second norm (L2): {}".format(secondL(vector)))
    print("Infinity norm (L∞): {}".format(infinityN(vector)))

    p = float(input("Enter a positive value for p: "))
    if p > 0:
        print("P-norm (p={}): {}".format(p, pNorm(vector, p)))
    else:
        print("Please enter a valid positive number for p.")

if __name__ == "__main__":
    main()
