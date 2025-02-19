import math

# Step 1: Convert large numbers to floats
# Step 2: Perform the exponentiation and modulus with floating-point arithmetic
# Step 3: Check if the cyphertexts match

# ciphertext = (plaintext^exponent) mod modulus is the formula I'm using
# 999998727899999 is a prime number. this is similar to 
# what is done in rsa.(in rsa a large prime number is taken).
modulus = 999998727899999
exponent = 17
plaintext = 123456789


ciphertext_int = pow(plaintext, exponent, modulus)


modulus_float = float(modulus)
plaintext_float = float(plaintext)


ciphertext_float = int((plaintext_float ** exponent) % modulus_float)


print("Correct integer-based ciphertext:", ciphertext_int)
print("Float-based ciphertext (with precision loss):", ciphertext_float)


if ciphertext_int == ciphertext_float:
    print("No precision error encountered.")
else:
    print("Precision error: ciphertexts do not match!")