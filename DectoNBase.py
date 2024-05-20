# For writing numbers, there is a system called N-base notation. This system uses only N-based symbols. It uses symbols that are listed as the first n symbols. Decimal and n-based notations are 0:0, 1:1, 2:2, …, 10:A, 11:B, …, 35:Z.

# Perform the function: Chats DectoNBase(int n, int num)

# This function only uses positive integers. Use a positive integer n and num to find out the n-base that is equal to num.

# Steps

# Select a decimal number and divide it by n. Consider this as an integer division.
# Denote the remainder as n-based notation.
# Again divide the quotient by n.
# Repeat the above steps until you get a 0 remainder.
# The remainders from last to first are the n-base values.

N = 12
Num = 718
answer = ""

Remainder_lst = []

while Num > 0:
    Remainder = Num % N
    Num = Num // N
    Remainder_lst.append(Remainder)


for i in range(len(Remainder_lst)-1 , -1 , -1):
    if Remainder_lst[i] > 9:
        letter = i - 10
        letter += 64
        answer += chr(letter)
    else:
        answer += str(Remainder_lst[i])

print(answer)