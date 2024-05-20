m = 6
n = 30

sum_divisible = 0
sum_ndivisible = 0

for i in range(31):
    if i % 6 == 0:
        sum_divisible += i
    else:
        sum_ndivisible += i


print("Sum Divisible %d" % sum_divisible)
print("Sum Not Divisible %d" % sum_ndivisible)