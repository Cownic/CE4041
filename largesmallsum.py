# 2. Execute the given function.
# def LargeSmallSum(arr)
# The function takes an integral arr which is of the size or length of its arguments. Return the sum of the second smallest element at odd position ‘arr’ and the second largest element at the even position.
# Assumption
# Every array element is unique.
# Array is 0 indexed.

arr = [3,2,1,7,5,4]
arr_even = []
arr_odd = []

for i in range(len(arr)):
    if i % 2 == 0:
        arr_even.append(arr[i])
    else:
        arr_odd.append(arr[i])
arr_even.sort()
arr_odd.sort()



print(arr_even[1])
print(arr_odd[1])



