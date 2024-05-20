x1 = "Listen"
x2 = "Silent"

x1 = x1.lower()
x2 = x2.lower()

x1_lst = [x for x in x1]
x2_lst = [x for x in x2]

x2_lst.sort()
x1_lst.sort()

print(x1_lst)
print(x2_lst)
if x1_lst == x2_lst:
    print("True")
else:
    print("False")