# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings
# Press the green button in the gutter to run the script.

fs = open('file.txt')
s = fs.readline()
inp = ''
while s != '':
    inp += s
    s = fs.readline()

print(inp)

inp = inp.lower().split()

d = set(inp)  # множество
max1 = 0
for word in d:
    if inp.count(word) > max1:
        max1 = inp.count(word)

res = dict()
for word in d:
    if inp.count(word) == max1:
        res[word] = max1

print(min(res), ' ', res[min(res)])
