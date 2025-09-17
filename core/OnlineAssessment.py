# Virtual Financial


# Write a function that, given an integer X, returns an integer that corresponds to the minimum number of steps
# required to change X to a Fibonacci number. In each step you can either increment or decrement the current number,
# i.e., you can change X to either X + 1 or X – 1. X will be between 0 and 1,000,000 inclusive.
# The Fibonacci sequence is defined as follows:F[0] = 0, F[1] = 1, for each i >= 2: F[i] = F[i-1] + F[i-2]
# The elements of the Fibonacci sequence are called Fibonacci numbers.

# Examples:
# For X = 15 the function should return 2.
# For X = 1 or X = 13 the function should return 0.

def min_steps_to_fibonacci(X):
    if X < 0 or X > 1000000:
        raise ValueError("X must be between 0 and 1,000,000 inclusive.")

    # Generate Fibonacci numbers up to 1,000,000
    fib = [0, 1]
    while True:
        next_fib = fib[-1] + fib[-2]
        if next_fib > 1000000:
            break
        fib.append(next_fib)

    # Find the closest Fibonacci number to X
    min_steps = float('inf')
    for f in fib:
        steps = abs(X - f)
        if steps < min_steps:
            min_steps = steps
    return min_steps

print(min_steps_to_fibonacci(15))
print(min_steps_to_fibonacci(13))

# Question 2
# Write a function that, given a zero-indexed array A consisting of N integers representing the initial test scores of
# a row of students, returns an array of integers representing their final test scores (in the same order).
#
# There is a group of students who sit next to each other in a row. Each day, the students study together and take
# a test at the end of the day. Test scores for a given student can only change once per day as follows:

# If a student sits immediately between two students with better scores, that student’s score will improve by 1
# when they take the next test.
# If a student sits immediately between two students with worse scores, that student’s score will decrease by 1.
#
# This process will repeat each day as long as at least one student’s score changes.
# Note that the first and last student in the row never change their scores as they never sit between two students.
# Return an array representing the final test scores for each student.
# You can assume that:
# The number of students is in the range 1 to 1,000.
# Scores are in the range 0 to 1,000.
#
# Example:
# Input:[1, 6, 3, 4, 3, 5]
# Output:[1, 4, 4, 4, 4, 5]

def final_scores(students):
    n = len(students)
    if n < 3:
        return students
    changed = True
    while changed:
        changed = False
        students_copy = students[:]
        for i in range(1, n-1, 1):
            if students[i] < students[i-1] and students[i] < students[i+1]:
                students_copy[i] += 1
                changed = True
            elif students[i] > students[i-1] and students[i] > students[i+1]:
                students_copy[i] += 1
                changed = True
        students = students_copy
    return students
print(final_scores([1, 6, 3, 4, 3, 5]))
print(final_scores([1, 6, 5, 4, 3, 2, 1, 10, 5]))

# Question 3
# Write a function that, given a string S encoding a decimal integer N, returns a string representing the HexSpeak
# representation H of N if H is a valid HexSpeak word, or else "ERROR".

# A decimal number can be converted to HexSpeak by:
# 1. Converting it to hexadecimal (in upper case).
# 2. Converting the number 0 to the letter "O" and the number 1 to the letter "I".

# A string is considered a valid HexSpeak word if it consists only of the letters A, B, C, D, E, F, I, O.
# The input string S will represent a decimal integer between 1 and 1,000,000,000,000 inclusive.

# Examples:
# If the input string S = "257", the decimal number it encodes is 257 which is written as 101 in hexadecimal.
# Since 1 and 0 represent "I" and "O", respectively, we should return "IOI".
# If the input string S = "3", it is written as 3 in hexadecimal, which does not represent a HexSpeak letter, so we return "ERROR".

dic = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
       10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F'}
def decimal_to_hexspeak(S):
    num = int(S)
    str = ''
    while num != 0:
        num, mod = divmod(num, 16)
        str = dic[mod] + str
    str = list(str)
    if '2' in str or '3' in str or '4' in str or '5' in str or '6' in str or '7' in str or '8' in str or '9' in str:
        return "Error"
    else:
        for i in range(len(str)):
            if str[i] == '0':
                str[i] = 'O'
            elif str[i] == '1':
                str[i] = 'I'
        return ''.join(str)
print(decimal_to_hexspeak("257"))
print(decimal_to_hexspeak("3"))
print(decimal_to_hexspeak("507"))

# Better way
def toHexSpeak(S: str) -> str:
    n = int(S)
    hex_str = hex(n)[2:].upper()
    hex_str = hex_str.replace("0", "O").replace("1", "I")
    valid_chars = set("ABCDEFIO")
    if all(c in valid_chars for c in hex_str):
        return hex_str
    else:
        return "ERROR"

# Question 4
# There is a box with a capacity of 5000 grams. The box may already contain some items, reducing its capacity.
# You’ll be adding apples to that box until it is full. Write a function that, given a zero-indexed array A
# consisting of N integers, representing the weight of items already in the box and each apple’s weight,
# returns the maximum number of apples that could fit in the box, without exceeding its capacity.
# The input array consists of an integer K as the first element, representing the sum of the weights of items already
# contained in the box, followed by zero or more integers representing individual apple weights.

# You may assume that A contains between 1 and 100 elements and that every number in it is between 0 and 5000.

# Example:
# Input:[4650, 150, 150, 150]
# Output:2

# Explanation: The box already contains 4650 grams of items, so only 2 more apples of weight 150 would fit
# (bringing the total weight to 4950, still below the capacity).

def max_apples(A):
    capacity = 5000
    current_weight = A[0]
    apples = sorted(A[1:])

    count = 0
    for apple in apples:
        if current_weight + apple <= capacity:
            current_weight += apple
            count += 1
        else:
            return apples
print(max_apples([4650, 150, 150, 150]))


# Question 5
# Write a function that, given a string S, returns an integer that represents the number of ways we can select a
# non-empty substring of S in which all of the characters are identical.

# Example:
# The string "zzzyz" contains 8 such substrings:
#
# Four instances of "z",
# Two of "zz",
# One of "zzz",
# And one of "y".

# String "k" contains only one such substring: "k".
#
# You may assume that the length of S is between 1 and 100, and each character in S is a lowercase letter (a–z).


def count_identical_substrings(S):
    n = len(S)
    count = n

    # Base case and use sliding window
    for i in range(n):
        j = i + 1
        while j < n and S[j] == S[i]:
            count += 1
            j += 1
    return count
print(count_identical_substrings("zzzyz"))  # 8 (z,z,z,zz,zz,zzz,y)
print(count_identical_substrings("k"))      # 1
print(count_identical_substrings("aaabbb")) # 6 + 6 = 12
print(count_identical_substrings("abc"))    # 3 (each single letter)

# Better solution
def count_identical_substrings(S: str) -> int:
    total = 0
    run_len = 1
    for i in range(1, len(S)):
        if S[i] == S[i-1]:
            run_len += 1
        else:
            total += run_len * (run_len + 1) // 2
            run_len = 1
    total += run_len * (run_len + 1) // 2  # last run
    return total

print(count_identical_substrings("zzzyz"))  # 8 (z,z,z,zz,zz,zzz,y)
print(count_identical_substrings("k"))      # 1
print(count_identical_substrings("aaabbb")) # 6 + 6 = 12
print(count_identical_substrings("abc"))    # 3 (each single letter)


