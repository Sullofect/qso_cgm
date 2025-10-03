# Virtual Financial


# Write a function that, given an integer X, returns an integer that corresponds to the minimum number of steps
# required to change X to a Fibonacci number. In each step you can either increment or decrement the current number,
# i.e., you can change X to either X + 1 or X – 1. X will be between 0 and 1,000,000 inclusive.
# The Fibonacci sequence is defined as follows:F[0] = 0, F[1] = 1, for each i >= 2: F[i] = F[i-1] + F[i-2]
# The elements of the Fibonacci sequence are called Fibonacci numbers.

# Examples:
# For X = 15 the function should return 2.
# For X = 1 or X = 13 the function should return 0.
#
# def min_steps_to_fibonacci(X):
#     if X < 0 or X > 1000000:
#         raise ValueError("X must be between 0 and 1,000,000 inclusive.")
#
#     # Generate Fibonacci numbers up to 1,000,000
#     fib = [0, 1]
#     while True:
#         next_fib = fib[-1] + fib[-2]
#         if next_fib > 1000000:
#             break
#         fib.append(next_fib)
#
#     # Find the closest Fibonacci number to X
#     min_steps = float('inf')
#     for f in fib:
#         steps = abs(X - f)
#         if steps < min_steps:
#             min_steps = steps
#     return min_steps
#
# print(min_steps_to_fibonacci(15))
# print(min_steps_to_fibonacci(13))

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

# def final_scores(students):
#     n = len(students)
#     if n < 3:
#         return students
#     changed = True
#     while changed:
#         changed = False
#         students_copy = students[:]
#         for i in range(1, n-1, 1):
#             if students[i] < students[i-1] and students[i] < students[i+1]:
#                 students_copy[i] += 1
#                 changed = True
#             elif students[i] > students[i-1] and students[i] > students[i+1]:
#                 students_copy[i] += 1
#                 changed = True
#         students = students_copy
#     return students
# print(final_scores([1, 6, 3, 4, 3, 5]))
# print(final_scores([1, 6, 5, 4, 3, 2, 1, 10, 5]))

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

# dic = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
#        10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F'}
# def decimal_to_hexspeak(S):
#     num = int(S)
#     str = ''
#     while num != 0:
#         num, mod = divmod(num, 16)
#         str = dic[mod] + str
#     str = list(str)
#     if '2' in str or '3' in str or '4' in str or '5' in str or '6' in str or '7' in str or '8' in str or '9' in str:
#         return "Error"
#     else:
#         for i in range(len(str)):
#             if str[i] == '0':
#                 str[i] = 'O'
#             elif str[i] == '1':
#                 str[i] = 'I'
#         return ''.join(str)
# print(decimal_to_hexspeak("257"))
# print(decimal_to_hexspeak("3"))
# print(decimal_to_hexspeak("507"))
#
# # Better way
# def toHexSpeak(S: str) -> str:
#     n = int(S)
#     hex_str = hex(n)[2:].upper()
#     hex_str = hex_str.replace("0", "O").replace("1", "I")
#     valid_chars = set("ABCDEFIO")
#     if all(c in valid_chars for c in hex_str):
#         return hex_str
#     else:
#         return "ERROR"

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

# def max_apples(A):
#     capacity = 5000
#     current_weight = A[0]
#     apples = sorted(A[1:])
#
#     count = 0
#     for apple in apples:
#         if current_weight + apple <= capacity:
#             current_weight += apple
#             count += 1
#         else:
#             return apples
# print(max_apples([4650, 150, 150, 150]))


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

#
# def count_identical_substrings(S):
#     n = len(S)
#     count = n
#
#     # Base case and use sliding window
#     for i in range(n):
#         j = i + 1
#         while j < n and S[j] == S[i]:
#             count += 1
#             j += 1
#     return count
# print(count_identical_substrings("zzzyz"))  # 8 (z,z,z,zz,zz,zzz,y)
# print(count_identical_substrings("k"))      # 1
# print(count_identical_substrings("aaabbb")) # 6 + 6 = 12
# print(count_identical_substrings("abc"))    # 3 (each single letter)
#
# # Better solution
# def count_identical_substrings(S: str) -> int:
#     total = 0
#     run_len = 1
#     for i in range(1, len(S)):
#         if S[i] == S[i-1]:
#             run_len += 1
#         else:
#             total += run_len * (run_len + 1) // 2
#             run_len = 1
#     total += run_len * (run_len + 1) // 2  # last run
#     return total
#
# print(count_identical_substrings("zzzyz"))  # 8 (z,z,z,zz,zz,zzz,y)
# print(count_identical_substrings("k"))      # 1
# print(count_identical_substrings("aaabbb")) # 6 + 6 = 12
# print(count_identical_substrings("abc"))    # 3 (each single letter)



# Goldman Sachs OA
# Version 1
# Question 1
# def findStarvation(priorities):
#     n = len(priorities)
#     starvation = [0] * n
#     for i in range(n):
#         for j in range(n-1, i, -1):
#             if priorities[j] < priorities[i]:
#                 starvation[i] = max(j - i, starvation[i])
#     return starvation
# print(findStarvation([8, 2, 5, 3]))
# print(findStarvation([6, 10, 9, 7]))
# print(findStarvation([8, 2, 11, 4, 9, 4, 7]))

# Question 2
# from itertools import permutations
#
# def calculateTotalPrefix(sequence, k):
#     def calculate10(subsequence):
#         count = 0
#         for i in range(len(subsequence)):
#             for j in range(i+1, len(subsequence)):
#                 if subsequence[i] + subsequence[j] == '10':
#                     count += 1
#         return count
#
#     count = 0
#     n = len(sequence)
#     for i in range(1, n):
#         prefix = sequence[:i]
#         if calculate10(prefix) == k:
#             count += 1
#             continue
#
#         # Generate all permutations of the remaining characters
#         permutations_list = permutations('01', n-i)
#         for permutation in permutations_list:
#             if calculate10(prefix + ''.join(permutation)) == k:
#                 count += 1
#     return count

# def calculateTotalPrefix(s, k):
#     n = len(s)
#     ones = zeros = count10 = 0
#     valid_count = 0
#
#     for i in range(n):
#         if s[i] == '1':
#             ones += 1
#         else:  # s[i] == '0'
#             zeros += 1
#             count10 += ones
#
#         if count10 > k:
#             continue
#         if count10 == k:
#             valid_count += 1
#         else:  # count10 < k
#             diff = k - count10
#             if ones == 0:
#                 valid += 1
#             elif ones > 0 and diff % ones == 0:
#                 valid_count += 1
#
#     return valid_count
# print('res 100 1', calculateTotalPrefix('100', 1))
# print('res 101 2', calculateTotalPrefix('101', 2))
# print('res 11 1', calculateTotalPrefix('11', 1))


# Question 3
# Data Reorganization
# def getMininumValue(data, maxOperations):
#     for k in range(maxOperations):
#         minValue, maxValue = float('inf'), 0
#         n = len(data)
#         for i in range(n-1):
#             for j in range(i+1, n):
#                 # maxValue = max(abs(data[i] - data[j]), maxValue)
#                 # minValue = min(abs(data[i] - data[j]), minValue)
#                 data.append(abs(data[i] - data[j]))
#         # data.append(maxValue)
#         # data.append(minValue)
#     print(data)
#     return min(data)

# print(getMininumValue([42, 47, 50, 54, 62, 79], 2))
# print(getMininumValue([4, 2, 5, 9, 3], 1))
# print(getMininumValue([5, 18, 3, 12, 11], 2))
# print(getMininumValue([5, 18, 3, 12, 10], 2))


# Question 4
# Bookshelf Organization
# def findMinimumOperations(bookshelf, k):
#     def locateMaxAuthor(author):
#         indice, max_count = (None, None), 0
#         for i in range(m):
#             for j in range(n):
#                 if bookshelf[i][j] != author:
#                     continue
#                 count = 1
#                 # Calculate how many same authors in the same row and same column
#                 ii_plus, jj_plus = i+1, j+1
#                 ii_minus, jj_minus = i-1, j-1
#                 while 0 <= ii_plus < m:
#                     if bookshelf[ii_plus][j] == bookshelf[i][j]:
#                         count += 1
#                     ii_plus += 1
#                 while 0 <= ii_minus < m:
#                     if bookshelf[ii_minus][j] == bookshelf[i][j]:
#                         count += 1
#                     ii_minus -= 1
#                 while 0 <= jj_plus < n:
#                     if bookshelf[i][jj_plus] == bookshelf[i][j]:
#                         count += 1
#                     jj_plus += 1
#                 while 0 <= jj_minus < n:
#                     if bookshelf[i][jj_minus] == bookshelf[i][j]:
#                         count += 1
#                     jj_minus -= 1
#                 if count > max_count:
#                     indice = (i, j)
#                 max_count = max(count, max_count)
#         return indice
#
#     authors = list(range(1, k+1, 1))
#     m, n = len(bookshelf), len(bookshelf[0])
#
#     # Determine the count
#     count = 0
#     elements = m * n
#     while elements > 0:
#         for author in authors:
#             i, j = locateMaxAuthor(author)
#             if i is None or j is None:
#                 continue
#             ii_plus, jj_plus = i+1, j+1
#             ii_minus, jj_minus = i-1, j-1
#             # Calculate how many same authors in the same row and same column
#             while 0 <= ii_plus < m:
#                 if bookshelf[ii_plus][j] == bookshelf[i][j]:
#                     bookshelf[ii_plus][j] = 0
#                     elements -= 1
#                 ii_plus += 1
#
#             while 0 <= ii_minus < m:
#                 if bookshelf[ii_minus][j] == bookshelf[i][j]:
#                     bookshelf[ii_minus][j] = 0
#                     elements -= 1
#                 ii_minus -= 1
#
#             while 0 <= jj_plus < n:
#                 if bookshelf[i][jj_plus] == bookshelf[i][j]:
#                     bookshelf[i][jj_plus] = 0
#                     elements -= 1
#                 jj_plus += 1
#
#             while 0 <= jj_minus < n:
#                 if bookshelf[i][jj_minus] == bookshelf[i][j]:
#                     bookshelf[i][jj_minus] = 0
#                     elements -= 1
#                 jj_minus -= 1
#             bookshelf[i][j] = 0
#             elements -= 1
#             count += 1
#     return count

# print(findMinimumOperations([[2, 2, 1], [1, 1, 1], [2, 3, 3]], 3))
# print(findMinimumOperations([[1, 1, 2], [2, 2, 2], [1, 2, 2]], 2))
# print(findMinimumOperations([[1, 2, 3], [4, 5, 6]], 6))


# Question 5
# Lock Code
# import math
#
# def _distinct_prime_factors(x):
#     """Return the set of distinct prime factors of x."""
#     s = set()
#     d = 2
#     while d * d <= x:
#         while x % d == 0:
#             s.add(d)
#             x //= d
#         d += 1 if d == 2 else 2  # after 2, try only odd
#     if x > 1:
#         s.add(x)
#     return s
#
# def _product_of_distinct_primes(arr):
#     """Return the product of distinct primes dividing any element in arr."""
#     primes = set()
#     for a in arr:
#         primes |= _distinct_prime_factors(a)
#     prod = 1
#     for p in primes:
#         prod *= p
#     return prod
#
# def _score_for_x(arr, x):
#     """Compute score for picking x (ensuring x exists, making others coprime)."""
#     non_coprime = sum(1 for a in arr if math.gcd(a, x) != 1)
#     has_x = any(a == x for a in arr)
#     # print(non_coprime)
#     # print(has_x)
#
#     # ensure x exists
#     if has_x or non_coprime > 0:
#         extra = 0
#     else:
#         extra = 1
#     return x - (non_coprime + extra)
#
# def decryptCodeLock(codeSequence, maxValue):
#     A = codeSequence
#
#     # Candidate 1: largest x <= maxValue that is coprime to *all* array numbers.
#     # We search downward from maxValue until gcd(x, G*) == 1.
#     Gstar = _product_of_distinct_primes(A)
#     x_all_coprime = None
#     x = maxValue
#     while x >= 1:
#         if math.gcd(x, Gstar) == 1:
#             x_all_coprime = x
#             break
#         x -= 1
#
#     best = -float('inf')
#     if x_all_coprime is not None:
#         best = max(best, _score_for_x(A, x_all_coprime))
#
#
#     # Candidate 2: each distinct value already present in the array.
#     # (Choosing an existing value avoids the "ensure x exists" extra when possible.)
#     for v in set(A):
#         if v <= maxValue:
#             best = max(best, _score_for_x(A, v))
#             print(_score_for_x(A, v))
#     return best

# print('res', decryptCodeLock([3, 2, 4], 6))
# print('res', decryptCodeLock([3, 6, 12], 15))
# print(decryptCodeLock([1, 2, 3], 6))
# print(decryptCodeLock([2, 4, 6, 8], 8))




# HRT

# Fancy Number
# A positive integer is "fancy" if its representation in base 4 only includes 0s and 1s. For example:
# SETTINGS
# 17 is fancy: its base-4 representation, 101, only includes Os and 1s.
# 18 is not fancy: its base-4 representation, 102, includes a 2.
#
# Given a positive integer n, find the number of fancy positive integers less than n
# Note that n may be as large as a billion! Your algorithm should be faster than iterating over values
# less than n and checking if each one is fancy.


def count_fancy(n: int) -> int:
    if n <= 1:
        return 0  # no positive fancy numbers < 1

    # base-4 digits of n, most significant first
    s = []
    x = n
    while x:
        s.append(x % 4)
        x //= 4
    s.reverse()
    k = len(s)

    # 1) all fancy numbers with fewer than k digits
    ans = (1 << (k - 1)) - 1  # 2^(k-1) - 1

    # 2) same length k
    # handle first digit
    if s[0] in (2, 3):
        ans += 1 << (k - 1)  # all k-digit fancy numbers
        return ans
    # s[0] must be 1 to have k-digit fancy; if s[0]==0 it can't happen

    # traverse remaining digits
    for i in range(1, k):
        if s[i] == 0:
            continue  # must place 0
        elif s[i] == 1:
            ans += 1 << (k - 1 - i)  # place 0 here, rest free
            # also can place 1 and continue
        else:  # s[i] in {2,3}
            ans += 1 << (k - i)  # choose 0 or 1 here, rest free
            return ans

    # if we finished without hitting >=2, we matched a {0,1}-only number exactly (don't count it)
    return ans
# print('Number of Fancy numbers', count_fancy(17))  # 0
# print('Number of Fancy numbers', count_fancy(18))  # 0
# print('Number of Fancy numbers', count_fancy(1000))


def count_fancy_via_predecessor(n: int) -> int:
    if n <= 1:
        return 0

    # 四进制展开（最高位在前）
    s = []
    x = n
    while x:
        s.append(x % 4)
        x //= 4
    s.reverse()

    t = s[:]  # 将要改造成 m 的四进制 0/1 串

    # 找第一处 >=2
    first_ge2 = next((i for i, d in enumerate(s) if d >= 2), None)
    if first_ge2 is not None:
        i = first_ge2
        # 前缀保持（必为 0/1），当前位置置 1，后缀全置 1
        t[i] = 1
        for j in range(i + 1, len(t)):
            t[j] = 1
    else:
        # n 的四进制本身只有 0/1，需要取严格更小的最大 0/1 串
        # 找最右侧的 1
        r = None
        for i in range(len(t) - 1, -1, -1):
            if t[i] == 1:
                r = i
                break
        if r is None:
            return 0  # 理论上只有 n==0 会到这，但前面已排除
        t[r] = 0
        for j in range(r + 1, len(t)):
            t[j] = 1
        # 去掉可能出现的前导 0
        while t and t[0] == 0:
            t.pop(0)

    # 现在 t 是 m 的四进制 0/1 串；把它当作二进制解析就是计数
    ans = 0
    for d in t:
        ans = (ans << 1) | d
    return ans



# Reversi (or Othello) is a 2-player game played on a N x N board. Player 1 plays the black pieces and Player 2 plays
# the  white pieces. On each turn, players take turns placing pieces on the board. Once a player places down a piece,
# the following process occurs: any sequence of opponent disks that are in a straight line
# (vertical, horizontal, or diagonal) and bounded by the player's just-played disk and another disk of the same color
# are turned into the moving player's color. (For example, if a sequence WBBBW had just added the bolded W, then the
# three B's would flip to W. However, WBB.W,'B' where. is an empty space, would not.)

# from typing import List, Tuple
#
# Cell = str  # '.', 'B', or 'W'
# Board = List[List[Cell]]
#
# DIRECTIONS: List[Tuple[int, int]] = [
#     (-1, -1), (-1, 0), (-1, +1),
#     ( 0, -1),          ( 0, +1),
#     (+1, -1), (+1, 0), (+1, +1),
# ]
#
# def in_bounds(N: int, r: int, c: int) -> bool:
#     return 0 <= r < N and 0 <= c < N
#
# def opponent(p: Cell) -> Cell:
#     return 'W' if p == 'B' else 'B'
#
# def find_flips(board: Board, r: int, c: int, p: Cell) -> List[Tuple[int, int]]:
#     """
#     If placing p at (r,c) is legal, return the list of coordinates to flip.
#     Otherwise return [].
#     """
#     if board[r][c] != '.':
#         return []
#     N = len(board)
#     opp = opponent(p)
#     flips: List[Tuple[int, int]] = []
#
#     for dr, dc in DIRECTIONS:
#         path = []
#         rr, cc = r + dr, c + dc
#         # First must see ≥1 opponent disk
#         if not in_bounds(N, rr, cc) or board[rr][cc] != opp:
#             continue
#         # Collect all contiguous opponent disks
#         while in_bounds(N, rr, cc) and board[rr][cc] == opp:
#             path.append((rr, cc))
#             rr += dr
#             cc += dc
#         # Now must end on a friendly disk to bound; otherwise no flips
#         if in_bounds(N, rr, cc) and board[rr][cc] == p and path:
#             flips.extend(path)
#
#     return flips
#
# def is_legal_move(board: Board, r: int, c: int, p: Cell) -> bool:
#     return len(find_flips(board, r, c, p)) > 0
#
# def apply_move(board: Board, r: int, c: int, p: Cell) -> bool:
#     """
#     Applies move if legal, mutating board; returns True if applied, False otherwise.
#     """
#     flips = find_flips(board, r, c, p)
#     if not flips:
#         return False
#     board[r][c] = p
#     for rr, cc in flips:
#         board[rr][cc] = p
#     return True
#
# def valid_moves(board: Board, p: Cell) -> List[Tuple[int, int]]:
#     N = len(board)
#     res = []
#     for r in range(N):
#         for c in range(N):
#             if is_legal_move(board, r, c, p):
#                 res.append((r, c))
#     return res
#
# def score(board: Board) -> Tuple[int, int]:
#     """
#     Returns (black_count, white_count).
#     """
#     b = sum(row.count('B') for row in board)
#     w = sum(row.count('W') for row in board)
#     return b, w
#

DIRS = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def apply_move(board, r, c, me):
    """Return a NEW board with the move applied, or None if illegal."""
    n = len(board)
    if board[r][c] != '.':
        return None
    opp = 'W' if me == 'B' else 'B'
    flips = []

    for dr, dc in DIRS:
        i, j = r + dr, c + dc
        run = []

        # collect opponent disks in this direction
        while 0 <= i < n and 0 <= j < n and board[i][j] == opp:
            run.append((i, j))
            i += dr
            j += dc

        # valid if we ended on our own color and saw at least 1 opp disk
        if run and 0 <= i < n and 0 <= j < n and board[i][j] == me:
            flips.extend(run)

    if not flips:
        return None  # no direction made a sandwich → illegal move

    # build new board with flips
    new = [row[:] for row in board]
    new[r][c] = me
    for i, j in flips:
        new[i][j] = me
    return new


board4 = [[".",".",".",".",".","."],
          [".",".",".",".",".","."],
          [".","B",".",".",".","."],
          [".","B",".",".",".","."],
          [".","W",".",".",".","."],
          [".",".",".",".",".","."]]

board5 = [["B",".",".","."],
          [".","W",".","."],
          [".",".","W","."],
          [".",".",".","."]]
print('board flip', apply_move(board4, 1, 1, 'W'))
print('board flip', apply_move(board5, 3, 3, 'B'))