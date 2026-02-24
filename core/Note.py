# Leetcode Study Node


# Python built in Sort (hybrid of Merge Sort and Insertion Sort)
# merge sort is divide and conquer algorithm
# Time complexity: O(nlogn)
# Space complexity: O(logn) to O(n) (worst case)


# subseq = [''.join(subseq) for subseq in combinations(s, 5)]
# Time complexity: O(n^5) for combinations
# Space complexity: O(n^5) for combinations

# Bit manipulation


# Merge Sort
def merge_sort(arr):
    # Base case: a list of 0 or 1 elements is already sorted
    if len(arr) <= 1:
        return arr
    # Step 1: Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    # Step 2: Merge
    return merge(left, right)
def merge(left, right):
    result = []
    i = j = 0
    # Merge sorted halves
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge(self, left, right):
    dummy = ListNode(0)
    curr = dummy
    while left and right:
        if left.val > right.val:
            curr.next = right
            right = right.next
        else:
            curr.next = left
            left = left.next
        curr = curr.next
    if left:
        curr.next = left
    if right:
        curr.next = right
    return dummy.next


# Tree
# In-order Left → Node → Right
# pre-order Node → Left → Right
# post-order Left → Right → Node
result = []

def inorder(root):
    if not root:
        return []

    inorder(root.left)
    result.append(root.val)
    inorder(root.right)
    return result

def preorder(root):
    if not root:
        return []

    result.append(root.val)
    preorder(root.left)
    preorder(root.right)
    return result

def postorder(root):
    if not root:
        return []

    postorder(root.left)
    postorder(root.right)
    result.append(root.val)
    return result


# Morris inorder
def morris_inorder(root):
    cur = root
    while cur:
        if not cur.left:
            print(cur.val)
            cur = cur.right
        else:
            pre = cur.left
            while pre.right and pre.right != cur:
                pre = pre.right
            if not pre.right:
                pre.right = cur
                cur = cur.left
            else:
                pre.right = None
                print(cur.val)
                cur = cur.right

# Bredth First Search (BFS)
# core
queue = deque()
queue.append(start)
curr = queue.popleft()
queue.append(curr.left)
queue.append(curr.right)

# DFS iterative
# core
queue = deque()
queue.append(start)
curr = queue.pop()
queue.append(curr.right)
queue.append(curr.left)

# Heapify
# Heap.pop/push
# Time complexity: O(log (n))
# Space complexity: O(1)

# Building a heap
# Inserting n elements one by one: O(n log n)
# Optimal method (heapify): O(n)


# Boyer-Moore Voting Algorithm
count = 0
candidate = None
for num in nums:
    if count == 0:
        candidate = num
    count += 1 if num == candidate else -1
return candidate


### Matrix ###
# Sudobu
# ij = i // 3 + (j // 3) * 3

# Spiral matrix
# m = len(matrix)
# n = len(matrix[0])
# direction = 1  # Start off going right
# i, j = 0, -1
# output = []
# while m * n > 0:
#     for _ in range(n):  # move horizontally
#         j += direction
#         output.append(matrix[i][j])
#     m -= 1
#     n -= 1
#     for _ in range(m):  # move vertically
#         i += direction
#         output.append(matrix[i][j])
#     direction *= -1  # flip direction
# return output

# Rotate image
# n = len(matrix[0])
# for i in range(n // 2 + n % 2):
#     for j in range(n // 2):
#         tmp = matrix[n - 1 - j][i]
#         matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
#         matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i]
#         matrix[j][n - 1 - i] = matrix[i][j]
#         matrix[i][j] = tmp


# Frog movement
# A: right moves, B: up moves, n: max consecutive moves
def count_paths_no_n_in_a_row(A, B, n):
    def dp(r, u, last, run):
        if r == A and u == B:
            return 1
        ways = 0
        # Try R
        if r < A:
            if last == 'R':
                if run < n - 1:
                    ways += dp(r + 1, u, 'R', run + 1)
            else:
                ways += dp(r + 1, u, 'R', 1)
        # Try U
        if u < B:
            if last == 'U':
                if run < n - 1:
                    ways += dp(r, u + 1, 'U', run + 1)
            else:
                ways += dp(r, u + 1, 'U', 1)
        return ways
    return dp(0, 0, None, 0)

# Example: original frog (A=7, B=4) with "no 3 in a row" => n=3
# print(count_paths_no_n_in_a_row(7, 4, 3))  # 30

def count_paths_no_n_in_a_row(A, B, n):
    dp = [[{'R': [0] * (n-1), 'U': [0] * (n-1)} for _ in range(B+1)] for _ in range(A+1)]

    # Initialize: first step can be R or U (run length = 1)
    if A >= 1:
        dp[1][0]['R'][0] = 1
    if B >= 1:
        dp[0][1]['U'][0] = 1

    for r in range(A+1):
        for u in range(B+1):
            # extend/switch from states that end with R
            Rruns = dp[r][u]['R']
            for k, ways in enumerate(Rruns):  # k = run_len-1
                if ways == 0:
                    continue
                # extend R-run if allowed
                if r+1 <= A and k+1 < n-1:
                    dp[r+1][u]['R'][k+1] += ways
                # switch to U (reset run to length 1)
                if u+1 <= B:
                    dp[r][u+1]['U'][0] += ways

            # extend/switch from states that end with U
            Uruns = dp[r][u]['U']
            for k, ways in enumerate(Uruns):
                if ways == 0:
                    continue
                # extend U-run if allowed
                if u+1 <= B and k+1 < n-1:
                    dp[r][u+1]['U'][k+1] += ways
                # switch to R
                if r+1 <= A:
                    dp[r+1][u]['R'][0] += ways

    # sum all ways at destination, regardless of last move or run length
    return sum(dp[A][B]['R']) + sum(dp[A][B]['U'])



# Newton's method
# x^2 = 37
# Make sure f(x) = 0 = x^2 - 37
f = lambda x: x**2 - 37
f_prime = lambda x: 2*x
def evaluate(f, f_prime, guess, maxiter, tolerance=1e-9):
    for _ in range(maxiter):
        x_n = guess - f(guess) / f_prime(guess)
        if abs(x_n - guess) < tolerance:
            return x_n
        guess = x_n
    return guess



# Trie/prefix tree
# Initialize links array and isEnd flag
class TrieNode:
    def __init__(self):
        self.links = [None] * 26
        self.is_end = False
    def contains_key(self, ch: str) -> bool:
        return self.links[ord(ch) - ord('a')] is not None
    def get(self, ch: str) -> 'TrieNode':
        return self.links[ord(ch) - ord('a')]
    def put(self, ch: str, node: 'TrieNode') -> None:
        self.links[ord(ch) - ord('a')] = node
    def set_end(self) -> None:
        self.is_end = True
    def is_end(self) -> bool:
        return self.is_end



# Backtracking
# Core
def backtrack(path, choices):
    if some_end_condition:
        output.append(path)
        return

    for choice in choices:
        # make choice
        path.append(choice)
        # explore
        backtrack(path, updated_choices)
        # undo choice
        path.pop()


# Manacher's Alogirhtm
class Solution:
    def longestPalindrome(self, s: str) -> str:
        s_prime = "#" + "#".join(s) + "#"
        print(s_prime)
        n = len(s_prime)
        palindrome_radii = [0] * n
        center = radius = 0
        for i in range(n):
            mirror = 2 * center - i
            if i < radius:
                palindrome_radii[i] = min(radius - i, palindrome_radii[mirror])
            while (i + 1 + palindrome_radii[i] < n and i - 1 - palindrome_radii[i] >= 0
                   and s_prime[i + 1 + palindrome_radii[i]] == s_prime[i - 1 - palindrome_radii[i]]):
                palindrome_radii[i] += 1
            if i + palindrome_radii[i] > radius:
                center = i
                radius = i + palindrome_radii[i]
        max_length = max(palindrome_radii)
        center_index = palindrome_radii.index(max_length)
        start_index = (center_index - max_length) // 2
        longest_palindrome = s[start_index : start_index + max_length]
        return longest_palindrome

# Kadane's algorithm

max_sum = float('-inf')
curr_sum = 0

for i in range(len(nums)):
    curr_sum += nums[i]
    if curr_sum > max_sum:
        max_sum = curr_sum
    if curr_sum < 0:
        curr_sum = 0


# Quick Select, Hoare's selection algorithm




# Binary search
# Exact match
l, r = 0, len(nums) - 1
while l <= r:
    mid = (l + r) // 2
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        l = mid + 1
    else:
        r = mid - 1
return -1

# Left boundary (first True / lower_bound)
l, r = 0, len(nums)
while l < r:
    mid = (l + r) // 2
    if nums[mid] < target:
        l = mid + 1
    else:
        r = mid
return l

# Right boundary (last True / upper_bound-1)
l, r = 0, len(nums)
while l < r:
    mid = (l + r) // 2
    if nums[mid] <= target:
        l = mid + 1
    else:
        r = mid
return l - 1


# if l = mid use mid = (l + r + 1) // 2 to avoid infinite loop
# if r = mid use normal mid = (l + r) // 2