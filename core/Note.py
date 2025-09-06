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

# Bredth First Search (BFS)
# core
queue = deque()
queue.append(start)
curr = queue.popleft()

# Heapify
# Heap.pop/push
# Time complexity: O(log (n)
# Space complexity: O(1)


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