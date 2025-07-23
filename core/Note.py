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