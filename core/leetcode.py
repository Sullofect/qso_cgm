

mod = 10 ** 9 + 7


def countBalancedClips(clipLength, diff):
    # Initiate DP arrays for 26 characters in the alphabet
    # curr = [0] * 26
    prev = [1] * 26 # prev as the base case where all single letter clips are balanced

    # Start iteration from 2 clips
    for i in range(2, clipLength + 1):
        curr = [0] * 26
        for c1 in range(26):
            for c2 in range(max(0, c1 - diff), min(25, c1 + diff) + 1):
                curr[c2] = (curr[c2] + prev[c1]) % mod

        prev = curr

    return sum(curr) % mod


print(countBalancedClips(2, 3))  # 2

def find(i):
    if parent[i] != i:
        parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j

def minCostToConnectServers(self, x, y):
    # Write your code here
    n = len(x)

    parent = list(range(n))

    # Create eges by sorting servers by x and y coord
    edges = []

    # sort by x-coordinate and add
    servers = [(x[i], y[i], i) for i in range(n)]
    servers.sort(key=lambda s: s[0])
    for i in range(n-1):
        cost = min(abs(servers[i][0] - servers[i+1][0]), abs(servers[i][1] - servers[i+1][1]))
        edges.append((cost, servers[i][2], servers[i+1][2]))

    # sort by y-coordinate and add
    servers.sort(key=lambda s: s[1])
    for i in range(n - 1):
        cost = min(abs(servers[i][0] - servers[i + 1][0]), abs(servers[i][1] - servers[i + 1][1]))
        edges.append((cost, servers[i][2], servers[i + 1][2]))

    # Use Kruskal's algorithm using a heap
    edges.sort()
    total_cost = 0
    edges_used = 0

    for cost, i, j in edges:
        if find(i) != find(j):
            union(i, j)
            total_cost += cost
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_cost


