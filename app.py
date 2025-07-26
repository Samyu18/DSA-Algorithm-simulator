from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import time
import heapq
from collections import defaultdict, deque

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# ==================== SORTING ALGORITHMS ====================

def bubble_sort(arr):
    trace = []
    a = arr.copy()
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            trace.append({'array': a[:], 'compare': [j, j+1], 'step': f'Comparing {a[j]} and {a[j+1]}'})
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                trace.append({'array': a[:], 'swap': [j, j+1], 'step': f'Swapped {a[j+1]} and {a[j]}'})
    trace.append({'array': a[:], 'step': 'Sorting complete!'})
    return trace

def selection_sort(arr):
    trace = []
    a = arr.copy()
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            trace.append({'array': a[:], 'compare': [min_idx, j], 'step': f'Comparing {a[min_idx]} and {a[j]}'})
            if a[min_idx] > a[j]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
            trace.append({'array': a[:], 'swap': [i, min_idx], 'step': f'Swapped {a[i]} and {a[min_idx]}'})
    trace.append({'array': a[:], 'step': 'Sorting complete!'})
    return trace

def insertion_sort(arr):
    trace = []
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        trace.append({'array': a[:], 'highlight': [i], 'step': f'Inserting {key} into sorted portion'})
        while j >= 0 and key < a[j]:
            trace.append({'array': a[:], 'compare': [j, i], 'step': f'Comparing {key} with {a[j]}'})
            a[j + 1] = a[j]
            trace.append({'array': a[:], 'move': [j, j+1], 'step': f'Moving {a[j+1]} to position {j+1}'})
            j -= 1
        a[j + 1] = key
        trace.append({'array': a[:], 'insert': [j+1, key], 'step': f'Inserted {key} at position {j+1}'})
    trace.append({'array': a[:], 'step': 'Sorting complete!'})
    return trace

def merge_sort(arr):
    trace = []
    a = arr.copy()
    
    def merge(left, mid, right):
        temp = []
        i, j = left, mid + 1
        while i <= mid and j <= right:
            trace.append({'array': a[:], 'compare': [i, j], 'step': f'Comparing {a[i]} and {a[j]}'})
            if a[i] <= a[j]:
                temp.append(a[i])
                i += 1
            else:
                temp.append(a[j])
                j += 1
        while i <= mid:
            temp.append(a[i])
            i += 1
        while j <= right:
            temp.append(a[j])
            j += 1
        for k in range(len(temp)):
            a[left + k] = temp[k]
            trace.append({'array': a[:], 'merge': [left + k, temp[k]], 'step': f'Merged {temp[k]} at position {left + k}'})
    
    def sort_helper(left, right):
        if left < right:
            mid = (left + right) // 2
            trace.append({'array': a[:], 'divide': [left, mid, right], 'step': f'Dividing array from {left} to {right}'})
            sort_helper(left, mid)
            sort_helper(mid + 1, right)
            merge(left, mid, right)
    
    sort_helper(0, len(a) - 1)
    trace.append({'array': a[:], 'step': 'Sorting complete!'})
    return trace

def quick_sort(arr):
    trace = []
    a = arr.copy()
    
    def partition(low, high):
        pivot = a[high]
        i = low - 1
        trace.append({'array': a[:], 'pivot': [high], 'step': f'Pivot: {pivot}'})
        for j in range(low, high):
            trace.append({'array': a[:], 'compare': [j, high], 'step': f'Comparing {a[j]} with pivot {pivot}'})
            if a[j] < pivot:
                i += 1
                a[i], a[j] = a[j], a[i]
                trace.append({'array': a[:], 'swap': [i, j], 'step': f'Swapped {a[i]} and {a[j]}'})
        a[i + 1], a[high] = a[high], a[i + 1]
        trace.append({'array': a[:], 'swap': [i + 1, high], 'step': f'Placed pivot {pivot} at position {i + 1}'})
        return i + 1
    
    def sort_helper(low, high):
        if low < high:
            pi = partition(low, high)
            sort_helper(low, pi - 1)
            sort_helper(pi + 1, high)
    
    sort_helper(0, len(a) - 1)
    trace.append({'array': a[:], 'step': 'Sorting complete!'})
    return trace

def heap_sort(arr):
    trace = []
    a = arr.copy()
    n = len(a)
    
    def heapify(n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and a[left] > a[largest]:
            largest = left
        if right < n and a[right] > a[largest]:
            largest = right
        if largest != i:
            a[i], a[largest] = a[largest], a[i]
            trace.append({'array': a[:], 'swap': [i, largest], 'step': f'Heapify: swapped {a[i]} and {a[largest]}'})
            heapify(n, largest)
    
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        trace.append({'array': a[:], 'swap': [0, i], 'step': f'Extracted max {a[0]} to position {i}'})
        heapify(i, 0)
    
    trace.append({'array': a[:], 'step': 'Sorting complete!'})
    return trace

def counting_sort(arr):
    trace = []
    a = arr.copy()
    max_val = max(a)
    count = [0] * (max_val + 1)
    output = [0] * len(a)
    
    for num in a:
        count[num] += 1
        trace.append({'array': a[:], 'count': [num], 'step': f'Counting occurrence of {num}'})
    
    for i in range(1, max_val + 1):
        count[i] += count[i - 1]
        trace.append({'array': a[:], 'step': f'Cumulative count for {i}: {count[i]}'})
    
    for i in range(len(a) - 1, -1, -1):
        output[count[a[i]] - 1] = a[i]
        count[a[i]] -= 1
        trace.append({'array': output[:], 'place': [count[a[i]], a[i]], 'step': f'Placed {a[i]} at position {count[a[i]]}'})
    
    trace.append({'array': output[:], 'step': 'Sorting complete!'})
    return trace

def radix_sort(arr):
    trace = []
    a = arr.copy()
    max_val = max(a)
    exp = 1
    
    while max_val // exp > 0:
        count = [0] * 10
        output = [0] * len(a)
        
        for num in a:
            digit = (num // exp) % 10
            count[digit] += 1
            trace.append({'array': a[:], 'digit': [digit], 'step': f'Counting digit {digit} at position {exp}'})
        
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        for i in range(len(a) - 1, -1, -1):
            digit = (a[i] // exp) % 10
            output[count[digit] - 1] = a[i]
            count[digit] -= 1
            trace.append({'array': output[:], 'place': [count[digit], a[i]], 'step': f'Placed {a[i]} based on digit {digit}'})
        
        a = output[:]
        exp *= 10
    
    trace.append({'array': a[:], 'step': 'Sorting complete!'})
    return trace

# ==================== SEARCHING ALGORITHMS ====================

def linear_search(arr, target):
    trace = []
    a = arr.copy()
    for i, val in enumerate(a):
        trace.append({'array': a[:], 'search': [i], 'step': f'Checking position {i}: {val}'})
        if val == target:
            trace.append({'array': a[:], 'found': [i], 'step': f'Found {target} at position {i}!'})
            return trace
    trace.append({'array': a[:], 'step': f'{target} not found in array'})
    return trace

def binary_search(arr, target):
    trace = []
    a = sorted(arr.copy())
    left, right = 0, len(a) - 1
    
    trace.append({'array': a[:], 'step': f'Searching for {target} in sorted array'})
    
    while left <= right:
        mid = (left + right) // 2
        trace.append({'array': a[:], 'search': [mid], 'step': f'Checking middle element at position {mid}: {a[mid]}'})
        
        if a[mid] == target:
            trace.append({'array': a[:], 'found': [mid], 'step': f'Found {target} at position {mid}!'})
            return trace
        elif a[mid] < target:
            trace.append({'array': a[:], 'range': [mid+1, right], 'step': f'{a[mid]} < {target}, searching right half'})
            left = mid + 1
        else:
            trace.append({'array': a[:], 'range': [left, mid-1], 'step': f'{a[mid]} > {target}, searching left half'})
            right = mid - 1
    
    trace.append({'array': a[:], 'step': f'{target} not found in array'})
    return trace

def jump_search(arr, target):
    trace = []
    a = sorted(arr.copy())
    n = len(a)
    step = int(n ** 0.5)
    
    trace.append({'array': a[:], 'step': f'Jump search for {target} with step size {step}'})
    
    prev = 0
    while prev < n and a[min(step, n) - 1] < target:
        trace.append({'array': a[:], 'jump': [prev, min(step, n) - 1], 'step': f'Jumping from {prev} to {min(step, n) - 1}'})
        prev = step
        step += int(n ** 0.5)
        if prev >= n:
            break
    
    while prev < min(step, n):
        trace.append({'array': a[:], 'search': [prev], 'step': f'Linear search at position {prev}: {a[prev]}'})
        if a[prev] == target:
            trace.append({'array': a[:], 'found': [prev], 'step': f'Found {target} at position {prev}!'})
            return trace
        prev += 1
    
    trace.append({'array': a[:], 'step': f'{target} not found in array'})
    return trace

def exponential_search(arr, target):
    trace = []
    a = sorted(arr.copy())
    n = len(a)
    
    if a[0] == target:
        trace.append({'array': a[:], 'found': [0], 'step': f'Found {target} at position 0!'})
        return trace
    
    i = 1
    while i < n and a[i] <= target:
        trace.append({'array': a[:], 'exponential': [i], 'step': f'Exponential jump to position {i}: {a[i]}'})
        i = i * 2
    
    left = i // 2
    right = min(i, n - 1)
    
    trace.append({'array': a[:], 'range': [left, right], 'step': f'Binary search in range [{left}, {right}]'})
    
    while left <= right:
        mid = (left + right) // 2
        trace.append({'array': a[:], 'search': [mid], 'step': f'Checking position {mid}: {a[mid]}'})
        
        if a[mid] == target:
            trace.append({'array': a[:], 'found': [mid], 'step': f'Found {target} at position {mid}!'})
            return trace
        elif a[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    trace.append({'array': a[:], 'step': f'{target} not found in array'})
    return trace

def ternary_search(arr, target):
    trace = []
    a = sorted(arr.copy())
    left, right = 0, len(a) - 1
    
    trace.append({'array': a[:], 'step': f'Ternary search for {target}'})
    
    while left <= right:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        trace.append({'array': a[:], 'search': [mid1, mid2], 'step': f'Checking positions {mid1} and {mid2}: {a[mid1]}, {a[mid2]}'})
        
        if a[mid1] == target:
            trace.append({'array': a[:], 'found': [mid1], 'step': f'Found {target} at position {mid1}!'})
            return trace
        if a[mid2] == target:
            trace.append({'array': a[:], 'found': [mid2], 'step': f'Found {target} at position {mid2}!'})
            return trace
        
        if target < a[mid1]:
            trace.append({'array': a[:], 'range': [left, mid1-1], 'step': f'{target} < {a[mid1]}, searching first third'})
            right = mid1 - 1
        elif target > a[mid2]:
            trace.append({'array': a[:], 'range': [mid2+1, right], 'step': f'{target} > {a[mid2]}, searching last third'})
            left = mid2 + 1
        else:
            trace.append({'array': a[:], 'range': [mid1+1, mid2-1], 'step': f'Searching middle third'})
            left = mid1 + 1
            right = mid2 - 1
    
    trace.append({'array': a[:], 'step': f'{target} not found in array'})
    return trace

# ==================== GREEDY ALGORITHMS ====================

def activity_selection(activities):
    trace = []
    if not activities:
        return trace
    
    # Sort by finish time
    activities = sorted(activities, key=lambda x: x[1])
    trace.append({'activities': activities[:], 'step': 'Sorted activities by finish time'})
    
    selected = [activities[0]]
    trace.append({'activities': activities[:], 'selected': [0], 'step': f'Selected activity 0: {activities[0]}'})
    
    for i in range(1, len(activities)):
        if activities[i][0] >= selected[-1][1]:
            selected.append(activities[i])
            trace.append({'activities': activities[:], 'selected': [i], 'step': f'Selected activity {i}: {activities[i]}'})
        else:
            trace.append({'activities': activities[:], 'rejected': [i], 'step': f'Rejected activity {i}: {activities[i]} (conflicts)'})
    
    trace.append({'activities': activities[:], 'result': selected, 'step': f'Final selection: {len(selected)} activities'})
    return trace

def fractional_knapsack(weights, values, capacity):
    trace = []
    items = list(zip(weights, values))
    items.sort(key=lambda x: x[1]/x[0], reverse=True)
    
    trace.append({'items': items[:], 'step': 'Sorted items by value/weight ratio'})
    
    total_value = 0
    remaining_capacity = capacity
    
    for i, (weight, value) in enumerate(items):
        if remaining_capacity >= weight:
            total_value += value
            remaining_capacity -= weight
            trace.append({'items': items[:], 'selected': [i], 'step': f'Added item {i} completely: weight={weight}, value={value}'})
        else:
            fraction = remaining_capacity / weight
            total_value += value * fraction
            trace.append({'items': items[:], 'partial': [i, fraction], 'step': f'Added {fraction:.2f} of item {i}: value={value*fraction:.2f}'})
            break
    
    trace.append({'items': items[:], 'result': total_value, 'step': f'Total value: {total_value:.2f}'})
    return trace

# ==================== DYNAMIC PROGRAMMING ====================

def knapsack_01(weights, values, capacity):
    trace = []
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
                trace.append({'dp': [row[:] for row in dp], 'step': f'Item {i-1}: max({dp[i-1][w]}, {dp[i-1][w-weights[i-1]]} + {values[i-1]}) = {dp[i][w]}'})
            else:
                dp[i][w] = dp[i-1][w]
                trace.append({'dp': [row[:] for row in dp], 'step': f'Item {i-1}: cannot fit, value = {dp[i][w]}'})
    
    trace.append({'dp': [row[:] for row in dp], 'result': dp[n][capacity], 'step': f'Maximum value: {dp[n][capacity]}'})
    return trace

def longest_common_subsequence(str1, str2):
    trace = []
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                trace.append({'dp': [row[:] for row in dp], 'match': [i-1, j-1], 'step': f'Match: {str1[i-1]} at positions {i-1}, {j-1}'})
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                trace.append({'dp': [row[:] for row in dp], 'step': f'No match: max({dp[i-1][j]}, {dp[i][j-1]}) = {dp[i][j]}'})
    
    # Backtrack to find LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    trace.append({'dp': [row[:] for row in dp], 'result': ''.join(lcs), 'step': f'LCS: {lcs}'})
    return trace

def fibonacci_dp(n):
    trace = []
    dp = [0] * (n + 1)
    dp[1] = 1
    
    trace.append({'dp': dp[:], 'step': 'Initialized: dp[0] = 0, dp[1] = 1'})
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
        trace.append({'dp': dp[:], 'step': f'F({i}) = F({i-1}) + F({i-2}) = {dp[i-1]} + {dp[i-2]} = {dp[i]}'})
    
    trace.append({'dp': dp[:], 'result': dp[n], 'step': f'Fibonacci({n}) = {dp[n]}'})
    return trace

# ==================== TREE ALGORITHMS ====================

def tree_traversals(nodes):
    trace = []
    
    # Inorder traversal
    def inorder(root):
        if root is None:
            return
        inorder(root.left)
        trace.append({'tree': nodes[:], 'visit': [root.val], 'step': f'Inorder visit: {root.val}'})
        inorder(root.right)
    
    # Preorder traversal
    def preorder(root):
        if root is None:
            return
        trace.append({'tree': nodes[:], 'visit': [root.val], 'step': f'Preorder visit: {root.val}'})
        preorder(root.left)
        preorder(root.right)
    
    # Postorder traversal
    def postorder(root):
        if root is None:
            return
        postorder(root.left)
        postorder(root.right)
        trace.append({'tree': nodes[:], 'visit': [root.val], 'step': f'Postorder visit: {root.val}'})
    
    # Build tree from nodes
    if nodes:
        root = TreeNode(nodes[0])
        for val in nodes[1:]:
            insert_bst(root, val)
        
        trace.append({'tree': nodes[:], 'step': 'Starting inorder traversal'})
        inorder(root)
        trace.append({'tree': nodes[:], 'step': 'Starting preorder traversal'})
        preorder(root)
        trace.append({'tree': nodes[:], 'step': 'Starting postorder traversal'})
        postorder(root)
    
    return trace

def inorder_traversal(nodes):
    trace = []
    def inorder(root):
        if root is None:
            return
        inorder(root.left)
        trace.append({'tree': nodes[:], 'visit': [root.val], 'traversal': 'inorder', 'step': f'Inorder visit: {root.val}'})
        inorder(root.right)
    if nodes:
        root = TreeNode(nodes[0])
        for val in nodes[1:]:
            insert_bst(root, val)
        inorder(root)
    return trace

def preorder_traversal(nodes):
    trace = []
    def preorder(root):
        if root is None:
            return
        trace.append({'tree': nodes[:], 'visit': [root.val], 'traversal': 'preorder', 'step': f'Preorder visit: {root.val}'})
        preorder(root.left)
        preorder(root.right)
    if nodes:
        root = TreeNode(nodes[0])
        for val in nodes[1:]:
            insert_bst(root, val)
        preorder(root)
    return trace

def postorder_traversal(nodes):
    trace = []
    def postorder(root):
        if root is None:
            return
        postorder(root.left)
        postorder(root.right)
        trace.append({'tree': nodes[:], 'visit': [root.val], 'traversal': 'postorder', 'step': f'Postorder visit: {root.val}'})
    if nodes:
        root = TreeNode(nodes[0])
        for val in nodes[1:]:
            insert_bst(root, val)
        postorder(root)
    return trace

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert_bst(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root

# ==================== GRAPH ALGORITHMS ====================

def bfs(graph, start):
    trace = []
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    trace.append({'graph': graph, 'visit': [start], 'step': f'Starting BFS from {start}'})
    
    while queue:
        vertex = queue.popleft()
        trace.append({'graph': graph, 'current': [vertex], 'step': f'Processing vertex {vertex}'})
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                trace.append({'graph': graph, 'visit': [neighbor], 'step': f'Discovered neighbor {neighbor}'})
    
    trace.append({'graph': graph, 'result': list(visited), 'step': f'BFS complete. Visited: {list(visited)}'})
    return trace

def dfs(graph, start):
    trace = []
    visited = set()
    
    def dfs_helper(vertex):
        visited.add(vertex)
        trace.append({'graph': graph, 'visit': [vertex], 'step': f'DFS visiting {vertex}'})
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs_helper(neighbor)
    
    trace.append({'graph': graph, 'step': f'Starting DFS from {start}'})
    dfs_helper(start)
    trace.append({'graph': graph, 'result': list(visited), 'step': f'DFS complete. Visited: {list(visited)}'})
    return trace

def dijkstra(graph, start):
    trace = []
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    trace.append({'graph': graph, 'distances': distances.copy(), 'step': f'Starting Dijkstra from {start}'})
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_dist > distances[current_node]:
            continue
        
        trace.append({'graph': graph, 'current': [current_node], 'step': f'Processing node {current_node} with distance {current_dist}'})
        
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
                trace.append({'graph': graph, 'update': [neighbor, distance], 'step': f'Updated distance to {neighbor}: {distance}'})
    
    trace.append({'graph': graph, 'result': distances, 'step': f'Shortest distances: {distances}'})
    return trace

def bellman_ford(graph, start):
    trace = []
    nodes = list(graph.keys())
    dist = {node: float('inf') for node in nodes}
    dist[start] = 0
    trace.append({'graph': graph, 'distances': dist.copy(), 'step': f'Starting Bellman-Ford from {start}'})
    for i in range(len(nodes) - 1):
        for u in graph:
            for v, w in (graph[u].items() if isinstance(graph[u], dict) else [(n, 1) for n in graph[u]]):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    trace.append({'graph': graph, 'update': [v, dist[v]], 'step': f'Updated distance to {v}: {dist[v]}'})
    trace.append({'graph': graph, 'result': dist, 'step': f'Final distances: {dist}'})
    return trace

def floyd_warshall(graph):
    trace = []
    nodes = list(graph.keys())
    n = len(nodes)
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    for u in nodes:
        dist[u][u] = 0
        for v, w in (graph[u].items() if isinstance(graph[u], dict) else [(n, 1) for n in graph[u]]):
            dist[u][v] = w
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    trace.append({'graph': graph, 'update': [i, j, dist[i][j]], 'step': f'Updated dist[{i}][{j}] to {dist[i][j]}'})
    trace.append({'graph': graph, 'result': dist, 'step': f'All pairs shortest paths: {dist}'})
    return trace

def kruskal(graph):
    trace = []
    parent = {}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        pu, pv = find(u), find(v)
        parent[pu] = pv
    edges = []
    for u in graph:
        for v, w in (graph[u].items() if isinstance(graph[u], dict) else [(n, 1) for n in graph[u]]):
            if u < v:
                edges.append((w, u, v))
    edges.sort()
    for node in graph:
        parent[node] = node
    mst = []
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, w))
            trace.append({'graph': graph, 'mst': mst[:], 'step': f'Added edge {u}-{v} (weight {w}) to MST'})
    trace.append({'graph': graph, 'result': mst, 'step': f'Final MST: {mst}'})
    return trace

def prim(graph, start):
    trace = []
    import heapq
    visited = set()
    min_edges = []
    mst = []
    visited.add(start)
    for v, w in (graph[start].items() if isinstance(graph[start], dict) else [(n, 1) for n in graph[start]]):
        heapq.heappush(min_edges, (w, start, v))
    while min_edges:
        w, u, v = heapq.heappop(min_edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, w))
            trace.append({'graph': graph, 'mst': mst[:], 'step': f'Added edge {u}-{v} (weight {w}) to MST'})
            for to, weight in (graph[v].items() if isinstance(graph[v], dict) else [(n, 1) for n in graph[v]]):
                if to not in visited:
                    heapq.heappush(min_edges, (weight, v, to))
    trace.append({'graph': graph, 'result': mst, 'step': f'Final MST: {mst}'})
    return trace

def topological_sort(graph):
    trace = []
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in (graph[u] if isinstance(graph[u], list) else graph[u].keys()):
            in_degree[v] += 1
    queue = [u for u in graph if in_degree[u] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        trace.append({'graph': graph, 'order': order[:], 'step': f'Added {u} to topological order'})
        for v in (graph[u] if isinstance(graph[u], list) else graph[u].keys()):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    trace.append({'graph': graph, 'result': order, 'step': f'Topological order: {order}'})
    return trace

def tarjan(graph):
    trace = []
    index = 0
    indices = {}
    lowlink = {}
    stack = []
    on_stack = set()
    sccs = []
    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in (graph[v] if isinstance(graph[v], list) else graph[v].keys()):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)
            trace.append({'graph': graph, 'scc': scc[:], 'step': f'Found SCC: {scc}'})
    for v in graph:
        if v not in indices:
            strongconnect(v)
    trace.append({'graph': graph, 'result': sccs, 'step': f'All SCCs: {sccs}'})
    return trace

def kosaraju(graph):
    trace = []
    visited = set()
    order = []
    def dfs(u):
        visited.add(u)
        for v in (graph[u] if isinstance(graph[u], list) else graph[u].keys()):
            if v not in visited:
                dfs(v)
        order.append(u)
    for u in graph:
        if u not in visited:
            dfs(u)
    # Transpose graph
    transpose = {u: [] for u in graph}
    for u in graph:
        for v in (graph[u] if isinstance(graph[u], list) else graph[u].keys()):
            transpose[v].append(u)
    visited.clear()
    sccs = []
    def dfs_rev(u, scc):
        visited.add(u)
        scc.append(u)
        for v in transpose[u]:
            if v not in visited:
                dfs_rev(v, scc)
    for u in reversed(order):
        if u not in visited:
            scc = []
            dfs_rev(u, scc)
            sccs.append(scc)
            trace.append({'graph': graph, 'scc': scc[:], 'step': f'Found SCC: {scc}'})
    trace.append({'graph': graph, 'result': sccs, 'step': f'All SCCs: {sccs}'})
    return trace

def union_find(graph, operations):
    trace = []
    parent = {u: u for u in graph}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for op in operations:
        if op['type'] == 'union':
            u, v = op['u'], op['v']
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
                trace.append({'graph': graph, 'union': [u, v], 'step': f'Union {u} and {v}'})
        elif op['type'] == 'find':
            u = op['u']
            pu = find(u)
            trace.append({'graph': graph, 'find': [u, pu], 'step': f'Find({u}) = {pu}'})
    trace.append({'graph': graph, 'parent': parent.copy(), 'step': f'Final parents: {parent}'})
    return trace

# ==================== BACKTRACKING ALGORITHMS ====================

def n_queens(n):
    trace = []
    
    def is_safe(board, row, col):
        for i in range(col):
            if board[row][i] == 1:
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        return True
    
    def solve_n_queens(board, col):
        if col >= n:
            trace.append({'board': [row[:] for row in board], 'step': 'Solution found!'})
            return True
        
        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 1
                trace.append({'board': [row[:] for row in board], 'place': [i, col], 'step': f'Placed queen at ({i}, {col})'})
                
                if solve_n_queens(board, col + 1):
                    return True
                
                board[i][col] = 0
                trace.append({'board': [row[:] for row in board], 'remove': [i, col], 'step': f'Removed queen from ({i}, {col})'})
        
        return False
    
    board = [[0 for _ in range(n)] for _ in range(n)]
    trace.append({'board': [row[:] for row in board], 'step': f'Starting {n}-queens problem'})
    solve_n_queens(board, 0)
    return trace

# ==================== BIT MANIPULATION ====================

def power_of_two(n):
    trace = []
    trace.append({'number': n, 'binary': bin(n), 'step': f'Checking if {n} is power of 2'})
    
    if n <= 0:
        trace.append({'number': n, 'result': False, 'step': 'Number is not positive'})
        return trace
    
    result = (n & (n - 1)) == 0
    trace.append({'number': n, 'binary': bin(n), 'result': result, 'step': f'Result: {result} (n & (n-1) = {n & (n-1)})'})
    return trace

def count_set_bits(n):
    trace = []
    count = 0
    original = n
    
    trace.append({'number': n, 'binary': bin(n), 'step': f'Counting set bits in {n}'})
    
    while n:
        count += n & 1
        trace.append({'number': n, 'binary': bin(n), 'count': count, 'step': f'LSB: {n & 1}, Total count: {count}'})
        n >>= 1
    
    trace.append({'number': original, 'binary': bin(original), 'result': count, 'step': f'Total set bits: {count}'})
    return trace

# ==================== SLIDING WINDOW ====================

def max_sum_subarray(arr, k):
    trace = []
    if len(arr) < k:
        return trace
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    trace.append({'array': arr[:], 'window': [0, k-1], 'sum': window_sum, 'step': f'Initial window sum: {window_sum}'})
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        trace.append({'array': arr[:], 'window': [i-k+1, i], 'sum': window_sum, 'step': f'Window sum: {window_sum} (removed {arr[i-k]}, added {arr[i]})'})
        max_sum = max(max_sum, window_sum)
    
    trace.append({'array': arr[:], 'result': max_sum, 'step': f'Maximum sum: {max_sum}'})
    return trace

def longest_substring_no_repeat(s):
    trace = []
    char_map = {}
    left = 0
    max_length = 0
    
    trace.append({'string': s, 'step': 'Starting longest substring without repeating characters'})
    
    for right, char in enumerate(s):
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1
            trace.append({'string': s, 'window': [left, right], 'step': f'Repeated character {char}, moved left to {left}'})
        
        char_map[char] = right
        max_length = max(max_length, right - left + 1)
        trace.append({'string': s, 'window': [left, right], 'length': max_length, 'step': f'Current length: {max_length}'})
    
    trace.append({'string': s, 'result': max_length, 'step': f'Longest substring length: {max_length}'})
    return trace

def variable_sliding_window(s):
    trace = []
    left = 0
    max_length = 0
    char_map = {}
    order = 1
    for right, char in enumerate(s):
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1
        char_map[char] = right
        max_length = max(max_length, right - left + 1)
        trace.append({
            'string': s,
            'window': [left, right],
            'length': max_length,
            'step': f'Window [{left},{right}] ("{s[left:right+1]}") length: {right-left+1}',
            'order': order
        })
        order += 1
    trace.append({'string': s, 'result': max_length, 'step': f'Longest substring length: {max_length}'})
    return trace

def two_pointer_pair_sum(arr, target):
    trace = []
    a = sorted(arr.copy())
    left, right = 0, len(a) - 1
    order = 1
    while left < right:
        current_sum = a[left] + a[right]
        trace.append({
            'array': a[:],
            'pointers': [left, right],
            'sum': current_sum,
            'step': f'Checking indices {left} and {right}: {a[left]}+{a[right]}={current_sum}',
            'order': order
        })
        order += 1
        if current_sum == target:
            trace.append({'array': a[:], 'found': [left, right], 'step': f'Found pair: {a[left]}+{a[right]}={target}'})
            return trace
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    trace.append({'array': a[:], 'step': f'No pair found for sum {target}'})
    return trace

# ==================== ALGORITHM REGISTRY ====================

ALGORITHMS = {
    'sorting': {
        'bubble': bubble_sort,
        'selection': selection_sort,
        'insertion': insertion_sort,
        'merge': merge_sort,
        'quick': quick_sort,
        'heap': heap_sort,
        'counting': counting_sort,
        'radix': radix_sort
    },
    'searching': {
        'linear': linear_search,
        'binary': binary_search,
        'jump': jump_search,
        'exponential': exponential_search,
        'ternary': ternary_search
    },
    'greedy': {
        'activity_selection': activity_selection,
        'fractional_knapsack': fractional_knapsack
    },
    'dp': {
        'knapsack_01': knapsack_01,
        'lcs': longest_common_subsequence,
        'fibonacci': fibonacci_dp
    },
    'tree': {
        'traversals': tree_traversals,
        'inorder': inorder_traversal,
        'preorder': preorder_traversal,
        'postorder': postorder_traversal
    },
    'graph': {
        'bfs': bfs,
        'dfs': dfs,
        'dijkstra': dijkstra,
        'bellman_ford': bellman_ford,
        'floyd_warshall': floyd_warshall,
        'kruskal': kruskal,
        'prim': prim,
        'topological_sort': topological_sort,
        'tarjan': tarjan,
        'kosaraju': kosaraju,
        'union_find': union_find
    },
    'backtracking': {
        'n_queens': n_queens
    },
    'bit_manipulation': {
        'power_of_two': power_of_two,
        'count_set_bits': count_set_bits
    },
    'sliding_window': {
        'max_sum_subarray': max_sum_subarray,
        'longest_substring_no_repeat': longest_substring_no_repeat,
        'variable': variable_sliding_window,
        'two_pointer': two_pointer_pair_sum
    }
}

ALGO_ARG_MAP = {
    'sorting': {
        'bubble': {'arr': 'array'},
        'selection': {'arr': 'array'},
        'insertion': {'arr': 'array'},
        'merge': {'arr': 'array'},
        'quick': {'arr': 'array'},
        'heap': {'arr': 'array'},
        'counting': {'arr': 'array'},
        'radix': {'arr': 'array'},
    },
    'searching': {
        'linear': {'arr': 'array', 'target': 'target'},
        'binary': {'arr': 'array', 'target': 'target'},
        'jump': {'arr': 'array', 'target': 'target'},
        'exponential': {'arr': 'array', 'target': 'target'},
        'ternary': {'arr': 'array', 'target': 'target'},
    },
    'greedy': {
        'activity_selection': {'activities': 'activities'},
        'fractional_knapsack': {'weights': 'weights', 'values': 'values', 'capacity': 'capacity'},
    },
    'dp': {
        'knapsack_01': {'weights': 'weights', 'values': 'values', 'capacity': 'capacity'},
        'lcs': {'str1': 'str1', 'str2': 'str2'},
        'fibonacci': {'n': 'n'},
    },
    'tree': {
        'traversals': {'nodes': 'nodes'},
        'inorder': {'nodes': 'nodes'},
        'preorder': {'nodes': 'nodes'},
        'postorder': {'nodes': 'nodes'},
    },
    'graph': {
        'bfs': {'graph': 'graph', 'start': 'start'},
        'dfs': {'graph': 'graph', 'start': 'start'},
        'dijkstra': {'graph': 'graph', 'start': 'start'},
        'bellman_ford': {'graph': 'graph', 'start': 'start'},
        'floyd_warshall': {'graph': 'graph'},
        'kruskal': {'graph': 'graph'},
        'prim': {'graph': 'graph', 'start': 'start'},
        'topological_sort': {'graph': 'graph'},
        'tarjan': {'graph': 'graph'},
        'kosaraju': {'graph': 'graph'},
        'union_find': {'graph': 'graph', 'operations': 'operations'}
    },
    'backtracking': {
        'n_queens': {'n': 'n'},
    },
    'bit_manipulation': {
        'power_of_two': {'n': 'n'},
        'count_set_bits': {'n': 'n'},
    },
    'sliding_window': {
        'max_sum_subarray': {'arr': 'array', 'k': 'k'},
        'longest_substring_no_repeat': {'s': 's'},
        'variable': {'s': 's'},
        'two_pointer': {'arr': 'array', 'target': 'target'},
    },
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/algorithms')
def get_algorithms():
    return jsonify(ALGORITHMS)

@app.route('/api/run_algorithm', methods=['POST'])
def run_algorithm():
    data = request.json
    category = data.get('category')
    algorithm = data.get('algorithm')
    params = data.get('params', {})
    
    if category not in ALGORITHMS or algorithm not in ALGORITHMS[category]:
        return jsonify({'error': 'Unknown algorithm'}), 400
    
    try:
        func = ALGORITHMS[category][algorithm]
        arg_map = ALGO_ARG_MAP.get(category, {}).get(algorithm, {})
        mapped_params = {py_arg: params[front_arg] for py_arg, front_arg in arg_map.items() if front_arg in params}
        trace = func(**mapped_params)
        return jsonify({'trace': trace})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 