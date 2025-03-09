import numpy as np
import pickle
from collections import deque
import math

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def dfs(vertex, goal, adj_matrix, depth, vis, path):
    vis[vertex] = depth 
    path.append(vertex)

    if vertex == goal:
        return True
    
    if depth <= 0:
        path.pop()
        return False
    
    for child in range(len(adj_matrix)):
        if adj_matrix[vertex][child] > 0 and (child not in vis or vis[child]<depth):
            if dfs(child, goal, adj_matrix, depth - 1, vis, path):
                return True

    path.pop()
    return False

def get_ids_path(adj_matrix, start_node, goal_node):
    total_nodes = len(adj_matrix)
    depth = 0
    
    while depth <= total_nodes:
        vis = {}
        path = []
        if dfs(start_node, goal_node, adj_matrix, depth, vis, path):
            return path 
        else:
            depth += 1
    
    return None


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def bfsForward(queue, adj_matrix, vis, par):
    last = len(queue)
    for i in range(last):
      current_node = queue.popleft()
      for child in range(len(adj_matrix)):
          if adj_matrix[current_node][child] > 0 and not vis[child]:
              queue.append(child)
              vis[child] = True
              par[child] = current_node

def bfsBackward(queue, adj_matrix, vis, par): # as directed graph
    last = len(queue)
    for i in range(last):
      current_node = queue.popleft()
      for child in range(len(adj_matrix)):
          if adj_matrix[child][current_node] > 0 and not vis[child]: # need forward edge
              queue.append(child)
              vis[child] = True
              par[child] = current_node

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    total_nodes = len(adj_matrix)
        
    forward_queue = deque()
    backward_queue = deque()
    forward_par = [None] * total_nodes
    backward_par = [None] * total_nodes
    forward_vis = [False] * total_nodes
    backward_vis = [False] * total_nodes

    forward_queue.append(start_node)
    backward_queue.append(goal_node)
    forward_vis[start_node] = True
    backward_vis[goal_node] = True

    while forward_queue and backward_queue:
        # Check if there is a direct connection between nodes in both queues
        for forward_node in forward_queue:
            for backward_node in backward_queue:
                if adj_matrix[forward_node][backward_node] > 0 or forward_node == backward_node:
                    path = []
                    while forward_node != start_node:
                        path.append(forward_node)
                        forward_node = forward_par[forward_node]
                    path.append(start_node)
                    path.reverse()
                    while backward_node != goal_node:
                        path.append(backward_node)
                        backward_node = backward_par[backward_node]
                    if start_node != goal_node:
                        path.append(goal_node) # for case if both are equal, only once
                    return path
        
        bfsForward(forward_queue, adj_matrix, forward_vis, forward_par)
        bfsBackward(backward_queue, adj_matrix, backward_vis, backward_par)

        # for i in range(len(forward_vis)):
        #     if (forward_vis[i]):
        #         print(i, end=" ")
        # print()
        # for i in range(len(backward_vis)):
        #     if (backward_vis[i]):
        #         print(i, end=" ")
        # print()
        
        # Check if some node is visited by both
        for i in range(total_nodes):
            if forward_vis[i] and backward_vis[i]:
                path = []
                node = i
                while node != start_node:
                    path.append(node)
                    node = forward_par[node]
                path.append(start_node)
                path.reverse()
                if start_node != goal_node:
                    path.pop() # to avoid repetition
                node = i
                while node != goal_node:
                    path.append(node)
                    node = backward_par[node]
                if start_node != goal_node:
                    path.append(goal_node)
                return path
          
    return None


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def euclideanDistance(node_u, node_v, node_attributes):
    x1 = node_attributes[node_u]['x']
    y1 = node_attributes[node_u]['y']
    x2 = node_attributes[node_v]['x']
    y2 = node_attributes[node_v]['y']
    return math.sqrt((float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)

# h(w) = dist(u, w) + dist(w, v)
def heuristicFunc(start_node, vertex, goal_node, node_attributes):
    return euclideanDistance(start_node, vertex, node_attributes) + euclideanDistance(vertex, goal_node, node_attributes)

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    total_nodes = len(adj_matrix)
    
    f_cost = {} # stores f-values
    g_cost = {i: float('inf') for i in range(total_nodes)} # stores g-cost from start
    explored_set = set()

    g_cost[start_node] = 0
    f_cost[start_node] = g_cost[start_node] + heuristicFunc(start_node, start_node, goal_node, node_attributes)
    
    parent = {i: None for i in range(total_nodes)}
    
    while f_cost:
        vertex = min(f_cost, key=f_cost.get)
        
        if vertex == goal_node:
            path = []
            while vertex is not None:
                path.append(vertex)
                vertex = parent[vertex]
            path.reverse()
            return path
        
        del f_cost[vertex]
        explored_set.add(vertex)
        
        for child in range(total_nodes):
            if adj_matrix[vertex][child] > 0 and child not in explored_set:
                g_temp = g_cost[vertex] + adj_matrix[vertex][child]
                
                if g_temp < g_cost[child]:
                    g_cost[child] = g_temp
                    parent[child] = vertex
                    
                    # f(n) = g(n) + h(n) 
                    f_cost[child] = g_cost[child] + heuristicFunc(start_node, child, goal_node, node_attributes)
                    
    return None


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]
def expandNodeForward(vertex, adj_matrix, g_cost, f_cost, parent, explored, start_node, goal_node, node_attributes):
    for child in range(len(adj_matrix)):
        if adj_matrix[vertex][child] > 0 and child not in explored:
            g_temp = g_cost[vertex] + adj_matrix[vertex][child]
            if g_temp < g_cost[child]:
                g_cost[child] = g_temp
                parent[child] = vertex
                f_cost[child] = g_temp + heuristicFunc(start_node, child, goal_node, node_attributes)

def expandNodeBackward(vertex, adj_matrix, g_cost, f_cost, parent, explored, start_node, goal_node, node_attributes):
    for child in range(len(adj_matrix)):
        if adj_matrix[child][vertex] > 0 and child not in explored:
            g_temp = g_cost[vertex] + adj_matrix[child][vertex]
            if g_temp < g_cost[child]:
                g_cost[child] = g_temp
                parent[child] = vertex
                f_cost[child] = g_temp + heuristicFunc(start_node, child, goal_node, node_attributes)

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    total_nodes = len(adj_matrix)
    
    # store f-values
    forward_f_cost = {} 
    backward_f_cost = {}

    # store g-values from start node
    forward_g_cost = {i: float('inf') for i in range(total_nodes)}
    backward_g_cost = {i: float('inf') for i in range(total_nodes)}
    
    forward_parent = {i: None for i in range(total_nodes)}
    backward_parent = {i: None for i in range(total_nodes)}
    
    forward_explored = set()
    backward_explored = set()
    
    forward_g_cost[start_node] = 0
    backward_g_cost[goal_node] = 0
    
    forward_f_cost[start_node] = forward_g_cost[start_node] + heuristicFunc(start_node, start_node, goal_node, node_attributes)
    backward_f_cost[goal_node] = backward_g_cost[goal_node] + heuristicFunc(goal_node, goal_node, start_node, node_attributes)
    
    mini = float('inf')
    common_node = None

    while forward_f_cost and backward_f_cost:
        if forward_f_cost:
            forward_vertex = min(forward_f_cost, key=forward_f_cost.get)
            
            if forward_vertex in backward_explored:
                temp_cost = forward_g_cost[forward_vertex] + backward_g_cost[forward_vertex]
                if temp_cost < mini:
                    mini = temp_cost
                    common_node = forward_vertex

            del forward_f_cost[forward_vertex]
            forward_explored.add(forward_vertex)
            expandNodeForward(forward_vertex, adj_matrix, forward_g_cost, forward_f_cost, forward_parent, forward_explored, start_node, goal_node, node_attributes)

        if backward_f_cost:
            backward_vertex = min(backward_f_cost, key=backward_f_cost.get)
            
            if backward_vertex in forward_explored:
                temp_cost = forward_g_cost[backward_vertex] + backward_g_cost[backward_vertex]
                if temp_cost < mini:
                    mini = temp_cost
                    common_node = backward_vertex

            del backward_f_cost[backward_vertex]
            backward_explored.add(backward_vertex)
            expandNodeBackward(backward_vertex, adj_matrix, backward_g_cost, backward_f_cost, backward_parent, backward_explored, goal_node, start_node, node_attributes)
  
    if common_node is None:
        return None
    
    path = []
    
    node = common_node
    while node is not None:
        path.append(node)
        node = forward_parent[node]
    path.reverse()

    node = backward_parent[common_node]
    while node is not None:
        path.append(node)
        node = backward_parent[node]

    return path


# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def make_undirected(adj_matrix):
    size = len(adj_matrix)
    for i in range(size):
        for j in range(i + 1, size):
            if adj_matrix[i][j] > 0 or adj_matrix[j][i] > 0:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
    return adj_matrix

# Help is taken from Take you forward DSA Course in writing the Finding Bridges code

def dfs_find_bridges(node, parent, discovery, low, visited, adj_matrix, bridges, time):
    visited[node] = True
    discovery[node] = low[node] = time[0]
    time[0] += 1

    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node][neighbor] == 0:
            continue

        if not visited[neighbor]:
            dfs_find_bridges(neighbor, node, discovery, low, visited, adj_matrix, bridges, time)
            low[node] = min(low[node], low[neighbor])

            if low[neighbor] > discovery[node]:
                bridges.append((node, neighbor))

        elif neighbor != parent:
            low[node] = min(low[node], discovery[neighbor])

def find_bridges(adj_matrix):
    size = len(adj_matrix)
    discovery = [-1] * size  
    low = [-1] * size  
    visited = [False] * size  
    bridges = []  
    time = [0] 

    for node in range(size):
        if not visited[node]:
            dfs_find_bridges(node, -1, discovery, low, visited, adj_matrix, bridges, time)

    # Remove duplicates: since the graph is undirected, (u, v) is the same as (v, u)
    unique_bridges = list(set(tuple(sorted(bridge)) for bridge in bridges))

    return unique_bridges if unique_bridges else []


def bonus_problem(adj_matrix):
  undirected_adj_matrix = make_undirected(adj_matrix)
  bridges = find_bridges(undirected_adj_matrix)
  return bridges

if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')
        

import time
import tracemalloc

# part (b)

def measure_performance_ids_bibfs(adj_matrix):
    total_nodes = len(adj_matrix)

    ids_memory_usage = 0
    ids_total_time = 0

    bidirectional_memory_usage = 0
    bidirectional_total_time = 0

    # Measure performance for IDS
    print("Measuring performance for Iterative Deepening Search (IDS)...")
    for start_node in range(total_nodes):
        for goal_node in range(total_nodes):
            tracemalloc.start()
            start_time = time.time()
            
            get_ids_path(adj_matrix, start_node, goal_node)

            ids_total_time += (time.time() - start_time)
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            ids_memory_usage += peak_memory
            tracemalloc.stop()
            print(f"Completed goal node {goal_node}")
        print(f"Completed start node {start_node}")

    # Measure performance for Bidirectional BFS
    print("Measuring performance for Bidirectional BFS...")
    for start_node in range(total_nodes):
        for goal_node in range(total_nodes):
            tracemalloc.start()
            start_time = time.time()
            
            get_bidirectional_search_path(adj_matrix, start_node, goal_node)

            bidirectional_total_time += (time.time() - start_time)
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            bidirectional_memory_usage += peak_memory
            tracemalloc.stop()
        print(f"Completed start node {start_node}")

    print("\n--- Performance Report ---")
    print(f"Total time for IDS: {ids_total_time:.4f} seconds")
    print(f"Total memory usage for IDS: {ids_memory_usage / 1e6:.4f} MB")
    
    print(f"Total time for Bidirectional BFS: {bidirectional_total_time:.4f} seconds")
    print(f"Total memory usage for Bidirectional BFS: {bidirectional_memory_usage / 1e6:.4f} MB")

# measure_performance_ids_bibfs(adj_matrix)


# Part (e)

def measure_performance_astar_bidir_heur(adj_matrix, node_attributes):
    total_nodes = len(adj_matrix)

    astar_memory_usage = 0
    astar_total_time = 0

    bidirectional_astar_memory_usage = 0
    bidirectional_astar_total_time = 0

    # Measure performance for A* Search
    print("Measuring performance for A* Search...")
    for start_node in range(total_nodes):
        for goal_node in range(total_nodes):
            if start_node != goal_node:
                tracemalloc.start()
                start_time = time.time()
                
                get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node)

                astar_total_time += (time.time() - start_time)
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                astar_memory_usage += peak_memory
                tracemalloc.stop()
        print(f"Completed start node {start_node}")

    # Measure performance for Bidirectional A* Search
    print("Measuring performance for Bidirectional A* Search...")
    for start_node in range(total_nodes):
        for goal_node in range(total_nodes):
            if start_node != goal_node:
                tracemalloc.start()
                start_time = time.time()
                
                get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node)

                bidirectional_astar_total_time += (time.time() - start_time)
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                bidirectional_astar_memory_usage += peak_memory
                tracemalloc.stop()
        print(f"Completed start node {start_node}")

    print("\n--- Performance Report ---")
    print(f"Total time for A* Search: {astar_total_time:.4f} seconds")
    print(f"Total memory usage for A* Search: {astar_memory_usage / 1e6:.4f} MB")
    
    print(f"Total time for Bidirectional A* Search: {bidirectional_astar_total_time:.4f} seconds")
    print(f"Total memory usage for Bidirectional A* Search: {bidirectional_astar_memory_usage / 1e6:.4f} MB")

# measure_performance_astar_bidir_heur(adj_matrix, node_attributes)


# Part (f)
import matplotlib.pyplot as plt
import time
import tracemalloc

def computePathCost(path, adj_matrix):
    if path is None:
        return 0
    cost = sum(adj_matrix[path[i-1]][path[i]] for i in range(1, len(path)))
    return cost

def measure_performance(algorithm_func, adj_matrix, node_attributes=None):
    data = {'time': [], 'space': [], 'cost': []}
    total_nodes = len(adj_matrix)
    
    for start_node in range(total_nodes):
        for goal_node in range(total_nodes):
            if start_node == goal_node:
                continue
            
            tracemalloc.start()
            start_time = time.time()
            path = algorithm_func(adj_matrix, node_attributes, start_node, goal_node)
            elapsed_time = time.time() - start_time
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            cost = computePathCost(path, adj_matrix)

            data['time'].append(elapsed_time)
            data['space'].append(peak_memory / 1e6) 
            data['cost'].append(cost)

            # print(f"Completed goal node: {goal_node}")
        
        print(f"Completed start node: {start_node}/{total_nodes}")
    
    return data

def get_ids_performance(adj_matrix, node_attributes=None):
    return measure_performance(get_ids_path, adj_matrix)

def get_bidirectional_bfs_performance(adj_matrix, node_attributes=None):
    return measure_performance(get_bidirectional_search_path, adj_matrix)

def get_astar_performance(adj_matrix, node_attributes):
    return measure_performance(get_astar_search_path, adj_matrix, node_attributes)

def get_bidirectional_astar_performance(adj_matrix, node_attributes):
    return measure_performance(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes)

def measure_all_algorithms_performance(adj_matrix, node_attributes):
    performance_data = {
        'ids': get_ids_performance(adj_matrix),
        'bidirectional_bfs': get_bidirectional_bfs_performance(adj_matrix),
        'astar': get_astar_performance(adj_matrix, node_attributes),
        'bidirectional_astar': get_bidirectional_astar_performance(adj_matrix, node_attributes)
    }
    
    return performance_data

def plot_performance_scatter(data):
    algorithms = {
        'ids': 'IDS',
        'bidirectional_bfs': 'Bidirectional BFS',
        'astar': 'A*',
        'bidirectional_astar': 'Bidirectional A*'
    }
    metrics = {
        'time': 'Time (seconds)',
        'space': 'Space (MB)',
        'cost': 'Cost (units)'
    }
    
    for algo_key, algo_name in algorithms.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for ax, (metric_key, metric_name) in zip(axes, metrics.items()):
            x_values = range(len(data[algo_key][metric_key]))
            y_values = data[algo_key][metric_key]
            
            ax.scatter(x_values, y_values, alpha=0.7, edgecolors='b', s=10)
            ax.set_title(f"{metric_name}", fontsize=12)
            ax.set_xlabel("Test Case Index", fontsize=10)
            ax.set_ylabel(f"{metric_name}", fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.suptitle(f"Performance of {algo_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

# data = measure_all_algorithms_performance(adj_matrix, node_attributes)
# plot_performance_scatter(data)
