import heapq
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

def uninformed_path_finder(cities, roads, start_city, goal_city, strategy):
 
    if strategy == "bfs":
        return bfs(roads, start_city, goal_city)
    elif strategy == "dfs":
        return dfs(roads, start_city, goal_city)
    elif strategy == "ucs":
        return ucs(roads, start_city, goal_city)
    else:
        raise ValueError("Invalid strategy! Choose 'bfs', 'dfs', or 'ucs'.")

def bfs(roads, start, goal):
    queue = deque([(start, [start], 0)])  # (current_city, path, cost)
    visited = set()
    
    while queue:
        current, path, cost = queue.popleft()
        if current == goal:
            return path, cost
        if current not in visited:
            visited.add(current)
            for neighbor, distance in roads[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], cost + 1))
    return None, float("inf")

def dfs(roads, start, goal):
    stack = [(start, [start], 0)]  # (current_city, path, cost)
    visited = set()
    
    while stack:
        current, path, cost = stack.pop()
        if current == goal:
            return path, cost
        if current not in visited:
            visited.add(current)
            for neighbor, distance in roads[current]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + 1))
    return None, float("inf")

def ucs(roads, start, goal):
    priority_queue = [(0, start, [start])]  # (cumulative_cost, current_city, path)
    visited = set()
    
    while priority_queue:
        cost, current, path = heapq.heappop(priority_queue)
        if current == goal:
            return path, cost
        if current not in visited:
            visited.add(current)
            for neighbor, distance in roads[current]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (cost + distance, neighbor, path + [neighbor]))
    return None, float("inf")

def traverse_all_cities(cities, roads, start_city, strategy):
    visited = set()
    path = []
    cost = 0
    
    def visit(city, current_cost):
        nonlocal path, cost
        visited.add(city)
        path.append(city)
        cost += current_cost

    def backtrack(city):
        for neighbor, distance in roads[city]:
            if neighbor not in visited:
                visit(neighbor, distance)
                backtrack(neighbor)
    
    visit(start_city, 0)
    backtrack(start_city)
    return path, cost

# Visualization
def visualize_graph(cities, roads, path=None):
    G = nx.Graph()
    for city in cities:
        G.add_node(city)
    for city, neighbors in roads.items():
        for neighbor, distance in neighbors:
            G.add_edge(city, neighbor, weight=distance)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, font_weight="bold")
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)
    plt.show()

# Example
cities = ['Addis Ababa', 'Bahir Dar', 'Gondar', 'Hawassa', 'Mekelle']
roads = {
    'Addis Ababa': [('Bahir Dar', 510), ('Hawassa', 275)],
    'Bahir Dar': [('Addis Ababa', 510), ('Gondar', 180)],
    'Gondar': [('Bahir Dar', 180), ('Mekelle', 300)],
    'Hawassa': [('Addis Ababa', 275)],
    'Mekelle': [('Gondar', 300)]
}

# Example Runs
start_city = "Addis Ababa"
goal_city = "Mekelle"
print("BFS:", uninformed_path_finder(cities, roads, start_city, goal_city, "bfs"))
print("DFS:", uninformed_path_finder(cities, roads, start_city, goal_city, "dfs"))
print("UCS:", uninformed_path_finder(cities, roads, start_city, goal_city, "ucs"))

visualize_graph(cities, roads, ['Addis Ababa', 'Bahir Dar', 'Gondar', 'Mekelle'])
