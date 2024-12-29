import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Input data
cities = ['Addis Ababa', 'Bahir Dar', 'Gondar', 'Hawassa', 'Mekelle']
roads = {
    'Addis Ababa': [('Bahir Dar', 510), ('Hawassa', 275)],
    'Bahir Dar': [('Addis Ababa', 510), ('Gondar', 180)],
    'Gondar': [('Bahir Dar', 180), ('Mekelle', 300)],
    'Hawassa': [('Addis Ababa', 275)],
    'Mekelle': [('Gondar', 300)]
}

# Create graph
def create_graph(cities, roads):
    graph = {}
    for city in cities:
        graph[city] = roads.get(city, [])
    return graph

graph = create_graph(cities, roads)

# Weighted BFS (using priority queue)
def weighted_bfs(graph, start_city, goal_city):
    """
    Parameters:
    - graph: Dictionary representing the adjacency list with weights.
    - start_city: The city to start the search.
    - goal_city: The target city to reach.

    Returns:
    - path: List of cities representing the shortest path (by distance) from start_city to goal_city.
    - cost: Total cost (distance) of the path.
    """
    visited = set()
    pq = [(0, start_city, [start_city])]  # Priority queue: (cost, city, path)
    
    while pq:
        current_cost, current_city, path = heapq.heappop(pq)
        
        if current_city == goal_city:
            return path, current_cost  # Return path and total cost
        
        if current_city not in visited:
            visited.add(current_city)
            for neighbor, cost in graph[current_city]:
                if neighbor not in visited:
                    heapq.heappush(pq, (current_cost + cost, neighbor, path + [neighbor]))
    
    return None, float('inf')  # Return None if no path is found

# Visualization
def visualize_graph(graph):
    """
    Visualizes the graph using NetworkX and Matplotlib.
    """
    G = nx.Graph()
    
    # Add edges and nodes
    for city, neighbors in graph.items():
        for neighbor, distance in neighbors:
            G.add_edge(city, neighbor, weight=distance)
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Positioning the nodes
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_weight="bold")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    plt.title("Road Network of Ethiopian Cities")
    plt.show()

# Test weighted BFS
weighted_path, weighted_cost = weighted_bfs(graph, "Addis Ababa", "Mekelle")
print("Weighted BFS Path:", weighted_path)
print("Weighted BFS Cost:", weighted_cost)

# Visualize the graph
visualize_graph(graph)
