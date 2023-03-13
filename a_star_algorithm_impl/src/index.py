from components.a_star import astar

# Define a simple graph with nodes 0, 1, 2, 3, 4 and edges (0,1), (1,2), (2,3), (3,4)
graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}

# Define the neighbors function as a lambda function that returns the neighbors of a node in the graph
neighbors = lambda n: graph[n]

# Define the distance function as a lambda function that returns 1 for adjacent nodes and 0 for non-adjacent nodes
distance = lambda a, b: 1 if b in graph[a] else 0

# Define the heuristic function as a lambda function that returns the distance to the goal node (4)
heuristic = lambda n: abs(n - 4)

# Call the astar function with starting node 0 and goal node 4
path = astar(0, 4, neighbors, distance, heuristic)

# Print the path from the starting node to the goal node
print("path 1: ",path)



"""
   First, let's define the graph as a dictionary where each key represents a city
    and the value is a list of tuples representing the city's neighbors and
      the distance between them. We'll use the example map shown below:

        A --- 5 --- B --- 7 --- C
        |           |           |
        8           4           2
        |           |           |
        D --- 6 --- E --- 3 --- F
"""
#We can represent this map as the following dictionary:
graph = {
    "A": [("B", 5), ("D", 8)],
    "B": [("A", 5), ("C", 7), ("E", 4)],
    "C": [("B", 7), ("F", 2)],
    "D": [("A", 8), ("E", 6)],
    "E": [("B", 4), ("D", 6), ("F", 3)],
    "F": [("C", 2), ("E", 3)]
}

# Define the neighbors function to return the neighbors of a given city
def neighbors(city):
    return [n[0] for n in graph[city]]

# Define the distance function to return the distance between two neighboring cities
def distance(city1, city2):
    for n in graph[city1]:
        if n[0] == city2:
            return n[1]

# Define the heuristic function to return the straight-line distance between a city and the goal city (C)
def heuristic(city):
    x = {"A": 0, "B": 1, "C": 2, "D": 0, "E": 1, "F": 2}[city]
    y = {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}[city]
    return ((x-2)**2 + (y-0)**2)**0.5



path = astar("A", "F", neighbors, distance, heuristic)

print("path 2: ",path)