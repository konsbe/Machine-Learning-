import heapq

def astar(start, goal, neighbors_fn, distance_fn, heuristic_fn):
    """
    Implements the A* algorithm to find the shortest path from a start node to a goal node
    in a graph. Returns the path as a list of nodes, or None if no path is found.

    Parameters:
    start (int): The index of the starting node in the graph.
    goal (int): The index of the goal node in the graph.
    neighbors_fn (function): A function that takes a node index as input and returns a list of
                             its neighboring nodes in the graph.
    distance_fn (function): A function that takes two node indices as input and returns the
                            distance between them.
    heuristic_fn (function): A function that takes a node index as input and returns an estimate
                             of the distance from that node to the goal node.

    Returns:
    list: The path as a list of nodes, or None if no path is found.
    """
    # Initialize the data structures
    open_set = [(0, start)]  # A priority queue of (f, node) pairs, ordered by f
    closed_set = set()  # A set of visited nodes
    g_scores = {start: 0}  # A dictionary of g scores for each node
    came_from = {}  # A dictionary of parent nodes for each node

    while open_set:
        # Pop the node with the lowest f score from the open set
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # If we've reached the goal node, construct and return the path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        # Mark the current node as visited
        closed_set.add(current)

        # Check the neighbors of the current node
        for neighbor in neighbors_fn(current):
            if neighbor in closed_set:
                # Ignore neighbors that have already been visited
                continue

            # Calculate the tentative g score for this neighbor
            tentative_g_score = g_scores[current] + distance_fn(current, neighbor)

            if neighbor not in [node[1] for node in open_set]:
                # If the neighbor is not in the open set, add it with the tentative g score and f score
                g_scores[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_fn(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current
            elif tentative_g_score < g_scores[neighbor]:
                # If the neighbor is already in the open set and this path to it is better than the previous path, update it
                g_scores[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_fn(neighbor)
                for i, (old_f_score, old_neighbor) in enumerate(open_set):
                    if old_neighbor == neighbor:
                        open_set[i] = (f_score, neighbor)
                        break
                heapq.heapify(open_set)
                came_from[neighbor] = current

    # If we've visited all nodes and haven't found a path, return None
    return None