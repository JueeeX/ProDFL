import networkx as nx
import random
import matplotlib.pyplot as plt


def generate_communication_graph(num_clients, p):
    """
    Generate a communication graph for DFL where the communication probability between
    each pair of clients is 'p'.

    :param num_clients: The number of clients (nodes in the graph)
    :param p: The probability of communication (edge between two nodes)
    :return: NetworkX graph object representing the communication topology
    """
    # Initialize an empty undirected graph
    G = nx.Graph()

    # Add nodes (clients) to the graph
    G.add_nodes_from(range(num_clients))

    # Iterate over all pairs of nodes to create edges based on probability p
    for i in range(num_clients):
        for j in range(i + 1, num_clients):  # Avoid duplicate pairs by ensuring i < j
            if random.random() < p:
                G.add_edge(i, j)

    return G


def plot_graph(G):
    """
    Plot the communication graph using matplotlib.

    :param G: The NetworkX graph object representing the communication topology
    """
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold")
    plt.title("Communication Topology for Decentralized Federated Learning")
    plt.show()


def print_adjacency_matrix(G):
    """
    Print the adjacency matrix of the graph to show the connectivity between clients.

    :param G: The NetworkX graph object representing the communication topology
    """
    adj_matrix = nx.to_numpy_array(G)
    print("Adjacency Matrix:")
    print(adj_matrix)


# Parameters
num_clients = 50  # Total number of clients (nodes)
communication_probability = 0.5  # Probability of communication between two clients

# Generate the communication graph
communication_graph = generate_communication_graph(num_clients, communication_probability)

# Plot the graph
plot_graph(communication_graph)

# Print the adjacency matrix to visualize the connectivity
print_adjacency_matrix(communication_graph)
