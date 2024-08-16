import matplotlib.pyplot as plt
import networkx as nx

def display_graph(G):
    pos = nx.spring_layout(G)#, seed=seed)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                        node_color = "#ffffff", node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)

    plt.show()
if __name__ == "__main__":

    seed = 13648  # Seed random number generators for reproducibility
    G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
    display_graph(G)