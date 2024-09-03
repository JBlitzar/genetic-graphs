import matplotlib.pyplot as plt
import networkx as nx

def display_graph(G):
    pos = nx.kamada_kawai_layout(G)
    
    # Truncate labels to the first 10 characters
    labels = {node: node[:10] for node, data in G.nodes(data=True)}
    
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                           node_color="#ffffff", node_size=500)
    nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)

    plt.show()


def save_graph(G, path):

    fig, ax = plt.subplots()

    pos = nx.kamada_kawai_layout(G)


    labels = {node: node[:10] for node, data in G.nodes(data=True)}


    nx.draw_networkx_nodes(G, pos, ax=ax, cmap=plt.get_cmap('jet'), 
                           node_color="#ffffff", node_size=500)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True, ax=ax)


    fig.savefig(path)
    

    plt.close(fig)


if __name__ == "__main__":

    seed = 13648  # Seed random number generators for reproducibility
    G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
    display_graph(G)