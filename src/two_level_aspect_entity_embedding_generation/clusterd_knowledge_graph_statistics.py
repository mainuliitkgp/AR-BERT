from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

if __name__ == "__main__":
    graph_file = sys.argv[1]
    #out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    #print(G_data)
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    count = 0
    max_node_degree = 0
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        else :
            count += G.degree(node)
            if G.degree(node)>max_node_degree:
                max_node_degree = G.degree(node)
    
    avg_node_degree = count/len(nodes)
    print(len(nodes), avg_node_degree, max_node_degree)
    print(nx.is_connected(G))
