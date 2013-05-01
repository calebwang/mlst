import networkx as nx
import nx_reader 
import reader
import matplotlib.pyplot as plt
import itertools as it
from networkx.algorithms import bipartite

def import_file(filename):
    """
    Read a hard.in file
    """
    r = nx_reader.InFileReader(open(filename))
    graphs = r.read_input_file()
    return graphs

def draw(graph):
    """
    Draw a graph.
    """
    nx.draw(graph)
    plt.show()

def argmax(dct, exceptions = []):
    """
    Find the key in a dictionary with the greatest value,
    omitting keys in exceptions.
    """
    if not dct:
        raise Exception("empty dictionary")
    max_value = 0
    max_key = None
    for k in dct:
        if k not in exceptions and dct[k] > max_value:
            max_value = dct[k]
            max_key = k
    return max_key

def argmax_in(dct, valid_set, exceptions = []):
    """
    Find the key in a dictionary in a given set of keys with the greatest value,
    omitting keys in exceptions.
    """
    if not dct:
        raise Exception("empty dictionary")
    max_value = 0
    max_key = None
    for k in valid_set:
        if k in dct and k not in exceptions and dct[k] > max_value:
            max_value = dct[k]
            max_key = k
    return max_key

def approximate_solution(graph):
    """
    Given a graph, construct a solution greedily using approximation methods.
    """
    new_graph = nx.Graph()
    degrees = nx.degree_centrality(graph) 
    largest = argmax(degrees)
    new_graph.add_node(largest)
    while new_graph.number_of_edges() < graph.number_of_nodes() - 1:
        neighbor_list = [nx.neighbors(graph, n) for n in new_graph.nodes()]
        neighbors = set()
        for lst in neighbor_list:
            neighbors = neighbors.union(lst)
        if not neighbors:
            break
        next_largest = argmax_in(degrees, neighbors, exceptions = new_graph.nodes())
        possible_edge_ends = [n for n in nx.neighbors(graph, next_largest) 
                              if graph.has_edge(n, next_largest) 
                              and n in new_graph.nodes()]
        new_graph.add_node(next_largest)
        edge_end = argmax_in(degrees, possible_edge_ends)
        new_graph.add_edge(edge_end, next_largest)
    print new_graph.edges()
    draw(new_graph)
    return new_graph

def local_search(graph, solution):
    """
    Given an original graph and a spanning tree of that graph,
    perform local search to optimize given solution.
    """
    pass
