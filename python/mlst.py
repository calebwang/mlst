import networkx as nx
import nx_reader 
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import logging

def process_file(filename):
    out_file = filename.split('.in')[0] + '.out'
    graphs = import_file(filename)
    solutions = [run(g) for g in graphs]
    write_output(solutions, out_file)

def import_file(filename):
    """
    Read a hard.in file,
    return list of graphs from input
    """
    r = nx_reader.InFileReader(open(filename))
    graphs = r.read_input_file()
    return graphs

def write_output(graphs, out_file):
    with open(out_file, 'w') as f:
        header = str(len(graphs))
        f.write(header + '\n')
        for graph in graphs:
            num_edges = str(graph.number_of_edges())
            f.write(num_edges + '\n')
            for edge in graph.edges():
                f.write( str(edge[0]) + ' ' + str(edge[1]) + '\n' )

def process_file(filename):
    graphs = import_file(filename)
    solutions = run_multiple(graphs)
    return solutions

def draw(graph):
    """
    Draw a graph.
    """
    nx.draw(graph)
    plt.show()

def generate_random_graph(p):
    """Generate random graph with 100 vertices with 
    p probability of edge creation
    """
    return nx.fast_gnp_random_graph(100, p)

def count_leaves(graph):
    """
    Count leaves in graph
    """
    return len([n for n in graph.nodes() if len(graph.neighbors(n)) == 1])

def get_leaves(graph):
    return [n for n in graph.nodes() if len(graph.neighbors(n)) == 1]

def get_nonleaves(graph):
    return [n for n in graph.nodes() if len(graph.neighbors(n)) != 1]

def get_vertices_with_degree(graph, degree):
    return [n for n in graph.nodes() if len(graph.neighbors(n)) == degree]
    
def argmax(dct, exceptions = []):
    """
    Find the key in a dictionary with the greatest value,
    omitting keys in exceptions.
    """
    if not dct:
        raise Exception("empty dictionary")
    max_value = -1
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
    max_value = -1
    max_key = None
    for k in valid_set:
        if k in dct and k not in exceptions and dct[k] > max_value:
            max_value = dct[k]
            max_key = k
    return max_key

def count_uncovered_degree(graph, new_graph, v):
    degree = len(nx.neighbors(graph, v))
    for n in nx.neighbors(graph, v):
        if n in new_graph:
            degree = degree - 1
    return degree
            

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
            return

        next_largest = argmax_in(degrees, neighbors, exceptions = new_graph.nodes())
        next_largest_list = [n for n in neighbors if (n not in new_graph.nodes()) and degrees[n] == degrees[next_largest]]
        best_edge = None
        best_score = 0
        for n_largest in next_largest_list:

            possible_edge_ends = [n for n in nx.neighbors(graph, n_largest) 
                                  if n in new_graph.nodes()]
            new_graph.add_node(n_largest)

            edge_end = argmax_in(degrees, possible_edge_ends)
            
            new_graph.add_edge(edge_end, n_largest)
            best_subscore = count_leaves(new_graph)
            best_end = edge_end
            new_graph.remove_edge(edge_end, n_largest)

            for end in possible_edge_ends:
                new_graph.add_edge(end, n_largest)
                subscore = count_leaves(new_graph)
                if subscore > best_subscore:
                    best_end = end
                    best_subscore = subscore
                new_graph.remove_edge(end, n_largest)
                
                
            new_graph.remove_node(n_largest)
            if best_subscore > best_score:
                best_edge = (n_largest, best_end)
                best_score = best_subscore
        new_graph.add_edge(best_edge[0], best_edge[1])

    return new_graph

def med_approximate_solution(graph):
    """
    Given a graph, construct a solution greedily using approximation methods.
    Performs roughly equal with approximate_solution in terms of optimality
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
        best_subscore = count_leaves(new_graph)
        best_end = edge_end
        new_graph.remove_edge(edge_end, next_largest)

        for end in possible_edge_ends:
            new_graph.add_edge(end, next_largest)
            subscore = count_leaves(new_graph)
            if subscore > best_subscore:
                best_end = end
                best_subscore = subscore
            new_graph.remove_edge(end, next_largest)
       
        new_graph.add_edge(best_end, next_largest)

    return new_graph


def fast_approximate_solution(graph):
    """
    Given a graph, construct a solution greedily using approximation methods.
    Performs roughly equal with approximate_solution in terms of optimality
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

    return new_graph

def fast_approximate_solution_two(graph):
    """
    Given a graph, construct a solution greedily using approximation methods.
    Performs bad.
    """
    new_graph = nx.Graph()
    degrees = nx.degree_centrality(graph) 
    largest = argmax(degrees)
    new_graph.add_node(largest)
    while new_graph.number_of_edges() < graph.number_of_nodes() - 1:
        degrees = {n: count_uncovered_degree(graph, new_graph, n) for n in nx.nodes(graph)}
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

    return new_graph



def basic_local_search(graph, solution):

    original = solution.copy()
    before = count_leaves(solution)
    best_solution = solution.copy()
    best_leaves = count_leaves(best_solution)
    for i in range(10):
        best = count_leaves(solution)
        candidates = set(get_vertices_with_degree(solution, 2))
        leaves = set(get_vertices_with_degree(solution, 1))
        leaf_neighbors = []
        for leaf in leaves:
            leaf_neighbors.extend(nx.neighbors(solution, leaf))
        leaf_neighbors = set(leaf_neighbors)
        vs = candidates.intersection(leaf_neighbors)
        for v in vs:
            leafs = [l for l in nx.neighbors(solution, v) if l in leaves]
            if leafs:
                leaf = leafs[0]
            else:
                break
            solution.remove_edge(v, leaf)
            neighbors = nx.neighbors(graph, leaf)
            for neighbor in neighbors:
                solution.add_edge(leaf, neighbor)
                new = count_leaves(solution)
                if new > best:
                    best = new
                else:
                    solution.remove_edge(leaf, neighbor)
            if not nx.is_connected(solution):
                solution.add_edge(v, leaf)
        if count_leaves(solution) < best_leaves:
            solution = best_solution.copy()
        elif count_leaves(solution) > best_leaves:
            best_solution = solution.copy()
            best_leaves = count_leaves(best_solution)

    after = count_leaves(solution)
    if before > after:
        solution = original.copy()
    if before != after:
        print 'before/after: ', before, after
    if before > after:
        raise Exception('you dun goofed')
    if not nx.is_connected(solution):
        raise Exception('you dun goofed')
    return solution

def local_search(graph, solution, k_value = 5):
    """
    This is BROKEN
    Given an original graph and a spanning tree of that graph,
    perform local search to optimize given solution.
    """
    
    before = count_leaves(solution)
    best_solution = solution.copy()
    best_leaves = count_leaves(best_solution)
    
    ks = range(k_value)
    temp = ks[:]
    temp.reverse()
    ks = ks + temp
    for k in ks:
        for i in range(5):
            best = count_leaves(solution)
            candidates = set(get_vertices_with_degree(solution, k + 1))
            leaves = set(get_vertices_with_degree(solution, k))
            leaf_neighbors = []
            for leaf in leaves:
                leaf_neighbors.extend(nx.neighbors(solution, leaf))
            leaf_neighbors = set(leaf_neighbors)
            vs = candidates.intersection(leaf_neighbors)
            for v in vs:
                leafs = [l for l in nx.neighbors(solution, v) if l in leaves]
                if leafs:
                    leaf = leafs[0]
                else:
                    break
                solution.remove_edge(v, leaf)
                neighbors = nx.neighbors(graph, leaf)
                for neighbor in neighbors:
                    solution.add_edge(leaf, neighbor)
                    new = count_leaves(solution)
                    if new > best:
                        best = new
                    else:
                        solution.remove_edge(leaf, neighbor)
                if not nx.is_connected(solution):
                    solution.add_edge(v, leaf)
            if count_leaves(solution) < best_leaves:
                solution = best_solution.copy()
            elif count_leaves(solution) > best_leaves:
                best_solution = solution.copy()
                best_leaves = count_leaves(best_solution)
        if count_leaves(solution) < best_leaves:
            solution = best_solution.copy()
        elif count_leaves(solution) > best_leaves:
            best_solution = solution.copy()
            best_leaves = count_leaves(best_solution)
    if count_leaves(solution) < best_leaves:
        solution = best_solution.copy()
    elif count_leaves(solution) > best_leaves:
        best_solution = solution.copy()
        best_leaves = count_leaves(best_solution)


    after = count_leaves(solution)
    if before != after:
        print 'before/after: ', before, after
    if before > after:
        raise Exception('you dun goofed')
    if not nx.is_connected(solution):
        raise Exception('you dun goofed')
    return solution


def test(p):
    g = generate_random_graph(p)
    return test_run(g)

def test_run(g):
    if not nx.is_connected(g):
        return None
    aprx_score = count_leaves(approximate_solution(g))
    med_score = count_leaves(med_approximate_solution(g))
    fast_score = count_leaves(fast_approximate_solution(g))
    fast_score_two = count_leaves(fast_approximate_solution_two(g))
    mst_score = count_leaves(nx.minimum_spanning_tree(g))
    print "Number of leaves in aprx: " + str(aprx_score)
    print "Number of leaves in med aprx: " + str(med_score)
    print "Number of leaves in fast aprx: " + str(fast_score)
    print "Number of leaves in fast2 aprx: " + str(fast_score_two)
    print "Number of leaves in MST solution: " + str(mst_score)
    return g, aprx_score, med_score, fast_score, fast_score_two, mst_score

def run(g):
    if not nx.is_connected(g):
        return None
    best = 0
    best_fn = None
    best_sol = None

    fns = [approximate_solution, med_approximate_solution, fast_approximate_solution, 
           fast_approximate_solution_two, nx.minimum_spanning_tree]

    print '--------------------------'
    for f in fns:
        sol = f(g)
        score = count_leaves(sol)

        new_sol = basic_local_search(g, sol.copy())
        if count_leaves(new_sol) > score:
            score = count_leaves(new_sol)
            sol = new_sol

        print f.func_name, score
        if score > best:
            best = score
            best_fn = f.func_name
            best_sol = sol

    print 'best: ', best_fn, best

    return best_sol

def run_multiple(graphs):
    return [run(g) for g in graphs]

def test_average(n):
    totals = np.array([0, 0, 0, 0, 0])
    for i in range(n):
        results = test(random.randrange(5, 15)/100.0)
        while results == None:
            results = test(random.randrange(5, 25)/100.0)
        results = np.array(results[1:])
        totals = totals + results

    return totals/float(i + 1)
        
