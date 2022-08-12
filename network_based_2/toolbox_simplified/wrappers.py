import sys
import os

sys.path.append(os.path.abspath("/Users/yanlixu/Desktop/git_code/drug_repurposing_202206/network_based_2/toolbox_simplified"))
import network_utilis as network_utilities
import os, numpy


def get_network(network_file, only_lcc):
    network = network_utilities.create_network_from_sif_file(network_file, use_edge_data=False, delim=None,include_unconnected=True)
    #print(network)
    # print len(network.nodes()), len(network.edges())
    if only_lcc and not network_file.endswith(".lcc"):
        print("Shrinking network to its LCC", len(network.nodes()), len(network.edges()))
        components = network_utilities.get_connected_components(network, False)
        network = network_utilities.get_subgraph(network, components[0])
        #print(components)
        #print(network)
        #print("Final shape:", len(network.nodes()), len(network.edges()))
        # print len(network.nodes()), len(network.edges())
        #network_lcc_file = network_file + ".lcc"
        #if not os.path.exists(network_lcc_file):
            #f = open(network_lcc_file, 'w')
            #for u, v in network.edges():
                #f.write("%s 1 %s\n" % (u, v))
            #f.close()
    return network


def calculate_closest_distance(network, nodes_from, nodes_to, lengths=None):
    values_outer = []
    if lengths is None:
        for node_from in nodes_from:
            values = []
            for node_to in nodes_to:
                val = network_utilities.get_shortest_path_length_between(network, node_from, node_to)
                values.append(val)
            d = min(values)
            # print d,
            values_outer.append(d)
    else:
        for node_from in nodes_from:
            values = []
            vals = lengths[node_from]
            for node_to in nodes_to:
                val = vals[node_to]
                values.append(val)
            d = min(values)
            values_outer.append(d)
    d = numpy.mean(values_outer)
    # print d
    return d


def get_random_nodes(nodes, network, bins=None, n_random=1000, min_bin_size=100, degree_aware=True, seed=None):
    if bins is None:
        # Get degree bins of the network
        bins = network_utilities.get_degree_binning(network, min_bin_size)
    nodes_random = network_utilities.pick_random_nodes_matching_selected(network, bins, nodes, n_random, degree_aware, seed=seed)
    return nodes_random

def calculate_proximity(network, nodes_from, nodes_to, nodes_from_random=None, nodes_to_random=None, bins=None, n_random=1000, min_bin_size=100, seed=452456, lengths=None, distance="closest"):
    """
    Calculate proximity from nodes_from to nodes_to
    If degree binning or random nodes are not given, they are generated
    lengths: precalculated shortest path length dictionary
    """
    nodes_network = set(network.nodes())
    nodes_from = set(nodes_from) & nodes_network
    nodes_to = set(nodes_to) & nodes_network

    if len(nodes_from) == 0 or len(nodes_to) == 0:
        return None  # At least one of the node group not in network
    if distance != "closest":
        lengths = network_utilities.get_shortest_path_lengths(network, "temp_n%d_e%d.sif.pcl" % (len(nodes_network), network.number_of_edges()))
        d = network_utilities.get_separation(network, lengths, nodes_from, nodes_to, distance, parameters={})
    else:
        d = calculate_closest_distance(network, nodes_from, nodes_to, lengths)
    if bins is None and (nodes_from_random is None or nodes_to_random is None):
        bins = network_utilities.get_degree_binning(network, min_bin_size, lengths)  # if lengths is given, it will only use those nodes
        #print(bins)
    if nodes_from_random is None:
        nodes_from_random = get_random_nodes(nodes_from, network, bins=bins, n_random=n_random, min_bin_size=min_bin_size, seed=seed)
        #print(nodes_from_random)
    if nodes_to_random is None:
        nodes_to_random = get_random_nodes(nodes_to, network, bins=bins, n_random=n_random, min_bin_size=min_bin_size, seed=seed)
        #print(nodes_to_random)
    random_values_list = zip(nodes_from_random, nodes_to_random)
    values = numpy.empty(len(nodes_from_random))  # n_random
    for i, values_random in enumerate(random_values_list):
        nodes_from, nodes_to = values_random
        if distance != "closest":
            values[i] = network_utilities.get_separation(network, lengths, nodes_from, nodes_to, distance, parameters={})
        else:
            values[i] = calculate_closest_distance(network, nodes_from, nodes_to, lengths)
    # pval = float(sum(values <= d)) / len(values) # needs high number of n_random
    m, s = numpy.mean(values), numpy.std(values)
    if s == 0:
        z = 0.0
    else:
        z = (d - m) / s
    return d, z, (m, s)  # (z, pval)


import pandas as pd

if __name__ == '__main__':

    #file_name = "/Users/yanlixu/Desktop/git_code/disease_drug_proximity/toolbox_simplified/toy.sif"
    #network = get_network(file_name, only_lcc = True)
    #nodes_from = ["A", "C"]
    #nodes_to = ["B", "D", "E"]
    #d, z, (mean, sd) = calculate_proximity(network, nodes_from, nodes_to, min_bin_size = 2, seed=452456)
    #print(d, z, (mean, sd))
    # get network
    file_name = "/Users/yanlixu/Desktop/git_code/drug_repurposing_202206/network_based_2/toolbox_simplified/ppi_chenfeixiong_w_edge.sif.txt"
    network = get_network(file_name, only_lcc = True)
    #print(network)
    # get disease genes
    her2_file = pd.read_excel('/Users/yanlixu/Desktop/pingan/合作/CDK12/network_based/1_her2_breast_cancer_20220620.xlsx', sheet_name='2_gwas_disgenet_genes')
    her2_breast_cancer_genes = [str(gene) for gene in her2_file['To'].tolist()]
    print(her2_breast_cancer_genes)
    # get drug genes
    

    #drug_genes = ['4843', '4846']
    #drug_genes = ['4843', '4846', '113451', '435', '445', '384']
    #d, z, (mean, sd) = calculate_proximity(network, her2_breast_cancer_genes, drug_genes, min_bin_size = 300, seed=452456)
    #print(d, z, (mean, sd))
    
    drug_genes = pd.read_excel('/Users/yanlixu/Desktop/pingan/合作/CDK12/network_based/related_datasets_20220619.xlsx', sheet_name='dti_from_chenfeixiong_compresse')
    for index, row in drug_genes.iterrows():
        drug_gene_list = row['Drug_Target (Gene Eentrez IDs)'].split(',')
        print(drug_gene_list)
        d, z, (mean, sd) = calculate_proximity(network, her2_breast_cancer_genes, drug_gene_list, min_bin_size = 300, seed=452456)
        print(d, z, (mean, sd))
    
    
    
    
    #her2_breast_cancer_genes = 
    #nodes_from = ["A", "C"]
    #nodes_to = ["B", "D", "E"]
    #d, z, (mean, sd) = calculate_proximity(network, nodes_from, nodes_to, min_bin_size = 2, seed=452456)
    #print(d, z, (mean, sd))
    
    
    
    
    
    
