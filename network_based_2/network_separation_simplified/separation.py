#! /usr/bin/env python

# revised by Yanli Xu on June 28th, 2022
import pandas as pd

"""
# -----------------------------------------------------------------------
#
# seperation.py
#
# by Joerg Menche
# Last Modified: 2014-12-06
#
# This code determines the network-based distance and sepration for
# two given sets of nodes on given network as described in 
# 
# Uncovering Disease-Disease Relationships Through The Human
# Interactome
#
# by Joerg Menche, Amitabh Sharma, Maksim Kitsak, Susan Dina
#    Ghiassian, Marc Vidal, Joseph Loscalzo & Albert-Laszlo Barabasi
# 
# J. Menche et al., Science 347, 1257601 (2015). DOI: 10.1126/science.1257601
# -----------------------------------------------------------------------
# 
# 
# This program will calculate the network-based distance d_AB and
# separation s_AB between two gene sets A and B.
# 
# * Required input:
# 
#   two files containing the gene sets A and B. The file must be in
#   form of a table, one gene per line. If the table contains several
#   columns, they must be tab-separated, only the first column will be
#   used. See the two files MS.txt and PD.txt for valid examples (they
#   contain genes for multiple sclerosis and peroxisomal disorders,
#   respectively).
# 
# * Optional input:  
# 
#   - file containing an interaction network. If now file is given, the
#     default network \"interactome.tsv\" will be used instead. The file
#     must contain an edgelist provided as a tab-separated table. The
#     first two columns of the table will be interpreted as an
#     interaction gene1 <==> gene2
# 
#  - filename for the output. If none is given,
#    \"separation_results.txt\" will be used
#  
# 
# Here's an example that should work, provided the files are in the same
# directory as this python script:
# 
# ./separation.py -n interactome.tsv --g1 MS.txt --g2 PD.txt -o output.txt
# 
#
# -----------------------------------------------------------------------
"""

import networkx as nx
import numpy as np
import optparse
import sys

"""
# =============================================================================

           S T A R T   D E F I N I T I O N S 

# =============================================================================
"""


# =============================================================================
def print_usage(option, opt, value, parser):
    usage_message = """

# ----------------------------------------------------------------------

This program will calculate the network-based distance d_AB and
separation s_AB between two gene sets A and B.

* Required input:

  two files containing the gene sets A and B. The file must be in form
  of a table, one gene per line. If the table contains several
  columns, they must be tab-separated, only the first column will be
  used. See the two files MS.txt and PD.txt for valid examples (they
  contain genes for multiple sclerosis and peroxisomal disorders,
  respectively).

* Optional input:  

  - file containing an interaction network. If now file is given, the
    default network \"interactome.tsv\" will be used instead. The file
    must contain an edgelist provided as a tab-separated table. The
    first two columns of the table will be interpreted as an
    interaction gene1 <==> gene2

 - filename for the output. If none is given,
   \"separation_results.txt\" will be used
 

Here's an example that should work, provided the files are in the same
directory as this python script:

./separation.py -n interactome.tsv --g1 MS.txt --g2 PD.txt -o output.txt

# ----------------------------------------------------------------------

    """

    print(usage_message)

    sys.exit()


# =============================================================================
def read_network(network_file):
    """
    Reads a network from an external file.

    * The edgelist must be provided as a tab-separated table. The
    first two columns of the table will be interpreted as an
    interaction gene1 <==> gene2

    * Lines that start with '#' will be ignored
    """

    G = nx.Graph()
    for line in open(network_file, 'r'):
        # lines starting with '#' will be ignored
        if line[0] == '#':
            continue
        # The first two columns in the line will be interpreted as an
        # interaction gene1 <=> gene2
        line_data = line.strip().split('\t')
        node1 = line_data[0]
        node2 = line_data[1]
        G.add_edge(node1, node2)

    print("\n> done loading network:")
    print("> network contains %s nodes and %s links" % (G.number_of_nodes(), G.number_of_edges()))
    return G


# =============================================================================
def read_gene_list(gene_file):
    """
    Reads a list genes from an external file.

    * The genes must be provided as a table. If the table has more
    than one column, they must be tab-separated. The first column will
    be used only.

    * Lines that start with '#' will be ignored
    """

    genes_set = set()
    for line in open(gene_file, 'r'):
        # lines starting with '#' will be ignored
        if line[0] == '#':
            continue
        # the first column in the line will be interpreted as a seed
        # gene:
        line_data = line.strip().split('\t')
        gene = line_data[0]
        genes_set.add(gene)

    print("\n> done reading genes:")
    print("> %s genes found in %s" % (len(genes_set), gene_file))
    return genes_set


# =============================================================================
def remove_self_links(G):
    #sl = G.selfloop_edges()
    sl = nx.selfloop_edges(G)
    G.remove_edges_from(sl)


# =============================================================================
def get_pathlengths_for_single_set(G, given_gene_set):
    """
    calculate the shortest paths of a given set of genes in a
    given network. The results are stored in a dictionary of
    dictionaries:
    all_path_lenghts[gene1][gene2] = l
    with gene1 < gene2, so each pair is stored only once!

    PARAMETERS:
    -----------
        - G: network
        - gene_set: gene set for which paths should be computed

    RETURNS:
    --------
        - all_path_lenghts[gene1][gene2] = l for all pairs of genes
          with gene1 < gene2

    """

    # remove all nodes that are not in the network
    all_genes_in_network = set(G.nodes())
    gene_set = given_gene_set & all_genes_in_network

    all_path_lenghts = {}

    # calculate the distance of all possible pairs
    for gene1 in gene_set:
        #if not all_path_lenghts.has_key(gene1):
        if gene1 not in all_path_lenghts:
            all_path_lenghts[gene1] = {}
        for gene2 in gene_set:
            if gene1 < gene2:
                try:
                    l = nx.shortest_path_length(G, source=gene1, target=gene2)
                    all_path_lenghts[gene1][gene2] = l
                except:
                    continue
    return all_path_lenghts


# =============================================================================
def get_pathlengths_for_two_sets(G, given_gene_set1, given_gene_set2):
    """
    calculate the shortest paths between two given set of genes in a
    given network. The results are stored in a dictionary of
    dictionaries: all_path_lenghts[gene1][gene2] = l with gene1 <
    gene2, so each pair is stored only once!

    PARAMETERS:
    -----------
        - G: network
        - gene_set1/2: gene sets for which paths should be computed

    RETURNS:
    --------
        - all_path_lenghts[gene1][gene2] = l for all pairs of genes
          with gene1 < gene2

    """

    # remove all nodes that are not in the network
    all_genes_in_network = set(G.nodes())
    gene_set1 = given_gene_set1 & all_genes_in_network
    gene_set2 = given_gene_set2 & all_genes_in_network

    all_path_lenghts = {}

    # calculate the distance of all possible pairs
    for gene1 in gene_set1:
        #if not all_path_lenghts.has_key(gene1):
        if gene1 not in all_path_lenghts:        
            all_path_lenghts[gene1] = {}
        for gene2 in gene_set2:
            if gene1 != gene2:
                try:
                    l = nx.shortest_path_length(G, source=gene1, target=gene2)
                    if gene1 < gene2:
                        all_path_lenghts[gene1][gene2] = l
                    else:
                        #if not all_path_lenghts.has_key(gene2):
                        if gene2 not in all_path_lenghts:
                            all_path_lenghts[gene2] = {}
                        all_path_lenghts[gene2][gene1] = l
                except:
                    continue

    return all_path_lenghts


# =============================================================================
def calc_single_set_distance(G, given_gene_set):
    """
    Calculates the mean shortest distance for a set of genes on a
    given network    
    

    PARAMETERS:
    -----------
        - G: network
        - gene_set: gene set for which distance will be computed 

    RETURNS:
    --------
         - mean shortest distance 

    """

    # remove all nodes that are not in the network, just to be safe
    all_genes_in_network = set(G.nodes())
    gene_set = given_gene_set & all_genes_in_network

    # get the network distances for all gene pairs:
    all_path_lenghts = get_pathlengths_for_single_set(G, gene_set)
    


    all_distances = []

    # going through all gene pairs
    for geneA in gene_set:

        all_distances_A = []
        for geneB in gene_set:

            # I have to check which gene is 'smaller' in order to know
            # where to look up the distance of that pair in the
            # all_path_lengths dict
            if geneA < geneB:
                #if all_path_lenghts[geneA].has_key(geneB):
                if geneB in all_path_lenghts[geneA]:                
                    all_distances_A.append(all_path_lenghts[geneA][geneB])
            else:
                #if all_path_lenghts[geneB].has_key(geneA):
                if geneA in all_path_lenghts[geneB]:
                    all_distances_A.append(all_path_lenghts[geneB][geneA])

        if len(all_distances_A) > 0:
            l_min = min(all_distances_A)
            all_distances.append(l_min)

    # calculate mean shortest distance
    mean_shortest_distance = np.mean(all_distances)

    return mean_shortest_distance


# =============================================================================
def calc_set_pair_distances(G, given_gene_set1, given_gene_set2):
    """
    Calculates the mean shortest distance between two sets of genes on
    a given network
    
    PARAMETERS:
    -----------
        - G: network
        - gene_set1/2: gene sets for which distance will be computed 

    RETURNS:
    --------
         - mean shortest distance 

    """

    # remove all nodes that are not in the network
    all_genes_in_network = set(G.nodes())
    gene_set1 = given_gene_set1 & all_genes_in_network
    gene_set2 = given_gene_set2 & all_genes_in_network

    # get the network distances for all gene pairs:
    all_path_lenghts = get_pathlengths_for_two_sets(G, gene_set1, gene_set2)

    all_distances = []

    # going through all pairs starting from set 1 
    for geneA in gene_set1:

        all_distances_A = []
        for geneB in gene_set2:

            # the genes are the same, so their distance is 0
            if geneA == geneB:
                all_distances_A.append(0)

            # I have to check which gene is 'smaller' in order to know
            # where to look up the distance of that pair in the
            # all_path_lengths dict
            else:
                if geneA < geneB:
                    try:
                        all_distances_A.append(all_path_lenghts[geneA][geneB])
                    except:
                        pass

                else:
                    try:
                        all_distances_A.append(all_path_lenghts[geneB][geneA])
                    except:
                        pass

        if len(all_distances_A) > 0:
            l_min = min(all_distances_A)
            all_distances.append(l_min)

    # going through all pairs starting from disease B
    for geneA in gene_set2:

        all_distances_A = []
        for geneB in gene_set1:

            # the genes are the same, so their distance is 0
            if geneA == geneB:
                all_distances_A.append(0)

            # I have to check which gene is 'smaller' in order to know
            # where to look up the distance of that pair in the
            # all_path_lengths dict
            else:
                if geneA < geneB:
                    try:
                        all_distances_A.append(all_path_lenghts[geneA][geneB])
                    except:
                        pass

                else:
                    try:
                        all_distances_A.append(all_path_lenghts[geneB][geneA])
                    except:
                        pass

        if len(all_distances_A) > 0:
            l_min = min(all_distances_A)
            all_distances.append(l_min)

    # calculate mean shortest distance
    mean_shortest_distance = np.mean(all_distances)

    return mean_shortest_distance

def calculate_distance_bw_2gene_sets(G, all_genes_in_network, glist1, glist2):
    # read gene set 1
    genes_A = set(glist1) & all_genes_in_network

    # read gene set 1
    genes_B = set(glist2) & all_genes_in_network

    # distances WITHIN the two gene sets:
    #d_A = calc_single_set_distance(G, genes_A)
    #d_B = calc_single_set_distance(G, genes_B)
    
    if len(genes_A) == 1:
        d_A = 0
    else:
        d_A = calc_single_set_distance(G, genes_A)
    if len(genes_B) == 1:
        d_B = 0
    else:
        d_B = calc_single_set_distance(G, genes_B)
    

    # distances BETWEEN the two gene sets:
    d_AB = calc_set_pair_distances(G, genes_A, genes_B)
    #print(d_A)
    #print(d_B)
    #print(d_AB)
    # calculate separation
    s_AB = d_AB - (d_A + d_B) / 2.
    
    return [d_A, d_B, d_AB, s_AB]


	
	

"""
# =============================================================================

           E N D    O F    D E F I N I T I O N S 

# =============================================================================
"""


if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option('-u', '--usage', help='print more info on how to use this script', action="callback", callback=print_usage)

    parser.add_option('-n', help='file containing the network edgelist [ppi_chenfeixiong.tsv]', dest='network_file', default='ppi_chenfeixiong.tsv',type="string")

    #parser.add_option('--g1',help='file containing gene set 1', dest='gene_file_1', default='none', type="string")

    #parser.add_option('--g2', help='file containing gene set 2', dest='gene_file_2', default='none', type="string")
    parser.add_option('-g', help='drug gene file', dest='drug_gene_file', default='none', type="string")

    parser.add_option('-o', help='file for results [separation_results.txt]', dest='results_file', default='separation_results.txt', type="string")

    (opts, args) = parser.parse_args()

    network_file = opts.network_file
    #gene_file_1 = opts.gene_file_1
    #gene_file_2 = opts.gene_file_2
    drug_gene_file = opts.drug_gene_file
    results_file = opts.results_file

    # checking for input:
    if drug_gene_file == 'none':
        error_message = """
        ERROR: you must specify drug target file, for example:
        ./separation.py -t drug_target_file

        For more information, type
        ./separation.py --usage
        
        """
        print(error_message)
        sys.exit(0)

    if network_file == 'ppi_chenfeixiong.tsv':
        print('> default network from "ppi_chenfeixiong.tsv" will be used')


    # --------------------------------------------------------
    #
    # LOADING NETWORK and DISEASE GENES
    #
    # --------------------------------------------------------

    # read network
    G = read_network(network_file)
    # get all genes ad remove self links
    all_genes_in_network = set(G.nodes())
    remove_self_links(G)

    
    # read data from excel
    drug_genes = pd.read_excel('/Users/yanlixu/Desktop/pingan/合作/CDK12/network_based/related_datasets_20220619.xlsx', sheet_name='dti_from_chenfeixiong_compresse')
    # convert genes column to list
    drug_genes['Drug_Target (Gene Eentrez IDs)'] = drug_genes['Drug_Target (Gene Eentrez IDs)'].apply(lambda x: x[1:-1].split(','))
    # dataframe to dictionary
    drug_genes.set_index('DrugID (DrugBank ID)', inplace=True)
    drug_gene_dic = drug_genes.to_dict()['Drug_Target (Gene Eentrez IDs)']
    
    #print(drug_gene_dic)
    
    # read gene set 1
    #genes_A = set(['3067', '3034', '3035']) & all_genes_in_network
    # read gene set 1
    #genes_B = set(['2162', '6744']) & all_genes_in_network    
    #result = calculate_distance_bw_2gene_sets(G, genes_A, genes_B)
    #print(result)
    fout = open('drug_genes_separation.txt', 'w')
    fout.write('drug_a' + '\t' + 'drug_a_genes' + '\t' + 'drug_b' + '\t' + 'drug_b_genes' + '\t' + 'dA' + '\t' + 'dB' + '\t' + 'dAB' + '\t' + 'sAB' + '\n')
    for drug_a in drug_gene_dic.keys():
        drug_a_genes = drug_gene_dic[drug_a]
        for drug_b in drug_gene_dic.keys():
            drug_b_genes = drug_gene_dic[drug_b]
            result = calculate_distance_bw_2gene_sets(G, all_genes_in_network, drug_a_genes, drug_b_genes)
            fout.write(drug_a + '\t' + '||'.join(drug_gene_dic[drug_a]) + '\t' + drug_b + '\t' + '||'.join(drug_gene_dic[drug_b])+ '\t' + str(result[0]) + '\t' + str(result[1]) + '\t' + str(result[2]) + '\t' + str(result[3]) + '\n')
    fout.close()
            
    
    
    

    









'''

if __name__ == '__main__':

    # "Hey Ho, Let's go!" -- The Ramones (1976)

    # --------------------------------------------------------
    # 
    # PARSING THE COMMAND LINE
    # 
    # --------------------------------------------------------

    parser = optparse.OptionParser()

    parser.add_option('-u', '--usage', help='print more info on how to use this script', action="callback", callback=print_usage)

    parser.add_option('-n', help='file containing the network edgelist [interactome.tsv]', dest='network_file', default='interactome.tsv',type="string")

    parser.add_option('--g1',help='file containing gene set 1', dest='gene_file_1', default='none', type="string")

    parser.add_option('--g2', help='file containing gene set 2', dest='gene_file_2', default='none', type="string")

    parser.add_option('-o', help='file for results [separation_results.txt]', dest='results_file', default='separation_results.txt', type="string")

    (opts, args) = parser.parse_args()

    network_file = opts.network_file
    gene_file_1 = opts.gene_file_1
    gene_file_2 = opts.gene_file_2
    results_file = opts.results_file

    # checking for input:
    if gene_file_1 == 'none' or gene_file_2 == 'none':
        error_message = """
        ERROR: you must specify two files with gene sets, for example:
        ./separation.py --g1 MS.txt --g2 PD.txt

        For more information, type
        ./separation.py --usage
        
        """
        print(error_message)
        sys.exit(0)

    if network_file == 'interactome.tsv':
        print('> default network from "interactome.tsv" will be used')


    # --------------------------------------------------------
    #
    # LOADING NETWORK and DISEASE GENES
    #
    # --------------------------------------------------------

    # read network
    G = read_network(network_file)
    # get all genes ad remove self links
    all_genes_in_network = set(G.nodes())
    remove_self_links(G)

    # read gene set 1
    genes_A_full = read_gene_list(gene_file_1)
    # removing genes that are not in the network:
    genes_A = genes_A_full & all_genes_in_network
    if len(genes_A_full) != len(genes_A):
        print("> ignoring %s genes that are not in the network" % (len(genes_A_full - all_genes_in_network)))
        print("> remaining number of genes: %s" % (len(genes_A)))


    # read gene set 1
    genes_B_full = read_gene_list(gene_file_2)
    # removing genes that are not in the network:
    genes_B = genes_B_full & all_genes_in_network
    if len(genes_B_full) != len(genes_B):
        print("> ignoring %s genes that are not in the network" % (len(genes_B_full - all_genes_in_network)))
        print("> remaining number of genes: %s" % (len(genes_B)))


    # --------------------------------------------------------
    #
    # CALCULATE NETWORK QUANTITIES
    #
    # --------------------------------------------------------

    # distances WITHIN the two gene sets:
    d_A = calc_single_set_distance(G, genes_A)
    d_B = calc_single_set_distance(G, genes_B)

    # distances BETWEEN the two gene sets:
    d_AB = calc_set_pair_distances(G, genes_A, genes_B)
    #print(d_A)
    #print(d_B)
    #print(d_AB)
    # calculate separation
    s_AB = d_AB - (d_A + d_B) / 2.

    # print and save results:

    results_message = """
> gene set A from \"%s\": %s genes, network-diameter d_A = %s
> gene set B from \"%s\": %s genes, network-diameter d_B = %s
> mean shortest distance between A & B: d_AB = %s 
> network separation of A & B:          s_AB = %s
""" % (gene_file_1, len(genes_A), d_A,
       gene_file_2, len(genes_B), d_B,
       d_AB, s_AB)

    print(results_message)

    fp = open(results_file, 'w')
    fp.write(results_message)
    fp.close()

    print("> results have been saved to %s" % (results_file))
'''
