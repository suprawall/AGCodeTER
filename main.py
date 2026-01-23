import networkx as nx
import matplotlib.pyplot as plt

from math import inf
from AlgorithmeGenetiqueTER import GeneticAlgorithm, getViablegenome, GetTrace, genereCommunauté
from display import displayResultGraph, displayResultMath, displayResultExp
from chrocos import (
    H,
    Kappa,
    Gamma,
    DrawColoredGraph,
    DrawChroCoS,
    RandomColoring,
    GenerateSeeds,
    MonochromeCommunityStructure,
    ChroCoDe
)

ITERATION = 1
POPSIZE = 5000
NB_GENERATION = 35
minimums = None
moyennes = None

for i in range(ITERATION):
    n = 40
    r = 4

    G = nx.connected_watts_strogatz_graph(n, 2, 0.6)

    position = nx.circular_layout(G)
    #gridposition = dict(zip(G, GD))  # define position as label of the initial graph

    seeds = GenerateSeeds(G, r)
    RandomColoring(G, seeds, density=0.3, transparency=0.0)
    colorProfile = nx.get_node_attributes(G, "color")
    nb_noeuds = len(G.nodes())

    graph_quotient = nx.quotient_graph(
                            G, MonochromeCommunityStructure(G)
                            )  # Quotient graph of the monochrome community structure.
    Po = set(graph_quotient.nodes())
    nbmax_communauté = len(Po)

    popsize = POPSIZE
    TABMAXCOMMU = [i for i in range(1, nbmax_communauté + 1)]
    genome = []
    pop0 = []
    progression = popsize / 10

    for i in range(popsize):
        if(len(pop0) > progression):
            print("pop0 : "+str(progression))
            progression += popsize / 10
        pop0.append(getViablegenome(G, TABMAXCOMMU))


    print("nombre de communautés dans le graph quotient: "+str(nbmax_communauté))

    iter_mutate_rate = 0.05
    iter_tournament = 2
    iter_maxbound = NB_GENERATION

    communautés_final = GeneticAlgorithm(pop0, TABMAXCOMMU,
                                            colorProfile,
                                            G,
                                            r,
                                            True,
                                            maxbound=iter_maxbound, 
                                            darwinianrate=1.0, crossrate=0.8, 
                                            mutaterate=iter_mutate_rate, tournament=iter_tournament,
                                            )
    print("meilleur génome avec iter_maxbound = "+str(iter_maxbound)+" : ")
    print(communautés_final)
        
    TraceMinimum = GetTrace()  
    if(minimums is None):
        minimums = [tup[0] for tup in TraceMinimum]
        moyennes = [tup[1] for tup in TraceMinimum]
    else:
        for i,tup in enumerate(TraceMinimum):
            if(minimums[i] != inf):
                minimums[i] = minimums[i] + tup[0]
            if(moyennes[i] != inf):
                moyennes[i] = moyennes[i] + tup[1]

min_m, moy_m = displayResultMath(minimums, moyennes, iter=ITERATION)

displayResultExp(TraceMinimum, min_m, moy_m)
displayResultGraph(G, genereCommunauté(communautés_final[0]), communautés_final[1], position)