# AUTHOR: Hesters Valentin
# DATE: 01/05/2024


from random import randint, choice, sample
from math import floor
from typing import TypeAlias, Callable
import numpy as np
from math import factorial, comb, ceil, exp, inf, log2
from typing import TypeAlias, Callable
from random import choices, random
from collections import Counter
from functools import reduce
from scipy.stats import gmean  # type: ignore
import random

import networkx as nx
import seaborn as sns  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from cdlib import NodeClustering, evaluation
from collections import defaultdict
from itertools import groupby
from ga import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import networkx as nx
import pandas as pd


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

from exampleOriginal import (
    graphexample
)

# fitness and viability function type
fitness_t: TypeAlias = Callable[[list], float]
viability_t: TypeAlias = Callable[[list], bool]

def union_list(liste1, liste2):
    return list(set(liste1) | set(liste2))

def getCommuPondéré(genome, pd):
    roue = {commu: genome.count(commu) for commu in genome}
    roue_trie = {k: v for k, v in sorted(roue.items(), key=lambda item: item[1], reverse=True)}
    proba = []
    for i, (_,v) in enumerate(roue_trie.items()):
        if(i == 0):
            proba.append(v * pd)
        else:
            proba.append(v)  
    total = sum(proba)
    proba = [p/total for p in proba]
    choix = random.random()
    somme_proba = 0
    for i, p in enumerate(proba):
        somme_proba += p
        if(choix <= somme_proba):
            return list(roue_trie.keys())[i]
    
def getCommuByEntropieSelection(genome, P, colorProfil, r):
    
    best = None
    min = None
    for p in P:
        h = H({p}, colorProfil, r, Gamma)
        if(min == None or (h != inf and h < min)):
            min = h
            best = p
    best_list = list(best)
    return genome[best_list[0]]
    

def Crossover(graph, g1: list, g2: list, pondération: bool, entropieSelection: bool, colorProfile: any, r: int) -> tuple:
    
    if(entropieSelection):
        c1 = getCommuByEntropieSelection(g1, genereCommunauté(g1), colorProfile, r)
        c2 = getCommuByEntropieSelection(g2, genereCommunauté(g2), colorProfile, r)
    else:
        if(pondération):
            c1 = getCommuPondéré(g1, 8)
            c2 = getCommuPondéré(g2, 8)
        else:  
            c1 = max(set(g1), key=g1.count)
            c2 = max(set(g2), key=g1.count)
        
    e1 = [-1 for _ in range(len(g1))]
    e2 = [-1 for _ in range(len(g1))]
    
    for i, elm in enumerate(g1):        
        if(elm == c1):
            e1[i] = c1
    for i, elm in enumerate(g2):        
        if(elm != c1):
            e1[i] = elm
    while(-1 in e1):                   
        for i, elm in enumerate(e1):
            if(elm == -1):
                voisins = graph.neighbors(i)
                commu_des_voisins = []
                for v in voisins:
                    if(not (e1[v] == -1)):
                        commu_des_voisins.append(e1[v])
                if(commu_des_voisins):
                    choix = rd.choice(commu_des_voisins)
                    e1[i] = choix
    
    for i, elm in enumerate(g2):       
        if(elm == c2):
            e2[i] = c2
    for i, elm in enumerate(g1):       
        if(elm != c2):
            e2[i] = elm
    it = 0
    while(-1 in e2): 
        it += 1
        if(it > 1000):
            break                 
        for i, elm in enumerate(e2):
            if(elm == -1):
                voisins = graph.neighbors(i)
                commu_des_voisins = []
                for v in voisins:
                    if(not (e2[v] == -1)):
                        commu_des_voisins.append(e2[v])
                if(commu_des_voisins):
                    choix = rd.choice(commu_des_voisins)
                    e2[i] = choix
    return(e1, e2)


def Mutate(graph, g: list, values: list | tuple) -> list:
    """Perform a mutation on 1 element of the genome

    Args:
        g (list): digital genome
        valuess (list or tuple): values

    Returns:
        list: a mutated genome
    """
    nodes = graph.nodes()

    n = len(g)
    mutate = randint(0, n - 1)
    voisins = list(graph.neighbors(mutate))
    
    for v in voisins:
        """print("le noeud : "+str(mutate)+", nodes[mutate] : "+str(nodes[mutate]))
        print("le voisin : "+str(v)+" a une couleur : "+str(nodes[v]))"""
        if(nodes[v] == nodes[mutate]):
            g[mutate] = g[v]
            return g
    choix = rd.choice(voisins)
    g[mutate] = g[choix]
    return g


def Tournament(pop: list, colorProfile:any, r: int, k: int = 2) -> list:
    """Select a genome by tournament

    Args:
        pop (list): population
        f (_type_): fitness function
        k (int, optional): number of genomes in a tournament Defaults to 2.

    Returns:
        list: _description_
    """
    tournament = sample(range(len(pop)), k=k)
    tournament_fitnesses = [H(genereCommunauté(pop[i]), colorProfile, r, Gamma) for i in tournament]
    if all(score == inf for score in tournament_fitnesses):
        #print("tous les génomes ont une entropie inf")
        return pop[np.random.choice(tournament)]
    else:
        valid_fitnesses = [fitness for fitness in tournament_fitnesses if fitness > 0 and fitness != inf]
        min_score = min(valid_fitnesses)
        #print("le gagnant est :"+str(pop[tournament[tournament_fitnesses.index(min_score)]])+" il a une entropie de: "+str(min_score))
        return pop[tournament[tournament_fitnesses.index(min_score)]]

# ! Main function ================================================================
def GeneticAlgorithm(
    pop0: list[list],
    values: list | tuple,
    colorProfile: any,
    graph,
    r: int,
    selectionByEntropie: bool,
    viable: viability_t = (lambda p: True),
    maxbound: int = 100,
    darwinianrate: float = 1.0,
    crossrate: float = 0.8,
    mutaterate: float = 0.02,
    tournament: int = 2,
) -> list:
    """Genetic Algorithm

    Args:
        pop0 (list): initial population
        values (list|tuple): values taken by the elements of the genome
        f (_type_): fitness function
        colorProfile(any): color profile of the gaph
        graph
        r(int): numbers of colors
        selectionByEntropie: wether or not the selection in crossover must be by entropie or by numbers of nodes in a community
        viability (tuple, optional): viability function. Defaults to (lambda p:True).
        maxbound (int): maximal number of iteration. Defaults to 1.0
        darwinianrate (float, optional):  rate of Darwinian selection. Defaults to 1.0.
        crossrate (float, optional): rate of  Cross over Defaults to 0.8. Defaults to 0.8.
        mutaterate (float, optional):  rate of mutation. Defaults to 0.02.
        tournament (int, optional): number of participants of a tournament.. Defaults to 2.

    Returns:
        list: best solution
    """
    assert 0 <= crossrate <= 1
    assert 0 <= mutaterate <= 1
    assert 0 <= darwinianrate <= 1
    assert 0 < maxbound
    assert 1 < tournament
    assert 1 < len(values)

    global GATRACE  # Trace of execution
    GATRACE = []

    pop = pop0
    for i in range(maxbound):
        viablepop = list(filter(viable, pop))  # select viable  genomes
        #GATRACE.append(viablepop)

        longueur = len(viablepop)
        somme = 0
        min = None  # Select the genome with the best fitness and always include it in the new population
        best = None
        for genome in viablepop:
            score = H(genereCommunauté(genome), colorProfile, r, Gamma)
            if(score == inf):
                somme += 0
                longueur -= 1
            else:
                somme += score
            if min is None or min > score:
                min = score
                best = genome
        newpop = [best]
        if(longueur == 0):
            moyenne = inf
            GATRACE.append((min, moyenne))
        else:
            moyenne = somme/longueur
            GATRACE.append((min, somme/longueur))
        print("meilleur génome de la génération "+str(i)+" : "+str(best)+", "+str(min)+", moyenne de cette génération : "+str(moyenne))

        nb = floor(len(viablepop) * darwinianrate) - 1  # Darwinian selection
        for _ in range(nb):
            newpop.append(Tournament(viablepop, colorProfile, r, tournament))

        n = len(newpop)

        nb = floor(n * crossrate)  # cross over
        for _ in range(nb):
            i1 = randint(0, n - 1)
            i2 = randint(0, n - 1)
            (newpop[i1], newpop[i2]) = Crossover(graph, newpop[i1], newpop[i2], pondération=True, entropieSelection=selectionByEntropie, colorProfile=colorProfile, r=r)
        
        nb = floor(n * mutaterate)
        for _ in range(nb):
            i1 = randint(0, n - 1)
            newpop[i1] = Mutate(graph, newpop[i1], values)
        
        if(not best in newpop):
            newpop.append(best)

        pop = newpop  # the offsprings become the parents
    # ! end of main loop

    min = None  # final selection of the best genome
    solution = None
    for genome in pop:
        score = H(genereCommunauté(genome), colorProfile, r, Gamma)
        if score == inf:
            continue
        elif min is None or min > score:
            min = score
            solution = genome

    return (solution, min)


def GetTrace() -> list:
    return GATRACE

def genereCommunauté(genome):
    result = pd.Series(range(len(genome))).groupby(genome, sort=False).apply(list).tolist()

    result_set = {frozenset(sublist) for sublist in result}
    return result_set


def Isviable(graph, genome): 
    Ensemble = []
    g = genome.copy()
    
    for i, communauté in enumerate(g):
        if(communauté == -1):
            continue
        if( any(i in e for e in Ensemble) ):
            continue
        frontiere = []
        explorer = []
        frontiere.append(i)
        while(frontiere):
            n = frontiere.pop(0)
            explorer.append(n)
            voisins = list(graph.neighbors(n))
            for v in voisins:
                if( (genome[v] == genome[n]) and (v not in explorer) and (v not in frontiere)):
                    frontiere.append(v)
        Ensemble.append(explorer)
    
    for ens1 in Ensemble:
        for ens2 in Ensemble:
            if( (ens1 != ens2) and (genome[ens1[0]] == genome[ens2[0]]) ):
                return False
    return True

def IsviableSingle(graph, genome, noeud, communauté_du_noeud):
    voisins = list(graph.neighbors(noeud))
    for j, commu in enumerate(genome):
        if( (commu != -1) and (commu == communauté_du_noeud) and (j not in voisins) and (noeud != j) ):
            return False
    return True

def getViablegenome(graph, tabMaxCommu):
    viableGenome = [-1 for _ in range(len(graph.nodes()))]
    longueur = len(viableGenome)
    start = randint(0, longueur - 1)
    explorer = []
    trace = []
    
    largeur = list(graph.neighbors(start))
    viableGenome[start] = rd.choice(tabMaxCommu)
    explorer.append(start)
    while(largeur):
        trace.append(largeur.copy())
        noeud = largeur.pop(0)
        copy = tabMaxCommu.copy()
        voisins_de_noeud = list(graph.neighbors(noeud))
        while(True):
            if not copy:
                v = voisins_de_noeud.copy()
                while(True):
                    ch = rd.choice(v)
                    c = viableGenome[ch]
                    if(c == -1):
                        v.remove(ch)
                    else:     
                        viableGenome[noeud] = c
                        break
            if not copy:
                break
            choix = rd.choice(copy)
            viableGenome[noeud] = choix
            if(IsviableSingle(graph, viableGenome, noeud, choix)):
                break
            else:
                copy.remove(choix)
        explorer.append(noeud)
        for v in voisins_de_noeud:
            if v not in explorer and v not in largeur:
                largeur.append(v) 
    
    return viableGenome
    
def displayResult(graph, setCommunauté, entropie, position):
    plt.figure(figsize=(10, 6))
    plt.subplot(221)
    plt.title("Détection de communauté sur ce graph ")
    DrawColoredGraph(graph, pos=position)
    plt.subplot(222)
    plt.title("Algorithme génétique, entropie = "+str(entropie))
    DrawChroCoS(graph, setCommunauté, theme="pastel", pos=position)
    
    P = ChroCoDe(G, r, radius=1, funenum=Gamma)
    cp = nx.get_node_attributes(G, "color")
    # Display the result
    plt.subplot(223)
    plt.title(
        "CHROCODE - Hg="
        + "{:.9e}".format(H(P, cp, 4, Gamma))
        
    )
    DrawChroCoS(G, P, pos=position)
    plt.show()
    
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
            
minimums_moyenne = [val/ITERATION for val in minimums]
moyennes_moyenne = [val/ITERATION for val in moyennes]

print("minimus_moyenne = "+str(minimums_moyenne))
print("moyennes_moyenne = "+str(moyennes_moyenne))

indices = list(range(1, len(TraceMinimum) + 1))  

plt.figure(figsize=(10, 5))  

plt.plot(indices, minimums_moyenne, marker='o', color='b', label='Minimums')
plt.plot(indices, moyennes_moyenne, marker='s', color='r', label='Moyennes')


plt.xlabel('Génération')
plt.ylabel('Valeur')
plt.title('Évolution des minimums, moyennes')
plt.legend()
plt.grid(True)
plt.show()

#displayResult(G, genereCommunauté(communautés_final[0]), communautés_final[1], position)









