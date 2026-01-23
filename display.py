import matplotlib.pyplot as plt
import networkx as nx

from chrocos import DrawColoredGraph, DrawChroCoS, ChroCoDe, H, Gamma

def displayResultGraph(graph, setCommunauté, entropie, position, g, r):
    plt.figure(figsize=(10, 6))
    plt.subplot(221)
    plt.title("Détection de communauté sur ce graph ")
    DrawColoredGraph(graph, pos=position)
    plt.subplot(222)
    plt.title("Algorithme génétique, entropie = "+str(entropie))
    DrawChroCoS(graph, setCommunauté, theme="pastel", pos=position)
    
    P = ChroCoDe(g, r, radius=1, funenum=Gamma)
    cp = nx.get_node_attributes(graph, "color")
    # Display the result
    plt.subplot(223)
    plt.title(
        "CHROCODE - Hg="
        + "{:.9e}".format(H(P, cp, 4, Gamma))
        
    )
    DrawChroCoS(graph, P, pos=position)
    plt.show()
    
def displayResultMath(minimums, moyennes, iter):
    minimums_moyenne = [val/iter for val in minimums]
    moyennes_moyenne = [val/iter for val in moyennes]

    print("minimus_moyenne = "+str(minimums_moyenne))
    print("moyennes_moyenne = "+str(moyennes_moyenne))
    return minimums_moyenne, moyennes_moyenne
    
def displayResultExp(TraceMinimum, minimums_moyenne, moyennes_moyenne):
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