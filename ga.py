# * * Python GENETIC ALGORITHM
# * * Author Franck Delaplace
# * * TUTORIAL OF MASTER
# * * Paris Saclay Unversity

from random import randint, choice, sample
from math import floor
from typing import TypeAlias, Callable
import numpy as np

# fitness and viability function type
fitness_t: TypeAlias = Callable[[list], float]
viability_t: TypeAlias = Callable[[list], bool]


# ! Basic functions =========================================================
def CrossOver(g1: list, g2: list) -> tuple:
    """Perform a cross over between g1 and g2

    Args:
        g1 (list): digital genome
        g2 (list): digital genome

    Returns:
        tuple: a pair of genomes.
    """
    n = min(len(g1), len(g2))
    split = randint(1, n - 1)
    return (g1[:split] + g2[split:], g2[:split] + g1[split:])


def Mutate(g: list, values: list | tuple) -> list:
    """Perform a mutation on 1 element of the genome

    Args:
        g (list): digital genome
        valuess (list or tuple): values

    Returns:
        list: a mutated genome
    """

    n = len(g)
    mutate = randint(0, n - 1)
    reducedvalues=values.copy()
    reducedvalues.remove(g[mutate])
    g[mutate] = choice(reducedvalues)
    return g


def Tournament(pop: list, f: fitness_t, k: int = 2) -> list:
    """Select a genome by tournament

    Args:
        pop (list): population
        f (_type_): fitness function
        k (int, optional): number of genomes in a tournament Defaults to 2.

    Returns:
        list: _description_
    """
    tournament = sample(range(len(pop)), k=k)
    tournament_fitnesses = [f(pop[i]) for i in tournament]
    winner_index = tournament[np.argmax(tournament_fitnesses)]
    return pop[winner_index]


# ! Main function ================================================================
def GeneticAlgorithm(
    pop0: list[list],
    values: list | tuple,
    f: fitness_t,
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
    for _ in range(maxbound):
        viablepop = list(filter(viable, pop))  # select viable  genomes

        GATRACE.append(viablepop)

        maxi = None  # Select the genome with the best fitness and always include it in the new population
        best = None
        for p in viablepop:
            if maxi is None or maxi < f(p):
                maxi = f(p)
                best = p
        newpop = [best]

        nb = floor(len(viablepop) * darwinianrate) - 1  # Darwinian selection
        for _ in range(nb):
            newpop.append(Tournament(viablepop, f, tournament))

        n = len(newpop)

        nb = floor(n * crossrate)  # cross over
        for _ in range(nb):
            i1 = randint(0, n - 1)
            i2 = randint(0, n - 1)
            (newpop[i1], newpop[i2]) = CrossOver(newpop[i1], newpop[i2])
        
        nb = floor(n * mutaterate)  # mutation
        for _ in range(nb):
            i1 = randint(0, n - 1)
            newpop[i1] = Mutate(newpop[i1], values)

        pop = newpop  # the offsprings become the parents
    # ! end of main loop

    GATRACE.append(
        list(filter(viable, pop))
    )  # keep the last viable population in the trace.

    maxi = None  # final selection of the best genome
    solution = None
    for p in pop:
        if maxi is None or maxi < f(p):
            maxi = f(p)
            solution = p

    return solution


def GetTrace() -> list:
    """Get the last trace of Genetic Algorithm run.

    Returns:
        list : a trace = list of populations
    """
    return GATRACE
