import random

import matplotlib.pyplot as plt

board_size = 8
population_size = 100
mutation_probability = 0.8
epochs = 1000


def initialize():
    """
    Generate random samples
    Representation: permutation
    """
    pop = list()
    for _ in range(population_size):
        x = [i for i in range(board_size)]
        random.shuffle(x)
        pop.append(x)
    return pop


def fitness(x):
    """
    Inverse sum of penalties
    Penalty: number of other queens a queen can check
    """
    f = 0
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if i != j and abs(i - j) == abs(xi - xj):
                f += 1
    return -1 * f


def select_parents(population):
    """
    Pick 2 best out of 5 random samples from the population
    Tournament-Based Selection
    """
    pop = random.choices(population, k=5)
    pop.sort(key=fitness)
    return pop[-1], pop[-2]


def recombination(parents):
    """
    Cut-and-Crossfill crossover with 100% probability
    """
    cross_point = random.randint(0, board_size + 1)
    p1, p2 = parents
    c1 = p1[:cross_point]
    c2 = p2[:cross_point]
    for i in range(board_size):
        p1i = p1[(cross_point + i) % board_size]
        p2i = p2[(cross_point + i) % board_size]
        if p1i not in p2[:cross_point]:
            c2.append(p1i)
        if p2i not in p1[:cross_point]:
            c1.append(p2i)
    return c1, c2


def mutation(offsprings):
    """
    Swapping values of two randomly chosen positions
    """
    c1, c2 = offsprings

    def m(c):
        if random.random() > mutation_probability:
            return c
        i = random.randint(0, board_size - 1)
        j = random.randint(0, board_size - 1)
        c[i], c[j] = c[j], c[i]
        return c

    return m(c1), m(c2)


def select_survivals(population, offsprings):
    """
    Replace 2 worst samples from the population with the new offsprings
    """
    population.sort(key=fitness)
    population[0:2] = offsprings
    return population


def evaluate(population):
    """
    Calculate the mean fitness of population
    """
    fs = map(fitness, population)
    return sum(fs) / len(population)


def show_solutions(population, show_configuration=False):
    """
    Print out each sample of the population
    """
    for i, x in enumerate(population):
        f = fitness(x)
        print(f'number=[{i + 1}/{population_size}], solution={x}, fitness={f}')
        if show_configuration:
            for j in range(board_size):
                for xi in x:
                    c = 'Q' if j == xi else '.'
                    print(c, end=' ')
                print()


def show_history(history):
    """
    Plot history of the metrics
    """
    x = range(epochs)
    plt.figure('N-Queens :: Genetic Algorithm')
    plt.plot(x, history)
    plt.xlabel('Epoch (Generation)')
    plt.ylabel('Metric (Average Fitness)')
    plt.show()


def run():
    history = list()
    population = initialize()
    for i in range(epochs):
        parents = select_parents(population)
        offsprings = recombination(parents)
        offsprings = mutation(offsprings)
        population = select_survivals(population, offsprings)
        metric = evaluate(population)
        history.append(metric)
        print(f'iteration=[{i + 1}/{epochs}], metric={metric}')
    show_solutions(population)
    show_history(history)
