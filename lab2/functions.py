import numpy as np


def roulette_wheel_selection(population, n_selection, items, knapsack_max_capacity, fitness_function):
    fitness_sum = sum([fitness_function(items, knapsack_max_capacity, individual) for individual in population])
    individual_probabilities = [fitness_function(items, knapsack_max_capacity, individual)/fitness_sum for individual in population]
    parents_indices = np.random.choice(len(population), n_selection, p=individual_probabilities)
    parents = [population[index] for index in parents_indices]
    return parents


def crossover(parents, population_size, cross_rate=0.5):
    generation = []

    while len(generation) < population_size:
        parents_indices = np.random.choice(len(parents), size=2)
        parent_a = parents[parents_indices[0]]
        parent_b = parents[parents_indices[1]]

        crossover_point = int(len(parent_a)*cross_rate)

        child_a = parent_a[:crossover_point] + parent_b[crossover_point:]
        child_b = parent_b[:crossover_point] + parent_a[crossover_point:]

        generation.extend([child_a, child_b])

    return generation[:population_size]

def mutate(population):
    mutated = []
    for individual in population:
        m = np.random.choice(len(individual))
        individual[m] = not individual[m]
        mutated.append(individual)

    return mutated

def elitism(population, n_elite, items, knapsack_max_capacity, fitness_function):
    sorted_population = sorted(population, key=lambda individual: fitness_function(items, knapsack_max_capacity, individual), reverse=True)
    return sorted_population[:n_elite]