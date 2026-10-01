from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def createOffspring(parent1, parent2):
    p1 = int(random.random()*len(parent1))
    p2 = int(random.random()*len(parent1))
    if(p2<p1):
        p1, p2 = p2, p1
    first_child = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    second_child= parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return first_child, second_child

def mutation (individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = not individual[i]
    return individual

def tournamentSelection(items, knapsack_max_capacity, population, n_selection):
    selected = []
    for _ in range(n_selection):
        tournament = random.sample(population, 2)
        best_individual = None
        best_individual_fitness = -1
        for individual in tournament:
            individual_fitness = fitness(items, knapsack_max_capacity, individual)
            if individual_fitness > best_individual_fitness:
                best_individual = individual
                best_individual_fitness = individual_fitness
        selected.append(best_individual)
    return selected

def findNBest(items, knapsack_max_capacity, population, n):
    bests=[]
    fitness_table = []
    for individual in population:
        fitness_table.append(fitness(items, knapsack_max_capacity, individual))
    population_fitness = [population.copy(), fitness_table.copy()]
    for i in range(n):
        bests.append (population_fitness[0][population_fitness[1].index(max(population_fitness[1]))])
        population_fitness[0].remove(bests[i])
        population_fitness[1].remove(max(population_fitness[1]))
        
    return bests


def genetic_algorithm(items, knapsack_max_capacity, population_size=100, generations=200,
                      n_selection=20, n_elite=5, mutation_rate=None):
    #returns the best solution, its fitness, the best fitness after every generation and all populations
    if mutation_rate is None:
        mutation_rate = 1/len(items)
    best_solution = None
    best_fitness = 0
    population_history = []
    best_history = []
    population = initial_population(len(items), population_size)
    for _ in range(generations):
        population_history.append(population)

        best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
        if best_individual_fitness > best_fitness:
            best_solution = best_individual
            best_fitness = best_individual_fitness
        best_history.append(best_fitness)



        selected = tournamentSelection(items, knapsack_max_capacity, population, n_selection)
        new_population = []
        for _ in range(int(population_size/2)):
            parent1, parent2 = random.sample(selected, 2)
            offspring1,offspring2 = createOffspring(parent1, parent2)
            offspring1 = mutation(offspring1, mutation_rate)
            offspring2 = mutation(offspring2, mutation_rate)
            new_population.append(offspring1)
            new_population.append(offspring2)
        #population_sorted_by_fitness = sorted(population, key=lambda individual: fitness(items, knapsack_max_capacity, individual), reverse=True)
        bests = findNBest(items, knapsack_max_capacity, population, n_elite)
        for i in range(n_elite):
            new_population[i] = bests[i]
        population = new_population
    return best_solution, best_fitness, best_history, population_history


if __name__ == "__main__":
    items, knapsack_max_capacity = get_big()
    print(items)

    start_time = time.time()
    best_solution, best_fitness, best_history, population_history = genetic_algorithm(items, knapsack_max_capacity)
    end_time = time.time()
    total_time = end_time - start_time
    print('Best solution:', list(compress(items['Name'], best_solution)))
    print('Best solution value:', best_fitness)
    print('Time: ', total_time)

    # plot generations
    x = []
    y = []
    top_best = 10
    for i, population in enumerate(population_history):
        plotted_individuals = min(len(population), top_best)
        x.extend([i] * plotted_individuals)
        population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
        population_fitnesses.sort(reverse=True)
        y.extend(population_fitnesses[:plotted_individuals])
    plt.scatter(x, y, marker='.')
    plt.plot(best_history, 'r')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
