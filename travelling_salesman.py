import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt

# Define the number of cities
num_cities = 10

# Generate the coordinates of the cities
cities = np.random.rand(num_cities, 2)

# Create a KD-tree for efficient nearest-neighbor search
tree = KDTree(cities)

# Define the mutation rate
mutation_rate = 0.1

# Define the number of iterations
num_iterations = 100

# Define the fitness function
def get_fitness(solution, cities, tree):
    distance = 0
    for i in range(len(solution) - 1):
        city1 = cities[solution[i]]
        city2 = cities[solution[i + 1]]
        dist, _ = tree.query([city1, city2])
        distance += dist[1]
    return distance

def crossover(parent1, parent2, mutation_rate):
    # Get the indices of the parent solutions
    parent1_index = parent1
    parent2_index = parent2

    # Get the parent solutions
    parent1 = population[parent1_index]
    parent2 = population[parent2_index[0]]

    child = parent1.copy()
    for i in range(len(parent1)):
        if np.random.rand() < mutation_rate:
            child[i] = parent2[i]
    return child


# Create an initial population of solutions
population = [np.random.permutation(num_cities) for _ in range(10)]

# Evaluate the fitness of each solution
fitness = [get_fitness(solution, cities, tree) for solution in population]

# Set up the plot
fig, ax = plt.subplots()
ax.scatter(cities[:, 0], cities[:, 1])
lines = []

# Repeat for a fixed number of iterations
for iteration in range(num_iterations):
    # Select the best solutions
    selected = sorted(range(len(fitness)), key=lambda i: fitness[i])[:5]

    # Create a new population of solutions
    population = [crossover(population[selected[i]][0], population[j], mutation_rate) for i in range(len(selected)) for j in range(len(population))]

    # Evaluate the fitness of each solution
    fitness = [get_fitness(solution, cities, tree) for solution in population]

    # Get the best solution
    best_solution_index = np.argmin(fitness)
    best_solution = population[best_solution_index]

    # Plot the solutions in the population
    for line in lines:
        line.remove()
    lines.clear()

    # Create a new figure for each iteration
    fig, ax = plt.subplots()

    for solution in population:
        x = cities[solution, 0]
        y = cities[solution, 1]
        line, = ax.plot(x, y, alpha=0.5)
        lines.append(line)

    # Plot the best solution
    x = cities[best_solution, 0]
    y = cities[best_solution, 1]
    best_line, = ax.plot(x, y, color="red")
    lines.append(best_line)

    # Update the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.show()

