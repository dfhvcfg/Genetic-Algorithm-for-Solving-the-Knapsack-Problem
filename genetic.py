from deap import base, creator, tools
import random
import numpy as np


weights = np.array([10, 20, 30, 40, 50])
values = np.array([60, 100, 120, 200, 300])
max_weight = 100


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(individual):
    total_weight = np.dot(individual, weights)
    total_value = np.dot(individual, values)
    if total_weight <= max_weight:
        return total_value,
    return 0,


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(weights))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


population = toolbox.population(n=50)
n_generations = 100

for gen in range(n_generations):
    # 评估适应度
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 选择
    offspring = toolbox.select(population, len(population))

    # 克隆选中的个体
    offspring = list(map(toolbox.clone, offspring))

    # 交叉
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.8:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 变异
    for mutant in offspring:
        if random.random() < 0.05:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 更新种群
    population[:] = offspring

# 找到最优解
best_ind = tools.selBest(population, 1)[0]
print('Best Individual:', best_ind)
print('Maximum Value:', best_ind.fitness.values[0])
