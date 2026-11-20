# Assignment 3
# Nicolas Jao
# 400450151
import random
import math

# constants
R = 30
V = 6.5 * (10 ** -6)

# bounds: mu, ms, Kt, K, C
low = [25, 400, 420000, 60000, 1900]
high = [40, 550, 700000, 90000, 3100]

def a_s(mu, ms, Kt, K, C):
    return math.sqrt(math.pi * R * V * ((Kt * C) / (2 * ms ** 1.5 * K ** 0.5) + ((mu + ms) * K ** 2) / (2 * C * ms ** 2)))

def make_person():
    p = []
    for i in range(5):
        p.append(random.uniform(low[i], high[i]))
    return p

pop_size = 100
generations = 1000
pop = []

# make the initial population
for i in range(pop_size):
    pop.append(make_person())

# run the genetic algorithm
for g in range(generations):
    # sort population from best to worst
    sorted_pop = []
    while len(pop) > 0:
        best = pop[0]
        best_value = a_s(best[0], best[1], best[2], best[3], best[4])
        for p in pop:
            value = a_s(p[0], p[1], p[2], p[3], p[4])
            if value < best_value:
                best = p
                best_value = value
        sorted_pop.append(best)
        pop.remove(best)
    pop = sorted_pop
    # keep the best half
    parents = pop[:pop_size // 2]
    # make children
    children = []
    while len(children) < pop_size - len(parents):
        mom = random.choice(parents)
        dad = random.choice(parents)
        child = []
        for i in range(5):
            # crossover: average or pick one
            if random.random() < 0.5:
                gene = (mom[i] + dad[i]) / 2
            else:
                gene = random.choice([mom[i], dad[i]])
            # mutation
            if random.random() < 0.1:
                gene = random.uniform(low[i], high[i])
            # keep in bounds
            if gene < low[i]:
                gene = low[i]
            if gene > high[i]:
                gene = high[i]
            child.append(gene)
        children.append(child)
    pop = parents + children
# sort population one last time
sorted_pop = []
while len(pop) > 0:
    best = pop[0]
    best_value = a_s(best[0], best[1], best[2], best[3], best[4])
    for p in pop:
        value = a_s(p[0], p[1], p[2], p[3], p[4])
        if value < best_value:
            best = p
            best_value = value
    sorted_pop.append(best)
    pop.remove(best)
pop = sorted_pop
best = pop[0]
print("mu =", best[0])
print("ms =", best[1])
print("Kt =", best[2])
print("K =", best[3])
print("C =", best[4])
print("function value =", a_s(best[0], best[1], best[2], best[3], best[4]))

'''
answers after running once:
mu = 25.003409345746803
ms = 549.9976347916739
Kt = 420014.4501740388
K = 60004.10741403773
C = 3099.737458776144
function value = 0.8958314189591444
'''