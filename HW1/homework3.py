import random, math
from turtle import distance

def ReadInput(input):
    with open(input) as f:
        lines = f.readlines()

    cities = []
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        x = int(line[0])
        y = int(line[1])
        z = int(line[2])
        cities.append((x, y, z))

    return cities



def CalculateDistance(city1, city2):
    x1 = city1[0]
    y1 = city1[1]
    z1 = city1[2]

    x2 = city2[0]
    y2 = city2[1]
    z2 = city2[2]

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    return distance



def Greedy(cities, start):
    cityDistance = {}
    unvisited = []
    visited = []
    
    for city in cities:
        cityDistance[city] = float('inf')
        unvisited.append(city)
    
    cityDistance[cities[start]] = 0

    while unvisited != []:
        currCity = min(cityDistance, key = cityDistance.get)
        unvisited.remove(currCity)
        visited.append(currCity)

        for adjCity in unvisited:
            if cityDistance[adjCity] > cityDistance[currCity] + CalculateDistance(currCity, adjCity):
                cityDistance[adjCity] =  cityDistance[currCity] + CalculateDistance(currCity, adjCity)

        del cityDistance[currCity]

    visited.append(cities[start])
    
    return visited



def CreateInitialPopulation(size, cities):
    initialPopulation = []

    for i in range(size):
        start = random.randint(0, len(cities) - 1)
        path = Greedy(cities, start)
        initialPopulation.append(path)
        
    return initialPopulation



def CalculatePathFitness(path):
    distance = 0
    for i in range(len(path) - 1):
        dist = CalculateDistance(path[i], path[i + 1])
        distance += dist

    fitness =  1 / distance
    return fitness



def RankPaths(population):
    pathStats = []
    for i in range(len(population)):
        pathFitness = CalculatePathFitness(population[i])
        pathStats.append((i, pathFitness))
    
    rankedPaths = sorted(pathStats, key = lambda x: x[1], reverse = True)

    return rankedPaths



def CreateMatingPool(population, rankedPaths):
    matingPool = []
    matingPoolSize = int(len(population) / 2)

    for i in range(matingPoolSize):
        matingPool.append(population[rankedPaths[i][0]])
        
    return matingPool



def Crossover(parent1, parent2, crossoverRate):
    rate = random.random()

    if rate <= crossoverRate:
        child = [0] * len(parent1)

        idx1 = random.randint(1, len(parent1) - 2)
        idx2 = random.randint(1, len(parent1) - 2)
        startIdx = min(idx1, idx2)
        endIdx = max(idx1, idx2)

        for i in range(startIdx, endIdx + 1):
            child[i] = parent1[i]

        ptr = 0
        while ptr < len(child):
            if child[ptr] == 0:
                for j in range(len(parent2)):
                    if parent2[j] not in child:
                        child[ptr] = parent2[j]
                        break
            ptr += 1
        
        child[-1] = child[0]
    else:
        child = parent2

    return child



def Mutation(child, mutationRate):
    rate = random.random()
    if rate <= mutationRate:
        idx1 = random.randint(1, len(child) - 2)
        idx2 = random.randint(1, len(child) - 2)
        child[idx1:idx2] = reversed(child[idx1:idx2])

    return child



def CreateChildrenPool(population, matingPool, rankedPaths, eliteSize, crossoverRate, mutationRate):
    childrenPool = []

    for i in range(eliteSize):
        childrenPool.append(population[rankedPaths[i][0]])

    for i in range(len(population) - eliteSize):
        parent1Idx = random.randint(0, len(matingPool) - 1)
        parent2Idx = random.randint(0, len(matingPool) - 1)

        child = Crossover(matingPool[parent1Idx], matingPool[parent2Idx], crossoverRate)
        child = Mutation(child, mutationRate)
        childrenPool.append(child)
    
    return childrenPool



def CreateNextPopulation(currentPopulation, eliteSize, crossoverRate, mutationRate):
    rankedPaths = RankPaths(currentPopulation)
    matingPool = CreateMatingPool(currentPopulation, rankedPaths)
    nextPopulation = CreateChildrenPool(currentPopulation, matingPool, rankedPaths, eliteSize, crossoverRate, mutationRate)
    return nextPopulation



def GeneticAlgorithm(population, size, eliteSize, crossoverRate, mutationRate, generations):
    currentPopulation = CreateInitialPopulation(size, population)

    for i in range(generations):
        currentPopulation = CreateNextPopulation(currentPopulation, eliteSize, crossoverRate, mutationRate)

    shortestPathIdx = RankPaths(currentPopulation)[0][0]
    shortestPath = currentPopulation[shortestPathIdx]

    return shortestPath



shortestPath = GeneticAlgorithm(population = ReadInput('input.txt'), size = 30, eliteSize = 10, crossoverRate = 0.8, mutationRate = 0.5, generations = 200)

f = open('output.txt', 'w')
for city in shortestPath:
    line = ' '.join(str(coord) for coord in city)
    f.write(line + '\n')
f.close()