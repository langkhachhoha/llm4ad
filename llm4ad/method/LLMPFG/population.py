from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np
from ...base import *
import numpy as np
import random
import matplotlib.pyplot as plt


# Population Management
def is_dominated(obj1, obj2):
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))

def fast_non_dominated_sort(population):
    fronts = []
    S = {}
    n = {}
    rank = {}

    for i, p in enumerate(population):
        S[i] = []
        n[i] = 0
        for j, q in enumerate(population):
            if i == j:
                continue
            if is_dominated(p.score, q.score):
                S[i].append(j)
            elif is_dominated(q.score, p.score):
                n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(i)

    i = 0
    while i < len(fronts):
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    rank[q_idx] = i + 1
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def calculate_crowding_distance(population, indices):
    distances = {i: 0.0 for i in indices}
    num_objectives = len(population[0].score)

    for m in range(num_objectives):
        indices.sort(key=lambda i: population[i].score[m])
        distances[indices[0]] = distances[indices[-1]] = float('inf')
        min_obj = population[indices[0]].score[m]
        max_obj = population[indices[-1]].score[m]
        if max_obj == min_obj:
            continue
        for k in range(1, len(indices) - 1):
            prev_obj = population[indices[k - 1]].score[m]
            next_obj = population[indices[k + 1]].score[m]
            distances[indices[k]] += (next_obj - prev_obj) / (max_obj - min_obj)
    return distances


def population_management(population, N):
    # population = [ind for ind in population if ind.score is not None]
    fronts = fast_non_dominated_sort(population)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            remaining = N - len(selected)
            distances = calculate_crowding_distance(population, front)
            sorted_front = sorted(front, key=lambda i: -distances[i])
            selected.extend(sorted_front[:remaining])
            break
    return [population[i] for i in selected]



# Selection
import numpy as np
import random
from copy import deepcopy



def cal_knee_point(pop):
    knee_point = np.zeros(len(pop[0].score))
    m = len(pop[0].score)
    for i in range(m):
        knee_point[i] = 1e9
    for indi in pop:
        for i in range(m):
            knee_point[i] = min(knee_point[i], indi.score[i])
    return knee_point


def cal_nadir_point(pop):
    m = len(pop[0].score)
    nadir_point = np.zeros(m)
    for i in range(m):
        nadir_point[i] = -1e9
    for indi in pop:
        for i in range(m):
            nadir_point[i] = max(nadir_point[i], indi.score[i])
    return nadir_point



def Generation_PFG(pop, GK, knee_point, nadir_point, sigma):
    m = len(knee_point)
    d = [(nadir_point[j] - knee_point[j] + 2 * sigma) / GK for j in range(m)]

    # Tính Grid cho từng cá thể
    Grid = []
    for indi in pop:
        grid_indi = [(indi.score[j] - knee_point[j] + sigma) // d[j] for j in range(m)]
        Grid.append(grid_indi)

    # Khởi tạo PFG: m mục tiêu, mỗi mục tiêu có GK đoạn
    PFG = [[[] for _ in range(GK)] for _ in range(m)]

    for i in range(m):  # với từng mục tiêu
        for j in range(GK):  # với từng đoạn
            # Tìm S_i(j): các cá thể thuộc đoạn thứ j của mục tiêu thứ i
            Sij = [idx for idx, g in enumerate(Grid) if g[i] == j]

            if not Sij:
                continue

            # Tìm giá trị nhỏ nhất theo Grid của các cá thể này (theo toàn bộ vector grid)
            g_min = min(Grid[idx][i] for idx in Sij)

            # Chọn những cá thể trong đoạn j có grid = g_min cho mục tiêu i
            for idx in Sij:
                if Grid[idx][i] == g_min:
                    PFG[i][j].append(pop[idx])

    return PFG


import random
def parent_selection(pop, m, GK = 4, sigma = 0.01, crossover_rate = 0.8):
    knee_point = cal_knee_point(pop)
    nadir_point = cal_nadir_point(pop)
    PFG = Generation_PFG(pop, GK, knee_point, nadir_point, sigma)
    if (random.random() > crossover_rate):
        parents = random.choices(pop, k=m)
    else:
        i = random.randint(0, len(knee_point)-1) 
        j = random.randint(0, len(PFG[i]) - 2)
        while len(PFG[i][j]) == 0:
            i = random.randint(0, len(knee_point)-1) 
            j = random.randint(0, len(PFG[i]) - 2)
        parents = random.choices(PFG[i][j] + PFG[i][j + 1], k=m)

    return parents


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # # if the score is None, we still put it into the population,
        # # we set the score to '-inf'
        if func.score is None:
            return
        try:
            self._lock.acquire()
            # register to next_gen
            self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                self._population = population_management(pop, self._pop_size)
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self, selection_num) -> List[Function]:
        return parent_selection(self._population, selection_num)
