from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np

from ...base import *

import numpy as np
import random
from codebleu import calc_codebleu


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
    
    # ----- Population_management ----
    def ast_similarity(self, code1, code2):
        return calc_codebleu([code1], [code2], "python")['syntax_match_score']

    def dominates(self, ind1, ind2):
        return all(x <= y for x, y in zip(ind1, ind2)) and any(x < y for x, y in zip(ind1, ind2))

    def compute_dissimilarity(self):
        N = len(self._population)
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    S[i, j] = -self.ast_similarity(str(self._population[i]), str(self._population[j])) 
        return S

    def compute_dominance_mask(self):
        N = len(self._population)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j and self.dominates(self._population[i].score, self._population[j].score):
                    D[i, j] = 1
        return D

    def population_management(self):
        S = self.compute_dissimilarity()
        D = self.compute_dominance_mask()
        S_prime = S * D
        v = np.sum(S_prime, axis=0)
        k = np.argsort(-v)
        new_population = [self._population[i] for i in k[:self._pop_size]]
        return new_population

    def register_function(self, func: Function):
        if (self._generation == 0 and func.score is None) or func.score is None:
            return
        try:
            self._lock.acquire()
            self._next_gen_pop.append(func)
            if len(self._next_gen_pop) >= self._pop_size:
                self._population = self._population + self._next_gen_pop
                self._population = self.population_management()
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
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

    def parent_selection(self, m):
        S = self.compute_dissimilarity()
        D = self.compute_dominance_mask()
        S_prime = S * D
        v = np.sum(S_prime, axis=0)
        pi = np.exp(v) / np.sum(np.exp(v))  

        parents = random.choices(self._population, weights=pi, k=m)
        return parents
    
    def selection(self, selection_num) -> List[Function]:
        try:
            return self.parent_selection(selection_num)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return []
