from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_kp.get_instance import GetData
from llm4ad.task.optimization.bi_kp.template import template_program, task_description
from pymoo.indicators.hv import HV
import random
import time

__all__ = ['BIKPEvaluation']


def knapsack_value(solution: np.ndarray, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
    if np.sum(solution * weight_lst) > capacity:
        return -1e10, -1e10  # Penalize infeasible solutions
    total_val1 = np.sum(solution * value1_lst)
    total_val2 = np.sum(solution * value2_lst)
    return total_val1, total_val2


def dominates(a, b):
    """True if a dominates b (maximization)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def random_solution(problem_size):
    return np.random.randint(0, 2, size=problem_size)


def evaluate(instance_data, n_instance, problem_size, ref_point, capacity, eva: callable):
    obj_1 = np.ones(n_instance)
    obj_2 = np.ones(n_instance)
    n_ins = 0
    for weight_lst, value1_lst, value2_lst in instance_data:
        start = time.time()
        s = [random_solution(problem_size) for _ in range(20)]
        Archive = [(s_, knapsack_value(s_, weight_lst, value1_lst, value2_lst, capacity)) for s_ in s if knapsack_value(s_, weight_lst, value1_lst, value2_lst, capacity)[0] > -1e5]
        for _ in range(2000):
            s_prime = np.array(eva(Archive, weight_lst, value1_lst, value2_lst, capacity))
            f_s_prime = knapsack_value(s_prime, weight_lst, value1_lst, value2_lst, capacity)

            if f_s_prime[0] < -1e5:
                continue  # Skip infeasible

            if not any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                Archive = [(a, f_a) for a, f_a in Archive if not dominates(f_s_prime, f_a)]
                Archive.append((s_prime, f_s_prime))
        end = time.time()
        objs = np.array([obj for _, obj in Archive]) * (-1)
        hv_indicator = HV(ref_point=ref_point)
        hv_value = hv_indicator(objs)
        obj_1[n_ins] = -hv_value
        obj_2[n_ins] = end - start
        n_ins += 1
    return np.mean(obj_1), np.mean(obj_2)


class BIKPEvaluation(Evaluation):
    """Evaluator for the Bi-objective Knapsack Problem (BI-KP) using a custom algorithm."""

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=20
        )
        self.n_instance = 4
        self.problem_size = 50
        getData = GetData(self.n_instance, self.problem_size)
        self._datasets, self.cap = getData.generate_instances() 
        self.ref_point = np.array([-5, -5]) 

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(self._datasets, self.n_instance, self.problem_size, self.ref_point, self.cap, callable_func)

    

# if __name__ == '__main__':
#     import numpy as np
#     from typing import List, Tuple

#     def select_neighbor(
#         archive: List[Tuple[np.ndarray, Tuple[float, float]]],
#         weight_lst: np.ndarray,
#         value1_lst: np.ndarray,
#         value2_lst: np.ndarray,
#         capacity: float
#     ) -> np.ndarray:
#         """
#         Select a promising solution from the archive and generate a valid neighbor solution from it.

#         Args:
#             archive: List of (solution, objective) pairs. Each solution is a binary array (0/1).
#             weight_lst: Array of item weights.
#             value1_lst: Array of profits for objective 1.
#             value2_lst: Array of profits for objective 2.
#             capacity: Maximum allowed total weight.

#         Returns:
#             A new valid neighbor solution (binary numpy array).
#         """
#         # Select the archive solution with the highest average value (objective 1 + 2)
#         best_solution = max(archive, key=lambda x: sum(x[1]))[0].copy()

#         # Generate neighbors by flipping one bit at a time, keeping only valid ones
#         for i in range(len(best_solution)):
#             neighbor = best_solution.copy()
#             neighbor[i] = 1 - neighbor[i]  # flip inclusion/exclusion

#             total_weight = np.sum(neighbor * weight_lst)
#             if total_weight <= capacity:
#                 return neighbor

#         # If no single-bit flip yields a valid neighbor, return the original best
#         return best_solution

    
#     tsp = BIKPEvaluation()
#     cost, tme = tsp.evaluate_program('_',select_neighbor)
#     print(cost, tme)



