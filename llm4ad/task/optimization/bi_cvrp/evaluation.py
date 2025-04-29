from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_cvrp.get_instance import GetData
from llm4ad.task.optimization.bi_cvrp.template import template_program, task_description
from pymoo.indicators.hv import HV 
import random
import time

__all__ = ['BICVRPEvaluation']

def compute_route_length(route: np.ndarray, distance_matrix: np.ndarray) -> float:
    if len(route) <= 1:
        return 0.0
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

def evaluate_solution(routes: list[np.ndarray], distance_matrix: np.ndarray):
    total_distance = 0.0
    longest_route = 0.0
    for route in routes:
        d = compute_route_length(route, distance_matrix)
        total_distance += d
        longest_route = max(longest_route, d)
    return total_distance, longest_route

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def random_solution(num_customers: int, capacity: float, demand: np.ndarray) -> list[np.ndarray]:
    customers = list(range(1, num_customers + 1))
    random.shuffle(customers)
    routes = []
    current_route = [0]
    current_load = 0.0
    for customer in customers:
        if current_load + demand[customer] <= capacity:
            current_route.append(customer)
            current_load += demand[customer]
        else:
            current_route.append(0)
            routes.append(np.array(current_route))
            current_route = [0, customer]
            current_load = demand[customer]
    current_route.append(0)
    routes.append(np.array(current_route))
    return routes

def is_feasible_solution(routes: List[np.ndarray], demand: np.ndarray, capacity: float) -> bool:
    """
    Check if all routes satisfy the vehicle capacity constraint, visit each customer exactly once,
    and have no duplicate visits. Routes are numpy arrays starting and ending with depot index 0.
    """
    all_customers = set(range(1, len(demand)))
    visited = []
    # Check capacity and collect visited customers
    for route in routes:
        # depot at start and end
        if route[0] != 0 or route[-1] != 0:
            return False
        # compute load excluding depot
        customers = route[1:-1]
        load = sum(demand[customers])
        if load > capacity:
            return False
        visited.extend(customers.tolist())
    # Check for duplicates and completeness
    visited_set = set(visited)
    if len(visited) != len(visited_set):
        return False
    if visited_set != all_customers:
        return False
    return True

def evaluate(instance_data, n_instance, ref_point, capacity, evaluate_func: callable):
    obj_1 = np.ones(n_instance)
    obj_2 = np.ones(n_instance)
    for i, (coords, demand, distance_matrix) in enumerate(instance_data):
        start = time.time()
        init_solutions = [random_solution(len(demand)-1, capacity, demand) for _ in range(10)]
        archive = [(s, evaluate_solution(s, distance_matrix)) for s in init_solutions]
        for _ in range(2000):
            s_prime = evaluate_func(archive, coords, demand, distance_matrix, capacity)
            if not is_feasible_solution(s_prime, demand, capacity):
                continue
            f_prime = evaluate_solution(s_prime, distance_matrix)
            if not any(dominates(f, f_prime) for _, f in archive):
                archive = [(s, f) for s, f in archive if not dominates(f_prime, f)]
                archive.append((s_prime, f_prime))
        end = time.time()
        objs = np.array([f for _, f in archive])
        hv_indicator = HV(ref_point=ref_point)
        obj_1[i] = -hv_indicator(objs)
        obj_2[i] = end - start
    return np.mean(obj_1), np.mean(obj_2)

class BICVRPEvaluation(Evaluation):
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
        self.ref_point = np.array([45, 8])

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(self._datasets, self.n_instance, self.ref_point, self.cap, callable_func)
    

if __name__ == '__main__':
    import numpy as np
    from typing import List, Tuple
    import random 

    def select_neighbor(
        archive: List[Tuple[np.ndarray, Tuple[float, float]]],
        coords: np.ndarray,
        demand: np.ndarray,
        distance_matrix: np.ndarray,
        capacity: float
    ) -> np.ndarray:
        """
        Select a promising solution from the archive and generate a neighbor solution from it.
        Args:
            archive: A list of tuples, where each tuple contains:
                - solution: A list of numpy arrays, each representing a vehicle route. 
                            Each route starts and ends at the depot (node index 0), e.g., [0, 3, 5, 0].
                - objective: A tuple of two float values (total_distance, makespan), 
                            representing the two objective values of the solution.
            
            coords: A numpy array of shape (n_nodes, 2), representing (x, y) coordinates of each node (depot + customers).
            demand: A numpy array of shape (n_nodes,), where demand[i] is the demand of node i. The depot has demand 0.
            distance_matrix: A numpy array of shape (n_nodes, n_nodes), where [i][j] is the Euclidean distance between node i and j.
            capacity: A float representing the maximum capacity of each vehicle.

        Returns:
            A new neighbor solution.
        """
        i = random.randint(0, len(archive) - 1)
        base_solution = archive[i][0].copy()
        new_solution = base_solution.copy()
        i = np.random.randint(1, len(base_solution) - 1)
        j = np.random.randint(1, len(base_solution) - 1)
        new_solution[i], new_solution[j] = new_solution[i], new_solution[j]

        return new_solution

    
    tsp = BICVRPEvaluation()
    cst, tme = tsp.evaluate_program('_',select_neighbor)
    print("Cost:", cst)
    print("Time:", tme)



