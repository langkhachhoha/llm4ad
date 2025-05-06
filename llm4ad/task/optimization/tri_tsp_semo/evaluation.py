# name: str: TSPEvaluation
# Parameters:
# timeout_seconds: int: 20
# end
from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.tri_tsp_semo.get_instance import GetData
from llm4ad.task.optimization.tri_tsp_semo.template import template_program, task_description
from pymoo.indicators.hv import HV 
import random
import time 

__all__ = ['TRITSPEvolution']


def tour_cost(instance, solution, problem_size):

        cost_1 = 0
        cost_2 = 0
        cost_3 = 0
        
        for j in range(problem_size - 1):
            node1, node2 = int(solution[j]), int(solution[j + 1])
            
            coord_1_node1, coord_2_node1, coord_3_node1 = instance[node1][:2], instance[node1][2:4], instance[node1][4:]
            coord_1_node2, coord_2_node2, coord_3_node2 = instance[node2][:2], instance[node2][2:4], instance[node2][4:]

            cost_1 += np.linalg.norm(coord_1_node1 - coord_1_node2)
            cost_2 += np.linalg.norm(coord_2_node1 - coord_2_node2)
            cost_3 += np.linalg.norm(coord_3_node1 - coord_3_node2)
        
        node_first, node_last = int(solution[0]), int(solution[-1])
        
        coord_1_first, coord_2_first, coord_3_first = instance[node_first][:2], instance[node_first][2:4], instance[node_first][4:]
        coord_1_last, coord_2_last, coord_3_last = instance[node_last][:2], instance[node_last][2:4], instance[node_last][4:]

        cost_1 += np.linalg.norm(coord_1_last - coord_1_first)
        cost_2 += np.linalg.norm(coord_2_last - coord_2_first)
        cost_3 += np.linalg.norm(coord_3_last - coord_3_first)

        return cost_1, cost_2, cost_3
    

def dominates(a, b):
        """True if a dominates b (minimization)."""
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def random_solution(problem_size):
        sol = list(range(problem_size))
        random.shuffle(sol)
        return np.array(sol)



def evaluate(instance_data, n_instance, problem_size, ref_point, eva: callable):
        obj_1 = np.ones(n_instance)
        obj_2 = np.ones(n_instance)
        obj_3 = np.ones(n_instance)
        n_ins = 0
        for instance, distance_matrix_1, distance_matrix_2, distance_matrix_3 in instance_data:
            start = time.time()
            s = [random_solution(problem_size) for _ in range(10)]
            Archive = [(s_, tour_cost(instance, s_, problem_size)) for s_ in s]
            for _ in range(2000):
                s_prime = eva(Archive, instance, distance_matrix_1, distance_matrix_2, distance_matrix_3)
                f_s_prime = tour_cost(instance, s_prime, problem_size)

                # Nếu không bị thống trị
                if not any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                    # Loại bỏ các phần tử bị thống trị bởi f_s_prime
                    Archive = [(a, f_a) for a, f_a in Archive if not dominates(f_s_prime, f_a)]
                    # Thêm nghiệm mới
                    Archive.append((s_prime, f_s_prime))
            end = time.time()
            objs = np.array([obj for _, obj in Archive])
            # Tính HV
            hv_indicator = HV(ref_point=ref_point)
            hv_value = hv_indicator(objs)
            obj_1[n_ins] = -hv_value
            obj_2[n_ins] = end - start
            n_ins += 1
        return np.mean(obj_1), np.mean(obj_2)
            





class TRITSPEvaluation(Evaluation):
    """Evaluator for the Bi-objective Traveling Salesman Problem (TSP) using a custom algorithm."""

    def __init__(self, **kwargs):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=20
        )

        self.n_instance = 8
        self.problem_size = 20
        getData = GetData(self.n_instance, self.problem_size)
        self._datasets = getData.generate_instances()
        self.ref_point = np.array([20.0, 20.0, 20.0])

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(self._datasets,self.n_instance,self.problem_size, self.ref_point, callable_func)
    

if __name__ == '__main__':
    import numpy as np
    from typing import List, Tuple
    import random 
    def select_neighbor(archive: List[Tuple[np.ndarray, Tuple[float, float]]], instance: np.ndarray, distance_matrix_1: np.ndarray, distance_matrix_2: np.ndarray, distance_matrix_3: np.ndarray) -> np.ndarray:
        """
        Select a promising solution from the archive and generate a neighbor solution from it.

        Args:
        archive: List of (solution, objective) pairs. Each solution is a numpy array of node IDs.
                Each objective is a tuple of three float values.
        instance: Numpy array of shape (N, 4). Each row corresponds to a node and contains its coordinates in two 2D spaces: (x1, y1, x2, y2).
        distance_matrix_1: Distance matrix in the first objective space.
        distance_matrix_2: Distance matrix in the second objective space.
        distance_matrix_3: Distance matrix in the third objective space.

        Returns:
        A new neighbor solution (numpy array).
        """
        best_solution, _ = min(archive, key=lambda x: sum(x[1]))

        # Step 2: Generate a neighbor solution using segment relocation
        # Create a neighbor solution using segment relocation
        n = len(best_solution)
        if n < 4:  # Need at least 4 nodes to perform a segment relocation
            return best_solution.copy()
        
        # Randomly select two segments to relocate
        start = np.random.randint(0, n - 2)
        end = np.random.randint(start + 2, n)  # Ensure segments don't overlap
        segment = best_solution[start:end].copy()
        
        # Remove the segment from the original solution
        new_solution = np.delete(best_solution, np.s_[start:end])
        
        # Randomly choose a position to insert the segment back
        insert_position = np.random.randint(0, len(new_solution) + 1)
        
        # Create the new neighbor solution
        neighbor_solution = np.insert(new_solution, insert_position, segment)
        
        return neighbor_solution
    
    tsp = TRITSPEvaluation()
    cst, _ = tsp.evaluate_program('_',select_neighbor)
    print(cst)



