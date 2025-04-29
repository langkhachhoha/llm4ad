
template_program = '''
import numpy as np
from typing import List, Tuple
import random 

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    instance: np.ndarray,
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray
) -> np.ndarray:
    """
    Select a promising solution from the archive and generate a neighbor solution from it.

    Args:
    archive: List of (solution, objective) pairs. Each solution is a numpy array of node IDs.
             Each objective is a tuple of two float values.
    instance: Numpy array of shape (N, 4). Each row corresponds to a node and contains its coordinates in two 2D spaces: (x1, y1, x2, y2).
    distance_matrix_1: Distance matrix in the first objective space.
    distance_matrix_2: Distance matrix in the second objective space.

    Returns:
    A new neighbor solution (numpy array).
    """
    base_solution = archive[0][0].copy()
    new_solution = base_solution.copy()
    new_solution[0], new_solution[1] = new_solution[1], new_solution[0]

    return new_solution
'''

task_description = "You are solving a Bi-objective Travelling Salesman Problem (bi-TSP), where each node has two different 2D coordinates: \
(x1, y1) and (x2, y2), representing its position in two objective spaces. The goal is to find a tour visiting each node exactly once and returning \
to the starting node, while minimizing two objectives simultaneously: the total tour length in each coordinate space. \
Given an archive of non-dominated solutions, where each solution is a numpy array representing a TSP tour, and its corresponding objective \
is a tuple of two values (cost in each space), design a heuristic function named 'select_neighbor' that selects one solution from the archive \
and generates a neighbor solution from it. Do not choose randomly. Instead, think about how to identify a solution that is promising for further  \
local improvement. Using a novel or creative strategy — not necessarily 2-opt. You can try swap, reinsertion, segment relocation, or invent your own local \
transformation logic.  The function should return the new neighbor solution."

