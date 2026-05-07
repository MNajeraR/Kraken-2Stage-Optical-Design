# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:21:18 2025

@author: MORGANRHAINAJERAROA
"""

# import numpy as np
# from scipy.special import erfinv

#--------------------------#
# Random Realization Class
#--------------------------#

"""
======================================
  Class: Random_RO
======================================

This class implements a randomized optimization strategy using Gaussian 
distribution for parameter sampling. It allows multiple iterations to 
refine the solution, progressively moving closer to the optimal result.

Parameters:
- num_range (int): The number of random values to generate for each center.
- centers (list): A list of initial center points for the Gaussian distribution.

Methods:


3. _process_iteration:
    Processes a single iteration of the optimization, evaluating the function 
    at each sampled point and selecting the best solution.

4. _process_new_iteration:
    Processes subsequent iterations, refining the sample space around 
    the best solution found so far.

======================================
"""


from typing import List, Tuple, Callable
import numpy as np


class Random_RO:
    def __init__(self, num_range: int, centers: List[float]):
        self.num_range = num_range
        self.centers = centers

    def Uniform_distribution(self, bounds: List[Tuple[float, float]]):
        self.dim_coor = []
        for (lo, hi) in bounds:
            samples = np.random.uniform(lo, hi, self.num_range).astype(np.float64)
            self.dim_coor.append(samples)
        return self.dim_coor

    def Gaussian_optimization(self, iterations: int, function: Callable,
                              bounds: List[Tuple[float, float]],
                              tol: float = 1e-6,
                              improvement_mode: str = "abs",
                              patience: int = 0):
        self.iterations = iterations
        if self.iterations < 1:
            print('Number of iterations must be greater than 1')
            return None

        if not bounds or len(bounds) != len(self.centers):
            raise ValueError("Debes proveer 'bounds' con una tupla (lo,hi) por dimensión.")

        self.history_vectors = []
        self.best_values_history = []
        self.early_stopped = False
        self.no_improve_count = 0
        self.tol = tol
        self.improvement_mode = improvement_mode
        self.patience = patience

        # Iteración 1
        self.set_values = self.Uniform_distribution(bounds)
        self.vector_2ev = []
        self.results = []
        self.num_vectors = len(self.set_values)
        self._process_iteration(function)
        self.history_vectors.append(self.vector_2ev)
        self.best_values_history.append(self.min_result)
        prev_best = self.min_result

        if self.iterations == 1:
            return self.best_solution

        # Iteraciones siguientes: muestreo uniforme global
        for _ in range(self.iterations - 1):
            self.set_new_values = self.Uniform_distribution(bounds)
            self._process_new_iteration(function)

            self.history_vectors.append(self.new_vector_2ev)
            self.best_values_history.append(self.min_result)

            if improvement_mode == "rel":
                denom = max(1.0, abs(prev_best))
                improvement = (prev_best - self.min_result) / denom
            else:
                improvement = prev_best - self.min_result

            if improvement > self.tol:
                self.no_improve_count = 0
                prev_best = self.min_result
            else:
                self.no_improve_count += 1
                if self.no_improve_count > self.patience:
                    self.early_stopped = True
                    break

        return self.best_solution
    # ======================================
    #    Process First Iteration
    # ======================================

    def _process_iteration(self, function: Callable):
        self.vector_2ev = []
        for i in range(self.num_range):
            vec = [self.set_values[d][i] for d in range(self.num_vectors)]
            self.vector_2ev.append(vec)

        self.results = [function(vec) for vec in self.vector_2ev]
        # print(self.results)
        self.min_index = np.argmin(self.results)
        self.min_result = self.results[self.min_index]
        self.best_solution = self.vector_2ev[self.min_index]

    # ======================================
    #    Process Subsequent Iterations
    # ======================================
    
    def _process_new_iteration(self, function: Callable):
        self.new_vector_2ev = []
        for i in range(self.num_range):
            vec = [self.set_new_values[d][i] for d in range(self.num_vectors)]
            self.new_vector_2ev.append(vec)

        self.new_results = [function(vec) for vec in self.new_vector_2ev]
        self.new_min_index = np.argmin(self.new_results)
        self.new_min_result = self.new_results[self.new_min_index]

        if self.new_min_result < self.min_result:
            self.min_result = self.new_min_result
            self.best_solution = self.new_vector_2ev[self.new_min_index]