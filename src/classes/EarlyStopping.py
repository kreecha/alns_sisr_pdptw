#-*- coding: utf-8 -*-
"""
Created on Thur August 14 04:26:17 2025

ALNS (Adaptive Large Neighborhood Search) implementation for PDPTW
Using the ALNS library https://github.com/N-Wouda/ALNS/tree/master
to solve Li-Lim benchmark instances download files from https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/

Data instances The format of the data files is as follows:

The first line gives the number of vehicles, the capacity, and the speed (not used)
From the second line, for each customer (starting with the depot):
The index of the customer
The x coordinate
The y coordinate
The demand
The earliest arrival
The latest arrival
The service time
The index of the corresponding pickup order (0 if the order is a pickup)
The index of the corresponding delivery order (0 if the order is a delivery)


@author: Kreecha_P

MIT License

Copyright (c) 2025 Kreecha Puphaiboon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from alns.stop import MaxIterations, StoppingCriterion

class EarlyStoppingCriterion(StoppingCriterion):
    """
    Enhanced early stopping criterion with adaptive improvement thresholds
    """

    def __init__(self, max_iterations: int, improvement_threshold: float = None,
                 patience_ratio: float = 0.5, adaptive_threshold: bool = True, is_verbose: bool = False):
        """
        Args:
            max_iterations: Maximum number of iterations
            improvement_threshold: Minimum improvement to reset counter
                                 (None = auto-calculate, float = fixed threshold)
            patience_ratio: Fraction of max_iterations to wait (default: 0.5 = 50%)
            adaptive_threshold: Whether to adapt threshold based on initial solution
        """
        self.max_iterations = max_iterations
        self.base_improvement_threshold = improvement_threshold
        self.patience = int(max_iterations * patience_ratio)
        self.adaptive_threshold = adaptive_threshold

        # Tracking variables
        self.iterations_without_improvement = 0
        self.best_objective = float('inf')
        self.current_iteration = 0
        self.initial_objective = None
        self.improvement_threshold = improvement_threshold  # Will be set dynamically

        # Statistics
        self.total_improvements = 0
        self.improvement_history = []

        # log
        self.is_verbose = is_verbose

    def _calculate_adaptive_threshold(self, initial_obj: float) -> float:
        """Calculate adaptive threshold based on initial objective value"""
        if initial_obj <= 0:
            return 1.0

        # Use 0.1% of initial objective as threshold, with reasonable bounds
        adaptive = initial_obj * 0.001  # 0.1% of initial objective

        # Set reasonable bounds: between 0.1 and 5.0
        adaptive = max(0.1, min(adaptive, 5.0))

        return round(adaptive, 2)

    def __call__(self, rng, best, current):
        """
        Check stopping criteria with enhanced threshold logic

        Args:
            rng: Random number generator
            best: Best solution state so far
            current: Current solution state

        Returns:
            bool: True if should stop, False otherwise
        """
        self.current_iteration += 1

        # Get best objective value
        try:
            current_best_objective = best.objective()
        except AttributeError:
            current_best_objective = float(best) if hasattr(best, '__float__') else self.best_objective

        # Set initial objective and adaptive threshold on first iteration
        if self.initial_objective is None:
            self.initial_objective = current_best_objective

            if self.adaptive_threshold and self.base_improvement_threshold is None:
                self.improvement_threshold = self._calculate_adaptive_threshold(self.initial_objective)
                print(f"Adaptive improvement threshold set to: {self.improvement_threshold}")
            elif self.base_improvement_threshold is not None:
                self.improvement_threshold = self.base_improvement_threshold
                print(f"Fixed improvement threshold: {self.improvement_threshold}")
            else:
                self.improvement_threshold = 1.0
                print(f"Default improvement threshold: {self.improvement_threshold}")

        # Check if we have improvement
        improvement = self.best_objective - current_best_objective
        if improvement >= self.improvement_threshold:
            # Found meaningful improvement, reset counter
            self.best_objective = current_best_objective
            self.iterations_without_improvement = 0
            self.total_improvements += 1
            self.improvement_history.append((self.current_iteration, improvement))

            # Optional: Print improvement notifications
            if self.is_verbose:
                if self.current_iteration % 50 == 0 and improvement > self.improvement_threshold * 5:
                    improvement_pct = (improvement / self.initial_objective * 100) if self.initial_objective > 0 else 0
                    print(f"  Improvement at iter {self.current_iteration}: -{improvement:.2f} ({improvement_pct:.2f}%)")
        else:
            # No significant improvement
            self.iterations_without_improvement += 1
            # Update best even if improvement is below threshold
            if current_best_objective < self.best_objective:
                self.best_objective = current_best_objective

        # Stop if reached max iterations
        if self.current_iteration >= self.max_iterations:
            print(f"\nStopping: Reached maximum iterations ({self.max_iterations})")
            self._print_final_stats()
            return True

        # Stop if no improvement for patience iterations
        if self.iterations_without_improvement >= self.patience:
            print(
                f"\nEarly stopping: No improvement ≥{self.improvement_threshold} for {self.iterations_without_improvement} iterations")
            print(
                f"(Patience limit: {self.patience} iterations, {self.patience / self.max_iterations * 100:.0f}% of max)")
            self._print_final_stats()
            return True

        return False

    def _print_final_stats(self):
        """Print final statistics about the optimization process"""
        if self.initial_objective is not None:
            total_improvement = self.initial_objective - self.best_objective
            improvement_pct = (total_improvement / self.initial_objective * 100) if self.initial_objective > 0 else 0

            print(f"Early Stopping Statistics:")
            print(f"  - Total iterations: {self.current_iteration}")
            print(f"  - Significant improvements: {self.total_improvements}")
            print(f"  - Total improvement: {total_improvement:.2f} ({improvement_pct:.1f}%)")
            print(f"  - Improvement threshold used: {self.improvement_threshold}")
            print(f"  - Iterations without improvement: {self.iterations_without_improvement}")
