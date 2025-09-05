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

class ALNSProgressLogger:
    """Progress logger that integrates with N-Wouda ALNS framework callbacks"""

    def __init__(self, log_mode: str = 'interval', interval: int = 100):
        """
        Initialize progress logger

        Args:
            log_mode: 'interval' to log every N iterations, 'improvement' to log only on improvements
            interval: Number of iterations between progress reports (if log_mode='interval')
        """
        self.log_mode = log_mode
        self.interval = interval
        self.iteration_count = 0
        self.best_objectives = []
        self.current_objectives = []
        self.improvements = []
        self.initial_objective = None

    def log_progress(self, candidate_solution, random_state, **kwargs):
        """
        Callback function called by ALNS framework on_accept

        Args:
            candidate_solution: The candidate solution being evaluated
            random_state: Random state (not used but required by callback signature)
            **kwargs: Additional parameters from ALNS framework
        """
        self.iteration_count += 1

        # The ALNS framework doesn't provide current and best in kwargs
        # We need to track the best ourselves
        candidate_obj = candidate_solution.objective()

        # Store initial objective on first call
        if self.initial_objective is None:
            self.initial_objective = candidate_obj

        # Update current objective
        self.current_objectives.append(candidate_obj)

        # Update best objective (maintain the best ever seen)
        if not self.best_objectives:
            # First iteration
            self.best_objectives.append(candidate_obj)
        else:
            # Keep the best (minimum) objective value
            current_best = min(self.best_objectives[-1], candidate_obj)
            self.best_objectives.append(current_best)

        # Check for improvement (only if we actually found a better solution)
        if len(self.best_objectives) > 1:
            if self.best_objectives[-1] < self.best_objectives[-2]:
                self.improvements.append(self.iteration_count)
                if self.log_mode == 'improvement':
                    self._report_improvement()

        # Report progress at intervals
        if self.log_mode == 'interval' and self.iteration_count % self.interval == 0:
            self._report_progress()

    def _report_progress(self):
        """Report current optimization progress"""
        if not self.best_objectives:
            return

        current_best = self.best_objectives[-1]
        current_current = self.current_objectives[-1]

        print(f"Iteration {self.iteration_count:4d}: "
              f"Current = {current_current:8.2f}, "
              f"Best = {current_best:8.2f}")

        if self.improvements:
            last_improvement = max(self.improvements)
            iterations_since = self.iteration_count - last_improvement
            print(f"                   Last improvement at iteration {last_improvement} "
                  f"({iterations_since} iterations ago)")

    def _report_improvement(self):
        """Report when improvement occurs"""
        current_best = self.best_objectives[-1]
        improvement_from_initial = ((self.initial_objective - current_best) / self.initial_objective * 100
                                    if self.initial_objective > 0 else 0)
        print(f"🎯 IMPROVEMENT at iteration {self.iteration_count}: "
              f"New best = {current_best:.2f} "
              f"(Total improvement: {improvement_from_initial:.1f}%)")

    def final_report(self):
        """Generate final optimization report"""
        if not self.best_objectives:
            print("No optimization data available")
            return

        initial_obj = self.initial_objective or self.best_objectives[0]
        final_obj = self.best_objectives[-1]
        improvement = initial_obj - final_obj
        improvement_pct = (improvement / initial_obj) * 100 if initial_obj > 0 else 0

        print(f"\n" + "=" * 60)
        print(f"OPTIMIZATION SUMMARY")
        print(f"=" * 60)
        print(f"Total iterations:     {self.iteration_count}")
        print(f"Initial objective:    {initial_obj:.2f}")
        print(f"Final objective:      {final_obj:.2f}")
        print(f"Total improvement:    {improvement:.2f} ({improvement_pct:.1f}%)")
        print(f"Number of improvements: {len(self.improvements)}")

        if self.improvements:
            print(f"Improvement iterations: {self.improvements}")
            if len(self.improvements) > 1:
                gaps = [self.improvements[i] - self.improvements[i - 1] for i in range(1, len(self.improvements))]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                print(f"Average gap between improvements: {avg_gap:.1f} iterations")

        print(f"=" * 60)


# FEASIBILITY-ENFORCING ACCEPTANCE CRITERION
class FeasibilityEnforcingAcceptance:
    """Custom acceptance criterion that rejects all infeasible solutions"""

    def __init__(self, base_criterion):
        self.base_criterion = base_criterion

    def __call__(self, current, candidate, random_state):
        # Never accept infeasible solutions
        if not candidate.is_feasible():
            return False

        # If candidate is feasible, use base criterion
        return self.base_criterion(current, candidate, random_state)

