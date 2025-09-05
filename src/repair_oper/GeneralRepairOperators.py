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

import numpy as np

from src.classes.AlnsProblem import PDPTWInstance, PDPTWSolution
from typing import List, Dict, Tuple

def regret_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Optimized regret insertion with caching and early termination"""
    """Improved regret insertion without problematic caching"""
    repaired = destroyed.copy()

    # Collect uninserted pickups
    all_pickups = {node['id'] for node in repaired.instance.nodes[1:] if node['demand'] > 0}
    inserted_pickups = set()
    for route in repaired.routes:
        inserted_pickups.update(node for node in route if node in all_pickups)

    uninserted = list(all_pickups - inserted_pickups)
    if not uninserted:
        return repaired

    while uninserted:
        best_insertion = None
        max_regret = -float('inf')

        for pickup in uninserted:
            delivery = repaired.instance.nodes[pickup]['delivery_index']
            insertion_costs = []

            # Pre-filter routes by capacity
            pair_demand = repaired.instance.nodes[pickup]['demand']
            candidate_routes = []
            for route_idx, route in enumerate(repaired.routes):
                route_load = sum(repaired.instance.nodes[n]['demand'] for n in route if n != 0)
                if route_load + pair_demand <= repaired.instance.capacity:
                    candidate_routes.append((route_idx, route))

            # Add new route option
            candidate_routes.append((len(repaired.routes), [0, 0]))

            # Evaluate insertions WITHOUT caching
            for route_idx, route in candidate_routes:
                is_new_route = route == [0, 0]
                route_best_cost = float('inf')
                route_best_pos = None

                # Limit search space for performance
                max_positions = min(len(route) + 1, 8)

                for i in range(1, max_positions):
                    for j in range(i + 1, max_positions + 1):
                        temp_route = route[:]
                        temp_route.insert(i, pickup)
                        temp_route.insert(j, delivery)

                        # DIRECT feasibility check without cache
                        if repaired._is_route_feasible(temp_route):
                            old_cost = 0 if is_new_route else sum(
                                repaired.instance.distances[route[k]][route[k + 1]]
                                for k in range(len(route) - 1)
                            )
                            new_cost = sum(
                                repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                for k in range(len(temp_route) - 1)
                            )
                            cost = new_cost - old_cost

                            if cost < route_best_cost:
                                route_best_cost = cost
                                route_best_pos = (route_idx, i, j)

                if route_best_pos:
                    insertion_costs.append((route_best_cost, *route_best_pos[1:]))

            # Calculate regret
            if len(insertion_costs) >= 2:
                insertion_costs.sort()
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                best_cost_info = (pickup, delivery, route_best_pos[0], *insertion_costs[0][1:])
            elif len(insertion_costs) == 1:
                regret = insertion_costs[0][0]
                best_cost_info = (pickup, delivery, route_best_pos[0], *insertion_costs[0][1:])
            else:
                continue  # No feasible insertion

            if regret > max_regret:
                max_regret = regret
                best_insertion = best_cost_info

        if best_insertion is None:
            break

        # Insert best pair
        pickup, delivery, route_idx, pickup_pos, delivery_pos = best_insertion
        uninserted.remove(pickup)

        if route_idx < len(repaired.routes):
            route = repaired.routes[route_idx]
            route.insert(pickup_pos, pickup)
            route.insert(delivery_pos, delivery)
        else:
            new_route = [0, pickup, delivery, 0]
            repaired.routes.append(new_route)

    repaired.calculate_objective()
    return repaired


def greedy_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Improved greedy insertion with stricter feasibility checking"""
    repaired = destroyed.copy()

    # Collect uninserted pickups
    all_pickups = {node['id'] for node in repaired.instance.nodes[1:] if node['demand'] > 0}
    inserted_pickups = set()
    for route in repaired.routes:
        inserted_pickups.update(node for node in route if node in all_pickups)

    uninserted = list(all_pickups - inserted_pickups)

    # Sort by time window urgency (earliest due date first)
    urgency_pairs = []
    for pickup in uninserted:
        pickup_info = repaired.instance.nodes[pickup]
        delivery_info = repaired.instance.nodes[pickup_info['delivery_index']]
        earliest_due = min(pickup_info['due_date'], delivery_info['due_date'])
        urgency_pairs.append((earliest_due, pickup))

    urgency_pairs.sort()
    uninserted = [pickup for _, pickup in urgency_pairs]

    max_failed_attempts = len(uninserted) * 2  # Prevent infinite loops
    failed_attempts = 0

    while uninserted and failed_attempts < max_failed_attempts:
        insertion_made = False

        for pickup_idx, pickup in enumerate(uninserted):
            delivery = repaired.instance.nodes[pickup]['delivery_index']
            best_insertion = None
            best_cost = float('inf')

            # Try existing routes
            for route_idx, route in enumerate(repaired.routes):
                # Check capacity first
                route_load = sum(repaired.instance.nodes[n]['demand'] for n in route if n != 0)
                pair_demand = repaired.instance.nodes[pickup]['demand']

                if route_load + pair_demand > repaired.instance.capacity:
                    continue

                # Try all position combinations
                for i in range(1, len(route) + 1):
                    for j in range(i + 1, len(route) + 2):
                        temp_route = route[:]
                        temp_route.insert(i, pickup)
                        temp_route.insert(j, delivery)

                        # STRICT feasibility check
                        if repaired._is_route_feasible_strict(temp_route):
                            old_cost = sum(repaired.instance.distances[route[k]][route[k + 1]]
                                           for k in range(len(route) - 1))
                            new_cost = sum(repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                           for k in range(len(temp_route) - 1))
                            cost = new_cost - old_cost

                            if cost < best_cost:
                                best_cost = cost
                                best_insertion = ('existing', route_idx, i, j)

            # Try new route
            new_route = [0, pickup, delivery, 0]
            if repaired._is_route_feasible_strict(new_route):
                cost = sum(repaired.instance.distances[new_route[k]][new_route[k + 1]]
                           for k in range(len(new_route) - 1))
                if cost < best_cost:
                    best_cost = cost
                    best_insertion = ('new', pickup, delivery)

            # Apply best insertion if found
            if best_insertion:
                if best_insertion[0] == 'existing':
                    _, route_idx, pickup_pos, delivery_pos = best_insertion
                    route = repaired.routes[route_idx]
                    route.insert(pickup_pos, pickup)
                    route.insert(delivery_pos, delivery)
                else:
                    _, pickup, delivery = best_insertion
                    repaired.routes.append([0, pickup, delivery, 0])

                uninserted.remove(pickup)
                insertion_made = True
                failed_attempts = 0
                break

        if not insertion_made:
            failed_attempts += 1

    # Final validation
    if not repaired.is_feasible():
        # If still infeasible, create separate routes for remaining pairs
        for pickup in uninserted[:]:
            delivery = repaired.instance.nodes[pickup]['delivery_index']
            new_route = [0, pickup, delivery, 0]
            if repaired._is_route_feasible_strict(new_route):
                repaired.routes.append(new_route)
                uninserted.remove(pickup)

    repaired.calculate_objective()
    return repaired


def k_regret_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState, k: int = 3) -> PDPTWSolution:
    """Optimized k-regret insertion with performance improvements"""
    repaired = destroyed.copy()

    # Collect uninserted pickups
    all_pickups = {node['id'] for node in repaired.instance.nodes[1:] if node['demand'] > 0}
    inserted_pickups = set()
    for route in repaired.routes:
        inserted_pickups.update(node for node in route if node in all_pickups)

    uninserted = list(all_pickups - inserted_pickups)
    if not uninserted:
        return repaired

    # Cache for expensive operations
    feasibility_cache = {}

    while uninserted:
        best_insertion = None
        max_regret = -float('inf')

        # Limit evaluation to most promising pairs
        evaluation_limit = min(len(uninserted), 8)  # Evaluate max 8 pairs per iteration
        candidate_pickups = random_state.choice(len(uninserted), evaluation_limit, replace=False)

        for pickup_idx in candidate_pickups:
            pickup = uninserted[pickup_idx]
            delivery = repaired.instance.nodes[pickup]['delivery_index']
            pair_demand = repaired.instance.nodes[pickup]['demand']
            insertion_costs = []

            # Pre-filter routes by capacity
            candidate_routes = []
            for route_idx, route in enumerate(repaired.routes):
                route_load = sum(repaired.instance.nodes[n]['demand'] for n in route if n != 0)
                if route_load + pair_demand <= repaired.instance.capacity:
                    candidate_routes.append((route_idx, route))

            # Add new route option
            candidate_routes.append((len(repaired.routes), [0, 0]))

            # Evaluate insertions with limited search
            for route_idx, route in candidate_routes:
                is_new_route = route == [0, 0]
                max_positions = min(len(route) + 1, 5)  # Further limit positions

                for i in range(1, max_positions):
                    for j in range(i + 1, max_positions + 1):
                        cache_key = (route_idx, pickup, delivery, i, j)

                        if cache_key not in feasibility_cache:
                            temp_route = route[:]
                            temp_route.insert(i, pickup)
                            temp_route.insert(j, delivery)
                            feasibility_cache[cache_key] = repaired._is_route_feasible(temp_route)

                        if feasibility_cache[cache_key]:
                            temp_route = route[:]
                            temp_route.insert(i, pickup)
                            temp_route.insert(j, delivery)

                            old_cost = 0 if is_new_route else sum(
                                repaired.instance.distances[route[m]][route[m + 1]]
                                for m in range(len(route) - 1)
                            )
                            new_cost = sum(
                                repaired.instance.distances[temp_route[m]][temp_route[m + 1]]
                                for m in range(len(temp_route) - 1)
                            )
                            insertion_cost = new_cost - old_cost
                            insertion_costs.append((insertion_cost, route_idx, i, j))

            if not insertion_costs:
                continue

            # Calculate k-regret efficiently
            insertion_costs.sort(key=lambda x: x[0])
            best_cost = insertion_costs[0][0]

            if len(insertion_costs) >= k + 1:
                k_cost_sum = sum(cost for cost, _, _, _ in insertion_costs[1:k + 1])
                regret = k_cost_sum - best_cost * k
            else:
                # Use available costs
                available_k = min(k, len(insertion_costs) - 1)
                if available_k > 0:
                    k_cost_sum = sum(cost for cost, _, _, _ in insertion_costs[1:available_k + 1])
                    regret = k_cost_sum - best_cost * available_k
                else:
                    regret = best_cost

            if regret > max_regret:
                max_regret = regret
                best_insertion = (pickup, delivery, *insertion_costs[0][1:])

        if best_insertion is None:
            break

        # Insert best pair
        pickup, delivery, route_idx, pickup_pos, delivery_pos = best_insertion
        uninserted.remove(pickup)

        if route_idx < len(repaired.routes):
            route = repaired.routes[route_idx]
            route.insert(pickup_pos, pickup)
            route.insert(delivery_pos, delivery)
        else:
            new_route = [0, pickup, delivery, 0]
            repaired.routes.append(new_route)

        # Clear cache periodically to prevent memory issues
        if len(feasibility_cache) > 1000:
            feasibility_cache.clear()

    repaired.calculate_objective()
    return repaired


def grasp_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState, rcl_size: int = 3) -> PDPTWSolution:
    """Optimized GRASP insertion with reduced RCL size and better filtering"""
    repaired = destroyed.copy()

    # Collect uninserted pickups
    all_pickups = {node['id'] for node in repaired.instance.nodes[1:] if node['demand'] > 0}
    inserted_pickups = set()
    for route in repaired.routes:
        inserted_pickups.update(node for node in route if node in all_pickups)

    uninserted = list(all_pickups - inserted_pickups)
    if not uninserted:
        return repaired

    while uninserted:
        # Select random pair to try inserting
        pickup_idx = random_state.choice(len(uninserted))
        pickup = uninserted[pickup_idx]
        delivery = repaired.instance.nodes[pickup]['delivery_index']
        pair_demand = repaired.instance.nodes[pickup]['demand']
        rcl = []

        # Pre-filter routes by capacity
        candidate_routes = []
        for route_idx, route in enumerate(repaired.routes):
            route_load = sum(repaired.instance.nodes[n]['demand'] for n in route if n != 0)
            if route_load + pair_demand <= repaired.instance.capacity:
                candidate_routes.append((route_idx, route))

        # Add new route option
        candidate_routes.append((len(repaired.routes), [0, 0]))

        # Evaluate insertions with limited positions
        for route_idx, route in candidate_routes:
            is_new_route = route == [0, 0]
            max_positions = min(len(route) + 1, 4)  # Reduced search space

            for i in range(1, max_positions):
                for j in range(i + 1, max_positions + 1):
                    temp_route = route[:]
                    temp_route.insert(i, pickup)
                    temp_route.insert(j, delivery)

                    if repaired._is_route_feasible(temp_route):
                        old_cost = 0 if is_new_route else sum(
                            repaired.instance.distances[route[m]][route[m + 1]]
                            for m in range(len(route) - 1)
                        )
                        new_cost = sum(
                            repaired.instance.distances[temp_route[m]][temp_route[m + 1]]
                            for m in range(len(temp_route) - 1)
                        )
                        insertion_cost = new_cost - old_cost
                        rcl.append((insertion_cost, route_idx, i, j))

        if not rcl:
            uninserted.pop(pickup_idx)
            continue

        # Select from reduced RCL
        rcl.sort(key=lambda x: x[0])
        rcl = rcl[:min(rcl_size, len(rcl))]  # Reduced RCL size
        selected_idx = random_state.choice(len(rcl))
        _, route_idx, pickup_pos, delivery_pos = rcl[selected_idx]

        # Insert pair
        if route_idx < len(repaired.routes):
            route = repaired.routes[route_idx]
            route.insert(pickup_pos, pickup)
            route.insert(delivery_pos, delivery)
        else:
            new_route = [0, pickup, delivery, 0]
            repaired.routes.append(new_route)

        uninserted.pop(pickup_idx)

    repaired.calculate_objective()
    return repaired

class RouteRegretInsertionWithNoise:
    """
    I12: Customer with highest route regret - at best position - with noise

    This operator:
    1. Calculates route regret for each unassigned pickup-delivery pair
    2. Route regret = difference between best and second-best route insertion costs
    3. Adds noise to the regret values for diversification
    4. Inserts the pair with highest (noisy) regret at its best position
    """

    def __init__(self, noise_factor: float = 0.1):
        self.name = "route_regret_insertion_with_noise"
        self.noise_factor = noise_factor  # Controls amount of noise added

    def calculate_insertion_cost(self, route: List[int], pickup_pos: int, delivery_pos: int,
                                 pickup_id: int, delivery_id: int,
                                 distances: np.ndarray) -> float:
        """Calculate the cost increase of inserting pickup-delivery pair into route"""
        if pickup_pos >= delivery_pos:
            return float('inf')

        # Calculate original route cost
        original_cost = sum(distances[route[i]][route[i + 1]] for i in range(len(route) - 1))

        # Create new route with insertions
        new_route = route[:]
        new_route.insert(pickup_pos, pickup_id)
        new_route.insert(delivery_pos, delivery_id)

        # Calculate new route cost
        new_cost = sum(distances[new_route[i]][new_route[i + 1]] for i in range(len(new_route) - 1))

        return new_cost - original_cost

    def find_best_insertion_in_route(self, route: List[int], pickup_id: int, delivery_id: int,
                                     solution: PDPTWSolution) -> Tuple[float, int, int]:
        """Find best insertion positions for pickup-delivery pair in a route"""
        best_cost = float('inf')
        best_pickup_pos = -1
        best_delivery_pos = -1

        # Try all valid position combinations
        for pickup_pos in range(1, len(route)):
            for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                # Create temporary route
                temp_route = route[:]
                temp_route.insert(pickup_pos, pickup_id)
                temp_route.insert(delivery_pos, delivery_id)

                # Check feasibility
                if solution._is_route_feasible(temp_route):
                    cost = self.calculate_insertion_cost(route, pickup_pos, delivery_pos,
                                                         pickup_id, delivery_id,
                                                         solution.instance.distances)

                    if cost < best_cost:
                        best_cost = cost
                        best_pickup_pos = pickup_pos
                        best_delivery_pos = delivery_pos

        return best_cost, best_pickup_pos, best_delivery_pos

    def calculate_route_regret(self, pickup_id: int, delivery_id: int,
                               solution: PDPTWSolution) -> Tuple[float, List[int], int, int]:
        """Calculate route regret for a pickup-delivery pair"""
        route_costs = []
        route_positions = []

        # Find best insertion cost for each route
        for route in solution.routes:
            cost, pickup_pos, delivery_pos = self.find_best_insertion_in_route(
                route, pickup_id, delivery_id, solution)

            if cost < float('inf'):
                route_costs.append(cost)
                route_positions.append((route, pickup_pos, delivery_pos))

        if len(route_costs) < 1:
            return 0.0, None, -1, -1

        # Calculate regret
        route_costs_sorted = sorted(route_costs)
        if len(route_costs_sorted) >= 2:
            regret = route_costs_sorted[1] - route_costs_sorted[0]
        else:
            regret = route_costs_sorted[0]  # Only one feasible route

        # Find the route with minimum cost
        min_cost_idx = route_costs.index(min(route_costs))
        best_route, best_pickup_pos, best_delivery_pos = route_positions[min_cost_idx]

        return regret, best_route, best_pickup_pos, best_delivery_pos

    def __call__(self, current_state: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
        """Insert pickup-delivery pairs using route regret with noise"""
        repaired = current_state.copy()

        # Get all unassigned pickup-delivery pairs
        all_pickups = set()
        for node in repaired.instance.nodes[1:]:
            if node['demand'] > 0:
                all_pickups.add(node['id'])

        assigned_pickups = set()
        for route in repaired.routes:
            for node in route:
                if node in all_pickups:
                    assigned_pickups.add(node)

        unassigned_pairs = []
        for pickup_id in all_pickups - assigned_pickups:
            delivery_id = repaired.instance.nodes[pickup_id]['delivery_index']
            if delivery_id > 0:
                unassigned_pairs.append((pickup_id, delivery_id))

        # Insert pairs one by one using route regret with noise
        while unassigned_pairs:
            best_insertion = None
            max_noisy_regret = -float('inf')

            # Calculate regret for each unassigned pair
            for pickup_id, delivery_id in unassigned_pairs:
                regret, best_route, pickup_pos, delivery_pos = self.calculate_route_regret(
                    pickup_id, delivery_id, repaired)

                if best_route is not None:
                    # Add noise to regret value
                    noise = random_state.normal(0, self.noise_factor * abs(regret))
                    noisy_regret = regret + noise

                    if noisy_regret > max_noisy_regret:
                        max_noisy_regret = noisy_regret
                        best_insertion = (pickup_id, delivery_id, best_route, pickup_pos, delivery_pos)

            # Insert the pair with highest noisy regret
            if best_insertion:
                pickup_id, delivery_id, route, pickup_pos, delivery_pos = best_insertion
                route.insert(pickup_pos, pickup_id)
                route.insert(delivery_pos, delivery_id)
                unassigned_pairs.remove((pickup_id, delivery_id))
            else:
                # No feasible insertion found, create new route if possible
                if unassigned_pairs and len(repaired.routes) < repaired.instance.vehicle_number:
                    pickup_id, delivery_id = unassigned_pairs[0]
                    new_route = [0, pickup_id, delivery_id, 0]
                    if repaired._is_route_feasible(new_route):
                        repaired.routes.append(new_route)
                        unassigned_pairs.remove((pickup_id, delivery_id))
                    else:
                        break  # Cannot create feasible route
                else:
                    break  # Cannot insert remaining pairs

        repaired.calculate_objective()
        return repaired

def route_regret_insertion_with_noise(current_state: PDPTWSolution,
                                      random_state: np.random.RandomState) -> PDPTWSolution:
    """Wrapper function for route regret insertion with noise operator"""
    operator = RouteRegretInsertionWithNoise(noise_factor=0.1)
    return operator(current_state, random_state)

