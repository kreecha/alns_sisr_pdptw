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
import os
import time
import itertools
import math
import random

from datetime import timedelta
from src.classes.AlnsProblem import PDPTWInstance, PDPTWSolution
from typing import List, Dict, Tuple


class InternalRoute:
    """Internal route representation for the advanced solver"""

    def __init__(self):
        self.customers = [0]  # Start at depot
        self.load = 0
        self.distance = 0.0

class AdvancedPDPTWSolver:
    """Advanced PDPTW solver that works with existing PDPTWInstance structure"""

    def __init__(self, instance, random_seed: int = 42):
        self.instance = instance
        self.random_state = np.random.RandomState(random_seed)
        random.seed(random_seed)

        # Extract problem characteristics
        self.n_vehicles = instance.vehicle_number
        self.capacity = instance.capacity
        self.nodes = instance.nodes
        self.distances = instance.distances

    def get_pickup_delivery_pairs(self) -> List[Tuple[int, int]]:
        """Get all pickup-delivery pairs from the instance"""
        pairs = []
        for node in self.nodes:
            if node['demand'] > 0:  # Pickup node
                delivery_id = node['delivery_index']
                if delivery_id > 0:
                    pairs.append((node['id'], delivery_id))
        return pairs

    def is_pickup(self, customer_id: int) -> bool:
        """Check if customer is a pickup location"""
        return self.nodes[customer_id]['demand'] > 0

    def is_delivery(self, customer_id: int) -> bool:
        """Check if customer is a delivery location"""
        return self.nodes[customer_id]['demand'] < 0

    def get_distance(self, i: int, j: int) -> float:
        """Get distance between customers i and j"""
        return self.distances[i][j]

    def calculate_route_distance(self, customers: List[int]) -> float:
        """Calculate total distance for a route"""
        total_distance = 0.0
        for i in range(len(customers) - 1):
            total_distance += self.get_distance(customers[i], customers[i + 1])
        return total_distance

    def is_route_feasible(self, customers: List[int]) -> bool:
        """Check if a route is feasible using existing PDPTWSolution logic"""
        # Create temporary PDPTWSolution to use existing feasibility check
        temp_solution = PDPTWSolution(self.instance)
        temp_solution.routes = [customers]
        return temp_solution._is_route_feasible(customers)

    def calculate_insertion_cost(self, route: InternalRoute, pickup: int, delivery: int,
                                 pickup_pos: int, delivery_pos: int) -> float:
        """Calculate cost of inserting pickup-delivery pair"""
        if pickup_pos > delivery_pos:
            return float('inf')

        customers = route.customers.copy()
        customers.insert(pickup_pos, pickup)
        customers.insert(delivery_pos + 1, delivery)

        # Calculate new route cost
        new_cost = self.calculate_route_distance(customers)
        return new_cost - route.distance

    def is_insertion_feasible(self, route: InternalRoute, pickup: int, delivery: int,
                              pickup_pos: int, delivery_pos: int) -> bool:
        """Check if insertion is feasible"""
        if pickup_pos > delivery_pos:
            return False

        # Check capacity
        pickup_demand = self.nodes[pickup]['demand']
        if route.load + pickup_demand > self.capacity:
            return False

        # Create temporary route for feasibility checking
        temp_customers = route.customers.copy()
        temp_customers.insert(pickup_pos, pickup)
        temp_customers.insert(delivery_pos + 1, delivery)

        return self.is_route_feasible(temp_customers)

    def calculate_removal_saving(self, route, pickup_idx, delivery_idx):
        """Fast removal saving calculation"""
        customers = route.customers

        # Simple distance-based saving (avoid complex calculations)
        if pickup_idx > 0 and pickup_idx < len(customers) - 1:
            pickup_saving = (self.get_distance(customers[pickup_idx - 1], customers[pickup_idx]) +
                             self.get_distance(customers[pickup_idx], customers[pickup_idx + 1]) -
                             self.get_distance(customers[pickup_idx - 1], customers[pickup_idx + 1]))
        else:
            pickup_saving = 0

        if delivery_idx > 0 and delivery_idx < len(customers) - 1:
            delivery_saving = (self.get_distance(customers[delivery_idx - 1], customers[delivery_idx]) +
                               self.get_distance(customers[delivery_idx], customers[delivery_idx + 1]) -
                               self.get_distance(customers[delivery_idx - 1], customers[delivery_idx + 1]))
        else:
            delivery_saving = 0

        return pickup_saving + delivery_saving

    def sequential_insertion_heuristic(self) -> List[InternalRoute]:
        """Optimized sequential insertion with early termination"""

        routes = []
        unrouted_pairs = self.get_pickup_delivery_pairs()

        # Sort pairs by pickup ready time (earliest first)
        unrouted_pairs.sort(key=lambda pair: self.nodes[pair[0]]['ready_time'])
        print("sequential_insertion_heuristic called ...")
        for pickup_id, delivery_id in unrouted_pairs:
            best_route = None
            best_cost = float('inf')
            best_positions = None

            pair_demand = self.nodes[pickup_id]['demand']

            # OPTIMIZATION 1: Pre-filter routes by capacity
            candidate_routes = [route for route in routes if route.load + pair_demand <= self.capacity]

            # Try inserting in existing routes
            for route in candidate_routes:
                # print("for 2")
                route_best_cost = float('inf')
                route_best_positions = None

                # OPTIMIZATION 2: Limit position search based on time windows
                max_pickup_pos = min(len(route.customers), 8)  # Limit search space

                for pickup_pos in range(1, max_pickup_pos + 1):
                    #print("for 3")
                    # OPTIMIZATION 3: Early time window check
                    if not self._can_insert_at_position(route, pickup_id, pickup_pos):
                        continue

                    # Calculate reasonable delivery positions
                    min_delivery_pos = pickup_pos
                    max_delivery_pos = min(pickup_pos + 6, len(route.customers))  # Limit delivery search

                    for delivery_pos in range(min_delivery_pos, max_delivery_pos + 1):

                        # OPTIMIZATION 4: Quick feasibility check before expensive validation
                        if not self._quick_insertion_check(route, pickup_id, delivery_id, pickup_pos, delivery_pos):
                            continue
                        #print("for 4")
                        # Only do expensive check if quick check passes
                        # is_insertion_feasible_optimized(route, pickup_id, delivery_id, pickup_pos, delivery_pos)
                        if self.is_insertion_feasible_optimized(route, pickup_id, delivery_id, pickup_pos, delivery_pos):

                            cost = self.calculate_insertion_cost(route, pickup_id, delivery_id, pickup_pos,
                                                                 delivery_pos)
                            if cost < route_best_cost:
                                route_best_cost = cost
                                route_best_positions = (pickup_pos, delivery_pos)

                                # OPTIMIZATION 5: Early termination if very good solution found
                                if cost < 10.0:  # Adjust threshold based on your problem scale
                                    break

                    if route_best_positions and route_best_cost < 10.0:
                        break  # Early termination for route

                # Update global best for this pair
                if route_best_cost < best_cost:
                    best_cost = route_best_cost
                    best_route = route
                    best_positions = route_best_positions

            # Insert into best route or create new one
            if best_route is not None:
                pickup_pos, delivery_pos = best_positions
                best_route.customers.insert(pickup_pos, pickup_id)
                best_route.customers.insert(delivery_pos + 1, delivery_id)
                best_route.load += pair_demand
                best_route.distance += best_cost
            else:
                # Create new route
                new_route = InternalRoute()
                new_route.customers = [0, pickup_id, delivery_id, 0]
                new_route.load = pair_demand
                new_route.distance = self.calculate_route_distance(new_route.customers)
                routes.append(new_route)

        return routes

    def _can_insert_at_position(self, route: InternalRoute, customer_id: int, position: int) -> bool:
        """Quick check if customer can be inserted at position based on time windows"""

        if position <= 0 or position > len(route.customers):
            return False

        customer = self.nodes[customer_id]

        # Check against previous customer
        if position > 0:
            prev_customer_id = route.customers[position - 1]
            prev_customer = self.nodes[prev_customer_id]

            earliest_arrival = (prev_customer.get('ready_time', 0) +
                                prev_customer.get('service_time', 0) +
                                self.get_distance(prev_customer_id, customer_id))

            if earliest_arrival > customer['due_date']:
                return False

        # Check against next customer
        if position < len(route.customers):
            next_customer_id = route.customers[position]
            next_customer = self.nodes[next_customer_id]

            earliest_departure = max(customer['ready_time'], 0) + customer.get('service_time', 0)
            travel_to_next = self.get_distance(customer_id, next_customer_id)
            arrival_at_next = earliest_departure + travel_to_next

            if arrival_at_next > next_customer['due_date']:
                return False

        return True

    def _quick_insertion_check(self, route: InternalRoute, pickup_id: int, delivery_id: int,
                               pickup_pos: int, delivery_pos: int) -> bool:
        """Quick feasibility check without full route simulation"""

        # Basic precedence
        if pickup_pos > delivery_pos:
            return False

        # Check if pickup and delivery time windows are compatible
        pickup_customer = self.nodes[pickup_id]
        delivery_customer = self.nodes[delivery_id]

        min_travel_time = self.get_distance(pickup_id, delivery_id)
        earliest_delivery_time = pickup_customer['ready_time'] + pickup_customer.get('service_time',
                                                                                     0) + min_travel_time

        if earliest_delivery_time > delivery_customer['due_date']:
            return False

        return True


    def regret_insertion_heuristic(self) -> List[InternalRoute]:
        """Regret-based insertion heuristic"""
        routes = []
        unrouted_pairs = self.get_pickup_delivery_pairs()

        # Caches to avoid redundant calculations
        feasibility_cache = {}
        cost_cache = {}

        while unrouted_pairs:
            best_pair = None
            best_route = None
            best_positions = None
            best_regret = -1

            # OPTIMIZATION 1: Pre-compute pair characteristics to skip expensive calculations
            pair_feasibility = {}  # pair -> list of (route, cost, positions)

            for pickup_id, delivery_id in unrouted_pairs:
                pair_demand = self.nodes[pickup_id]['demand']
                feasible_insertions = []
                costs = []

                # OPTIMIZATION 2: Pre-filter routes by capacity
                candidate_routes = [route for route in routes if route.load + pair_demand <= self.capacity]

                # OPTIMIZATION 3: Early termination if no routes have capacity
                if not candidate_routes:
                    # Must create new route - set infinite regret to prioritize
                    pair_feasibility[(pickup_id, delivery_id)] = ([], [float('inf')])
                    continue

                for route in candidate_routes:
                    route_best_cost = float('inf')
                    route_best_pos = None

                    # OPTIMIZATION 4: Limit position search based on route size
                    max_pickup_pos = min(len(route.customers), 8)  # Limit search space

                    for pickup_pos in range(1, max_pickup_pos + 1):
                        # OPTIMIZATION 5: Early time window check
                        if not self._can_insert_at_position(route, pickup_id, pickup_pos):
                            continue

                        # Limit delivery position search
                        max_delivery_pos = min(pickup_pos + 6, len(route.customers))

                        for delivery_pos in range(pickup_pos, max_delivery_pos + 1):
                            cache_key = (id(route), pickup_id, delivery_id, pickup_pos, delivery_pos)

                            # OPTIMIZATION 6: Use cached feasibility if available
                            if cache_key not in feasibility_cache:
                                # Quick check first
                                if not self._quick_insertion_check(route, pickup_id, delivery_id, pickup_pos,
                                                                   delivery_pos):
                                    feasibility_cache[cache_key] = False
                                else:
                                    feasibility_cache[cache_key] = self.is_insertion_feasible_optimized(
                                        route, pickup_id, delivery_id, pickup_pos, delivery_pos)

                            if feasibility_cache[cache_key]:
                                # Calculate cost only if feasible
                                if cache_key not in cost_cache:
                                    cost_cache[cache_key] = self.calculate_insertion_cost(
                                        route, pickup_id, delivery_id, pickup_pos, delivery_pos)

                                cost = cost_cache[cache_key]
                                if cost < route_best_cost:
                                    route_best_cost = cost
                                    route_best_pos = (route, pickup_pos, delivery_pos)

                            # OPTIMIZATION 7: Early termination for very good solutions
                            if route_best_cost < 5.0:  # Adjust threshold based on your problem
                                break

                        if route_best_cost < 5.0:
                            break

                    if route_best_pos is not None:
                        costs.append(route_best_cost)
                        feasible_insertions.append(route_best_pos)

                pair_feasibility[(pickup_id, delivery_id)] = (feasible_insertions, costs)

            # OPTIMIZATION 8: Calculate regret only for pairs with multiple options
            for pickup_id, delivery_id in unrouted_pairs:
                feasible_insertions, costs = pair_feasibility[(pickup_id, delivery_id)]

                # Calculate regret
                if len(costs) >= 2:
                    costs_sorted = sorted(costs)
                    regret = costs_sorted[1] - costs_sorted[0]
                elif len(costs) == 1:
                    regret = costs[0]  # Only one option available
                else:
                    regret = float('inf')  # Must create new route - highest priority

                if regret > best_regret:
                    best_regret = regret
                    best_pair = (pickup_id, delivery_id)

                    if feasible_insertions:
                        # Find the best insertion among feasible options
                        best_insertion_idx = costs.index(min(costs))
                        best_route_info = feasible_insertions[best_insertion_idx]
                        best_route, pickup_pos, delivery_pos = best_route_info
                        best_positions = (pickup_pos, delivery_pos)
                    else:
                        best_route = None

            # Insert best pair
            if best_pair:
                pickup_id, delivery_id = best_pair
                unrouted_pairs.remove(best_pair)

                if best_route is not None:
                    # Insert into existing route
                    pickup_pos, delivery_pos = best_positions
                    best_route.customers.insert(pickup_pos, pickup_id)
                    best_route.customers.insert(delivery_pos + 1, delivery_id)
                    best_route.load += self.nodes[pickup_id]['demand']

                    # Calculate actual cost increase (recalculate to ensure accuracy)
                    cost_increase = self.calculate_insertion_cost(best_route, pickup_id, delivery_id,
                                                                  pickup_pos, delivery_pos)
                    best_route.distance += cost_increase
                else:
                    # Create new route
                    new_route = InternalRoute()
                    new_route.customers = [0, pickup_id, delivery_id, 0]
                    new_route.load = self.nodes[pickup_id]['demand']
                    new_route.distance = self.calculate_route_distance(new_route.customers)
                    routes.append(new_route)

                # OPTIMIZATION 9: Clear relevant cache entries to avoid stale data
                keys_to_remove = [k for k in feasibility_cache.keys()
                                  if k[1] == pickup_id or k[2] == delivery_id]
                for key in keys_to_remove:
                    feasibility_cache.pop(key, None)
                    cost_cache.pop(key, None)

        return routes


    def push_forward_insertion_heuristic(self) -> List[InternalRoute]:
        """
        Push-forward insertion heuristic (PFIH), performance issue can happen...

        Note: poor performance
        :return:
        """
        # Start with individual routes
        pairs = self.get_pickup_delivery_pairs()
        routes = []
        for pickup_id, delivery_id in pairs:
            route = InternalRoute()
            route.customers = [0, pickup_id, delivery_id, 0]
            route.load = self.nodes[pickup_id]['demand']
            route.distance = self.calculate_route_distance(route.customers)
            routes.append(route)

        # CRITICAL FIX 1: Limit iterations to prevent infinite loops
        max_iterations = 5  # Instead of while True
        distance_cache = {}  # Cache distance calculations

        for iteration in range(max_iterations):
            best_move = None
            max_savings = 0.0
            moves_tested = 0
            max_moves_per_iteration = 1000  # CRITICAL: Limit search space

            # CRITICAL FIX 2: Early termination if no improvement
            if iteration > 0 and max_savings == 0:
                break

            # Find best move with limited search
            for i in range(len(routes)):
                if moves_tested >= max_moves_per_iteration:
                    break

                route_i = routes[i]
                if len(route_i.customers) <= 2:  # Skip empty routes
                    continue

                # CRITICAL FIX 3: Only test promising routes (capacity-based)
                for j in range(len(routes)):
                    if i == j or routes[j].load + route_i.load > self.capacity:
                        continue

                    route_j = routes[j]

                    # CRITICAL FIX 4: Test only a few pickup-delivery pairs per route
                    tested_pairs = 0
                    max_pairs_per_route = 3

                    for pickup_idx in range(1, len(route_i.customers) - 1):
                        if tested_pairs >= max_pairs_per_route:
                            break

                        customer_id = route_i.customers[pickup_idx]
                        if not self.is_pickup(customer_id):
                            continue

                        delivery_id = self.nodes[customer_id]['delivery_index']
                        if delivery_id not in route_i.customers:
                            continue

                        delivery_idx = route_i.customers.index(delivery_id)

                        # CRITICAL FIX 5: Limit insertion positions tested
                        max_positions = min(5, len(route_j.customers))

                        for pickup_pos in range(1, max_positions):
                            for delivery_pos in range(pickup_pos, max_positions):
                                moves_tested += 1

                                # CRITICAL FIX 6: Quick feasibility check first
                                if not self._quick_move_feasible(route_j, customer_id, delivery_id,
                                                                 pickup_pos, delivery_pos):
                                    continue

                                if self.is_insertion_feasible(route_j, customer_id, delivery_id,
                                                              pickup_pos, delivery_pos):

                                    # Use cached distances
                                    cache_key = (pickup_idx, delivery_idx, pickup_pos, delivery_pos)
                                    if cache_key not in distance_cache:
                                        removal_saving = self._calculate_removal_saving_fast(
                                            route_i, pickup_idx, delivery_idx)
                                        insertion_cost = self.calculate_insertion_cost(
                                            route_j, customer_id, delivery_id, pickup_pos, delivery_pos)
                                        distance_cache[cache_key] = (removal_saving, insertion_cost)
                                    else:
                                        removal_saving, insertion_cost = distance_cache[cache_key]

                                    net_savings = removal_saving - insertion_cost
                                    if net_savings > max_savings:
                                        max_savings = net_savings
                                        best_move = {
                                            'from_route_idx': i, 'to_route_idx': j,
                                            'pickup_id': customer_id, 'delivery_id': delivery_id,
                                            'pickup_pos': pickup_pos, 'delivery_pos': delivery_pos,
                                            'pickup_idx_i': pickup_idx, 'delivery_idx_i': delivery_idx
                                        }

                        tested_pairs += 1

            # CRITICAL FIX 7: Higher threshold for moves
            if best_move and max_savings > 5.0:  # Require significant improvement
                # Perform the move (same logic as before)
                i = best_move['from_route_idx']
                j = best_move['to_route_idx']
                # ... move implementation ...
            else:
                break  # No good moves found, terminate

        # Remove empty routes
        routes = [route for route in routes if len(route.customers) > 2]
        return routes

    def _quick_move_feasible(self, route, pickup_id, delivery_id, pickup_pos, delivery_pos):
        """Ultra-fast feasibility check to avoid expensive validation"""

        # Basic checks
        if pickup_pos > delivery_pos:
            return False

        # Capacity check
        pair_demand = self.nodes[pickup_id]['demand']
        if route.load + pair_demand > self.capacity:
            return False

        # Quick time window compatibility
        pickup_ready = self.nodes[pickup_id]['ready_time']
        delivery_due = self.nodes[delivery_id]['due_date']
        min_service_time = self.nodes[pickup_id].get('service_time', 0) + \
                           self.get_distance(pickup_id, delivery_id)

        if pickup_ready + min_service_time > delivery_due:
            return False

        return True

    def cluster_first_route_second(self) -> List[InternalRoute]:
        """Cluster-first, route-second heuristic"""
        pairs = self.get_pickup_delivery_pairs()

        # OPTIMIZATION 1: Cache distance calculations
        distance_cache = {}

        def cached_distance(i, j):
            if (i, j) not in distance_cache:
                distance_cache[(i, j)] = self.get_distance(i, j)
            return distance_cache[(i, j)]

        clusters = []
        unassigned_pairs = pairs.copy()

        while unassigned_pairs:
            # Start new cluster with pair having earliest pickup time
            unassigned_pairs.sort(key=lambda p: self.nodes[p[0]]['ready_time'])
            seed_pair = unassigned_pairs.pop(0)
            cluster = [seed_pair]
            cluster_demand = self.nodes[seed_pair[0]]['demand']

            # OPTIMIZATION 2: Pre-calculate cluster center once
            center_x = (self.nodes[seed_pair[0]]['x'] + self.nodes[seed_pair[1]]['x']) / 2
            center_y = (self.nodes[seed_pair[0]]['y'] + self.nodes[seed_pair[1]]['y']) / 2

            # OPTIMIZATION 3: Pre-filter by capacity before distance calculations
            capacity_candidates = [
                (pickup_id, delivery_id) for pickup_id, delivery_id in unassigned_pairs
                if cluster_demand + self.nodes[pickup_id]['demand'] <= self.capacity
            ]

            # Add compatible pairs to cluster
            i = 0
            while i < len(capacity_candidates):
                pickup_id, delivery_id = capacity_candidates[i]

                # OPTIMIZATION 4: Use cached distance
                pair_center_x = (self.nodes[pickup_id]['x'] + self.nodes[delivery_id]['x']) / 2
                pair_center_y = (self.nodes[pickup_id]['y'] + self.nodes[delivery_id]['y']) / 2

                distance_to_cluster = math.sqrt((center_x - pair_center_x) ** 2 +
                                                (center_y - pair_center_y) ** 2)

                if distance_to_cluster < 50:  # Same threshold
                    # Time compatibility check
                    earliest_pickup = min(self.nodes[p[0]]['ready_time'] for p in cluster)
                    latest_delivery = max(self.nodes[p[1]]['due_date'] for p in cluster)

                    if (self.nodes[pickup_id]['ready_time'] <= latest_delivery and
                            self.nodes[delivery_id]['due_date'] >= earliest_pickup):
                        cluster.append((pickup_id, delivery_id))
                        cluster_demand += self.nodes[pickup_id]['demand']
                        unassigned_pairs.remove((pickup_id, delivery_id))
                        capacity_candidates.pop(i)

                        # OPTIMIZATION 5: Incremental center update (more efficient)
                        total_customers = len(cluster) * 2
                        center_x = ((center_x * (total_customers - 2)) +
                                    self.nodes[pickup_id]['x'] + self.nodes[delivery_id]['x']) / total_customers
                        center_y = ((center_y * (total_customers - 2)) +
                                    self.nodes[pickup_id]['y'] + self.nodes[delivery_id]['y']) / total_customers
                        continue

                i += 1

            clusters.append(cluster)

        # Route each cluster (same as original)
        routes = []
        for cluster in clusters:
            route = self.route_cluster_optimally(cluster, cached_distance)
            if route:
                routes.append(route)

        return routes


    def route_cluster_optimally(self, cluster_pairs: List[Tuple[int, int]]) -> InternalRoute:
        """Route a cluster of pickup-delivery pairs optimally"""
        if not cluster_pairs:
            return None

        # Collect all customers in cluster
        customers_in_cluster = []
        for pickup_id, delivery_id in cluster_pairs:
            customers_in_cluster.extend([pickup_id, delivery_id])

        # Use modified nearest neighbor with precedence constraints
        route = InternalRoute()
        route.customers = [0]
        unvisited = set(customers_in_cluster)
        visited_pickups = set()
        current_pos = 0
        current_time = 0.0

        while unvisited:
            best_next = None
            best_cost = float('inf')

            for customer_id in unvisited:
                # Check precedence constraints
                if self.is_delivery(customer_id):
                    pickup_id = self.nodes[customer_id]['pickup_index']
                    if pickup_id not in visited_pickups:
                        continue

                # Check feasibility
                travel_time = self.get_distance(current_pos, customer_id)
                arrival_time = current_time + travel_time

                # Check capacity
                temp_load = route.load + self.nodes[customer_id]['demand']
                if temp_load > self.capacity:
                    continue

                # Check time window
                customer = self.nodes[customer_id]
                actual_arrival = max(arrival_time, customer['ready_time'])
                if actual_arrival > customer['due_date']:
                    continue

                # Calculate cost (distance + time penalty)
                waiting_time = max(0, customer['ready_time'] - arrival_time)
                cost = travel_time + waiting_time * 0.1

                if cost < best_cost:
                    best_cost = cost
                    best_next = customer_id

            if best_next is None:
                break

            # Add customer to route
            route.customers.append(best_next)
            unvisited.remove(best_next)

            if self.is_pickup(best_next):
                visited_pickups.add(best_next)

            route.load += self.nodes[best_next]['demand']

            # Update time and position
            travel_time = self.get_distance(current_pos, best_next)
            current_time += travel_time
            customer = self.nodes[best_next]
            current_time = max(current_time, customer['ready_time']) + customer['service_time']
            current_pos = best_next

        # Return to depot
        route.customers.append(0)
        route.distance = self.calculate_route_distance(route.customers)

        return route if len(unvisited) == 0 else None

    def savings_algorithm_pdptw(self) -> List[InternalRoute]:
        """
        Clarke-Wright Savings adapted for PDPTW
        Note: poor performance
        :return:
        """


        pairs = self.get_pickup_delivery_pairs()

        # Initialize: one route per pickup-delivery pair
        routes = []
        for pickup_id, delivery_id in pairs:
            route = InternalRoute()
            route.customers = [0, pickup_id, delivery_id, 0]
            route.load = self.nodes[pickup_id]['demand']
            route.distance = self.calculate_route_distance(route.customers)
            routes.append(route)

        # Calculate savings between all pairs of routes
        savings_list = []

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route_i, route_j = routes[i], routes[j]

                # Skip if combined capacity exceeds limit
                if route_i.load + route_j.load > self.capacity:
                    continue

                # Calculate savings for both merge orders
                savings_ij = self._calculate_merge_savings(route_i, route_j)
                savings_ji = self._calculate_merge_savings(route_j, route_i)

                # Store the better option
                if savings_ij >= savings_ji and savings_ij > 0:
                    savings_list.append((savings_ij, i, j, 'ij'))
                elif savings_ji > 0:
                    savings_list.append((savings_ji, i, j, 'ji'))

        # Sort savings in descending order
        savings_list.sort(reverse=True)

        # Keep track of active routes (avoid index management issues)
        active_routes = {i: route for i, route in enumerate(routes)}

        # Merge routes based on savings
        for saving, i, j, order in savings_list:
            # Check if both routes still exist
            if i not in active_routes or j not in active_routes:
                continue

            route_i = active_routes[i]
            route_j = active_routes[j]

            # Double-check capacity (routes might have changed)
            if route_i.load + route_j.load > self.capacity:
                continue

            # Try merging based on the saved order
            if order == 'ij':
                merged_route = self._merge_routes(route_i, route_j)
            else:
                merged_route = self._merge_routes(route_j, route_i)

            if merged_route:
                # Successfully merged - update active routes
                active_routes[i] = merged_route
                del active_routes[j]

        return list(active_routes.values())

    def _calculate_merge_savings(self, route_first, route_second):
        """Calculate savings for merging route_first + route_second"""

        # Get connection points
        last_customer_first = route_first.customers[-2]  # Last customer before depot
        first_customer_second = route_second.customers[1]  # First customer after depot

        # Calculate distance savings
        current_distance = (self.get_distance(last_customer_first, 0) +
                            self.get_distance(0, first_customer_second))
        new_distance = self.get_distance(last_customer_first, first_customer_second)

        distance_saving = current_distance - new_distance

        # Quick feasibility check - if this fails, no point in detailed checking
        if distance_saving <= 0:
            return 0.0

        # Check if merge is time-feasible (quick approximation)
        last_ready_time = self.nodes[last_customer_first]['ready_time']
        last_service_time = self.nodes[last_customer_first].get('service_time', 0)
        travel_time = self.get_distance(last_customer_first, first_customer_second)

        earliest_arrival_at_second = last_ready_time + last_service_time + travel_time
        first_due_time = self.nodes[first_customer_second]['due_date']

        if earliest_arrival_at_second > first_due_time:
            return 0.0  # Time infeasible

        return distance_saving

    def _merge_routes(self, route_first, route_second):
        """Attempt to merge two routes, return merged route if feasible"""

        # Create merged customer sequence
        merged_customers = (route_first.customers[:-1] +  # Remove final depot
                            route_second.customers[1:])  # Remove initial depot

        # Check if merged route is feasible
        if self.is_route_feasible(merged_customers):
            merged_route = InternalRoute()
            merged_route.customers = merged_customers
            merged_route.load = route_first.load + route_second.load
            merged_route.distance = self.calculate_route_distance(merged_customers)
            return merged_route

        return None  # Merge not feasible

    def temporal_clustering_heuristic(self) -> List[InternalRoute]:
        """Temporal clustering approach"""
        pairs = self.get_pickup_delivery_pairs()

        # Sort pairs by pickup ready time
        pairs.sort(key=lambda p: self.nodes[p[0]]['ready_time'])

        routes = []
        current_route = InternalRoute()
        current_route.customers = [0]

        for pickup_id, delivery_id in pairs:
            # Check if pair can be added to current route
            if (current_route.load + self.nodes[pickup_id]['demand'] <= self.capacity and
                    len(current_route.customers) > 1):

                # Find best insertion position
                best_cost = float('inf')
                best_positions = None

                for pickup_pos in range(1, len(current_route.customers)):
                    for delivery_pos in range(pickup_pos, len(current_route.customers)):
                        if self.is_insertion_feasible(current_route, pickup_id, delivery_id,
                                                      pickup_pos, delivery_pos):
                            cost = self.calculate_insertion_cost(current_route, pickup_id, delivery_id,
                                                                 pickup_pos, delivery_pos)
                            if cost < best_cost:
                                best_cost = cost
                                best_positions = (pickup_pos, delivery_pos)

                if best_positions:
                    # Insert into current route
                    pickup_pos, delivery_pos = best_positions
                    current_route.customers.insert(pickup_pos, pickup_id)
                    current_route.customers.insert(delivery_pos + 1, delivery_id)
                    current_route.load += self.nodes[pickup_id]['demand']
                    current_route.distance += best_cost
                    continue

            # Cannot add to current route, finalize it and start new one
            if len(current_route.customers) > 1:
                current_route.customers.append(0)
                current_route.distance = self.calculate_route_distance(current_route.customers)
                routes.append(current_route)

            # Start new route
            current_route = InternalRoute()
            current_route.customers = [0, pickup_id, delivery_id]
            current_route.load = self.nodes[pickup_id]['demand']

        # Finalize last route
        if len(current_route.customers) > 1:
            current_route.customers.append(0)
            current_route.distance = self.calculate_route_distance(current_route.customers)
            routes.append(current_route)

        return routes

    def greedy_randomized_construction(self) -> List[InternalRoute]:
        """GRASP-style randomized construction"""
        routes = []
        unrouted_pairs = self.get_pickup_delivery_pairs()
        alpha = 0.3

        # OPTIMIZATION 1: Pre-calculate urgency scores once
        pair_urgency = {}
        for pickup_id, delivery_id in unrouted_pairs:
            pickup_slack = (self.nodes[pickup_id]['due_date'] -
                            self.nodes[pickup_id]['ready_time'])
            delivery_slack = (self.nodes[delivery_id]['due_date'] -
                              self.nodes[delivery_id]['ready_time'])
            pair_urgency[(pickup_id, delivery_id)] = pickup_slack + delivery_slack

        while unrouted_pairs:
            # Build candidate list based on pre-calculated urgency
            candidates = [(pair_urgency[pair], pair[0], pair[1]) for pair in unrouted_pairs]
            candidates.sort()

            # Create restricted candidate list (RCL)
            min_urgency = candidates[0][0]
            max_urgency = candidates[-1][0]
            threshold = min_urgency + alpha * (max_urgency - min_urgency)

            rcl = [(pickup, delivery) for urgency, pickup, delivery in candidates
                   if urgency <= threshold]

            # Randomly select from RCL
            selected_idx = self.random_state.choice(len(rcl))
            pickup_id, delivery_id = rcl[selected_idx]

            # OPTIMIZATION 2: Try existing routes with capacity pre-filtering
            pair_demand = self.nodes[pickup_id]['demand']
            candidate_routes = [route for route in routes
                                if route.load + pair_demand <= self.capacity]

            inserted = False

            # OPTIMIZATION 3: Limit insertion attempts per route
            for route in candidate_routes:
                # Try optimized insertion (same pattern as other optimizations)
                if self._try_insert_pair(route, pickup_id, delivery_id):
                    inserted = True
                    break

            if not inserted:
                # Create new route
                new_route = InternalRoute()
                new_route.customers = [0, pickup_id, delivery_id, 0]
                new_route.load = pair_demand
                new_route.distance = self.calculate_route_distance(new_route.customers)
                routes.append(new_route)

            unrouted_pairs.remove((pickup_id, delivery_id))

        return routes

    def _quick_pair_time_check(self, pickup_id: int, delivery_id: int, route: InternalRoute) -> bool:
        """Quick time compatibility check for the pair"""

        pickup_ready = self.nodes[pickup_id]['ready_time']
        pickup_due = self.nodes[pickup_id]['due_date']
        delivery_ready = self.nodes[delivery_id]['ready_time']
        delivery_due = self.nodes[delivery_id]['due_date']

        # Basic time window overlap check
        if pickup_due < delivery_ready:  # Impossible timing
            return False

        # Check if pair timing is compatible with route's time span
        if len(route.customers) > 2:
            route_customers = route.customers[1:-1]  # Exclude depots
            route_earliest = min(self.nodes[c]['ready_time'] for c in route_customers)
            route_latest = max(self.nodes[c]['due_date'] for c in route_customers)

            # Check if pair fits within route's time span with some buffer
            if pickup_ready > route_latest + 100 or delivery_due < route_earliest - 100:
                return False

        return True

    def _try_insert_pair(self, route: InternalRoute, pickup_id: int, delivery_id: int) -> bool:
        """Optimized insertion with early termination"""

        # OPTIMIZATION 4: Quick capacity check
        pair_demand = self.nodes[pickup_id]['demand']
        if route.load + pair_demand > self.capacity:
            return False

        # OPTIMIZATION 5: Quick time window compatibility check
        if not self._quick_pair_time_check(pickup_id, delivery_id, route):
            return False

        # OPTIMIZATION 6: Limit position search space
        max_positions = min(6, len(route.customers))

        for pickup_pos in range(1, max_positions):
            for delivery_pos in range(pickup_pos, max_positions):

                # OPTIMIZATION 7: Quick feasibility pre-check
                if not self._quick_insertion_check(route, pickup_id, delivery_id,
                                                         pickup_pos, delivery_pos):
                    continue

                # Only do expensive check if quick check passes
                if self.is_insertion_feasible(route, pickup_id, delivery_id,
                                              pickup_pos, delivery_pos):
                    # Perform insertion
                    route.customers.insert(pickup_pos, pickup_id)
                    route.customers.insert(delivery_pos + 1, delivery_id)
                    route.load += pair_demand

                    # Update distance
                    cost = self.calculate_insertion_cost(route, pickup_id, delivery_id,
                                                         pickup_pos, delivery_pos)
                    route.distance += cost
                    return True

        return False

    def local_search_route_minimization(self, routes: List[InternalRoute]) -> List[InternalRoute]:
        """Apply local search to minimize number of routes"""
        improved = True

        while improved:
            improved = False

            # Try to merge routes
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    if j >= len(routes):  # Route may have been removed
                        break

                    # Try merging route j into route i
                    if self.can_merge_routes(routes[i], routes[j]):
                        merged_route = self.merge_routes(routes[i], routes[j])
                        if merged_route:
                            routes[i] = merged_route
                            routes.pop(j)
                            improved = True
                            break
                if improved:
                    break

            # Try inter-route relocations to enable merges
            if not improved:
                for i in range(len(routes)):
                    for j in range(len(routes)):
                        if i != j and len(routes[i].customers) > 3:  # More than just depot-depot
                            # Try moving pickup-delivery pairs from route i to route j
                            for k in range(1, len(routes[i].customers) - 1):
                                customer_id = routes[i].customers[k]
                                if self.is_pickup(customer_id):
                                    delivery_id = self.nodes[customer_id]['delivery_index']
                                    if delivery_id in routes[i].customers:
                                        # Try relocating this pair
                                        if self.try_relocate_pair(routes[i], routes[j], customer_id, delivery_id):
                                            improved = True
                                            break
                            if improved:
                                break
                    if improved:
                        break

        # Remove empty routes (routes with only depot nodes)
        routes = [route for route in routes if len(route.customers) > 2]

        return routes

    def try_relocate_pair(self, from_route: InternalRoute, to_route: InternalRoute,
                          pickup_id: int, delivery_id: int) -> bool:
        """Try to relocate a pickup-delivery pair between routes"""
        # Check capacity constraint
        pair_demand = self.nodes[pickup_id]['demand']
        if to_route.load + pair_demand > self.capacity:
            return False

        # Find best insertion positions in destination route
        best_cost = float('inf')
        best_positions = None

        for pickup_pos in range(1, len(to_route.customers)):
            for delivery_pos in range(pickup_pos, len(to_route.customers)):
                if self.is_insertion_feasible(to_route, pickup_id, delivery_id, pickup_pos, delivery_pos):
                    insertion_cost = self.calculate_insertion_cost(to_route, pickup_id, delivery_id,
                                                                   pickup_pos, delivery_pos)
                    if insertion_cost < best_cost:
                        best_cost = insertion_cost
                        best_positions = (pickup_pos, delivery_pos)

        if best_positions:
            # Calculate removal savings from source route
            pickup_idx = from_route.customers.index(pickup_id)
            delivery_idx = from_route.customers.index(delivery_id)
            removal_saving = self.calculate_removal_saving(from_route, pickup_idx, delivery_idx)

            # Only relocate if it's beneficial
            if removal_saving > best_cost:
                # Remove from source route
                from_route.customers.remove(pickup_id)
                from_route.customers.remove(delivery_id)
                from_route.load -= pair_demand
                from_route.distance = self.calculate_route_distance(from_route.customers)

                # Insert into destination route
                pickup_pos, delivery_pos = best_positions
                to_route.customers.insert(pickup_pos, pickup_id)
                to_route.customers.insert(delivery_pos + 1, delivery_id)
                to_route.load += pair_demand
                to_route.distance = self.calculate_route_distance(to_route.customers)

                return True

        return False

    def can_merge_routes(self, route1: InternalRoute, route2: InternalRoute) -> bool:
        """Check if two routes can be merged"""
        return route1.load + route2.load <= self.capacity

    def merge_routes(self, route1: InternalRoute, route2: InternalRoute) -> InternalRoute:
        """Merge two routes optimally"""
        if not self.can_merge_routes(route1, route2):
            return None

        # Try different merging strategies
        customers1 = route1.customers[1:-1]  # Remove depot
        customers2 = route2.customers[1:-1]  # Remove depot

        best_merged = None
        best_distance = float('inf')

        # Strategy 1: Append route2 to route1
        merged_customers = [0] + customers1 + customers2 + [0]
        if self.is_route_feasible(merged_customers):
            distance = self.calculate_route_distance(merged_customers)
            if distance < best_distance:
                best_distance = distance
                best_merged = merged_customers

        # Strategy 2: Append route1 to route2
        merged_customers = [0] + customers2 + customers1 + [0]
        if self.is_route_feasible(merged_customers):
            distance = self.calculate_route_distance(merged_customers)
            if distance < best_distance:
                best_distance = distance
                best_merged = merged_customers

        if best_merged:
            new_route = InternalRoute()
            new_route.customers = best_merged
            new_route.load = route1.load + route2.load
            new_route.distance = best_distance
            return new_route

        return None

    def solve_all_methods(self) -> Dict[str, List[InternalRoute]]:
        """Solve using all implemented methods and return results"""
        methods = {
            'Sequential Insertion': self.sequential_insertion_heuristic,
            'Regret Insertion': self.regret_insertion_heuristic,
            # 'Push-Forward (PFIH)': self.push_forward_insertion_heuristic,
            # 'Cluster-First Route-Second': self.cluster_first_route_second,
            # 'Savings Algorithm': self.savings_algorithm_pdptw,
            # 'Temporal Clustering': self.temporal_clustering_heuristic,
            # 'Greedy Randomized': self.greedy_randomized_construction,
            'Cheapest Insertion': self.cheapest_insertion_with_route_minimization
        }

        results = {}
        for name, method in methods.items():
            try:
                print(f"Running {name}...")
                routes = method()
                # Apply local search for route minimization
                if routes:
                    routes = self.local_search_route_minimization(routes)
                results[name] = routes
                if routes:
                    print(f"{name}: {len(routes)} routes, total distance: {sum(r.distance for r in routes):.2f}")
                else:
                    print(f"{name}: Failed to generate solution")
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = []

        return results

    def get_best_solution(self, results: Dict[str, List[InternalRoute]]) -> Tuple[str, List[InternalRoute]]:
        """Get the best solution (minimum routes, then minimum distance)"""
        best_method = None
        best_routes = None
        best_num_routes = float('inf')
        best_distance = float('inf')

        for method, routes in results.items():
            if not routes:
                continue

            num_routes = len(routes)
            total_distance = sum(r.distance for r in routes)

            if (num_routes < best_num_routes or
                    (num_routes == best_num_routes and total_distance < best_distance)):
                best_num_routes = num_routes
                best_distance = total_distance
                best_method = method
                best_routes = routes

        return best_method, best_routes

    def convert_to_pdptw_solution(self, routes: List[InternalRoute]):
        """Convert internal routes to PDPTWSolution format"""

        solution = PDPTWSolution(self.instance)
        solution.routes = []

        for route in routes:
            # Ensure route has proper depot structure
            route_customers = route.customers[:]
            if not route_customers:
                route_customers = [0, 0]
            elif route_customers[0] != 0:
                route_customers = [0] + route_customers
            if route_customers[-1] != 0:
                route_customers = route_customers + [0]

            solution.routes.append(route_customers)

        # Ensure all routes are properly formatted and calculate objective
        solution.ensure_depot_nodes()
        solution.calculate_objective()

        return solution

    def validate_solution(self, routes: List[InternalRoute]) -> bool:
        """Validate that solution is feasible"""
        all_customers = set()
        pickup_delivery_map = {}

        # Build pickup-delivery mapping
        for node in self.nodes[1:]:  # Skip depot
            if node['demand'] > 0:  # Pickup
                pickup_delivery_map[node['id']] = node['delivery_index']

        for route in routes:
            visited_pickups = set()
            current_time = 0.0
            current_load = 0

            for i in range(len(route.customers) - 1):
                curr_id = route.customers[i]
                next_id = route.customers[i + 1]

                if curr_id != 0:  # Not depot
                    all_customers.add(curr_id)

                    # Check precedence
                    if self.is_delivery(curr_id):
                        pickup_id = self.nodes[curr_id]['pickup_index']
                        if pickup_id not in visited_pickups:
                            print(f"Precedence violation: delivery {curr_id} before pickup {pickup_id}")
                            return False
                    else:
                        visited_pickups.add(curr_id)

                # Update load
                current_load += self.nodes[curr_id]['demand']
                if current_load > self.capacity:
                    print(f"Capacity violation: {current_load} > {self.capacity}")
                    return False

                # Check time windows
                travel_time = self.get_distance(curr_id, next_id)
                current_time += travel_time

                if next_id != 0:  # Not returning to depot
                    customer = self.nodes[next_id]
                    arrival_time = max(current_time, customer['ready_time'])

                    if arrival_time > customer['due_date']:
                        print(
                            f"Time window violation at customer {next_id}: arrival {arrival_time} > due {customer['due_date']}")
                        return False

                    current_time = arrival_time + customer['service_time']

        # Check that all pickup-delivery pairs are served
        expected_customers = set()
        for pickup_id, delivery_id in pickup_delivery_map.items():
            expected_customers.add(pickup_id)
            expected_customers.add(delivery_id)

        if all_customers != expected_customers:
            print(f"Missing customers: {expected_customers - all_customers}")
            return False

        print("Solution is feasible!")
        return True

    def is_insertion_feasible_optimized(self, route: InternalRoute, pickup: int, delivery: int,
                                        pickup_pos: int, delivery_pos: int) -> bool:
        """Optimized feasibility check with early termination"""

        # EARLY CHECK 1: Basic precedence (fastest check first)
        if pickup_pos > delivery_pos:
            return False

        # EARLY CHECK 2: Capacity constraint (avoid expensive route validation)
        pickup_demand = self.nodes[pickup]['demand']
        if route.load + pickup_demand > self.capacity:
            return False

        # EARLY CHECK 3: Quick distance/time feasibility estimate
        # Skip expensive full validation if insertion looks impossible
        if not self._quick_time_feasibility_check(route, pickup, delivery, pickup_pos, delivery_pos):
            return False

        # ONLY NOW do the expensive full route validation
        temp_customers = route.customers.copy()
        temp_customers.insert(pickup_pos, pickup)
        temp_customers.insert(delivery_pos + 1, delivery)

        # Use lightweight feasibility check instead of creating PDPTWSolution
        return self._lightweight_route_feasibility_check(temp_customers)

    def _quick_time_feasibility_check(self, route: InternalRoute, pickup: int, delivery: int,
                                      pickup_pos: int, delivery_pos: int) -> bool:
        """Fast preliminary time window check without full route simulation"""

        # Check if pickup can be served after previous customer
        if pickup_pos > 0:
            prev_customer = route.customers[pickup_pos - 1]
            earliest_arrival_at_pickup = (
                    self.nodes[prev_customer]['ready_time'] +
                    self.nodes[prev_customer]['service_time'] +
                    self.get_distance(prev_customer, pickup)
            )

            if earliest_arrival_at_pickup > self.nodes[pickup]['due_date']:
                return False

        # Check if delivery can be served after pickup
        min_time_at_delivery = (
                max(self.nodes[pickup]['ready_time'], earliest_arrival_at_pickup) +
                self.nodes[pickup]['service_time'] +
                self.get_distance(pickup, delivery)
        )

        if min_time_at_delivery > self.nodes[delivery]['due_date']:
            return False

        return True

    def _lightweight_route_feasibility_check(self, customers: List[int]) -> bool:
        """Lightweight route feasibility without creating PDPTWSolution objects"""

        current_time = 0.0
        current_load = 0
        visited_pickups = set()

        for i in range(len(customers) - 1):
            curr_customer = customers[i]
            next_customer = customers[i + 1]

            # Update load and check capacity
            current_load += self.nodes[curr_customer]['demand']
            if current_load > self.capacity:
                return False

            # Check precedence constraints
            if curr_customer != 0:  # Not depot
                if self.is_delivery(curr_customer):
                    pickup_id = self.nodes[curr_customer]['pickup_index']
                    if pickup_id not in visited_pickups:
                        return False
                else:
                    visited_pickups.add(curr_customer)

            # Update time and check time windows
            if next_customer != 0:  # Not returning to depot
                travel_time = self.get_distance(curr_customer, next_customer)
                current_time += travel_time

                # Check time window
                customer = self.nodes[next_customer]
                arrival_time = max(current_time, customer['ready_time'])

                if arrival_time > customer['due_date']:
                    return False

                current_time = arrival_time + customer['service_time']

        return True

    def cheapest_insertion_with_route_minimization(self) -> List[InternalRoute]:
        """Optimized version with reduced feasibility checking"""

        routes = []
        unrouted_pairs = self.get_pickup_delivery_pairs()

        # Cache to avoid redundant calculations
        feasibility_cache = {}
        cost_cache = {}

        while unrouted_pairs:
            best_insertion = None
            best_cost = float('inf')
            create_new_route = True

            # Try inserting each pair into existing routes
            for pickup_id, delivery_id in unrouted_pairs:

                # OPTIMIZATION: Check capacity early for all routes
                pair_demand = self.nodes[pickup_id]['demand']
                candidate_routes = [route for route in routes if route.load + pair_demand <= self.capacity]

                if not candidate_routes:
                    continue  # Skip to next pair if no routes have capacity

                for route in candidate_routes:
                    # OPTIMIZATION: Limit search positions based on route characteristics
                    max_positions = min(len(route.customers), 10)  # Limit position search

                    for pickup_pos in range(1, max_positions):
                        for delivery_pos in range(pickup_pos, max_positions):

                            # Use cache key to avoid redundant calculations
                            cache_key = (id(route), pickup_id, delivery_id, pickup_pos, delivery_pos)

                            # Check feasibility with optimized function
                            if cache_key not in feasibility_cache:
                                feasibility_cache[cache_key] = self.is_insertion_feasible_optimized(
                                    route, pickup_id, delivery_id, pickup_pos, delivery_pos)

                            if feasibility_cache[cache_key]:
                                # Calculate cost only if feasible
                                if cache_key not in cost_cache:
                                    cost_cache[cache_key] = self.calculate_insertion_cost(
                                        route, pickup_id, delivery_id, pickup_pos, delivery_pos)

                                cost = cost_cache[cache_key]
                                if cost < best_cost:
                                    best_cost = cost
                                    best_insertion = (route, pickup_id, delivery_id, pickup_pos, delivery_pos)
                                    create_new_route = False

            # Insert best pair or create new route
            if not create_new_route and best_insertion:
                route, pickup_id, delivery_id, pickup_pos, delivery_pos = best_insertion
                route.customers.insert(pickup_pos, pickup_id)
                route.customers.insert(delivery_pos + 1, delivery_id)
                route.load += self.nodes[pickup_id]['demand']
                route.distance += best_cost
                unrouted_pairs.remove((pickup_id, delivery_id))
            else:
                # Create new route with most urgent pair
                urgency_pairs = [(
                    self.nodes[pickup_id]['due_date'] - self.nodes[pickup_id]['ready_time'] +
                    self.nodes[delivery_id]['due_date'] - self.nodes[delivery_id]['ready_time'],
                    pickup_id, delivery_id
                ) for pickup_id, delivery_id in unrouted_pairs]

                urgency_pairs.sort()
                _, pickup_id, delivery_id = urgency_pairs[0]

                new_route = InternalRoute()
                new_route.customers = [0, pickup_id, delivery_id, 0]
                new_route.load = self.nodes[pickup_id]['demand']
                new_route.distance = self.calculate_route_distance(new_route.customers)
                routes.append(new_route)
                unrouted_pairs.remove((pickup_id, delivery_id))

        return routes

def create_original_initial_solution(instance: PDPTWInstance) -> PDPTWSolution:
    """Create initial solution using greedy insertion prioritizing time window flexibility"""
    solution = PDPTWSolution(instance)
    random_state = np.random.RandomState(42)  # Consistent random seed

    # Get pickup nodes (excluding depot)
    pickup_nodes = [i for i, node in enumerate(instance.nodes)
                    if node['demand'] > 0 and i > 0]
    uninserted = list(pickup_nodes)  # Track uninserted pickup nodes
    random_state.shuffle(uninserted)  # Randomize order for diversity

    while uninserted:
        best_insertion = None
        best_cost = float('inf')
        best_time_slack = -float('inf')  # Track time window slack for flexibility
        selected_pickup = None

        for pickup in uninserted:
            delivery = instance.nodes[pickup]['delivery_index']

            # Try inserting into existing routes
            for route_idx, route in enumerate(solution.routes + [[0, 0]]):  # Include new route option
                is_new_route = route == [0, 0]
                for i in range(1, len(route)):  # Pickup insertion positions
                    for j in range(i + 1, len(route) + 1):  # Delivery insertion positions
                        # Create temporary route for insertion
                        temp_route = route[:]
                        temp_route.insert(i, pickup)
                        temp_route.insert(j, delivery)

                        # Check feasibility
                        if solution._is_route_feasible(temp_route):
                            # Calculate insertion cost (distance increase)
                            old_cost = 0 if is_new_route else sum(
                                instance.distances[temp_route[k]][temp_route[k + 1]]
                                for k in range(len(temp_route) - 1)
                            )
                            new_cost = sum(
                                instance.distances[temp_route[k]][temp_route[k + 1]]
                                for k in range(len(temp_route) - 1)
                            )
                            insertion_cost = new_cost - old_cost

                            # Calculate time window slack (sum of due_date - arrival_time)
                            current_time = 0
                            time_slack = 0
                            for k in range(len(temp_route) - 1):
                                current_node = temp_route[k]
                                next_node = temp_route[k + 1]
                                travel_time = instance.distances[current_node][next_node]
                                current_time += travel_time
                                node_info = instance.nodes[next_node]
                                arrival_time = max(current_time, node_info['ready_time'])
                                if arrival_time <= node_info['due_date']:
                                    time_slack += node_info['due_date'] - arrival_time
                                current_time = arrival_time + node_info['service_time']

                            # Prioritize based on time slack, then cost
                            if time_slack > best_time_slack or (time_slack == best_time_slack and insertion_cost < best_cost):
                                best_time_slack = time_slack
                                best_cost = insertion_cost
                                best_insertion = (route_idx, i, j, pickup, delivery)

        if best_insertion is None:
            break  # No feasible insertion found

        route_idx, pickup_pos, delivery_pos, pickup, delivery = best_insertion
        uninserted.remove(pickup)

        # Insert into existing or new route
        if route_idx < len(solution.routes):
            route = solution.routes[route_idx]
            route.insert(pickup_pos, pickup)
            route.insert(delivery_pos, delivery)
        else:
            new_route = [0, pickup, delivery, 0]
            solution.routes.append(new_route)

    # Ensure depot nodes and calculate objective
    solution.ensure_depot_nodes()
    solution.calculate_objective()
    return solution

def create_advanced_initial_solution(instance, method: str = 'best', random_seed: int = 42):
    """
    Create advanced initial solution using multiple construction heuristics

    Args:
        instance: PDPTWInstance object from your existing framework
        method: 'best', 'sequential', 'regret', 'push_forward', 'cluster_first',
               'savings', 'temporal', 'greedy_randomized', 'cheapest'
        random_seed: Random seed for reproducibility

    Returns:
        PDPTWSolution object with optimized routes
    """

    # Initialize the advanced solver with your instance
    solver = AdvancedPDPTWSolver(instance, random_seed)

    if method == 'best':
        # Run all methods and return the best
        results = solver.solve_all_methods()
        best_method, best_routes = solver.get_best_solution(results)
        print(f"Best method: {best_method} with {len(best_routes)} routes")
        return solver.convert_to_pdptw_solution(best_routes)
    else:
        # Run specific method
        method_map = {
            'sequential': solver.sequential_insertion_heuristic,
            'regret': solver.regret_insertion_heuristic,
            # 'push_forward': solver.push_forward_insertion_heuristic,
            # 'cluster_first': solver.cluster_first_route_second,
            # 'savings': solver.savings_algorithm_pdptw,
            'temporal': solver.temporal_clustering_heuristic,
            'greedy_randomized': solver.greedy_randomized_construction,
            'cheapest': solver.cheapest_insertion_with_route_minimization
        }

        if method in method_map:
            routes = method_map[method]()
            routes = solver.local_search_route_minimization(routes)
            return solver.convert_to_pdptw_solution(routes)
        else:
            raise ValueError(f"Unknown method: {method}")

