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

import os
import numpy as np
import requests
import random
import math
import time
import matplotlib.pyplot as plt

from datetime import timedelta
from alns import ALNS, State
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations


from typing import List, Tuple, Dict

from src.sisr_alns_pdptw import add_sisr_pdptw_operators


class PDPTWInstance:
    """Represents a PDPTW instance from Li-Lim benchmark"""

    def __init__(self, filename: str = None, data: str = None):
        if filename and not data:
            self.load_from_file(filename)
        elif data and not filename:
            self.parse_data(data)
        elif filename and data:
            # If both are provided, treat 'data' as the folder path and 'filename' as the file
            import os
            full_path = os.path.join(data, filename)
            self.load_from_file(full_path)
        else:
            raise ValueError("Either filename or data must be provided")

    def load_from_file(self, filepath: str):
        """Load PDPTW data from file"""
        try:
            with open(filepath, 'r') as f:
                file_content = f.read()
            self.parse_data(file_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {filepath}")
        except Exception as e:
            raise Exception(f"Error reading file {filepath}: {str(e)}")

    def parse_data(self, data: str):
        """Parse PDPTW data format"""
        lines = data.strip().split('\n')

        # First line: vehicle_number, capacity
        vehicle_info = lines[0].split()
        self.vehicle_number = int(vehicle_info[0])
        self.capacity = int(vehicle_info[1])

        # Parse nodes
        self.nodes = []
        for i, line in enumerate(lines[1:], 1):
            parts = line.split()
            if len(parts) >= 9:
                node = {
                    'id': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'demand': int(parts[3]),
                    'ready_time': int(parts[4]),
                    'due_date': int(parts[5]),
                    'service_time': int(parts[6]),
                    'pickup_index': int(parts[7]) if parts[7] != '0' else 0,
                    'delivery_index': int(parts[8]) if parts[8] != '0' else 0
                }
                self.nodes.append(node)

        self.n_customers = len([n for n in self.nodes if n['demand'] != 0]) // 2
        self.depot = self.nodes[0]

        # Calculate distance matrix
        self.calculate_distances()

    def calculate_distances(self):
        """Calculate Euclidean distance matrix"""
        n = len(self.nodes)
        self.distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.nodes[i]['x'] - self.nodes[j]['x']
                    dy = self.nodes[i]['y'] - self.nodes[j]['y']
                    self.distances[i][j] = math.sqrt(dx * dx + dy * dy)



class PDPTWSolution:
    """Represents a solution to PDPTW"""

    def __init__(self, instance: PDPTWInstance):
        self.instance = instance
        self.routes = []
        self._objective_value = float('inf')

    def copy(self):
        """Create a deep copy of the solution"""
        new_solution = PDPTWSolution(self.instance)
        new_solution.routes = [route[:] for route in self.routes]
        new_solution._objective_value = self._objective_value
        return new_solution

    def objective(self):
        """Return the objective value (ALNS expects this as a method)"""
        return self._objective_value

    def calculate_objective_with_noise(self, random_state: np.random.RandomState = None, noise_factor: float = 0.015):
        """Calculate total distance with optional noise to escape local optima"""
        self.ensure_depot_nodes()
        total_distance = 0
        max_arc_distance = 0

        # Calculate total distance and max arc distance
        for route in self.routes:
            if len(route) > 2:
                for i in range(len(route) - 1):
                    arc_dist = self.instance.distances[route[i]][route[i + 1]]
                    total_distance += arc_dist
                    if arc_dist > max_arc_distance:
                        max_arc_distance = arc_dist

        # Apply noise only if random_state is provided
        if random_state is not None:
            max_noise = noise_factor * max_arc_distance
            noise = random_state.uniform(-max_noise, max_noise)
            noisy_distance = max(0, total_distance + noise)
            # Choose between noisy and normal distance randomly
            noise_success = 1
            no_noise_success = 1
            rand_val = random_state.random()
            if rand_val < no_noise_success / (noise_success + no_noise_success):
                self._objective_value = total_distance
            else:
                self._objective_value = noisy_distance
        else:
            self._objective_value = total_distance

        return self._objective_value

    def calculate_objective(self):
        """Calculate total distance of all routes"""
        # Ensure routes start and end with depot
        self.ensure_depot_nodes()

        total_distance = 0

        for route in self.routes:
            # print(f" ====== {route} ===========")
            if len(route) > 2:  # More than just depot-depot
                route_distance = 0
                for i in range(len(route) - 1):
                    route_distance += self.instance.distances[route[i]][route[i + 1]]
                total_distance += route_distance

        self._objective_value = total_distance
        return total_distance

    def is_feasible(self):
        """Check if solution is feasible"""
        # Ensure routes start and end with depot
        self.ensure_depot_nodes()

        for route in self.routes:
            if not self._is_route_feasible(route):
                return False
        return True

    def _is_route_feasible(self, route):
        """Check if a single route is feasible"""
        if len(route) < 2:
            return True

        current_time = 0
        current_load = 0

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # Travel time
            travel_time = self.instance.distances[current_node][next_node]
            current_time += travel_time

            # Service at next node
            node_info = self.instance.nodes[next_node]

            # Time window check
            if current_time < node_info['ready_time']:
                current_time = node_info['ready_time']
            elif current_time > node_info['due_date']:
                return False

            # Update load
            current_load += node_info['demand']
            if current_load > self.instance.capacity or current_load < 0:
                return False

            # Service time
            current_time += node_info['service_time']

            # Pickup-delivery precedence check
            pickup_idx = node_info['pickup_index']
            delivery_idx = node_info['delivery_index']

            if pickup_idx > 0:  # This is a delivery node
                # Check if pickup is visited before delivery in this route
                try:
                    pickup_pos = route.index(pickup_idx)
                    delivery_pos = route.index(next_node)
                    if pickup_pos >= delivery_pos:
                        return False
                except ValueError:
                    return False

        return True

    def get_all_nodes(self) -> List[int]:
        """Return non-depot nodes in the solution"""
        nodes = set()
        for route in self.routes:
            nodes.update(node for node in route if node != 0)
        return list(nodes)

    def remove_node(self, node_id: int) -> None:
        """Remove a specific node from all routes in the solution."""
        for route in self.routes:
            if node_id in route:
                route.remove(node_id)
        # Clean up empty routes (routes with only depot nodes [0, 0])
        self.routes = [route for route in self.routes if len(route) > 2]
        # Ensure routes start and end with depot
        self.ensure_depot_nodes()
        self.calculate_objective()

    def ensure_depot_nodes(self):
        """Ensure all routes start and end with depot (0)"""
        for i, route in enumerate(self.routes):
            if not route:  # Empty route
                self.routes[i] = [0, 0]  # Minimum valid route
            else:
                if route[0] != 0:
                    self.routes[i] = [0] + route
                if route[-1] != 0:
                    self.routes[i] = self.routes[i] + [0]
        # Remove empty routes (only depot nodes) if they exist
        self.routes = [route for route in self.routes if len(route) > 2 or route == [0, 0]]

def create_initial_solution(instance: PDPTWInstance) -> PDPTWSolution:
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

# Destroy operators
def random_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Remove random pickup-delivery pairs"""
    destroyed = current.copy()

    if not destroyed.routes:
        return destroyed

    # Collect all pickup-delivery pairs
    pairs = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:  # Exclude depot
            node_info = destroyed.instance.nodes[node]
            if node_info['demand'] > 0:  # Pickup node
                delivery_idx = node_info['delivery_index']
                if delivery_idx in route:
                    pairs.append((route_idx, node, delivery_idx))

    if not pairs:
        return destroyed

    # Remove 1-3 random pairs
    n_remove = min(random_state.randint(1, 4), len(pairs))
    removed_pairs = random_state.choice(len(pairs), n_remove, replace=False)

    # Sort by route index to remove from back to front
    pairs_to_remove = [pairs[i] for i in removed_pairs]
    pairs_to_remove.sort(key=lambda x: x[0], reverse=True)

    for route_idx, pickup, delivery in pairs_to_remove:
        route = destroyed.routes[route_idx]
        if pickup in route:
            route.remove(pickup)
        if delivery in route:
            route.remove(delivery)

        # Remove empty routes
        if len(route) <= 2:
            destroyed.routes.pop(route_idx)

    destroyed.calculate_objective()
    return destroyed

def worst_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Remove most expensive pickup-delivery pairs"""
    destroyed = current.copy()

    # Calculate cost of each pickup-delivery pair
    pair_costs = []
    for route_idx, route in enumerate(destroyed.routes):
        for i, node in enumerate(route[1:-1], 1):
            node_info = destroyed.instance.nodes[node]
            if node_info['demand'] > 0:  # Pickup node
                delivery_idx = node_info['delivery_index']
                try:
                    delivery_pos = route.index(delivery_idx)
                    # Calculate cost as sum of distances
                    cost = 0
                    if i > 1:
                        cost += destroyed.instance.distances[route[i - 1]][node]
                    if i < len(route) - 1:
                        cost += destroyed.instance.distances[node][route[i + 1]]
                    if delivery_pos > 1:
                        cost += destroyed.instance.distances[route[delivery_pos - 1]][delivery_idx]
                    if delivery_pos < len(route) - 1:
                        cost += destroyed.instance.distances[delivery_idx][route[delivery_pos + 1]]

                    pair_costs.append((cost, route_idx, node, delivery_idx))
                except ValueError:
                    continue

    if not pair_costs:
        return destroyed

    # Sort by cost and remove worst ones
    pair_costs.sort(reverse=True)
    n_remove = min(2, len(pair_costs))

    for i in range(n_remove):
        _, route_idx, pickup, delivery = pair_costs[i]
        if route_idx < len(destroyed.routes):
            route = destroyed.routes[route_idx]
            if pickup in route:
                route.remove(pickup)
            if delivery in route:
                route.remove(delivery)

    # Clean up empty routes
    destroyed.routes = [route for route in destroyed.routes if len(route) > 2]
    destroyed.calculate_objective()
    return destroyed

def shaw_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Remove pickup-delivery pairs based on Shaw relatedness (distance, time windows, demand)"""
    destroyed = current.copy()
    if not destroyed.routes:
        return destroyed

    # Collect all pickup-delivery pairs
    pairs = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:  # Exclude depot
            node_info = destroyed.instance.nodes[node]
            if node_info['demand'] > 0:  # Pickup node
                delivery_idx = node_info['delivery_index']
                if delivery_idx in route:
                    pairs.append((route_idx, node, delivery_idx))

    if not pairs:
        return destroyed

    # Select a random starting pair
    start_pair_idx = random_state.choice(len(pairs))
    start_route_idx, start_pickup, start_delivery = pairs[start_pair_idx]
    start_pickup_info = destroyed.instance.nodes[start_pickup]
    start_delivery_info = destroyed.instance.nodes[start_delivery]

    # Calculate relatedness for all other pairs
    relatednesses = []
    for pair_idx, (route_idx, pickup, delivery) in enumerate(pairs):
        if pair_idx == start_pair_idx:
            continue
        pickup_info = destroyed.instance.nodes[pickup]
        delivery_info = destroyed.instance.nodes[delivery]

        # Distance relatedness (average of pickup and delivery distances)
        dist_pickup = destroyed.instance.distances[start_pickup][pickup]
        dist_delivery = destroyed.instance.distances[start_delivery][delivery]
        dist_relatedness = (dist_pickup + dist_delivery) / 2

        # Time window relatedness (average of ready_time differences)
        tw_pickup = abs(start_pickup_info['ready_time'] - pickup_info['ready_time'])
        tw_delivery = abs(start_delivery_info['ready_time'] - delivery_info['ready_time'])
        tw_relatedness = (tw_pickup + tw_delivery) / 2

        # Demand relatedness
        demand_relatedness = abs(start_pickup_info['demand'] - pickup_info['demand'])

        # Normalize and combine
        max_dist = max(destroyed.instance.distances.max(), 1)
        max_tw = max(max(node['due_date'] - node['ready_time'] for node in destroyed.instance.nodes), 1)
        max_demand = max(max(abs(node['demand']) for node in destroyed.instance.nodes if node['demand'] != 0), 1)

        normalized_dist = dist_relatedness / max_dist
        normalized_tw = tw_relatedness / max_tw
        normalized_demand = demand_relatedness / max_demand

        relatedness = normalized_dist + normalized_tw + normalized_demand
        relatednesses.append((relatedness, route_idx, pickup, delivery))

    # Sort by relatedness and select 1-3 pairs to remove
    relatednesses.sort(key=lambda x: x[0])
    n_remove = min(random_state.randint(1, 4), len(relatednesses))
    pairs_to_remove = [(r[1], r[2], r[3]) for r in relatednesses[:n_remove]]

    # Add the starting pair to remove
    pairs_to_remove.append((start_route_idx, start_pickup, start_delivery))
    pairs_to_remove.sort(key=lambda x: x[0], reverse=True)  # Sort by route index for safe removal

    # Remove selected pairs
    for route_idx, pickup, delivery in pairs_to_remove:
        route = destroyed.routes[route_idx]
        if pickup in route:
            route.remove(pickup)
        if delivery in route:
            route.remove(delivery)
        if len(route) <= 2:
            destroyed.routes.pop(route_idx)

    destroyed.calculate_objective()
    return destroyed

def proximity_based_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Remove pickup-delivery pairs based on proximity to a randomly selected node"""
    destroyed = current.copy()
    if not destroyed.routes:
        return destroyed

    # Select a random pickup node
    pickup_nodes = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:
            if destroyed.instance.nodes[node]['demand'] > 0:
                pickup_nodes.append((route_idx, node))

    if not pickup_nodes:
        return destroyed

    start_idx = random_state.choice(len(pickup_nodes))
    start_route_idx, start_pickup = pickup_nodes[start_idx]
    start_delivery = destroyed.instance.nodes[start_pickup]['delivery_index']

    # Find closest pairs based on pickup node proximity
    distances = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:
            if destroyed.instance.nodes[node]['demand'] > 0 and node != start_pickup:
                delivery_idx = destroyed.instance.nodes[node]['delivery_index']
                if delivery_idx in route:
                    dist = destroyed.instance.distances[start_pickup][node]
                    distances.append((dist, route_idx, node, delivery_idx))

    # Sort by distance and select 1-3 pairs to remove
    distances.sort(key=lambda x: x[0])
    n_remove = min(random_state.randint(1, 4), len(distances))
    pairs_to_remove = [(r[1], r[2], r[3]) for r in distances[:n_remove]]

    # Add the starting pair
    pairs_to_remove.append((start_route_idx, start_pickup, start_delivery))
    pairs_to_remove.sort(key=lambda x: x[0], reverse=True)

    # Remove selected pairs
    for route_idx, pickup, delivery in pairs_to_remove:
        route = destroyed.routes[route_idx]
        if pickup in route:
            route.remove(pickup)
        if delivery in route:
            route.remove(delivery)
        if len(route) <= 2:
            destroyed.routes.pop(route_idx)

    destroyed.calculate_objective()
    return destroyed


def time_based_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Remove pickup-delivery pairs based on time window similarity"""
    destroyed = current.copy()
    if not destroyed.routes:
        return destroyed

    # Select a random pickup node
    pickup_nodes = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:
            if destroyed.instance.nodes[node]['demand'] > 0:
                pickup_nodes.append((route_idx, node))

    if not pickup_nodes:
        return destroyed

    start_idx = random_state.choice(len(pickup_nodes))
    start_route_idx, start_pickup = pickup_nodes[start_idx]
    start_delivery = destroyed.instance.nodes[start_pickup]['delivery_index']
    start_pickup_info = destroyed.instance.nodes[start_pickup]

    # Find pairs with similar time windows
    tw_diffs = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:
            if destroyed.instance.nodes[node]['demand'] > 0 and node != start_pickup:
                delivery_idx = destroyed.instance.nodes[node]['delivery_index']
                if delivery_idx in route:
                    tw_diff = abs(start_pickup_info['ready_time'] - destroyed.instance.nodes[node]['ready_time'])
                    tw_diffs.append((tw_diff, route_idx, node, delivery_idx))

    # Sort by time window difference and select 1-3 pairs to remove
    tw_diffs.sort(key=lambda x: x[0])
    n_remove = min(random_state.randint(1, 4), len(tw_diffs))
    pairs_to_remove = [(r[1], r[2], r[3]) for r in tw_diffs[:n_remove]]

    # Add the starting pair
    pairs_to_remove.append((start_route_idx, start_pickup, start_delivery))
    pairs_to_remove.sort(key=lambda x: x[0], reverse=True)

    # Remove selected pairs
    for route_idx, pickup, delivery in pairs_to_remove:
        route = destroyed.routes[route_idx]
        if pickup in route:
            route.remove(pickup)
        if delivery in route:
            route.remove(delivery)
        if len(route) <= 2:
            destroyed.routes.pop(route_idx)

    destroyed.calculate_objective()
    return destroyed


# Repair operators
def regret_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Insert removed nodes using regret insertion to prioritize time window flexibility"""
    repaired = destroyed.copy()

    # Collect uninserted pickup nodes
    all_pickups = set()
    for node_info in repaired.instance.nodes[1:]:
        if node_info['demand'] > 0:
            all_pickups.add(node_info['id'])

    inserted_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                inserted_pickups.add(node)

    uninserted = list(all_pickups - inserted_pickups)
    random_state.shuffle(uninserted)  # Randomize for diversity

    while uninserted:
        best_insertion = None
        max_regret = -float('inf')

        for pickup in uninserted:
            delivery = repaired.instance.nodes[pickup]['delivery_index']
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_insertion_for_pickup = None

            # Evaluate all insertion positions
            for route_idx, route in enumerate(repaired.routes + [[0, 0]]):  # Include new route
                is_new_route = route == [0, 0]
                for i in range(1, len(route)):
                    for j in range(i + 1, len(route) + 1):
                        temp_route = route[:]
                        temp_route.insert(i, pickup)
                        temp_route.insert(j, delivery)

                        if repaired._is_route_feasible(temp_route):
                            old_cost = 0 if is_new_route else sum(
                                repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                for k in range(len(temp_route) - 1)
                            )
                            new_cost = sum(
                                repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                for k in range(len(temp_route) - 1)
                            )
                            insertion_cost = new_cost - old_cost

                            if insertion_cost < best_cost:
                                second_best_cost = best_cost
                                best_cost = insertion_cost
                                best_insertion_for_pickup = (route_idx, i, j, pickup, delivery)
                            elif insertion_cost < second_best_cost:
                                second_best_cost = insertion_cost

            # Calculate regret (difference between best and second-best insertion cost)
            regret = second_best_cost - best_cost if second_best_cost != float('inf') else best_cost
            if regret > max_regret:
                max_regret = regret
                best_insertion = best_insertion_for_pickup

        if best_insertion is None:
            break  # No feasible insertion

        route_idx, pickup_pos, delivery_pos, pickup, delivery = best_insertion
        uninserted.remove(pickup)

        # Insert into existing or new route
        if route_idx < len(repaired.routes):
            route = repaired.routes[route_idx]
            route.insert(pickup_pos, pickup)
            route.insert(delivery_pos, delivery)
        else:
            new_route = [0, pickup, delivery, 0]
            repaired.routes.append(new_route)

    repaired.calculate_objective()
    return repaired

# Repair operators
def greedy_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Insert removed nodes using greedy strategy"""
    repaired = destroyed.copy()

    # Collect all pickup nodes that need to be inserted
    all_pickups = set()
    for node_info in repaired.instance.nodes[1:]:  # Exclude depot
        if node_info['demand'] > 0:
            all_pickups.add(node_info['id'])

    # Find which pickups are already in routes
    inserted_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                inserted_pickups.add(node)

    uninserted = list(all_pickups - inserted_pickups)

    while uninserted:
        best_insertion = None
        best_cost = float('inf')

        for pickup in uninserted[:]:
            delivery = repaired.instance.nodes[pickup]['delivery_index']

            # Try inserting into existing routes
            for route_idx, route in enumerate(repaired.routes):
                for i in range(1, len(route)):
                    for j in range(i + 1, len(route)):
                        # Try inserting pickup at position i and delivery at position j
                        new_route = route[:]
                        new_route.insert(i, pickup)
                        new_route.insert(j + 1, delivery)

                        if repaired._is_route_feasible(new_route):
                            # Calculate insertion cost
                            old_cost = sum(repaired.instance.distances[route[k]][route[k + 1]]
                                           for k in range(len(route) - 1))
                            new_cost = sum(repaired.instance.distances[new_route[k]][new_route[k + 1]]
                                           for k in range(len(new_route) - 1))
                            cost = new_cost - old_cost

                            if cost < best_cost:
                                best_cost = cost
                                best_insertion = (route_idx, i, j + 1, pickup, delivery)

            # Try creating new route
            new_route = [0, pickup, delivery, 0]
            if repaired._is_route_feasible(new_route):
                cost = sum(repaired.instance.distances[new_route[k]][new_route[k + 1]]
                           for k in range(len(new_route) - 1))
                if cost < best_cost:
                    best_cost = cost
                    best_insertion = (-1, pickup, delivery)

        if best_insertion is None:
            break

        if len(best_insertion) == 3:  # New route
            _, pickup, delivery = best_insertion
            new_route = [0, pickup, delivery, 0]
            repaired.routes.append(new_route)
            uninserted.remove(pickup)
        else:  # Insert into existing route
            route_idx, pickup_pos, delivery_pos, pickup, delivery = best_insertion
            route = repaired.routes[route_idx]
            route.insert(pickup_pos, pickup)
            route.insert(delivery_pos, delivery)
            uninserted.remove(pickup)

    repaired.calculate_objective()
    return repaired

def k_regret_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState, k: int = 3) -> PDPTWSolution:
    """Insert pickup-delivery pairs using k-regret strategy, inspired by alns_wouda.py"""
    repaired = destroyed.copy()

    # Collect uninserted pickup nodes
    all_pickups = set()
    for node_info in repaired.instance.nodes[1:]:
        if node_info['demand'] > 0:
            all_pickups.add(node_info['id'])

    inserted_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                inserted_pickups.add(node)

    uninserted = list(all_pickups - inserted_pickups)
    random_state.shuffle(uninserted)  # Randomize for diversity

    while uninserted:
        best_insertion = None
        max_regret = -float('inf')

        for pickup in uninserted:
            delivery = repaired.instance.nodes[pickup]['delivery_index']
            insertion_costs = []

            # Evaluate all feasible insertion positions
            for route_idx, route in enumerate(repaired.routes + [[0, 0]]):  # Include new route
                is_new_route = route == [0, 0]
                for i in range(1, len(route)):  # Pickup positions
                    for j in range(i + 1, len(route) + 1):  # Delivery positions
                        temp_route = route[:]
                        temp_route.insert(i, pickup)
                        temp_route.insert(j, delivery)
                        if repaired._is_route_feasible(temp_route):
                            old_cost = 0 if is_new_route else sum(
                                repaired.instance.distances[temp_route[m]][temp_route[m + 1]]
                                for m in range(len(temp_route) - 1)
                            )
                            new_cost = sum(
                                repaired.instance.distances[temp_route[m]][temp_route[m + 1]]
                                for m in range(len(temp_route) - 1)
                            )
                            insertion_cost = new_cost - old_cost
                            insertion_costs.append((insertion_cost, route_idx, i, j))

            if not insertion_costs:
                continue  # No feasible insertion for this pair

            # Sort insertion costs and calculate k-regret
            insertion_costs.sort(key=lambda x: x[0])
            best_cost = insertion_costs[0][0]
            k_cost_sum = sum(cost for cost, _, _, _ in insertion_costs[1:k + 1]) if len(
                insertion_costs) > 1 else best_cost
            regret = k_cost_sum - best_cost * min(k, len(insertion_costs) - 1)

            if regret > max_regret:
                max_regret = regret
                best_insertion = (pickup, delivery, *insertion_costs[0][1:])  # route_idx, i, j

        if best_insertion is None:
            break  # No feasible insertion found

        pickup, delivery, route_idx, pickup_pos, delivery_pos = best_insertion
        uninserted.remove(pickup)

        # Insert into existing or new route
        if route_idx < len(repaired.routes):
            route = repaired.routes[route_idx]
            route.insert(pickup_pos, pickup)
            route.insert(delivery_pos, delivery)
        else:
            new_route = [0, pickup, delivery, 0]
            repaired.routes.append(new_route)

    repaired.calculate_objective()
    return repaired

def grasp_insertion(destroyed: PDPTWSolution, random_state: np.random.RandomState, rcl_size: int = 5) -> PDPTWSolution:
    """Insert pickup-delivery pairs using GRASP strategy, inspired by alns_wouda.py"""
    repaired = destroyed.copy()

    # Collect uninserted pickup nodes
    all_pickups = set()
    for node_info in repaired.instance.nodes[1:]:
        if node_info['demand'] > 0:
            all_pickups.add(node_info['id'])

    inserted_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                inserted_pickups.add(node)

    uninserted = list(all_pickups - inserted_pickups)
    random_state.shuffle(uninserted)  # Randomize for diversity

    while uninserted:
        # Select a random uninserted pair
        pickup_idx = random_state.choice(len(uninserted))
        pickup = uninserted[pickup_idx]
        delivery = repaired.instance.nodes[pickup]['delivery_index']
        rcl = []

        # Evaluate all feasible insertion positions
        for route_idx, route in enumerate(repaired.routes + [[0, 0]]):  # Include new route
            is_new_route = route == [0, 0]
            for i in range(1, len(route)):  # Pickup positions
                for j in range(i + 1, len(route) + 1):  # Delivery positions
                    temp_route = route[:]
                    temp_route.insert(i, pickup)
                    temp_route.insert(j, delivery)
                    if repaired._is_route_feasible(temp_route):
                        old_cost = 0 if is_new_route else sum(
                            repaired.instance.distances[temp_route[m]][temp_route[m + 1]]
                            for m in range(len(temp_route) - 1)
                        )
                        new_cost = sum(
                            repaired.instance.distances[temp_route[m]][temp_route[m + 1]]
                            for m in range(len(temp_route) - 1)
                        )
                        insertion_cost = new_cost - old_cost
                        rcl.append((insertion_cost, route_idx, i, j))

        if not rcl:
            uninserted.pop(pickup_idx)
            continue  # No feasible insertion for this pair

        # Sort RCL by cost and select top rcl_size
        rcl.sort(key=lambda x: x[0])
        rcl = rcl[:min(rcl_size, len(rcl))]

        # Randomly select an insertion from RCL
        selected_idx = random_state.choice(len(rcl))
        _, route_idx, pickup_pos, delivery_pos = rcl[selected_idx]

        # Insert into existing or new route
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


def solve_pdptw_with_alns(instance: PDPTWInstance, max_iterations: int = 1000,
                          report_interval: int = 100, is_plot=True) -> tuple[State, ALNSProgressLogger]:
    """Solve PDPTW using ALNS with progress tracking"""
    # Create ALNS instance with random state
    random_state = np.random.RandomState(42)
    # Create ALNS instance
    alns = ALNS(random_state)
    
    # Add destroy operators
    alns.add_destroy_operator(random_removal, name='random_removal')
    alns.add_destroy_operator(worst_removal, name='worst_removal')
    alns.add_destroy_operator(shaw_removal, name='shaw_removal')
    alns.add_destroy_operator(proximity_based_removal, name='proximity_based_removal')
    alns.add_destroy_operator(time_based_removal, name='time_based_removal')

    # Add repair operators
    alns.add_repair_operator(greedy_insertion, name='greedy_insertion')
    alns.add_repair_operator(k_regret_insertion, name='k_regret_insertion')
    add_sisr_pdptw_operators(alns)

    # Create initial solution
    initial_solution = create_initial_solution(instance)
    print(f"Initial solution: {len(initial_solution.routes)} routes, "
          f"objective = {initial_solution.objective():.2f}")

    # Set up progress logger with N-Wouda framework callbacks
    progress_logger = ALNSProgressLogger(log_mode='interval', interval=report_interval)
    alns.on_accept(progress_logger.log_progress)  # This is the key difference!

    # Set up SA parameters inspired by baseline_alns_pdptw.py
    tau = 0.011658000847676892              # 0.03  # Control parameter for starting temperature
    initial_distance = initial_solution.objective()
    start_temperature = -tau * initial_distance / np.log(0.5) if initial_distance > 0 else 1000.0
    cooling_rate = 0.9923992759758272       # 0.9990  # Slower cooling rate
    criterion = SimulatedAnnealing(
        start_temperature=round(start_temperature, 4),
        end_temperature=6.315734773880141, # 10,
        step=cooling_rate
    )

    # Override objective method for ALNS iteration to include noise
    original_objective = PDPTWSolution.objective

    def objective_with_noise(self):
        return self.calculate_objective_with_noise(random_state)

    PDPTWSolution.objective = objective_with_noise

    # Set up stopping criterion and selection
    stop_criterion = MaxIterations(max_iterations)
    select = RouletteWheel([13, 4, 3, 0], 0.7615516997372846, 4, 2)

    print(f"\nStarting ALNS optimization for {max_iterations} iterations...")
    print(f"Progress will be reported every {report_interval} iterations.")
    print("=" * 60)

    # Run ALNS - this will automatically call progress_logger.log_progress via callback
    result = alns.iterate(initial_solution, select, criterion, stop_criterion)

    # Restore original objective method
    PDPTWSolution.objective = original_objective

    # Final report
    progress_logger.final_report()

    if is_plot:
        _, ax = plt.subplots(figsize=(12, 6))
        result.plot_objectives(ax=ax)
        plt.show()

    return result.best_state, progress_logger


def main():
    """Main function to demonstrate ALNS on PDPTW"""

    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the data file
    full_path = os.path.join("..", "data", "lc102.txt")
    
    print("Loading PDPTW instance...")

    #
    instance = PDPTWInstance(filename=full_path)

    print(f"Instance loaded:")
    print(f"- Vehicles: {instance.vehicle_number}")
    print(f"- Capacity: {instance.capacity}")
    print(f"- Customers: {instance.n_customers}")
    print(f"- Total nodes: {len(instance.nodes)}")

    print("\nSolving with ALNS...")
    solution, tracker = solve_pdptw_with_alns(instance, max_iterations=3_500, report_interval=150)

    print(f"\nFinal solution:")
    print(f"- Routes: {len(solution.routes)}")
    print(f"- Total distance: {solution.objective():.2f}")
    print(f"- Feasible: {solution.is_feasible()}")

    # Check if all nodes are in the solution
    check_all_nodes_in_solution(solution, instance)

    # Use detailed feasibility check instead of basic check
    feasibility_report = detailed_feasibility_check(solution, instance)

    print("\nRoute details:")
    for i, route in enumerate(solution.routes):
        route_distance = sum(instance.distances[route[j]][route[j + 1]]
                             for j in range(len(route) - 1))
        print(f"Route {i + 1}: {route} (distance: {route_distance:.2f})")


def check_all_nodes_in_solution(solution: PDPTWSolution, instance: PDPTWInstance) -> bool:
    """Check if all nodes from the instance are present in the solution's routes."""
    # Collect all nodes in the solution's routes (excluding depot duplicates)
    solution_nodes = set()
    for route in solution.routes:
        # Exclude depot (node 0) from uniqueness check, as it appears in every route
        solution_nodes.update(node for node in route if node != 0)

    # Get all expected nodes from the instance (excluding depot if appropriate)
    expected_nodes = set(range(1, len(instance.nodes)))  # Exclude depot (node 0)

    # Check if all expected nodes are in the solution
    missing_nodes = expected_nodes - solution_nodes
    extra_nodes = solution_nodes - expected_nodes

    # Print results
    if not missing_nodes and not extra_nodes:
        print(f"Node check: All {len(expected_nodes)} nodes are present in the solution.")
        return True
    else:
        if missing_nodes:
            print(f"Node check failed: Missing nodes {missing_nodes}")
        if extra_nodes:
            print(f"Node check failed: Extra nodes {extra_nodes}")
        return False


def detailed_feasibility_check(solution, instance) -> dict:
    """
    Comprehensive feasibility check that explains all constraint violations
    Returns a dictionary with detailed violation information
    """
    violations = {
        'is_feasible': True,
        'capacity_violations': [],
        'time_window_violations': [],
        'precedence_violations': [],
        'route_structure_violations': [],
        'total_violations': 0
    }

    print("\n" + "=" * 60)
    print("DETAILED FEASIBILITY ANALYSIS")
    print("=" * 60)

    for route_idx, route in enumerate(solution.routes):
        print(f"\nRoute {route_idx + 1}: {route}")

        # Check route structure
        if len(route) < 2:
            violation = f"Route {route_idx + 1} has insufficient nodes: {route}"
            violations['route_structure_violations'].append(violation)
            violations['is_feasible'] = False
            print(f"  X STRUCTURE: {violation}")
            continue

        # Check if route starts and ends with depot
        if route[0] != 0 or route[-1] != 0:
            violation = f"Route {route_idx + 1} doesn't start/end with depot: {route}"
            violations['route_structure_violations'].append(violation)
            violations['is_feasible'] = False
            print(f"  X STRUCTURE: {violation}")

        # Initialize route tracking variables
        current_time = 0
        current_load = 0
        pickup_visited = set()  # Track pickups visited in this route

        # Check each segment in the route
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # Travel time to next node
            travel_time = instance.distances[current_node][next_node]
            current_time += travel_time

            # Skip depot analysis
            if next_node == 0:
                continue

            node_info = instance.nodes[next_node]

            # 1. TIME WINDOW CONSTRAINTS
            original_arrival_time = current_time

            # Check early arrival (waiting time)
            if current_time < node_info['ready_time']:
                waiting_time = node_info['ready_time'] - current_time
                print(f"  WAIT Node {next_node}: Early arrival at {current_time:.1f}, "
                      f"ready at {node_info['ready_time']}, waiting {waiting_time:.1f} min")
                current_time = node_info['ready_time']  # Wait until opening

            # Check late arrival
            if current_time > node_info['due_date']:
                violation_time = current_time - node_info['due_date']
                violation = f"Route {route_idx + 1}, Node {next_node}: Late arrival at {current_time:.1f}, " \
                            f"due by {node_info['due_date']}, violation: {violation_time:.1f} min"
                violations['time_window_violations'].append(violation)
                violations['is_feasible'] = False
                print(f"  X TIME WINDOW: {violation}")
            else:
                print(f"  OK Node {next_node}: Arrival {current_time:.1f} "
                      f"(window: {node_info['ready_time']}-{node_info['due_date']})")

            # 2. CAPACITY CONSTRAINTS
            old_load = current_load
            current_load += node_info['demand']

            if node_info['demand'] > 0:  # Pickup
                pickup_visited.add(next_node)
                print(f"  PICKUP {next_node}: Load {old_load} + {node_info['demand']} = {current_load}")
            else:  # Delivery
                print(f"  DELIVERY {next_node}: Load {old_load} + ({node_info['demand']}) = {current_load}")

            if current_load > instance.capacity:
                violation = f"Route {route_idx + 1}, Node {next_node}: Capacity exceeded, " \
                            f"load: {current_load}, capacity: {instance.capacity}"
                violations['capacity_violations'].append(violation)
                violations['is_feasible'] = False
                print(f"  X CAPACITY: {violation}")
            elif current_load < 0:
                violation = f"Route {route_idx + 1}, Node {next_node}: Negative load: {current_load}"
                violations['capacity_violations'].append(violation)
                violations['is_feasible'] = False
                print(f"  X CAPACITY: {violation}")

            # 3. PRECEDENCE CONSTRAINTS (pickup before delivery)
            if node_info['demand'] < 0:  # This is a delivery node
                pickup_node_id = node_info['pickup_index']
                if pickup_node_id > 0:  # Has corresponding pickup
                    if pickup_node_id not in pickup_visited:
                        # Check if pickup is later in this route
                        if pickup_node_id in route[i + 1:]:
                            violation = f"Route {route_idx + 1}: Delivery {next_node} before pickup {pickup_node_id}"
                            violations['precedence_violations'].append(violation)
                            violations['is_feasible'] = False
                            print(f"  X PRECEDENCE: {violation}")
                        else:
                            # Pickup not in this route at all - check if it's in any other route
                            pickup_found_elsewhere = False
                            for other_route in solution.routes:
                                if pickup_node_id in other_route:
                                    pickup_found_elsewhere = True
                                    break

                            if not pickup_found_elsewhere:
                                violation = f"Route {route_idx + 1}: Delivery {next_node} without pickup {pickup_node_id} anywhere"
                                violations['precedence_violations'].append(violation)
                                violations['is_feasible'] = False
                                print(f"  X PRECEDENCE: {violation}")
                            else:
                                # This might be OK if we allow cross-route precedence
                                print(f"  NOTE: Delivery {next_node} has pickup {pickup_node_id} in different route")
                    else:
                        print(f"  OK Precedence: Pickup {pickup_node_id} before delivery {next_node}")

            # Add service time
            current_time += node_info['service_time']

        print(f"  SUMMARY: Final load: {current_load}, Total time: {current_time:.1f}")

        # Check if route ends with balanced load (should be 0)
        if current_load != 0:
            violation = f"Route {route_idx + 1} ends with non-zero load: {current_load}"
            violations['capacity_violations'].append(violation)
            violations['is_feasible'] = False
            print(f"  X LOAD BALANCE: {violation}")

    # Calculate total violations
    violations['total_violations'] = (len(violations['capacity_violations']) +
                                      len(violations['time_window_violations']) +
                                      len(violations['precedence_violations']) +
                                      len(violations['route_structure_violations']))

    # Print summary
    print(f"\n" + "=" * 60)
    print("FEASIBILITY SUMMARY")
    print("=" * 60)
    print(f"Overall feasible: {violations['is_feasible']}")
    print(f"Total violations: {violations['total_violations']}")
    print(f"  - Capacity violations: {len(violations['capacity_violations'])}")
    print(f"  - Time window violations: {len(violations['time_window_violations'])}")
    print(f"  - Precedence violations: {len(violations['precedence_violations'])}")
    print(f"  - Structure violations: {len(violations['route_structure_violations'])}")

    if not violations['is_feasible']:
        print(f"\nDETAILED VIOLATIONS:")
        for category, violation_list in violations.items():
            if category != 'is_feasible' and category != 'total_violations' and violation_list:
                print(f"\n{category.upper().replace('_', ' ')}:")
                for violation in violation_list:
                    print(f"  - {violation}")

    print("=" * 60)
    return violations

if __name__ == "__main__":
    # Start timer
    start_time = time.time()
    #
    main()
    #
    end_timer = time.time()
    end_time = end_timer - start_time

    print("================================================================")
    print('Finished performing everything, time elapsed {}'.format(str(timedelta(seconds=end_time))))

    print("================================================================")
