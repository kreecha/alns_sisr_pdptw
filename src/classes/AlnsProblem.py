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

import itertools
import math
import os
from typing import List, Dict, Tuple

import numpy as np


class PDPTWInstance:
    """Represents a PDPTW instance from Li-Lim benchmark"""

    def __init__(self, filename: str = None, data: str = None):
        self.filename = None  # Initialize filename attribute
        if filename and not data:
            self.load_from_file(filename)
        elif data and not filename:
            self.parse_data(data)
            self.filename = "from_data"  # Default for data-based instances
        elif filename and data:
            # If both are provided, treat 'data' as the folder path and 'filename' as the file
            import os
            full_path = os.path.join(data, filename)
            self.load_from_file(full_path)
        else:
            raise ValueError("Either filename or data must be provided")

    def load_from_file(self, filepath: str):
        """Load PDPTW data from file"""
        self.filename = os.path.basename(filepath)  # Store filename
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

    def _is_route_feasible_original(self, route):
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

    def _is_route_feasible(self, route):
        """Check if a single route is feasible with improved time window handling"""
        if len(route) < 2:
            return True

        current_time = 0.0  # Use float for precise time calculations
        current_load = 0
        pickup_visited = set()  # Track pickups visited in this route

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # Travel time to next node
            travel_time = self.instance.distances[current_node][next_node]
            current_time += travel_time

            # Skip depot analysis (depot has no constraints typically)
            if next_node == 0:
                continue

            node_info = self.instance.nodes[next_node]

            # TIME WINDOW CHECK
            # Handle early arrival (waiting time)
            if current_time < node_info['ready_time']:
                current_time = float(node_info['ready_time'])  # Wait until ready time

            # Check late arrival - this is a hard constraint violation
            if current_time > node_info['due_date']:
                return False  # Route is infeasible due to time window violation

            # CAPACITY CHECK
            current_load += node_info['demand']

            # Check capacity constraints
            if current_load > self.instance.capacity or current_load < 0:
                return False

            # PRECEDENCE CHECK for pickup-delivery constraints
            if node_info['demand'] > 0:  # This is a pickup
                pickup_visited.add(next_node)
            elif node_info['demand'] < 0:  # This is a delivery
                pickup_idx = node_info['pickup_index']
                if pickup_idx > 0:  # Has corresponding pickup
                    # Check if pickup was visited before delivery in this route
                    if pickup_idx not in pickup_visited:
                        # Check if pickup appears later in route (which would be invalid)
                        if pickup_idx in route[i + 1:]:
                            return False  # Delivery before pickup
                        # If pickup not in route at all, this might be cross-route
                        # For strict PDPTW, pickup and delivery should be in same route
                        else:
                            return False  # Pickup not found in route

            # Add service time after all checks
            current_time += node_info['service_time']

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

    # =========== Time window violation prevention ============= #
    # 1. IMPROVED FEASIBILITY CHECKING WITH STRICT VALIDATION
    def _is_route_feasible_strict(self, route, debug=False):
        """Strict feasibility check that catches all violations"""
        if len(route) < 2:
            return True

        current_time = 0.0
        current_load = 0
        pickup_visited = set()

        if debug:
            print(f"Checking route: {route}")

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # Travel time to next node
            travel_time = float(self.instance.distances[current_node][next_node])
            current_time += travel_time

            # Skip depot analysis
            if next_node == 0:
                continue

            node_info = self.instance.nodes[next_node]

            # TIME WINDOW CHECK - STRICT
            # Wait if arriving early
            if current_time < node_info['ready_time']:
                current_time = float(node_info['ready_time'])

            # HARD CONSTRAINT: Cannot arrive late
            if current_time > node_info['due_date']:
                if debug:
                    print(f"  TIME VIOLATION: Node {next_node} arrival {current_time} > due {node_info['due_date']}")
                return False

            # CAPACITY CHECK - STRICT
            current_load += node_info['demand']
            if current_load > self.instance.capacity or current_load < 0:
                if debug:
                    print(f"  CAPACITY VIOLATION: Load {current_load}, capacity {self.instance.capacity}")
                return False

            # PRECEDENCE CHECK - STRICT
            if node_info['demand'] > 0:  # Pickup
                pickup_visited.add(next_node)
            elif node_info['demand'] < 0:  # Delivery
                pickup_idx = node_info['pickup_index']
                if pickup_idx > 0:
                    if pickup_idx not in pickup_visited:
                        if debug:
                            print(f"  PRECEDENCE VIOLATION: Delivery {next_node} before pickup {pickup_idx}")
                        return False

            # Add service time
            current_time += node_info['service_time']

        return True

    # 2. PENALTY-BASED OBJECTIVE FOR INFEASIBLE SOLUTIONS
    def calculate_objective_with_penalties(self, penalty_factor=10000):
        """Calculate objective with heavy penalties for infeasible solutions"""
        base_distance = self.calculate_objective()

        # Check feasibility and add penalties
        penalty = 0

        for route in self.routes:
            if not self._is_route_feasible_strict(route):
                # Add penalty for each infeasible route
                penalty += penalty_factor

                # Additional penalties for specific violations
                penalty += self._calculate_violation_penalties(route, penalty_factor)

        self._objective_value = base_distance + penalty
        return self._objective_value

    def _calculate_violation_penalties(self, route, base_penalty):
        """Calculate specific penalties for different types of violations"""
        penalty = 0
        current_time = 0.0
        current_load = 0
        pickup_visited = set()

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            travel_time = float(self.instance.distances[current_node][next_node])
            current_time += travel_time

            if next_node == 0:
                continue

            node_info = self.instance.nodes[next_node]

            # Time window penalties
            if current_time < node_info['ready_time']:
                current_time = float(node_info['ready_time'])
            elif current_time > node_info['due_date']:
                # Heavy penalty for late arrival
                lateness = current_time - node_info['due_date']
                penalty += base_penalty * lateness * 0.1

            # Capacity penalties
            current_load += node_info['demand']
            if current_load > self.instance.capacity:
                penalty += base_penalty * (current_load - self.instance.capacity)
            elif current_load < 0:
                penalty += base_penalty * abs(current_load)

            # Precedence penalties
            if node_info['demand'] > 0:
                pickup_visited.add(next_node)
            elif node_info['demand'] < 0:
                pickup_idx = node_info['pickup_index']
                if pickup_idx > 0 and pickup_idx not in pickup_visited:
                    penalty += base_penalty

            current_time += node_info['service_time']

        return penalty


class ValidatedPDPTWSolution(PDPTWSolution):
    """Wrapper that validates all operations"""

    def __init__(self, instance: PDPTWInstance):
        super().__init__(instance)

    def copy(self):
        new_solution = ValidatedPDPTWSolution(self.instance)
        new_solution.routes = [route[:] for route in self.routes]
        new_solution._objective_value = self._objective_value
        return new_solution

    def objective(self):
        # Always return penalized objective during optimization
        return self.calculate_objective_with_penalties()

    def is_feasible(self):
        """Enhanced feasibility check"""
        for route in self.routes:
            if not self._is_route_feasible_strict(route):
                return False
        return True


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