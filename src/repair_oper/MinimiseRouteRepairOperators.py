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


# =============================================================================
# ROUTE MINIMIZATION REPAIR OPERATORS
# =============================================================================

def route_aware_regret_insertion(current_state, random_state: np.random.RandomState):
    """
    Modified regret insertion that heavily penalizes new route creation
    """
    repaired = current_state.copy()

    # Get unassigned pairs
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

    while unassigned_pairs:
        best_insertion = None
        max_regret = -float('inf')

        for pickup_id, delivery_id in unassigned_pairs:
            costs = []
            feasible_insertions = []

            # Find all feasible insertions in EXISTING routes only
            for route in repaired.routes:
                route_best_cost = float('inf')
                route_best_pos = None

                for pickup_pos in range(1, len(route)):
                    for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                        temp_route = route[:]
                        temp_route.insert(pickup_pos, pickup_id)
                        temp_route.insert(delivery_pos, delivery_id)

                        if repaired._is_route_feasible(temp_route):
                            old_cost = sum(repaired.instance.distances[route[k]][route[k + 1]]
                                           for k in range(len(route) - 1))
                            new_cost = sum(repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                           for k in range(len(temp_route) - 1))
                            cost = new_cost - old_cost

                            if cost < route_best_cost:
                                route_best_cost = cost
                                route_best_pos = (route, pickup_pos, delivery_pos)

                if route_best_pos is not None:
                    costs.append(route_best_cost)
                    feasible_insertions.append(route_best_pos)

            # Calculate regret with heavy penalty for new route creation
            if len(costs) >= 2:
                costs.sort()
                regret = costs[1] - costs[0]
            elif len(costs) == 1:
                regret = costs[0]
            else:
                # No existing route insertion possible - massive penalty
                regret = 1000.0  # Heavy penalty for new route

            if regret > max_regret:
                max_regret = regret
                best_insertion = (pickup_id, delivery_id, feasible_insertions)

        if best_insertion:
            pickup_id, delivery_id, feasible_insertions = best_insertion
            unassigned_pairs.remove((pickup_id, delivery_id))

            if feasible_insertions:
                # Insert into best existing route
                best_route_info = min(feasible_insertions,
                                      key=lambda x: sum(repaired.instance.distances[x[0][k]][x[0][k + 1]]
                                                        for k in range(len(x[0]) - 1)))
                route, pickup_pos, delivery_pos = best_route_info
                route.insert(pickup_pos, pickup_id)
                route.insert(delivery_pos, delivery_id)
            else:
                # Create new route only if no choice
                if len(repaired.routes) < repaired.instance.vehicle_number:
                    new_route = [0, pickup_id, delivery_id, 0]
                    repaired.routes.append(new_route)
                else:
                    # Cannot create more routes - skip this pair
                    break
        else:
            break

    repaired.calculate_objective()
    return repaired

def consolidation_repair(current_state, random_state: np.random.RandomState):
    """
    Repair by aggressively trying to fit removed customers into existing routes
    Strongly biases against creating new routes
    pdp_200\lr2_2_8.txt
    3 routes, distance: 2127.33, feasible: True
    overall performance gap is 3.50%
    time elapsed 0:23:52.907987
    """
    repaired = current_state.copy()

    # Get unassigned pickup-delivery pairs
    all_pickups = set()
    for node in repaired.instance.nodes[1:]:
        if node['demand'] > 0:
            all_pickups.add(node['id'])

    assigned_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                assigned_pickups.add(node)

    unassigned = list(all_pickups - assigned_pickups)
    unassigned_pairs = []

    for pickup_id in unassigned:
        delivery_id = repaired.instance.nodes[pickup_id]['delivery_index']
        if delivery_id > 0 and delivery_id not in assigned_pickups:
            unassigned_pairs.append((pickup_id, delivery_id))

    # Sort pairs by total time window flexibility (tightest first)
    unassigned_pairs.sort(key=lambda pair:
    (repaired.instance.nodes[pair[0]]['due_date'] -
     repaired.instance.nodes[pair[0]]['ready_time']) +
    (repaired.instance.nodes[pair[1]]['due_date'] -
     repaired.instance.nodes[pair[1]]['ready_time']))

    for pickup_id, delivery_id in unassigned_pairs:
        # Try VERY hard to insert into existing routes
        best_insertion = None
        best_cost = float('inf')

        # Check all possible insertions with relaxed cost criteria
        for route in repaired.routes:
            for pickup_pos in range(1, len(route)):
                for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                    # Create temporary route
                    temp_route = route[:]
                    temp_route.insert(pickup_pos, pickup_id)
                    temp_route.insert(delivery_pos, delivery_id)

                    # Check feasibility
                    if repaired._is_route_feasible(temp_route):
                        # Calculate insertion cost
                        old_cost = sum(repaired.instance.distances[route[k]][route[k + 1]]
                                       for k in range(len(route) - 1))
                        new_cost = sum(repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                       for k in range(len(temp_route) - 1))
                        cost = new_cost - old_cost

                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (route, pickup_pos, delivery_pos)

        # Insert into existing route if possible (strongly preferred)
        if best_insertion:
            route, pickup_pos, delivery_pos = best_insertion
            route.insert(pickup_pos, pickup_id)
            route.insert(delivery_pos, delivery_id)
        else:
            # Only create new route if absolutely necessary
            # But penalize heavily by making it less likely to be accepted
            if len(repaired.routes) < repaired.instance.vehicle_number:
                new_route = [0, pickup_id, delivery_id, 0]
                if repaired._is_route_feasible(new_route):
                    repaired.routes.append(new_route)

    repaired.calculate_objective()
    return repaired


def consolidation_repair_with_capacity_filter(current_state, random_state: np.random.RandomState):
    """
    Fixed consolidation repair with capacity pre-filtering
    Only adds 5-10 lines but provides 2-3x speed improvement

    pdp_200\lr2_2_8.txt
    3 routes, distance: 2127.33, feasible: True
    overall performance gap is 3.50%
    time elapsed 0:18:30.366961

    Final solution: 3 routes, distance: 2127.33, feasible: True
    """
    repaired = current_state.copy()

    # Get unassigned pickup-delivery pairs (same as original)
    all_pickups = set()
    for node in repaired.instance.nodes[1:]:
        if node['demand'] > 0:
            all_pickups.add(node['id'])

    assigned_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                assigned_pickups.add(node)

    unassigned = list(all_pickups - assigned_pickups)
    unassigned_pairs = []

    for pickup_id in unassigned:
        delivery_id = repaired.instance.nodes[pickup_id]['delivery_index']
        if delivery_id > 0 and delivery_id not in assigned_pickups:
            unassigned_pairs.append((pickup_id, delivery_id))

    # Sort pairs by total time window flexibility (same as original)
    unassigned_pairs.sort(key=lambda pair:
    (repaired.instance.nodes[pair[0]]['due_date'] -
     repaired.instance.nodes[pair[0]]['ready_time']) +
    (repaired.instance.nodes[pair[1]]['due_date'] -
     repaired.instance.nodes[pair[1]]['ready_time']))

    for pickup_id, delivery_id in unassigned_pairs:
        best_insertion = None
        best_cost = float('inf')

        # CAPACITY PRE-FILTERING - This is the key fix
        pair_demand = repaired.instance.nodes[pickup_id]['demand']

        # Pre-filter routes by capacity before trying any insertions
        feasible_routes = []
        for route in repaired.routes:
            current_load = sum(repaired.instance.nodes[n]['demand'] for n in route if n != 0)
            if current_load + pair_demand <= repaired.instance.capacity:
                feasible_routes.append(route)

        # Only check feasible routes (MAJOR SPEEDUP)
        for route in feasible_routes:
            for pickup_pos in range(1, len(route)):
                for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                    # Create temporary route
                    temp_route = route[:]
                    temp_route.insert(pickup_pos, pickup_id)
                    temp_route.insert(delivery_pos, delivery_id)

                    # Now we only call expensive feasibility check on capacity-feasible routes
                    if repaired._is_route_feasible(temp_route):
                        # Calculate insertion cost (same as original)
                        old_cost = sum(repaired.instance.distances[route[k]][route[k + 1]]
                                       for k in range(len(route) - 1))
                        new_cost = sum(repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                       for k in range(len(temp_route) - 1))
                        cost = new_cost - old_cost

                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (route, pickup_pos, delivery_pos)

        # Insert into existing route if possible (same as original)
        if best_insertion:
            route, pickup_pos, delivery_pos = best_insertion
            route.insert(pickup_pos, pickup_id)
            route.insert(delivery_pos, delivery_id)
        else:
            # Only create new route if absolutely necessary (same as original)
            if len(repaired.routes) < repaired.instance.vehicle_number:
                new_route = [0, pickup_id, delivery_id, 0]
                if repaired._is_route_feasible(new_route):
                    repaired.routes.append(new_route)

    repaired.calculate_objective()
    return repaired


def consolidation_repair_with_capacity_bounded_search_filter(current_state, random_state: np.random.RandomState,
                                        position_limit_ratio: float = 0.9):
    """
    Consolidation repair with capacity and bounded position search filter.

    pdp_200\lr2_2_8.txt
    3 routes, distance: 2172.64, feasible: True

    time elapsed 0:15:36.166171

    Final solution: 3 routes, distance: 2127.33, feasible: True

    Args:
        position_limit_ratio: Fraction of positions to search (0.9 = 90% of positions)
    """
    repaired = current_state.copy()

    # Get unassigned pickup-delivery pairs (same as before)
    all_pickups = set()
    for node in repaired.instance.nodes[1:]:
        if node['demand'] > 0:
            all_pickups.add(node['id'])

    assigned_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                assigned_pickups.add(node)

    unassigned = list(all_pickups - assigned_pickups)
    unassigned_pairs = []

    for pickup_id in unassigned:
        delivery_id = repaired.instance.nodes[pickup_id]['delivery_index']
        if delivery_id > 0 and delivery_id not in assigned_pickups:
            unassigned_pairs.append((pickup_id, delivery_id))

    # Sort pairs by total time window flexibility
    unassigned_pairs.sort(key=lambda pair:
    (repaired.instance.nodes[pair[0]]['due_date'] -
     repaired.instance.nodes[pair[0]]['ready_time']) +
    (repaired.instance.nodes[pair[1]]['due_date'] -
     repaired.instance.nodes[pair[1]]['ready_time']))

    for pickup_id, delivery_id in unassigned_pairs:
        best_insertion = None
        best_cost = float('inf')

        # Capacity pre-filtering
        pair_demand = repaired.instance.nodes[pickup_id]['demand']
        feasible_routes = []
        for route in repaired.routes:
            current_load = sum(repaired.instance.nodes[n]['demand'] for n in route if n != 0)
            if current_load + pair_demand <= repaired.instance.capacity:
                feasible_routes.append(route)

        # Bounded position search for each feasible route
        for route in feasible_routes:
            # BOUNDED SEARCH - Limit positions to search
            max_pickup_positions = max(1, int((len(route) - 1) * position_limit_ratio))

            for pickup_pos in range(1, max_pickup_positions + 1):
                # For delivery positions, also apply the limit
                max_delivery_start = pickup_pos + 1
                max_delivery_end = min(len(route) + 1,
                                       pickup_pos + 1 + int((len(route) - pickup_pos) * position_limit_ratio))

                for delivery_pos in range(max_delivery_start, max_delivery_end + 1):
                    temp_route = route[:]
                    temp_route.insert(pickup_pos, pickup_id)
                    temp_route.insert(delivery_pos, delivery_id)

                    if repaired._is_route_feasible(temp_route):
                        old_cost = sum(repaired.instance.distances[route[k]][route[k + 1]]
                                       for k in range(len(route) - 1))
                        new_cost = sum(repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                       for k in range(len(temp_route) - 1))
                        cost = new_cost - old_cost

                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (route, pickup_pos, delivery_pos)

        # Insert into existing route if possible
        if best_insertion:
            route, pickup_pos, delivery_pos = best_insertion
            route.insert(pickup_pos, pickup_id)
            route.insert(delivery_pos, delivery_id)
        else:
            # Create new route if necessary
            if len(repaired.routes) < repaired.instance.vehicle_number:
                new_route = [0, pickup_id, delivery_id, 0]
                if repaired._is_route_feasible(new_route):
                    repaired.routes.append(new_route)

    repaired.calculate_objective()
    return repaired

def forced_merge_repair(current_state, random_state: np.random.RandomState):
    """
    Repair that attempts to merge routes during the repair process
    """
    repaired = current_state.copy()

    # First do standard repair
    repaired = consolidation_repair(repaired, random_state)

    # Then attempt aggressive route merging
    merged_any = True
    while merged_any:
        merged_any = False

        # Try all possible route pairs for merging
        for i in range(len(repaired.routes)):
            for j in range(i + 1, len(repaired.routes)):
                if j >= len(repaired.routes):
                    break

                route1 = repaired.routes[i]
                route2 = repaired.routes[j]

                # Calculate combined load
                load1 = sum(repaired.instance.nodes[c]['demand'] for c in route1 if c != 0)
                load2 = sum(repaired.instance.nodes[c]['demand'] for c in route2 if c != 0)

                if abs(load1 + load2) <= repaired.instance.capacity:
                    # Try different merge strategies
                    customers1 = [c for c in route1 if c != 0]
                    customers2 = [c for c in route2 if c != 0]

                    # Strategy 1: route1 + route2
                    merged = [0] + customers1 + customers2 + [0]
                    if repaired._is_route_feasible(merged):
                        repaired.routes[i] = merged
                        repaired.routes.pop(j)
                        merged_any = True
                        break

                    # Strategy 2: route2 + route1
                    merged = [0] + customers2 + customers1 + [0]
                    if repaired._is_route_feasible(merged):
                        repaired.routes[i] = merged
                        repaired.routes.pop(j)
                        merged_any = True
                        break

            if merged_any:
                break

    repaired.calculate_objective()
    return repaired


def greedy_route_consolidation_repair(current_state, random_state: np.random.RandomState):
    """
    Repair that greedily tries to maximize route utilization
    """
    repaired = current_state.copy()

    # Get unassigned pairs
    unassigned_pairs = []
    all_pickups = set()
    for node in repaired.instance.nodes[1:]:
        if node['demand'] > 0:
            all_pickups.add(node['id'])

    assigned_pickups = set()
    for route in repaired.routes:
        for node in route:
            if node in all_pickups:
                assigned_pickups.add(node)

    for pickup_id in all_pickups - assigned_pickups:
        delivery_id = repaired.instance.nodes[pickup_id]['delivery_index']
        if delivery_id > 0:
            unassigned_pairs.append((pickup_id, delivery_id))

    # Calculate route utilization and sort routes by available capacity
    route_capacities = []
    for i, route in enumerate(repaired.routes):
        current_load = abs(sum(repaired.instance.nodes[c]['demand'] for c in route if c != 0))
        available_capacity = repaired.instance.capacity - current_load
        route_capacities.append((available_capacity, i, route))

    # Sort by available capacity (most available first)
    route_capacities.sort(key=lambda x: x[0], reverse=True)

    # Try to insert pairs into routes with most available capacity
    for pickup_id, delivery_id in unassigned_pairs:
        pair_demand = repaired.instance.nodes[pickup_id]['demand']
        inserted = False

        for available_cap, route_idx, route in route_capacities:
            if available_cap >= pair_demand:
                # Try to insert this pair
                best_cost = float('inf')
                best_positions = None

                for pickup_pos in range(1, len(route)):
                    for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                        temp_route = route[:]
                        temp_route.insert(pickup_pos, pickup_id)
                        temp_route.insert(delivery_pos, delivery_id)

                        if repaired._is_route_feasible(temp_route):
                            old_cost = sum(repaired.instance.distances[route[k]][route[k + 1]]
                                           for k in range(len(route) - 1))
                            new_cost = sum(repaired.instance.distances[temp_route[k]][temp_route[k + 1]]
                                           for k in range(len(temp_route) - 1))
                            cost = new_cost - old_cost

                            if cost < best_cost:
                                best_cost = cost
                                best_positions = (pickup_pos, delivery_pos)

                if best_positions:
                    pickup_pos, delivery_pos = best_positions
                    route.insert(pickup_pos, pickup_id)
                    route.insert(delivery_pos, delivery_id)
                    # Update available capacity
                    route_capacities[route_capacities.index((available_cap, route_idx, route))] = (
                        available_cap - pair_demand, route_idx, route)
                    inserted = True
                    break

        if not inserted:
            # Create new route only as last resort
            if len(repaired.routes) < repaired.instance.vehicle_number:
                new_route = [0, pickup_id, delivery_id, 0]
                if repaired._is_route_feasible(new_route):
                    repaired.routes.append(new_route)

    repaired.calculate_objective()
    return repaired
