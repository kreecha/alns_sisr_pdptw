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

from src.classes.AlnsProblem import PDPTWSolution



def random_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Optimized random removal - the original was mostly fine but can be slightly improved"""
    destroyed = current.copy()

    if not destroyed.routes:
        return destroyed

    # More efficient pair collection using list comprehension
    pairs = []
    for route_idx, route in enumerate(destroyed.routes):
        pickup_nodes = [node for node in route[1:-1] if destroyed.instance.nodes[node]['demand'] > 0]
        for pickup in pickup_nodes:
            delivery_idx = destroyed.instance.nodes[pickup]['delivery_index']
            if delivery_idx in route:
                pairs.append((route_idx, pickup, delivery_idx))

    if not pairs:
        return destroyed

    # Limit removal size to prevent excessive destruction
    n_remove = min(random_state.randint(1, 4), len(pairs), 3)  # Cap at 3 pairs max
    removed_pairs = random_state.choice(len(pairs), n_remove, replace=False)

    # More efficient removal - sort by route index descending for safe removal
    pairs_to_remove = sorted([pairs[i] for i in removed_pairs], key=lambda x: x[0], reverse=True)

    # Remove pairs and clean up empty routes in one pass
    routes_to_remove = set()
    for route_idx, pickup, delivery in pairs_to_remove:
        if route_idx < len(destroyed.routes):
            route = destroyed.routes[route_idx]
            if pickup in route: route.remove(pickup)
            if delivery in route: route.remove(delivery)
            if len(route) <= 2:
                routes_to_remove.add(route_idx)

    # Remove empty routes efficiently
    if routes_to_remove:
        destroyed.routes = [route for i, route in enumerate(destroyed.routes)
                            if i not in routes_to_remove]

    destroyed.calculate_objective()
    return destroyed


def proximity_based_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Optimized proximity-based removal with better distance calculations"""
    destroyed = current.copy()
    if not destroyed.routes:
        return destroyed

    # Collect pickup nodes more efficiently
    pickup_nodes = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:
            if destroyed.instance.nodes[node]['demand'] > 0:
                delivery_idx = destroyed.instance.nodes[node]['delivery_index']
                if delivery_idx in route:  # Ensure delivery is in same route
                    pickup_nodes.append((route_idx, node, delivery_idx))

    if not pickup_nodes:
        return destroyed

    # Select random starting pair
    start_idx = random_state.choice(len(pickup_nodes))
    start_route_idx, start_pickup, start_delivery = pickup_nodes[start_idx]

    # Pre-compute distances for all pickup nodes to avoid repeated calculations
    distances = []
    for route_idx, pickup, delivery in pickup_nodes:
        if pickup != start_pickup:  # Skip the starting pickup
            dist = destroyed.instance.distances[start_pickup][pickup]
            distances.append((dist, route_idx, pickup, delivery))

    if not distances:
        # Only one pair available, remove just the starting pair
        pairs_to_remove = [(start_route_idx, start_pickup, start_delivery)]
    else:
        # Sort by distance and select closest pairs
        distances.sort(key=lambda x: x[0])
        n_remove = min(random_state.randint(1, 3), len(distances))  # Reduced max from 4 to 3

        pairs_to_remove = [(r[1], r[2], r[3]) for r in distances[:n_remove]]
        pairs_to_remove.append((start_route_idx, start_pickup, start_delivery))

    # Remove pairs efficiently (sort by route index for safe removal)
    pairs_to_remove.sort(key=lambda x: x[0], reverse=True)
    routes_to_remove = set()

    for route_idx, pickup, delivery in pairs_to_remove:
        if route_idx < len(destroyed.routes):
            route = destroyed.routes[route_idx]
            if pickup in route: route.remove(pickup)
            if delivery in route: route.remove(delivery)
            if len(route) <= 2:
                routes_to_remove.add(route_idx)

    # Clean up empty routes
    if routes_to_remove:
        destroyed.routes = [route for i, route in enumerate(destroyed.routes)
                            if i not in routes_to_remove]

    destroyed.calculate_objective()
    return destroyed


def time_based_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Optimized time-based removal with better time window calculations"""
    destroyed = current.copy()
    if not destroyed.routes:
        return destroyed

    # Collect pickup nodes with their time windows
    pickup_nodes = []
    for route_idx, route in enumerate(destroyed.routes):
        for node in route[1:-1]:
            if destroyed.instance.nodes[node]['demand'] > 0:
                delivery_idx = destroyed.instance.nodes[node]['delivery_index']
                if delivery_idx in route:
                    pickup_nodes.append((route_idx, node, delivery_idx))

    if not pickup_nodes:
        return destroyed

    # Select random starting pair
    start_idx = random_state.choice(len(pickup_nodes))
    start_route_idx, start_pickup, start_delivery = pickup_nodes[start_idx]
    start_pickup_info = destroyed.instance.nodes[start_pickup]

    # Calculate time window similarities more efficiently
    tw_similarities = []
    for route_idx, pickup, delivery in pickup_nodes:
        if pickup != start_pickup:
            pickup_info = destroyed.instance.nodes[pickup]

            # Simple time window difference calculation
            tw_diff = abs(start_pickup_info['ready_time'] - pickup_info['ready_time'])
            tw_similarities.append((tw_diff, route_idx, pickup, delivery))

    if not tw_similarities:
        # Only one pair available
        pairs_to_remove = [(start_route_idx, start_pickup, start_delivery)]
    else:
        # Sort by time window similarity (smallest difference first = most similar)
        tw_similarities.sort(key=lambda x: x[0])
        n_remove = min(random_state.randint(1, 3), len(tw_similarities))  # Reduced max from 4 to 3

        pairs_to_remove = [(r[1], r[2], r[3]) for r in tw_similarities[:n_remove]]
        pairs_to_remove.append((start_route_idx, start_pickup, start_delivery))

    # Remove pairs efficiently
    pairs_to_remove.sort(key=lambda x: x[0], reverse=True)
    routes_to_remove = set()

    for route_idx, pickup, delivery in pairs_to_remove:
        if route_idx < len(destroyed.routes):
            route = destroyed.routes[route_idx]
            if pickup in route: route.remove(pickup)
            if delivery in route: route.remove(delivery)
            if len(route) <= 2:
                routes_to_remove.add(route_idx)

    # Clean up empty routes
    if routes_to_remove:
        destroyed.routes = [route for i, route in enumerate(destroyed.routes)
                            if i not in routes_to_remove]

    destroyed.calculate_objective()
    return destroyed


def shaw_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Optimized Shaw removal with reduced computational complexity"""
    destroyed = current.copy()
    if not destroyed.routes:
        return destroyed

    # Collect pickup-delivery pairs
    pairs = []
    for route_idx, route in enumerate(destroyed.routes):
        pickup_nodes = [node for node in route[1:-1] if destroyed.instance.nodes[node]['demand'] > 0]
        for pickup in pickup_nodes:
            delivery_idx = destroyed.instance.nodes[pickup]['delivery_index']
            if delivery_idx in route:
                pairs.append((route_idx, pickup, delivery_idx))

    if not pairs:
        return destroyed

    # Select random starting pair
    start_pair_idx = random_state.choice(len(pairs))
    start_route_idx, start_pickup, start_delivery = pairs[start_pair_idx]

    # Pre-compute normalization factors once
    max_dist = destroyed.instance.distances.max()
    max_tw = max(node['due_date'] - node['ready_time'] for node in destroyed.instance.nodes[1:])
    max_demand = max(abs(node['demand']) for node in destroyed.instance.nodes[1:] if node['demand'] != 0)

    start_pickup_info = destroyed.instance.nodes[start_pickup]
    start_delivery_info = destroyed.instance.nodes[start_delivery]

    # Calculate relatedness
    relatednesses = []
    for pair_idx, (route_idx, pickup, delivery) in enumerate(pairs):
        if pair_idx == start_pair_idx:
            continue

        pickup_info = destroyed.instance.nodes[pickup]
        delivery_info = destroyed.instance.nodes[delivery]

        # Combined relatedness calculation
        dist_relatedness = (destroyed.instance.distances[start_pickup][pickup] +
                            destroyed.instance.distances[start_delivery][delivery]) / 2
        tw_relatedness = (abs(start_pickup_info['ready_time'] - pickup_info['ready_time']) +
                          abs(start_delivery_info['ready_time'] - delivery_info['ready_time'])) / 2
        demand_relatedness = abs(start_pickup_info['demand'] - pickup_info['demand'])

        relatedness = (dist_relatedness / max_dist +
                       tw_relatedness / max_tw +
                       demand_relatedness / max_demand)

        relatednesses.append((relatedness, route_idx, pickup, delivery))

    # IMPROVED: Allow more removal for better diversification
    relatednesses.sort(key=lambda x: x[0])
    n_remove = min(random_state.randint(1, 5), len(relatednesses))  # Increased to 5
    pairs_to_remove = [(r[1], r[2], r[3]) for r in relatednesses[:n_remove]]
    pairs_to_remove.append((start_route_idx, start_pickup, start_delivery))

    # Remove pairs efficiently
    pairs_to_remove.sort(key=lambda x: x[0], reverse=True)
    for route_idx, pickup, delivery in pairs_to_remove:
        if route_idx < len(destroyed.routes):
            route = destroyed.routes[route_idx]
            if pickup in route: route.remove(pickup)
            if delivery in route: route.remove(delivery)
            if len(route) <= 2:
                destroyed.routes.pop(route_idx)

    destroyed.calculate_objective()
    return destroyed


def worst_removal(current: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
    """Optimized worst removal with better cost calculation"""
    destroyed = current.copy()

    # Pre-compute all pair costs in one pass
    pair_costs = []
    for route_idx, route in enumerate(destroyed.routes):
        for i, node in enumerate(route[1:-1], 1):
            if destroyed.instance.nodes[node]['demand'] > 0:  # Pickup node
                delivery_idx = destroyed.instance.nodes[node]['delivery_index']
                if delivery_idx in route:
                    delivery_pos = route.index(delivery_idx)

                    # Optimized cost calculation
                    cost = 0
                    # Pickup removal cost
                    if 1 < i < len(route) - 1:
                        cost += (destroyed.instance.distances[route[i - 1]][node] +
                                 destroyed.instance.distances[node][route[i + 1]] -
                                 destroyed.instance.distances[route[i - 1]][route[i + 1]])

                    # Delivery removal cost
                    if 1 < delivery_pos < len(route) - 1:
                        cost += (destroyed.instance.distances[route[delivery_pos - 1]][delivery_idx] +
                                 destroyed.instance.distances[delivery_idx][route[delivery_pos + 1]] -
                                 destroyed.instance.distances[route[delivery_pos - 1]][route[delivery_pos + 1]])

                    pair_costs.append((cost, route_idx, node, delivery_idx))

    if not pair_costs:
        return destroyed

    # Remove top 1-2 worst pairs (reduced from potentially more)
    pair_costs.sort(reverse=True)
    n_remove = min(2, len(pair_costs))

    for i in range(n_remove):
        _, route_idx, pickup, delivery = pair_costs[i]
        if route_idx < len(destroyed.routes):
            route = destroyed.routes[route_idx]
            if pickup in route: route.remove(pickup)
            if delivery in route: route.remove(delivery)

    destroyed.routes = [route for route in destroyed.routes if len(route) > 2]
    destroyed.calculate_objective()
    return destroyed
