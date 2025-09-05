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


# =============================================================================
# ROUTE MINIMIZATION DESTROY OPERATORS
# =============================================================================

def route_minimization_destroy(current_state: PDPTWSolution, random_state: np.random.RandomState):
    """
    Remove all customers from routes with fewest customers to enable consolidation
    This targets routes that are good candidates for merging
    """
    destroyed = current_state.copy()

    if len(destroyed.routes) <= 1:
        return destroyed

    # Calculate route sizes (excluding depot nodes)
    route_sizes = []
    for i, route in enumerate(destroyed.routes):
        non_depot_customers = [c for c in route if c != 0]
        route_sizes.append((len(non_depot_customers), i, route))

    # Sort by size (smallest first)
    route_sizes.sort(key=lambda x: x[0])

    # Remove 1-2 smallest routes entirely
    num_routes_to_clear = min(2, max(1, len(destroyed.routes) // 3))

    nodes_to_remove = set()
    for i in range(num_routes_to_clear):
        if i < len(route_sizes):
            size, route_idx, route = route_sizes[i]
            for customer in route:
                if customer != 0:  # Don't remove depot
                    nodes_to_remove.add(customer)

    # Remove the nodes
    for node_id in nodes_to_remove:
        destroyed.remove_node(node_id)

    return destroyed


def weak_route_destroy(current_state, random_state: np.random.RandomState):
    """
    Identify and empty routes that are candidates for elimination
    Targets routes with low utilization or poor geographic distribution
    """
    destroyed = current_state.copy()

    if len(destroyed.routes) <= 2:
        return destroyed

    # Calculate route efficiency metrics
    route_metrics = []
    for i, route in enumerate(destroyed.routes):
        if len(route) <= 2:  # Skip empty routes
            continue

        # Calculate utilization
        route_load = sum(destroyed.instance.nodes[c]['demand'] for c in route if c != 0)
        utilization = abs(route_load) / destroyed.instance.capacity if destroyed.instance.capacity > 0 else 0

        # Calculate geographic spread
        if len(route) > 2:
            coords = [(destroyed.instance.nodes[c]['x'], destroyed.instance.nodes[c]['y'])
                      for c in route if c != 0]
            if len(coords) >= 2:
                min_x, max_x = min(coords, key=lambda p: p[0])[0], max(coords, key=lambda p: p[0])[0]
                min_y, max_y = min(coords, key=lambda p: p[1])[1], max(coords, key=lambda p: p[1])[1]
                spread = (max_x - min_x) + (max_y - min_y)
            else:
                spread = 0
        else:
            spread = 0

        # Lower score = better candidate for removal
        efficiency_score = utilization - spread * 0.01
        route_metrics.append((efficiency_score, i, route))

    # Sort by efficiency (worst first)
    route_metrics.sort(key=lambda x: x[0])

    # Remove customers from 1-2 least efficient routes
    num_routes_to_target = min(2, len(route_metrics))
    nodes_to_remove = set()

    for i in range(num_routes_to_target):
        score, route_idx, route = route_metrics[i]
        # Remove some customers from this route
        non_depot_customers = [c for c in route if c != 0]
        num_to_remove = max(1, len(non_depot_customers) // 2)

        customers_to_remove = random_state.choice(non_depot_customers,
                                                  size=num_to_remove,
                                                  replace=False)

        for customer in customers_to_remove:
            # Check if it's a pickup - if so, also remove its delivery
            if destroyed.instance.nodes[customer]['demand'] > 0:
                delivery_id = destroyed.instance.nodes[customer]['delivery_index']
                nodes_to_remove.add(customer)
                if delivery_id > 0:
                    nodes_to_remove.add(delivery_id)
            # If it's a delivery, also remove its pickup
            elif destroyed.instance.nodes[customer]['demand'] < 0:
                pickup_id = destroyed.instance.nodes[customer]['pickup_index']
                nodes_to_remove.add(customer)
                if pickup_id > 0:
                    nodes_to_remove.add(pickup_id)

    # Remove the selected nodes
    for node_id in nodes_to_remove:
        destroyed.remove_node(node_id)

    return destroyed


def adjacent_route_destroy(current_state, random_state: np.random.RandomState):
    """
    Remove customers from geographically adjacent routes to enable merging
    """
    destroyed = current_state.copy()

    if len(destroyed.routes) <= 2:
        return destroyed

    # Calculate route centroids
    route_centroids = []
    for i, route in enumerate(destroyed.routes):
        if len(route) <= 2:
            continue
        coords = [(destroyed.instance.nodes[c]['x'], destroyed.instance.nodes[c]['y'])
                  for c in route if c != 0]
        if coords:
            center_x = sum(p[0] for p in coords) / len(coords)
            center_y = sum(p[1] for p in coords) / len(coords)
            route_centroids.append((center_x, center_y, i, route))

    if len(route_centroids) < 2:
        return destroyed

    # Find two closest routes
    min_distance = float('inf')
    closest_pair = None

    for i in range(len(route_centroids)):
        for j in range(i + 1, len(route_centroids)):
            x1, y1, idx1, route1 = route_centroids[i]
            x2, y2, idx2, route2 = route_centroids[j]

            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_pair = (idx1, idx2)

    if closest_pair:
        # Remove some customers from both routes to enable merging
        nodes_to_remove = set()

        for route_idx in closest_pair:
            route = destroyed.routes[route_idx]
            non_depot_customers = [c for c in route if c != 0]

            # Remove 30-50% of customers from each route
            num_to_remove = max(1, len(non_depot_customers) // 3)
            if len(non_depot_customers) >= num_to_remove:
                customers_to_remove = random_state.choice(non_depot_customers,
                                                          size=num_to_remove,
                                                          replace=False)

                for customer in customers_to_remove:
                    # Remove pickup-delivery pairs
                    if destroyed.instance.nodes[customer]['demand'] > 0:
                        delivery_id = destroyed.instance.nodes[customer]['delivery_index']
                        nodes_to_remove.add(customer)
                        if delivery_id > 0:
                            nodes_to_remove.add(delivery_id)

        # Remove the selected nodes
        for node_id in nodes_to_remove:
            destroyed.remove_node(node_id)

    return destroyed


