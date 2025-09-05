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

class KruskalClusterRemoval:
    """
    R10: All customers - from one of two Kruskal clusters from randomly selected route

    This operator:
    1. Selects a random route
    2. Creates a minimum spanning tree (MST) of customers in that route using Kruskal's algorithm
    3. Removes the most expensive edge to create two clusters
    4. Randomly selects one cluster and removes all its pickup-delivery pairs
    """

    def __init__(self):
        self.name = "kruskal_cluster_removal"

    def find_parent(self, parent: Dict[int, int], node: int) -> int:
        """Find root parent for Union-Find structure"""
        if parent[node] != node:
            parent[node] = self.find_parent(parent, parent[node])
        return parent[node]

    def union_sets(self, parent: Dict[int, int], rank: Dict[int, int], x: int, y: int):
        """Union two sets in Union-Find structure"""
        root_x = self.find_parent(parent, x)
        root_y = self.find_parent(parent, y)

        if root_x != root_y:
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

    def kruskal_mst(self, nodes: List[int], distances: np.ndarray) -> List[Tuple[float, int, int]]:
        """
        Build Minimum Spanning Tree using Kruskal's algorithm
        Returns list of edges (distance, node1, node2) in MST
        """
        if len(nodes) < 2:
            return []

        # Create all edges between nodes
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                dist = distances[node1][node2]
                edges.append((dist, node1, node2))

        # Sort edges by distance
        edges.sort()

        # Initialize Union-Find
        parent = {node: node for node in nodes}
        rank = {node: 0 for node in nodes}

        mst_edges = []

        for dist, node1, node2 in edges:
            if self.find_parent(parent, node1) != self.find_parent(parent, node2):
                self.union_sets(parent, rank, node1, node2)
                mst_edges.append((dist, node1, node2))

                # MST complete when we have n-1 edges
                if len(mst_edges) == len(nodes) - 1:
                    break

        return mst_edges

    def __call__(self, current_state: PDPTWSolution, random_state: np.random.RandomState) -> PDPTWSolution:
        """Remove customers from one of two Kruskal clusters"""
        destroyed = current_state.copy()

        if not destroyed.routes:
            return destroyed

        # Select a random route
        non_empty_routes = [route for route in destroyed.routes if len(route) > 2]
        if not non_empty_routes:
            return destroyed

        selected_route = random_state.choice(non_empty_routes)

        # Get non-depot customers from the route
        customers = [node for node in selected_route if node != 0]

        if len(customers) < 3:  # Need at least 3 customers to form meaningful clusters
            return destroyed

        # Build MST using Kruskal's algorithm
        mst_edges = self.kruskal_mst(customers, destroyed.instance.distances)

        if not mst_edges:
            return destroyed

        # Find the most expensive edge in MST to remove (creates two clusters)
        most_expensive_edge = max(mst_edges, key=lambda x: x[0])
        mst_edges.remove(most_expensive_edge)

        # Build clusters by traversing remaining MST
        clusters = []
        visited = set()

        # Build adjacency list from remaining MST edges
        adj_list = {node: [] for node in customers}
        for _, node1, node2 in mst_edges:
            adj_list[node1].append(node2)
            adj_list[node2].append(node1)

        # DFS to find connected components (clusters)
        def dfs(node: int, cluster: List[int]):
            visited.add(node)
            cluster.append(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for node in customers:
            if node not in visited:
                cluster = []
                dfs(node, cluster)
                if cluster:  # Only add non-empty clusters
                    clusters.append(cluster)

        if len(clusters) < 2:
            return destroyed

        # Randomly select one cluster to remove
        cluster_to_remove = random_state.choice(clusters)

        # Collect pickup-delivery pairs to remove
        nodes_to_remove = set()
        for node in cluster_to_remove:
            node_info = destroyed.instance.nodes[node]
            if node_info['demand'] > 0:  # It's a pickup
                delivery_id = node_info['delivery_index']
                nodes_to_remove.add(node)
                if delivery_id > 0:
                    nodes_to_remove.add(delivery_id)
            elif node_info['demand'] < 0:  # It's a delivery
                pickup_id = node_info['pickup_index']
                nodes_to_remove.add(node)
                if pickup_id > 0:
                    nodes_to_remove.add(pickup_id)

        # Remove the nodes
        for node_id in nodes_to_remove:
            destroyed.remove_node(node_id)

        destroyed.calculate_objective()
        return destroyed
