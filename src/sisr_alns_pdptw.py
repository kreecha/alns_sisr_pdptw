"""
SISR-inspired ALNS Operators for PDPTW (Pickup and Delivery Problem with Time Windows)
Adapted from Slack Induction String Removal (SISR) for the N-Wouda ALNS framework

This converts SISR logic for PDPTW problems where:
- Each pickup node has a corresponding delivery node
- Pickup must precede its delivery (precedence constraint)
- Routes contain mixed pickup/delivery sequences
- Time windows and capacity constraints apply
"""
import os

import numpy as np
import math
import random
from typing import List, Tuple, Dict, Set, Optional
from copy import deepcopy

from src.pdptw import PDPTWInstance, ALNSProgressLogger, create_initial_solution, k_regret_insertion, PDPTWSolution, \
    detailed_feasibility_check
from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations


class SISRPDPTWParameters:
    """Container for SISR parameters adapted for PDPTW"""

    def __init__(self,
                 c_bar: float = 8.0,  # Reduced for PDPTW complexity
                 L_max: float = 6.0,  # Shorter strings due to precedence
                 beta: float = 0.02,  # Split depth parameter
                 alpha: float = 0.6,  # Higher preference for string removal
                 pair_removal_prob: float = 0.7):  # Probability of removing pairs together
        self.c_bar = c_bar
        self.L_max = L_max
        self.beta = beta
        self.alpha = alpha
        self.pair_removal_prob = pair_removal_prob



# =============================================================================
# SISR-INSPIRED DESTROY OPERATORS FOR PDPTW
# =============================================================================
def sisr_paired_string_removal(current_state, random_state: np.random.RandomState):
    """
     SISR String Removal adapted for PDPTW with pickup-delivery pair awareness

     This prioritizes removing pickup-delivery pairs as units to maintain
     the problem structure while applying SISR string logic.
    """
    destroyed = current_state.copy()
    # SISR Parameters for PDPTW
    params = SISRPDPTWParameters(destroyed.instance, r=0.15, k=3)
    all_nodes = [node for node in destroyed.get_all_nodes() if node != 0]
    if not all_nodes:
        return destroyed
    neighbors = _create_pdptw_neighbor_lists(destroyed.instance)
    current_routes = _convert_pdptw_state_to_routes(destroyed)
    if not current_routes:
        return destroyed
    ruined_routes, removed_nodes = _sisr_paired_ruin_logic(current_routes, neighbors, params, random_state, destroyed.instance)
    for node_id in removed_nodes:
        destroyed.remove_node(node_id)
    return destroyed

def sisr_precedence_aware_removal(current_state, random_state: np.random.RandomState):
    destroyed = current_state.copy()
    params = SISRPDPTWParameters(destroyed.instance, r=0.15, k=3)
    all_nodes = [node for node in destroyed.get_all_nodes() if node != 0]
    if not all_nodes:
        return destroyed
    neighbors = _create_pdptw_neighbor_lists(destroyed.instance)
    current_routes = _convert_pdptw_state_to_routes(destroyed)
    if not current_routes:
        return destroyed
    ruined_routes, removed_nodes = _sisr_precedence_ruin_logic(current_routes, neighbors, params, random_state, destroyed.instance)
    for node_id in removed_nodes:
        destroyed.remove_node(node_id)
    return destroyed

def sisr_split_pair_removal(current_state, random_state: np.random.RandomState):
    destroyed = current_state.copy()
    params = SISRPDPTWParameters(destroyed.instance, r=0.15, k=3)
    all_nodes = [node for node in destroyed.get_all_nodes() if node != 0]
    if not all_nodes:
        return destroyed
    neighbors = _create_pdptw_neighbor_lists(destroyed.instance)
    current_routes = _convert_pdptw_state_to_routes(destroyed)
    if not current_routes:
        return destroyed
    ruined_routes, removed_nodes = _sisr_split_ruin_logic(current_routes, neighbors, params, random_state, destroyed.instance)
    for node_id in removed_nodes:
        destroyed.remove_node(node_id)
    return destroyed

# =============================================================================
# SISR-INSPIRED REPAIR OPERATORS FOR PDPTW
# =============================================================================
def sisr_paired_recreate(current_state, random_state: np.random.RandomState):
    repaired = current_state.copy()
    repaired.ensure_depot_nodes()
    unassigned_pairs = _find_unassigned_pairs(repaired)
    if not unassigned_pairs:
        return repaired
    unassigned_pairs = random_state.permutation(unassigned_pairs).tolist()
    for pickup_id, delivery_id in unassigned_pairs:
        _sisr_insert_pickup_delivery_pair(repaired, pickup_id, delivery_id)
    repaired.ensure_depot_nodes()
    return repaired

def sisr_precedence_recreate(current_state, random_state: np.random.RandomState):
    repaired = current_state.copy()
    repaired.ensure_depot_nodes()
    unassigned_pairs = _find_unassigned_pairs(repaired)
    if not unassigned_pairs:
        return repaired
    depot_id = 0
    unassigned_pairs.sort(key=lambda pair: _get_pair_depot_distance(repaired.instance, pair))
    for pickup_id, delivery_id in unassigned_pairs:
        _sisr_insert_pair_with_precedence(repaired, pickup_id, delivery_id)
    repaired.ensure_depot_nodes()
    return repaired

def sisr_flexible_recreate(current_state, random_state: np.random.RandomState):
    repaired = current_state.copy()
    repaired.ensure_depot_nodes()
    all_assigned = set(repaired.get_all_nodes())
    all_nodes = set(range(1, len(repaired.instance.nodes)))
    unassigned = list(all_nodes - all_assigned)
    if not unassigned:
        return repaired
    unassigned_pairs = _find_unassigned_pairs(repaired)
    unassigned_pairs = random_state.permutation(unassigned_pairs).tolist()
    for pickup_id, delivery_id in unassigned_pairs:
        _sisr_insert_pair_with_precedence(repaired, pickup_id, delivery_id)
    remaining_unassigned = [n for n in unassigned if n not in repaired.get_all_nodes()]
    for node_id in random_state.permutation(remaining_unassigned):
        node_info = repaired.instance.nodes[node_id]
        is_pickup = node_info.get('demand', 0) > 0
        if is_pickup:
            delivery_id = node_info.get('delivery_index', 0)
            if delivery_id and delivery_id not in repaired.get_all_nodes():
                continue  # Skip pickup if delivery is unassigned
            if delivery_id in repaired.get_all_nodes():
                _sisr_insert_single_node(repaired, node_id, must_precede=delivery_id)
        else:
            pickup_id = node_info.get('pickup_index', 0)
            if pickup_id and pickup_id not in repaired.get_all_nodes():
                continue  # Skip delivery if pickup is unassigned
            if pickup_id in repaired.get_all_nodes():
                _sisr_insert_single_node(repaired, node_id, must_follow=pickup_id)
    repaired.ensure_depot_nodes()
    return repaired

# =============================================================================
# HELPER FUNCTIONS (SISR LOGIC IMPLEMENTATION FOR PDPTW)
# =============================================================================
def _create_pdptw_neighbor_lists(instance) -> List[List[int]]:
    n_nodes = len(instance.nodes)
    neighbors = [[] for _ in range(n_nodes)]
    for node_id in range(1, n_nodes):
        node_info = instance.nodes[node_id]
        other_neighbors = []
        if node_info.get('demand', 0) > 0:
            delivery_id = node_info.get('delivery_index', 0)
            if delivery_id > 0:
                neighbors[node_id].append(delivery_id)
        elif node_info.get('demand', 0) < 0:
            pickup_id = node_info.get('pickup_index', 0)
            if pickup_id > 0:
                neighbors[node_id].append(pickup_id)
        for other_id in range(1, n_nodes):
            if other_id != node_id:
                dist = _calculate_distance(instance, node_id, other_id)
                other_neighbors.append((other_id, dist))
        other_neighbors.sort(key=lambda x: x[1])
        close_neighbors = [nid for nid, _ in other_neighbors[:8]]
        neighbors[node_id].extend(close_neighbors)
        seen = set()
        neighbors[node_id] = [x for x in neighbors[node_id] if not (x in seen or seen.add(x))]
    return neighbors

def _convert_pdptw_state_to_routes(state) -> List[List[int]]:
    routes = []
    for route in state.routes:
        if route:
            routes.append(route[:])
    return routes

def _sisr_paired_ruin_logic(last_routes: List[List[int]], neighbors: List[List[int]], params: SISRPDPTWParameters, random_state: np.random.RandomState, instance) -> Tuple[List[List[int]], List[int]]:
    if not last_routes:
        return [[0, 0]], []
    avg_route_length = np.mean([len(route) for route in last_routes])
    l_s_max = min(params.L_max, avg_route_length)
    k_s_max = max(1, int(4.0 * params.c_bar / (1.0 + l_s_max) - 1.0))
    k_s = max(1, int(random_state.random() * k_s_max + 1.0))
    all_customers = []
    for route in last_routes:
        all_customers.extend([node for node in route if node != 0])
    if not all_customers:
        return [[0, 0]], []
    seed_node = random_state.choice(all_customers)
    removed_nodes = []
    processed_routes = set()
    candidates = [seed_node]
    if seed_node < len(neighbors):
        candidates.extend(neighbors[seed_node][:k_s])
    for candidate in candidates:
        if len(processed_routes) >= k_s:
            break
        route_idx = None
        for i, route in enumerate(last_routes):
            if candidate in route and i not in processed_routes:
                route_idx = i
                break
        if route_idx is None:
            continue
        route = last_routes[route_idx]
        # Always remove pairs to maintain precedence
        pairs_removed = _remove_pairs_from_route(route, instance, random_state, int(l_s_max))
        removed_nodes.extend(pairs_removed)
        processed_routes.add(route_idx)
    current_routes = []
    for route in last_routes:
        new_route = [node for node in route if node not in removed_nodes]
        if not new_route:
            new_route = [0, 0]
        elif new_route[0] != 0:
            new_route = [0] + new_route
        elif new_route[-1] != 0:
            new_route = new_route + [0]
        if len(new_route) >= 2:
            current_routes.append(new_route)
    return current_routes, removed_nodes

def _sisr_precedence_ruin_logic(last_routes: List[List[int]], neighbors: List[List[int]], params: SISRPDPTWParameters, random_state: np.random.RandomState, instance) -> Tuple[List[List[int]], List[int]]:
    if not last_routes:
        return [[0, 0]], []
    removed_nodes = set()
    pickup_nodes = []
    for route in last_routes:
        for node in route:
            if node != 0 and instance.nodes[node].get('demand', 0) > 0:
                pickup_nodes.append(node)
    if not pickup_nodes:
        return [[0, 0]], []
    seed_pickup = random_state.choice(pickup_nodes)
    seed_delivery = instance.nodes[seed_pickup].get('delivery_index', 0)
    removed_nodes.add(seed_pickup)
    if seed_delivery > 0:
        removed_nodes.add(seed_delivery)
    max_pairs = max(1, min(4, len(pickup_nodes) // 2))
    for _ in range(max_pairs - 1):
        if not pickup_nodes or seed_pickup >= len(neighbors):
            break
        related_pickup = None
        for neighbor in neighbors[seed_pickup]:
            if neighbor in pickup_nodes and neighbor not in removed_nodes and instance.nodes[neighbor].get('demand', 0) > 0:
                related_pickup = neighbor
                break
        if related_pickup:
            related_delivery = instance.nodes[related_pickup].get('delivery_index', 0)
            removed_nodes.add(related_pickup)
            if related_delivery > 0:
                removed_nodes.add(related_delivery)
            seed_pickup = related_pickup
        else:
            break
    current_routes = []
    for route in last_routes:
        new_route = [node for node in route if node not in removed_nodes]
        if not new_route:
            new_route = [0, 0]
        elif new_route[0] != 0:
            new_route = [0] + new_route
        elif new_route[-1] != 0:
            new_route = new_route + [0]
        if len(new_route) >= 2:
            current_routes.append(new_route)
    return current_routes, list(removed_nodes)


def _sisr_split_ruin_logic(last_routes: List[List[int]], neighbors: List[List[int]], params: SISRPDPTWParameters, random_state: np.random.RandomState, instance) -> Tuple[List[List[int]], List[int]]:
    return _sisr_paired_ruin_logic(last_routes, neighbors, params, random_state, instance)

def _remove_pairs_from_route(route: List[int], instance, random_state, max_pairs: int) -> List[int]:
    removed = []
    pickup_delivery_pairs = []
    for node in route:
        if node != 0 and instance.nodes[node].get('demand', 0) > 0:
            delivery = instance.nodes[node].get('delivery_index', 0)
            if delivery in route:
                pickup_delivery_pairs.append((node, delivery))
    num_to_remove = min(max_pairs, len(pickup_delivery_pairs))
    if num_to_remove > 0:
        pairs_to_remove = random_state.choice(len(pickup_delivery_pairs), size=num_to_remove, replace=False)
        for idx in pairs_to_remove:
            pickup, delivery = pickup_delivery_pairs[idx]
            removed.extend([pickup, delivery])
    return removed

def _apply_string_removal(route: List[int], candidate: int, length: int, random_state, params) -> List[int]:
    if candidate not in route or len(route) <= 2:
        return []
    candidate_idx = route.index(candidate)
    start_range = max(1, candidate_idx + 1 - length)
    end_range = min(candidate_idx, len(route) - length - 1) + 1
    if start_range >= end_range:
        return [candidate]
    if isinstance(random_state, np.random.RandomState):
        start_pos = random_state.randint(start_range, end_range)
    else:
        start_pos = random_state.integers(start_range, end_range)
    removed = [node for node in route[start_pos:start_pos + length] if node != 0]
    return removed

def _sisr_insert_pickup_delivery_pair(state, pickup_id: int, delivery_id: int):
    best_cost = float('inf')
    best_insertion = None
    state.ensure_depot_nodes()
    for route_idx, route in enumerate(state.routes):
        route_without_depots = route[1:-1] if len(route) > 2 else []
        pickup_demand = state.instance.nodes[pickup_id].get('demand', 0)
        delivery_demand = state.instance.nodes[delivery_id].get('demand', 0)
        for pickup_pos in range(len(route_without_depots) + 1):
            for delivery_pos in range(pickup_pos + 1, len(route_without_depots) + 2):
                temp_route = route[:]
                temp_route.insert(pickup_pos + 1, pickup_id)
                temp_route.insert(delivery_pos + 1, delivery_id)
                # Check full route load feasibility
                current_load = 0
                load_feasible = True
                for node in temp_route[1:-1]:  # Exclude depots
                    current_load += state.instance.nodes[node].get('demand', 0)
                    if current_load > state.instance.capacity or current_load < 0:
                        load_feasible = False
                        break
                if load_feasible and state._is_route_feasible(temp_route):
                    cost = _calculate_pair_insertion_cost_with_depot(state, route_idx, pickup_id, delivery_id, pickup_pos + 1, delivery_pos + 1)
                    if cost < best_cost:
                        best_cost = cost
                        best_insertion = (route_idx, pickup_pos + 1, delivery_pos + 1)
    new_route = [0, pickup_id, delivery_id, 0]
    new_route_cost = _calculate_new_route_cost(state, pickup_id, delivery_id)
    # Check new route load feasibility
    current_load = state.instance.nodes[pickup_id].get('demand', 0) + state.instance.nodes[delivery_id].get('demand', 0)
    if state._is_route_feasible(new_route) and current_load <= state.instance.capacity and current_load >= 0:
        if new_route_cost < best_cost:
            best_insertion = None
    if best_insertion is None:
        state.routes.append(new_route)
    else:
        route_idx, pickup_pos, delivery_pos = best_insertion
        route = state.routes[route_idx]
        route.insert(pickup_pos, pickup_id)
        route.insert(delivery_pos, delivery_id)
    state.ensure_depot_nodes()
    state.calculate_objective()

def _sisr_insert_single_node(state, node_id: int, must_precede: int = None, must_follow: int = None):
    state.ensure_depot_nodes()
    best_cost = float('inf')
    best_insertion = None
    node_info = state.instance.nodes[node_id]
    node_demand = node_info.get('demand', 0)
    for route_idx, route in enumerate(state.routes):
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            continue
        valid_positions = []
        for pos in range(1, len(route)):
            temp_route = route[:]
            temp_route.insert(pos, node_id)
            # Check full route load feasibility
            current_load = 0
            load_feasible = True
            for node in temp_route[1:-1]:  # Exclude depots
                current_load += state.instance.nodes[node].get('demand', 0)
                if current_load > state.instance.capacity or current_load < 0:
                    load_feasible = False
                    break
            if load_feasible and state._is_route_feasible(temp_route):
                if must_precede and must_precede in route and route.index(must_precede) < pos:
                    continue
                if must_follow and must_follow in route and route.index(must_follow) >= pos:
                    continue
                valid_positions.append(pos)
        for pos in valid_positions:
            cost = _calculate_single_insertion_cost(state, route_idx, node_id, pos)
            if cost < best_cost:
                best_cost = cost
                best_insertion = (route_idx, pos)
    new_route = [0, node_id, 0]
    current_load = node_demand
    if state._is_route_feasible(new_route) and current_load <= state.instance.capacity and current_load >= 0:
        new_cost = state.instance.distances[0][node_id] + state.instance.distances[node_id][0]
        if new_cost < best_cost:
            best_insertion = None
    if best_insertion:
        route_idx, pos = best_insertion
        state.routes[route_idx].insert(pos, node_id)
    else:
        state.routes.append(new_route)
    state.ensure_depot_nodes()
    state.calculate_objective()

def _sisr_insert_pair_with_precedence(state, pickup_id: int, delivery_id: int):
    _sisr_insert_pickup_delivery_pair(state, pickup_id, delivery_id)

def _calculate_route_load(state, route: List[int]) -> int:
    """Calculate the total load of a route."""
    current_load = 0
    for node_id in route:
        if node_id != 0:  # Exclude depot
            current_load += state.instance.nodes[node_id].get('demand', 0)
    return current_load

def _calculate_pair_insertion_cost_with_depot(state, route_idx: int, pickup_id: int, delivery_id: int, pickup_pos: int, delivery_pos: int) -> float:
    route = state.routes[route_idx]
    if pickup_pos == 1:
        if len(route) <= 2:
            pickup_cost = state.instance.distances[0][pickup_id]
        else:
            next_node = route[1]
            pickup_cost = (state.instance.distances[0][pickup_id] + state.instance.distances[pickup_id][next_node] - state.instance.distances[0][next_node])
    elif pickup_pos == len(route) - 1:
        prev_node = route[pickup_pos - 1]
        pickup_cost = (state.instance.distances[prev_node][pickup_id] + state.instance.distances[pickup_id][0] - state.instance.distances[prev_node][0])
    else:
        prev_node = route[pickup_pos - 1]
        next_node = route[pickup_pos]
        pickup_cost = (state.instance.distances[prev_node][pickup_id] + state.instance.distances[pickup_id][next_node] - state.instance.distances[prev_node][next_node])
    delivery_cost = state.instance.distances[pickup_id][delivery_id]
    return pickup_cost + delivery_cost

def _find_unassigned_pairs(state) -> List[Tuple[int, int]]:
    assigned = set(state.get_all_nodes())
    pairs = []
    for node_id in range(1, len(state.instance.nodes)):
        if node_id not in assigned:
            node_info = state.instance.nodes[node_id]
            if node_info.get('demand', 0) > 0:
                delivery_id = node_info.get('delivery_index', 0)
                if delivery_id > 0 and delivery_id not in assigned:
                    pairs.append((node_id, delivery_id))
    return pairs

def _get_pair_depot_distance(instance, pair: Tuple[int, int]) -> float:
    pickup_id, delivery_id = pair
    depot_dist_pickup = instance.distances[0][pickup_id]
    depot_dist_delivery = instance.distances[0][delivery_id]
    return (depot_dist_pickup + depot_dist_delivery) / 2

def _calculate_distance(instance, node1: int, node2: int) -> float:
    return instance.distances[node1][node2]

def _calculate_pair_insertion_cost(state, route_idx: int, pickup_id: int, delivery_id: int, pickup_pos: int, delivery_pos: int) -> float:
    route = state.routes[route_idx]
    if pickup_pos == 0:
        if len(route) == 0:
            pickup_cost = state.instance.distances[0][pickup_id]
        else:
            pickup_cost = (state.instance.distances[0][pickup_id] + state.instance.distances[pickup_id][route[0]] - state.instance.distances[0][route[0]])
    else:
        prev_node = route[pickup_pos - 1]
        if pickup_pos < len(route):
            next_node = route[pickup_pos]
            pickup_cost = (state.instance.distances[prev_node][pickup_id] + state.instance.distances[pickup_id][next_node] - state.instance.distances[prev_node][next_node])
        else:
            pickup_cost = (state.instance.distances[prev_node][pickup_id] + state.instance.distances[pickup_id][0] - state.instance.distances[prev_node][0])
    delivery_cost = state.instance.distances[pickup_id][delivery_id]
    return pickup_cost + delivery_cost

def _calculate_new_route_cost(state, pickup_id: int, delivery_id: int) -> float:
    return (state.instance.distances[0][pickup_id] + state.instance.distances[pickup_id][delivery_id] + state.instance.distances[delivery_id][0])

def _calculate_single_insertion_cost(state, route_idx: int, node_id: int, pos: int) -> float:
    if route_idx >= len(state.routes):
        return state.instance.distances[0][node_id] + state.instance.distances[node_id][0]
    route = state.routes[route_idx]
    if pos == 0:
        if len(route) == 0:
            return state.instance.distances[0][node_id] + state.instance.distances[node_id][0]
        else:
            return (state.instance.distances[0][node_id] + state.instance.distances[node_id][route[0]] - state.instance.distances[0][route[0]])
    elif pos == len(route):
        prev_node = route[-1]
        return (state.instance.distances[prev_node][node_id] + state.instance.distances[node_id][0] - state.instance.distances[prev_node][0])
    else:
        prev_node = route[pos - 1]
        next_node = route[pos]
        return (state.instance.distances[prev_node][node_id] + state.instance.distances[node_id][next_node] - state.instance.distances[prev_node][next_node])

# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def add_sisr_pdptw_operators(alns_instance):
    """Add SISR-PDPTW operators to ALNS instance"""

    # Add SISR destroy operators for PDPTW
    alns_instance.add_destroy_operator(sisr_paired_string_removal, name='sisr_paired_string_removal')
    alns_instance.add_destroy_operator(sisr_precedence_aware_removal, name='sisr_precedence_aware_removal')
    alns_instance.add_destroy_operator(sisr_split_pair_removal, name='sisr_split_pair_removal')

    # Add SISR repair operators for PDPTW
    alns_instance.add_repair_operator(sisr_paired_recreate, name='sisr_paired_recreate')
    alns_instance.add_repair_operator(sisr_precedence_recreate, name='sisr_precedence_recreate')
    alns_instance.add_repair_operator(sisr_flexible_recreate, name='sisr_flexible_recreate')

    print("Added SISR-PDPTW operators:")
    print("  Destroy: sisr_paired_string_removal, sisr_precedence_aware_removal, sisr_split_pair_removal")
    print("  Repair: sisr_paired_recreate, sisr_precedence_recreate, sisr_flexible_recreate")


if __name__ == "__main__":
    full_path = os.path.join("..", "data", "lc102.txt")
    instance = PDPTWInstance(filename=full_path)
    random_state = np.random.RandomState(42)

    alns = ALNS(random_state)
    sisr_params = SISRPDPTWParameters(instance, r=0.15, k=3)
    add_sisr_pdptw_operators(alns, sisr_params)  # Pass params here

    n = instance.n_customers
    m = instance.vehicle_number
    k_s_max = max(1, int(4 * sisr_params.c_bar / (1 + sisr_params.L_max) - 1))
    s = sisr_params.alpha * k_s_max * (sisr_params.L_max / n) + (
                1 - sisr_params.alpha) * sisr_params.pair_removal_prob * sisr_params.beta
    k = round(1 / s + sisr_params.beta * m)
    alns.add_repair_operator(lambda state, rs: k_regret_insertion(state, rs, k=k), name=f'k_regret_insertion_k{k}')

    initial_solution = create_initial_solution(instance)
    print(f"Initial solution: {len(initial_solution.routes)} routes, objective = {initial_solution.objective():.2f}")
    progress_logger = ALNSProgressLogger(log_mode='interval', interval=150)
    alns.on_accept(progress_logger.log_progress)

    tau = 0.011658000847676892
    initial_distance = initial_solution.objective()
    start_temperature = -tau * initial_distance / np.log(0.5) if initial_distance > 0 else 1000.0
    cooling_rate = 0.9923992759758272
    criterion = SimulatedAnnealing(start_temperature=round(start_temperature, 4), end_temperature=6.315734773880141,
                                   step=cooling_rate)

    original_objective = PDPTWSolution.objective


    def objective_with_noise(self):
        return self.calculate_objective_with_noise(random_state)


    PDPTWSolution.objective = objective_with_noise
    max_iterations = 3_500
    report_interval = 500
    stop_criterion = MaxIterations(max_iterations)
    select = RouletteWheel([13, 4, 3, 0], 0.7615516997372846, 3, 3)
    print(f"\nStarting ALNS optimization for {max_iterations} iterations...")
    print(f"Progress will be reported every {report_interval} iterations.")
    print("=" * 60)

    result = alns.iterate(initial_solution, select, criterion, stop_criterion)
    PDPTWSolution.objective = original_objective
    progress_logger.final_report()
    print(f"\nFinal solution:")
    print(f"- Routes: {len(result.best_state.routes)}")
    print(f"- Total distance: {result.best_state.objective():.2f}")
    print(f"- Feasible: {result.best_state.is_feasible()}")

    feasibility_report = detailed_feasibility_check(result.best_state, instance)
