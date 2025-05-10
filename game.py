import pygame
import sys
from collections import deque
import copy
import time
from queue import PriorityQueue
import traceback
import math
import random
import itertools
import uuid

# Pygame Initialization and Constants
pygame.init()
MENU_WIDTH = 200
WIDTH = 1100
HEIGHT = 800
GRID_SIZE = 3
CELL_SIZE = 100
GRID_PADDING = 40
GRID_DISPLAY_WIDTH = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (30, 144, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FONT = pygame.font.SysFont('Arial', 60)
BUTTON_FONT = pygame.font.SysFont('Arial', 24)
INFO_FONT = pygame.font.SysFont('Arial', 18)
TITLE_FONT = pygame.font.SysFont('Arial', 26, bold=True)
MENU_COLOR = (50, 50, 50)
MENU_BUTTON_COLOR = (70, 70, 70)
MENU_HOVER_COLOR = (90, 90, 90)
MENU_SELECTED_COLOR = pygame.Color('dodgerblue')
INFO_COLOR = (50, 50, 150)
INFO_BG = (245, 245, 245)
POPUP_WIDTH = 600
POPUP_HEIGHT = 400

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LamVanDi-23110191 - 8 Puzzle Solver")

# --- FIXED INITIAL AND GOAL STATES ---
_FIXED_INITIAL_STATE_FLAT = [2,6,5,0,8,7,4,3,1]
_FIXED_GOAL_STATE_FLAT = [1,2,3,4,5,6,7,8,0]

initial_state_fixed_global = [[_FIXED_INITIAL_STATE_FLAT[i*GRID_SIZE+j] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
goal_state_fixed_global = [[_FIXED_GOAL_STATE_FLAT[i*GRID_SIZE+j] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]

# Global state variables used by the application.
# `initial_state` is the problem's defined start, `goal_state` is the target.
initial_state = copy.deepcopy(initial_state_fixed_global)
goal_state = copy.deepcopy(goal_state_fixed_global) # This is THE goal for all algorithms.
# --- END FIXED STATES ---


# Các hàm tiện ích
def find_empty(state):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] == 0:
                return i, j
    return -1, -1

def is_goal(state): # Checks against the global `goal_state`
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            return False
    return state == goal_state # Relies on global goal_state

def get_neighbors(state):
    neighbors = []
    empty_i, empty_j = find_empty(state)
    if empty_i == -1: # Should not happen in a valid 8-puzzle state
        return []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
    for di, dj in directions:
        new_i, new_j = empty_i + di, empty_j + dj
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
            new_state = copy.deepcopy(state)
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    return neighbors

def state_to_tuple(state):
    if not isinstance(state, list):
        return None
    try:
        return tuple(tuple(row) for row in state)
    except TypeError:
        print(f"Error converting state to tuple. State: {state}")
        return None

def tuple_to_list(state_tuple):
    if not isinstance(state_tuple, tuple):
        return None
    try:
        return [list(row) for row in state_tuple]
    except TypeError:
        print(f"Error converting tuple to list. Tuple: {state_tuple}")
        return None

def apply_action_to_state(state_list, action):
    if state_list is None:
        return None
    new_state = copy.deepcopy(state_list)
    empty_i, empty_j = find_empty(new_state)
    if empty_i == -1:
        return new_state # Or handle error
    di, dj = 0, 0
    if action == 'Up':    di = -1
    elif action == 'Down':  di = 1
    elif action == 'Left':  dj = -1
    elif action == 'Right': dj = 1
    else:
        return new_state # Unknown action

    new_i, new_j = empty_i + di, empty_j + dj
    if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
        new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
        return new_state
    return state_list # Return original if move is invalid (should not happen if action is from valid plan)


def manhattan_distance(state): # Calculates heuristic to the global `goal_state`
    distance = 0
    goal_pos = {} # Cache goal positions for faster lookup

    # Pre-calculate goal positions only if not cached or goal_state changed (it's fixed now)
    # This could be a global cache if performance is an issue, but for 3x3 it's fine.
    for r_goal in range(GRID_SIZE):
        for c_goal in range(GRID_SIZE):
            val = goal_state[r_goal][c_goal] # Use global goal_state
            if val != 0 : # We don't count distance for the empty tile
                goal_pos[val] = (r_goal, c_goal)

    for r_curr in range(GRID_SIZE):
        for c_curr in range(GRID_SIZE):
            tile = state[r_curr][c_curr]
            if tile != 0 and tile in goal_pos: # Tile is not empty and is in goal
                goal_r, goal_c = goal_pos[tile]
                distance += abs(r_curr - goal_r) + abs(c_curr - goal_c)
    return distance

def is_valid_state_for_solve(state_to_check): # Checks if a state is fully defined (0-8 once)
    flat_state = []
    try:
        for row in state_to_check:
            for tile in row:
                flat_state.append(tile)
    except TypeError:
        return False # Not a 2D list
        
    if len(flat_state) != GRID_SIZE * GRID_SIZE:
        return False
    
    seen_numbers = set()
    for tile in flat_state:
        if not isinstance(tile, int) or not (0 <= tile < GRID_SIZE * GRID_SIZE):
            return False # Invalid number or type
        if tile in seen_numbers:
            return False # Duplicate number
        seen_numbers.add(tile)
    return len(seen_numbers) == GRID_SIZE * GRID_SIZE


def execute_plan(start_state, action_plan): # For Sensorless
    if not action_plan:
        return [start_state]
    current_state = copy.deepcopy(start_state)
    state_sequence = [current_state]
    for action in action_plan:
        next_state = apply_action_to_state(current_state, action)
        current_state = next_state
        state_sequence.append(copy.deepcopy(current_state))
    return state_sequence


# Các thuật toán tìm kiếm (BFS, DFS, IDS, UCS, A*, Greedy, IDA*, Hill Climbing variants, SA, Beam, AND-OR)
# These algorithms will use the global `initial_state` (passed as parameter)
# and the global `goal_state` (used by `is_goal` and `manhattan_distance`).

def bfs(start_node_state, time_limit=30):
    start_time = time.time()
    queue = deque([(start_node_state, [start_node_state])])
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None
    visited = {init_tuple}

    while queue:
        if time.time() - start_time > time_limit:
            print(f"BFS Timeout ({time_limit}s)")
            return None
        current_s, path = queue.popleft()
        if is_goal(current_s):
            return path
        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor_s, path + [neighbor_s]))
    return None

def dfs(start_node_state, max_depth=30, time_limit=30):
    start_time = time.time()
    # Stack: (state, path_to_state, depth)
    stack = [(start_node_state, [start_node_state], 0)]
    # Visited: store {state_tuple: depth} to revisit if a shorter path to it is found (not typical for basic DFS, but good for graph search variant)
    visited = {} 

    while stack:
        if time.time() - start_time > time_limit:
            print(f"DFS Timeout ({time_limit}s)")
            return None
        
        current_s, path, depth = stack.pop()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        # If already visited at a shallower or equal depth, skip
        if current_tuple in visited and visited[current_tuple] <= depth:
            continue
        
        visited[current_tuple] = depth # Mark visited at this depth

        if is_goal(current_s):
            return path

        if depth >= max_depth: # Depth limit exceeded
            continue

        # Add neighbors in reverse order to explore "leftmost" branches first (convention)
        neighbors = get_neighbors(current_s)
        for neighbor_s in reversed(neighbors): # Process in a consistent order
            # No need to check visited here if we check at pop, but can optimize
            # if neighbor_tuple not in visited or visited[neighbor_tuple] > depth + 1:
            stack.append((neighbor_s, path + [neighbor_s], depth + 1))
            
    return None


def ids(start_node_state, max_depth_limit=30, time_limit=30):
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    for depth_limit in range(max_depth_limit + 1):
        if time.time() - start_time_global > time_limit:
            print(f"IDS Global Timeout ({time_limit}s)")
            return None

        # For each iteration of DLS:
        stack = [(start_node_state, [start_node_state], 0)]  # (state, path, depth)
        visited_in_iteration = {init_tuple: 0} # Visited for current depth limit to handle cycles

        # DLS part
        while stack:
            # Check global timeout inside DLS loop as well
            if time.time() - start_time_global > time_limit:
                print(f"IDS Iteration Timeout during DLS({depth_limit})")
                return None

            current_s, path, depth = stack.pop()

            if is_goal(current_s):
                return path

            if depth < depth_limit: # Only expand if not at current depth limit
                neighbors = get_neighbors(current_s)
                for neighbor_s in reversed(neighbors): # Consistent order
                    neighbor_tuple = state_to_tuple(neighbor_s)
                    if neighbor_tuple is not None:
                        # Add to stack if not visited in this iteration or found via shorter path
                        if neighbor_tuple not in visited_in_iteration or visited_in_iteration[neighbor_tuple] > depth + 1:
                            visited_in_iteration[neighbor_tuple] = depth + 1
                            stack.append((neighbor_s, path + [neighbor_s], depth + 1))
    return None # No solution found within max_depth_limit

def ucs(start_node_state, time_limit=30):
    start_time = time.time()
    # Priority queue stores (cost, state, path)
    # Cost is uniform (1 per step for 8-puzzle)
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None
    
    frontier.put((0, start_node_state, [start_node_state]))
    # Visited stores {state_tuple: cost_to_reach_state}
    visited = {init_tuple: 0}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            print(f"UCS Timeout ({time_limit}s)")
            return None

        cost, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s) # Should exist if it was added

        if current_tuple is None: continue # Should not happen

        # If we found a shorter path to this state previously, skip current one
        # This check is important because UCS might add multiple paths to same state in frontier
        if cost > visited[current_tuple]:
            continue

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            new_cost = cost + 1 # Assuming cost of each move is 1
            
            # If neighbor not visited, or found a cheaper path to it
            if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                visited[neighbor_tuple] = new_cost
                frontier.put((new_cost, neighbor_s, path + [neighbor_s]))
    return None

def astar(start_node_state, time_limit=30):
    start_time = time.time()
    frontier = PriorityQueue() # Stores (f_score, g_score, state, path)
    
    g_init = 0
    h_init = manhattan_distance(start_node_state)
    f_init = g_init + h_init
    
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    frontier.put((f_init, g_init, start_node_state, [start_node_state]))
    
    # visited stores {state_tuple: g_score_to_reach_state}
    # This is equivalent to 'closed set' and also helps update costs in 'open set (frontier)'
    visited_g_scores = {init_tuple: g_init}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            print(f"A* Timeout ({time_limit}s)")
            return None

        f_score_curr, g_score_curr, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        # Optimization: If we pulled a state from PQ whose g_score is worse than one already found
        # (can happen if state was added to PQ multiple times with different f/g scores before a better one was processed)
        if g_score_curr > visited_g_scores.get(current_tuple, float('inf')):
            continue
            
        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            tentative_g_score = g_score_curr + 1 # Cost of move is 1

            # If this path to neighbor is better than any previous one
            if tentative_g_score < visited_g_scores.get(neighbor_tuple, float('inf')):
                visited_g_scores[neighbor_tuple] = tentative_g_score
                h_neighbor = manhattan_distance(neighbor_s)
                f_neighbor = tentative_g_score + h_neighbor
                frontier.put((f_neighbor, tentative_g_score, neighbor_s, path + [neighbor_s]))
    return None

def greedy(start_node_state, time_limit=30): # Greedy Best-First Search
    start_time = time.time()
    # Priority queue stores (heuristic_value, state, path)
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    frontier.put((manhattan_distance(start_node_state), start_node_state, [start_node_state]))
    visited = {init_tuple} # To avoid cycles and redundant explorations

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            print(f"Greedy Timeout ({time_limit}s)")
            return None

        h_val, current_s, path = frontier.get()

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None or neighbor_tuple in visited:
                continue
            
            visited.add(neighbor_tuple)
            h_neighbor = manhattan_distance(neighbor_s)
            frontier.put((h_neighbor, neighbor_s, path + [neighbor_s]))
            
    return None


# IDA* helper function (recursive search)
def _search_ida_recursive(path_ida, g_score, threshold, visited_in_iteration_ida, start_time_ida, time_limit_ida):
    if time.time() - start_time_ida >= time_limit_ida:
        return "Timeout", float('inf')

    current_s_ida = path_ida[-1]
    h_ida = manhattan_distance(current_s_ida)
    f_score_ida = g_score + h_ida

    if f_score_ida > threshold:
        return None, f_score_ida # Return new threshold candidate

    if is_goal(current_s_ida):
        return path_ida, threshold # Solution found

    min_new_threshold = float('inf')

    for neighbor_s_ida in get_neighbors(current_s_ida):
        neighbor_tuple_ida = state_to_tuple(neighbor_s_ida)
        if neighbor_tuple_ida is None: continue
        
        new_g_ida = g_score + 1
        # Avoid cycles within the current search path or re-visiting with higher cost
        # For IDA*, typically, we only need to avoid direct cycles in path.
        # A more robust visited check considers cost for graph search like behavior.
        if neighbor_tuple_ida not in visited_in_iteration_ida or new_g_ida < visited_in_iteration_ida[neighbor_tuple_ida]:
            visited_in_iteration_ida[neighbor_tuple_ida] = new_g_ida # Update cost or add
            path_ida.append(neighbor_s_ida)
            
            result_ida, recursive_threshold_ida = _search_ida_recursive(path_ida, new_g_ida, threshold, visited_in_iteration_ida, start_time_ida, time_limit_ida)
            
            path_ida.pop() # Backtrack

            if result_ida == "Timeout":
                return "Timeout", float('inf')
            if result_ida is not None: # Solution found in recursive call
                return result_ida, threshold
            
            min_new_threshold = min(min_new_threshold, recursive_threshold_ida)
            
            # After backtracking, remove from visited for this path to allow other paths
            # This depends on the visited strategy. If visited is for the entire iteration, don't remove.
            # If visited is path-dependent (to avoid cycles in current path), then remove or handle carefully.
            # For IDA*, visited_in_iteration helps prune branches already explored with better/equal cost in this iteration.
            # Let's stick to the standard: visited_in_iteration prevents re-exploration if cost is not better.

    return None, min_new_threshold


def ida_star(start_node_state, time_limit=60): # Iterative Deepening A*
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    threshold = manhattan_distance(start_node_state)
    
    while True: # Loop indefinitely, increasing threshold
        if time.time() - start_time_global >= time_limit:
            print(f"IDA* Global Timeout ({time_limit}s)")
            return None

        # Visited for the current iteration, stores {state_tuple: g_score}
        # Reset for each new threshold to allow re-exploration if needed.
        visited_this_iteration = {init_tuple: 0} 
        
        # Initial path for the recursive search
        current_path = [start_node_state]
        
        # Call the recursive search function
        # _search_ida_recursive(path, g, threshold, visited_for_iteration, global_start_time, global_time_limit)
        result, new_threshold_candidate = _search_ida_recursive(current_path, 0, threshold, visited_this_iteration, start_time_global, time_limit)

        if result == "Timeout":
            print("IDA* Timeout during search iteration.")
            return None
        if result is not None: # Solution found
            return result # This is the path

        if new_threshold_candidate == float('inf'): # No solution possible
            print("IDA*: Search exhausted, no solution found.")
            return None
        
        threshold = new_threshold_candidate # Update threshold for next iteration
        print(f"IDA* new threshold: {threshold}")


# Local Search Algorithms
def simple_hill_climbing(start_node_state, time_limit=30): # Takes first better neighbor
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)

    while True:
        if time.time() - start_time > time_limit:
            print("Simple Hill Climbing: Timeout or stuck too long.")
            return path # Return path so far

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break # Should not happen in 8-puzzle

        best_neighbor_found = None
        moved = False
        for neighbor_s in neighbors: # Order might matter slightly
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor < current_h:
                best_neighbor_found = neighbor_s
                current_h = h_neighbor # Update current best h
                moved = True
                break # Take the first better neighbor

        if not moved or best_neighbor_found is None: # No better neighbor found
            print("Simple Hill Climbing: Reached local optimum or plateau.")
            return path # Stuck

        current_s = best_neighbor_found
        path.append(current_s)


def steepest_hill_climbing(start_node_state, time_limit=30): # Takes the best (steepest) neighbor
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)

    while True:
        if time.time() - start_time > time_limit:
            print("Steepest Hill Climbing: Timeout or stuck too long.")
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        best_next_s = None
        best_next_h = current_h # Must be strictly better

        for neighbor_s in neighbors:
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor < best_next_h: # If this neighbor is better than current best_next
                best_next_h = h_neighbor
                best_next_s = neighbor_s
        
        if best_next_s is None: # No neighbor is strictly better
            print("Steepest Hill Climbing: Reached local optimum or plateau.")
            return path # Stuck

        current_s = best_next_s
        current_h = best_next_h
        path.append(current_s)

def random_hill_climbing(start_node_state, time_limit=30, max_iter_no_improve=500): # Stochastic Hill Climbing
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    iter_no_improve = 0

    while True:
        if time.time() - start_time > time_limit:
            print("Stochastic Hill Climbing: Timeout.")
            return path
        
        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors:
            print("Stochastic Hill Climbing: No neighbors (should not happen).")
            break
        
        random_neighbor = random.choice(neighbors)
        neighbor_h = manhattan_distance(random_neighbor)

        if neighbor_h <= current_h: # If neighbor is better or equal
            if neighbor_h < current_h:
                iter_no_improve = 0 # Reset counter if strictly better
            else: # Moved to a state with equal heuristic (plateau)
                iter_no_improve += 1
            
            current_s = random_neighbor
            current_h = neighbor_h
            path.append(current_s)
        else: # Neighbor is worse
            iter_no_improve += 1

        if iter_no_improve >= max_iter_no_improve:
            print(f"Stochastic Hill Climbing: Stuck after {max_iter_no_improve} iterations without strict improvement.")
            return path
    return path


def simulated_annealing(start_node_state, initial_temp=1000, cooling_rate=0.99, min_temp=0.1, time_limit=30):
    start_time = time.time()
    current_s = start_node_state
    current_h = manhattan_distance(current_s)
    path = [current_s] # Path taken
    
    best_s_so_far = current_s # Keep track of the absolute best state found
    best_h_so_far = current_h

    temp = initial_temp

    while temp > min_temp:
        if time.time() - start_time > time_limit:
            print(f"Simulated Annealing: Timeout ({time_limit}s). Temp was {temp:.2f}")
            # Return path to best state found, not necessarily current path
            # For consistency with others, return path taken. If goal found, it's in path.
            return path 

        if is_goal(current_s):
            print("Simulated Annealing: Goal found.")
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        next_s = random.choice(neighbors)
        next_h = manhattan_distance(next_s)

        delta_h = next_h - current_h # If positive, next_s is worse

        if delta_h < 0: # next_s is better
            current_s = next_s
            current_h = next_h
            path.append(current_s)
            if current_h < best_h_so_far: # Update absolute best if this is better
                best_s_so_far = current_s
                best_h_so_far = current_h
        else: # next_s is worse or equal
            # Probability of accepting a worse state
            if temp > 0: # Avoid division by zero if temp somehow becomes 0
                acceptance_prob = math.exp(-delta_h / temp)
                if random.random() < acceptance_prob:
                    current_s = next_s
                    current_h = next_h
                    path.append(current_s)
            # No else needed: if not accepted, current_s and current_h remain unchanged
        
        temp *= cooling_rate
    
    print(f"Simulated Annealing: Cooled down. Final temp {temp:.2f}. Best h found: {best_h_so_far}")
    # If goal not found, path ends at the last state visited.
    # It might be useful to return the path leading to best_s_so_far if goal not found.
    # However, current structure returns the actual path of exploration.
    return path


def beam_search(start_node_state, beam_width=5, time_limit=30):
    start_time = time.time()
    if is_goal(start_node_state): return [start_node_state]

    # Beam stores tuples: (state, path_to_state, heuristic_of_state)
    # Using heuristic (Manhattan distance) to guide the beam.
    h_init = manhattan_distance(start_node_state)
    beam = [(start_node_state, [start_node_state], h_init)]
    
    # Visited set to avoid re-exploring states within the beam's history at a level
    # More complex visited handling might be needed for optimal paths, but for basic beam search:
    visited_tuples_global = {state_to_tuple(start_node_state)}

    best_goal_path_found = None

    while beam: # While there are states in the current beam
        if time.time() - start_time > time_limit:
            print(f"Beam Search Timeout ({time_limit}s)")
            return best_goal_path_found # Return best goal found so far, or None

        next_level_candidates = [] # Candidates for the next beam

        # Expand all states in the current beam
        for s_beam, path_beam, _ in beam:
            for neighbor_s_beam in get_neighbors(s_beam):
                neighbor_tuple_beam = state_to_tuple(neighbor_s_beam)
                if neighbor_tuple_beam is None or neighbor_tuple_beam in visited_tuples_global:
                    continue
                
                # visited_tuples_global.add(neighbor_tuple_beam) # Add when selected for next beam, or here?
                                                              # Adding here prevents considering it via another parent in same level.
                                                              # Standard is often to add when chosen for *next* beam.
                                                              # For simplicity here, let's add when generated if not looking for optimal.

                new_path_beam = path_beam + [neighbor_s_beam]
                
                if is_goal(neighbor_s_beam):
                    # Found a goal. Since Beam Search is not optimal, store it and continue
                    # to see if other paths in beam also reach goal (maybe shorter, though unlikely with h-sort)
                    if best_goal_path_found is None or len(new_path_beam) < len(best_goal_path_found):
                        best_goal_path_found = new_path_beam
                    # We could choose to terminate here if any goal is fine.
                    # Or continue to fill the beam width for this level.
                    # For now, let's collect all candidates for the level.

                h_neighbor_beam = manhattan_distance(neighbor_s_beam)
                next_level_candidates.append((neighbor_s_beam, new_path_beam, h_neighbor_beam))
        
        if not next_level_candidates: # No new states generated
            break

        # Sort candidates by heuristic (lower is better)
        next_level_candidates.sort(key=lambda x: x[2])
        
        # Select the best 'beam_width' candidates for the next beam
        new_beam = []
        for cand_s, cand_path, cand_h in next_level_candidates[:beam_width]:
            cand_tuple = state_to_tuple(cand_s)
            # Ensure we don't add duplicates to the new beam if multiple parents led to same child
            # And add to global visited here.
            if cand_tuple not in visited_tuples_global: # Check before adding to new_beam
                 new_beam.append((cand_s, cand_path, cand_h))
                 visited_tuples_global.add(cand_tuple)
        beam = new_beam
        
        # If a goal was found and the beam is now empty or all states in beam are worse than goal,
        # we can potentially stop. However, standard beam search fills the beam.

    if best_goal_path_found:
        print(f"Beam Search: Goal found, path length {len(best_goal_path_found)}.")
    else:
        print("Beam Search: Beam empty or timeout/limit before finding goal.")
    return best_goal_path_found


# AND-OR Search (Simplified for puzzle solving - treat as OR search for now)
# A full AND-OR graph is more for problems with AND nodes (e.g., subproblems that ALL must be solved)
# For 8-puzzle, each move leads to an OR choice. So this will behave like a form of DFS/BFS.
# For a true AND-OR, problem structure needs to define AND/OR nodes.
# Here, we'll implement a recursive DFS-like search that tracks solved/unsolved.
def _and_or_recursive(state_ao, path_ao, solved_states_ao, unsolved_states_ao, start_time_ao, time_limit_ao, depth_ao, max_depth_ao=50):
    state_tuple_ao = state_to_tuple(state_ao)
    if state_tuple_ao is None: return "UNSOLVED", None

    if time.time() - start_time_ao > time_limit_ao:
        return "Timeout", None
    
    if depth_ao > max_depth_ao: # Depth limit to prevent infinite loops in non-optimal paths
        return "UNSOLVED", None 

    if is_goal(state_ao):
        solved_states_ao.add(state_tuple_ao)
        return "SOLVED", path_ao
    
    if state_tuple_ao in solved_states_ao: # Already known to be solvable
        # This part is tricky: path_ao is current path. If state_tuple_ao is solved,
        # it means there's *some* path from it to goal. We need *that* path segment.
        # For simplicity, if we hit a state marked 'SOLVED', assume this path is part of it.
        # A full AND-OR would reconstruct the plan. Here, we just confirm solvability.
        return "SOLVED", path_ao # Simplification: current path led to a solvable state.

    if state_tuple_ao in unsolved_states_ao: # Already known to be unsolvable from here
        return "UNSOLVED", None

    # Mark as currently trying to solve (part of recursion stack)
    # This is akin to adding to 'unsolved' optimistically, or a 'visiting' set.
    # For this simplified version, let's assume an OR graph logic:
    
    # Add to 'visiting' to detect cycles in current path of recursion
    # Let's use unsolved_states_ao as a general "explored and not (yet) proven SOLVED"
    unsolved_states_ao.add(state_tuple_ao) # Mark as being explored

    for neighbor_ao in get_neighbors(state_ao):
        neighbor_tuple_ao = state_to_tuple(neighbor_ao)
        if neighbor_tuple_ao is None: continue

        # If neighbor is already part of the current recursive path (cycle)
        # This check is implicitly handled if neighbor_tuple_ao is in unsolved_states_ao from a deeper recursion that returned UNSOLVED
        # Or if it's in solved_states_ao.
        # Basic cycle check: if neighbor_s in path_ao[:-1] -> skip (already in path)
        # However, visited sets (solved/unsolved) are more general.

        if neighbor_tuple_ao not in solved_states_ao and neighbor_tuple_ao not in unsolved_states_ao:
            # If truly new, or if previously in unsolved but from a different branch.
            status_ao, solution_path_ao = _and_or_recursive(
                neighbor_ao, path_ao + [neighbor_ao], 
                solved_states_ao, unsolved_states_ao, 
                start_time_ao, time_limit_ao, depth_ao + 1, max_depth_ao
            )

            if status_ao == "Timeout":
                return "Timeout", None
            if status_ao == "SOLVED":
                # If any OR branch leads to SOLVED, then current state_ao is SOLVED
                solved_states_ao.add(state_tuple_ao)
                if state_tuple_ao in unsolved_states_ao: # Move from unsolved to solved
                    unsolved_states_ao.remove(state_tuple_ao)
                return "SOLVED", solution_path_ao # Return the successful path
    
    # If all OR branches failed or were timeouts that didn't yield solution
    # current state_ao remains in unsolved_states_ao (or is confirmed unsolvable if all branches explored)
    return "UNSOLVED", None


def and_or_search(start_node_state, time_limit=30): # Simplified to OR-graph search
    start_time = time.time()
    solved_states = set()    # Tuples of states known to lead to a goal
    unsolved_states = set()  # Tuples of states known not to lead to a goal or currently exploring

    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None:
        print("AND-OR (Simplified): Invalid initial state.")
        return None

    status, solution_path = _and_or_recursive(
        start_node_state, [start_node_state], 
        solved_states, unsolved_states, 
        start_time, time_limit, 0
    )

    if status == "SOLVED":
        print("AND-OR (Simplified): Solution found.")
        return solution_path
    elif status == "Timeout":
        print(f"AND-OR (Simplified): Timeout ({time_limit}s).")
        return None
    else: # UNSOLVED
        print("AND-OR (Simplified): No solution found (exhausted/depth limit).")
        return None

# --- MODIFIED BACKTRACKING SEARCH ---
def backtracking_search(selected_sub_algorithm_name, max_attempts=50, time_limit_overall=60):
    global initial_state # This global is initial_state_fixed_global, used for UI. Not the start for these attempts.
                        # The goal is the global `goal_state` (which is goal_state_fixed_global)

    start_time_overall = time.time()
    attempts = 0

    algo_map = {
        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 'Greedy': greedy,
        'IDA*': ida_star, 
        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing, 
        'Beam Search': beam_search, 'AND-OR': and_or_search
    }
    sub_algo_func = algo_map.get(selected_sub_algorithm_name)
    if not sub_algo_func:
        return None, f"Sub-algorithm '{selected_sub_algorithm_name}' not supported.", None

    states_to_try_q = deque()
    tried_start_state_tuples = set()

    # 1. Initial state for the very first attempt by backtracking
    base_attempt_flat = [0, 1, 2, 3, 4, 5, 6, 7, 8] # As per request
    states_to_try_q.append(list(base_attempt_flat)) # Store as list
    tried_start_state_tuples.add(tuple(base_attempt_flat))
    
    generated_neighbors_of_base = False
    time_limit_per_sub_solve = max(1.0, time_limit_overall / max_attempts if max_attempts > 0 else time_limit_overall / 5) # Ensure some time
    time_limit_per_sub_solve = min(time_limit_per_sub_solve, 15) # Cap sub-solve time

    solution_found_path = None
    actual_start_state_of_solution = None

    while states_to_try_q and attempts < max_attempts:
        if time.time() - start_time_overall > time_limit_overall:
            return None, "Backtracking: Global timeout.", None
        
        current_attempt_flat_start = states_to_try_q.popleft()
        attempts += 1

        current_attempt_2d_start = [
            [current_attempt_flat_start[r*GRID_SIZE + c] for c in range(GRID_SIZE)] 
            for r in range(GRID_SIZE)
        ]

        print(f"Backtracking Attempt {attempts}/{max_attempts} with {selected_sub_algorithm_name}: "
              f"Trying from {current_attempt_2d_start}")
        
        # Prepare arguments for the sub-algorithm
        # Most sub-algorithms take (initial_state_for_attempt, optional_time_limit)
        algo_params = [current_attempt_2d_start] 
        sub_algo_func_varnames = sub_algo_func.__code__.co_varnames[:sub_algo_func.__code__.co_argcount]
        
        # Add time_limit if the sub-algorithm expects it
        # Some algos have specific time limits (e.g. IDA*), others can take a general one.
        # We will pass our calculated `time_limit_per_sub_solve`.
        if 'time_limit' in sub_algo_func_varnames:
            if selected_sub_algorithm_name == 'IDA*': # IDA* has its own default, use that or a larger one
                 algo_params.append(max(time_limit_per_sub_solve, 30)) # Give IDA* a bit more
            else:
                algo_params.append(time_limit_per_sub_solve)
        elif selected_sub_algorithm_name == 'DFS': # DFS needs max_depth
            algo_params.append(30) # Default max_depth for DFS if not specified
            if 'time_limit' in sub_algo_func_varnames: # If it also takes time_limit
                 algo_params.append(time_limit_per_sub_solve)

        path_from_sub_algo = None
        try:
            path_from_sub_algo = sub_algo_func(*algo_params)
        except Exception as e:
            print(f"Backtracking: Sub-algorithm {selected_sub_algorithm_name} error: {str(e)}")
            traceback.print_exc() # For debugging
        
        if path_from_sub_algo and len(path_from_sub_algo) > 0:
            # Check if the path actually reaches the global goal_state
            if is_goal(path_from_sub_algo[-1]):
                solution_found_path = path_from_sub_algo
                actual_start_state_of_solution = current_attempt_2d_start
                print(f"Backtracking: Found solution with {selected_sub_algorithm_name} "
                      f"from {actual_start_state_of_solution} after {attempts} attempts.")
                break # Exit while loop, solution found
            else:
                print(f"Backtracking: Sub-algo {selected_sub_algorithm_name} finished but did not reach goal from {current_attempt_2d_start}.")
        else:
            print(f"Backtracking: Sub-algo {selected_sub_algorithm_name} found no path from {current_attempt_2d_start}.")

        # If no solution from current attempt AND queue is empty, generate more states to try
        if not solution_found_path and not states_to_try_q and attempts < max_attempts:
            if not generated_neighbors_of_base:
                print("Backtracking: Base state attempt failed. Generating its single-swap neighbors...")
                parent_for_new_states = base_attempt_flat # Generate from the original [0,1,2...]
                newly_added_count = 0
                for i in range(len(parent_for_new_states)):
                    for j in range(i + 1, len(parent_for_new_states)):
                        if len(states_to_try_q) > max_attempts * 2: break # Limit queue size
                        
                        neighbor_flat_list = list(parent_for_new_states)
                        neighbor_flat_list[i], neighbor_flat_list[j] = neighbor_flat_list[j], neighbor_flat_list[i]
                        neighbor_tuple_to_check = tuple(neighbor_flat_list)

                        if neighbor_tuple_to_check not in tried_start_state_tuples:
                            states_to_try_q.append(neighbor_flat_list)
                            tried_start_state_tuples.add(neighbor_tuple_to_check)
                            newly_added_count +=1
                    if len(states_to_try_q) > max_attempts * 2: break
                
                generated_neighbors_of_base = True
                if newly_added_count > 0:
                    temp_list_for_shuffle = list(states_to_try_q)
                    random.shuffle(temp_list_for_shuffle)
                    states_to_try_q = deque(temp_list_for_shuffle)
                    print(f"Backtracking: Added {newly_added_count} single-swap neighbors of base to try.")
                else:
                    print("Backtracking: No novel single-swap neighbors of base found (all already tried or will be).")

            else: # If single-swap neighbors of base also tried/exhausted, try some random permutations
                print("Backtracking: Single-swap neighbors attempted. Generating random permutations...")
                newly_added_count = 0
                for _ in range(min(20, max_attempts - attempts)): # Generate a few random ones
                    if len(states_to_try_q) > max_attempts * 2: break

                    random_flat_list = list(range(GRID_SIZE * GRID_SIZE))
                    random.shuffle(random_flat_list)
                    random_tuple_to_check = tuple(random_flat_list)

                    if random_tuple_to_check not in tried_start_state_tuples:
                        states_to_try_q.append(random_flat_list)
                        tried_start_state_tuples.add(random_tuple_to_check)
                        newly_added_count +=1
                if newly_added_count > 0:
                     print(f"Backtracking: Added {newly_added_count} random permutations to try.")
                else:
                    print("Backtracking: Failed to generate novel random permutations (many might have been tried).")
    
    if solution_found_path:
        return solution_found_path, None, actual_start_state_of_solution
    else:
        msg = f"Backtracking: No solution found after {attempts} attempts with {selected_sub_algorithm_name}."
        if time.time() - start_time_overall > time_limit_overall:
            msg = "Backtracking: Global timeout. " + msg
        return None, msg, None


# Sensorless Search related functions
def generate_belief_states(partial_state): # Not used with fixed initial state
    # This function's original purpose was for partially observable states.
    # For the 8-puzzle with a fully defined initial_state, there's only one belief state: initial_state itself.
    # If the initial_state were e.g. [[-1, 1, 2], ...], then it would generate possibilities.
    # Given current setup (fixed initial_state), this might not be used by Sensorless in the same way.
    # Sensorless for 8-puzzle implies the agent doesn't know *its own* state after an action.
    # It would start with all possible (solvable) 8-puzzle states as the initial belief set.
    # This is computationally very expensive.
    # The existing sensorless_search likely assumes a small number of initial belief states from a template.
    # For 8-puzzle, it usually means starting with a set of all possible states.
    # We will keep the provided logic for now. If `initial_state` has -1, it generates.
    # With fixed initial_state, it will return [initial_state] if that state is solvable.
    
    flat_state = []
    is_template = False
    for r in partial_state:
        for tile in r:
            flat_state.append(tile)
            if tile == -1: is_template = True
    
    if not is_template: # If it's a full state, just check solvability
        if is_solvable(partial_state): # is_solvable needs to be defined or imported
            return [partial_state]
        else:
            return []

    used_numbers = [tile for tile in flat_state if tile != -1]
    missing_numbers = [i for i in range(GRID_SIZE * GRID_SIZE) if i not in used_numbers]
    unknown_positions_indices = [i for i, tile in enumerate(flat_state) if tile == -1]
    
    belief_states_generated = []
    if len(missing_numbers) != len(unknown_positions_indices):
        print("Sensorless: Mismatch between missing numbers and unknown positions.")
        return [] # Mismatch, cannot form valid states

    for perm_nums in itertools.permutations(missing_numbers): # Permute the missing numbers
        new_flat_state_perm = list(flat_state) # Copy
        for idx, अज्ञात_pos_idx in enumerate(unknown_positions_indices):
            new_flat_state_perm[अज्ञात_pos_idx] = perm_nums[idx]
        
        # Convert flat state back to 2D
        state_2d_perm = []
        for i in range(0, len(new_flat_state_perm), GRID_SIZE):
            state_2d_perm.append(new_flat_state_perm[i:i+GRID_SIZE])
        
        # For 8-puzzle, typically you'd check if this permutation is solvable relative to the goal
        # if is_solvable(state_2d_perm): # is_solvable needs to be defined
        belief_states_generated.append(state_2d_perm) # Add all for now
    
    return belief_states_generated


def is_belief_state_goal(list_of_belief_states): # All states in belief set must be goal
    if not list_of_belief_states: return False
    for state_b in list_of_belief_states:
        if not is_goal(state_b): # is_goal checks against global goal_state
            return False
    return True

def apply_action_to_belief_states(list_of_belief_states, action_str):
    new_belief_states_set = set() # Use set to store tuples of new states to avoid duplicates
    
    for state_b in list_of_belief_states:
        # apply_action_to_state moves the '0' tile.
        # For sensorless, the action is applied, and all possible outcomes form the new belief state.
        # In deterministic 8-puzzle, one action leads to one state.
        next_s_b = apply_action_to_state(state_b, action_str)
        if next_s_b: # If action was valid and resulted in a state
            new_belief_states_set.add(state_to_tuple(next_s_b))
            
    return [tuple_to_list(s_tuple) for s_tuple in new_belief_states_set]


def sensorless_search(start_belief_state_template, time_limit=60):
    # For 8-puzzle, a true sensorless problem starts with agent not knowing its state.
    # Initial belief set = all valid (e.g. solvable) 8-puzzle configurations. This is huge (9!/2).
    # The provided `generate_belief_states` seems to be for partially specified initial states.
    # If `start_belief_state_template` is our fixed `initial_state`, it generates just `[initial_state]`.
    # This makes it behave like a regular search on that single state if actions are deterministic.
    
    start_time = time.time()
    
    # If initial_state is fully specified, belief_states will be [initial_state]
    current_belief_set = generate_belief_states(start_belief_state_template) 
    if not current_belief_set:
        print("Sensorless: No valid initial belief states generated from template.")
        return None
    
    # Queue stores (action_plan_so_far, current_set_of_belief_states)
    queue_sensorless = deque([ ( [], current_belief_set ) ])
    
    # Visited set for belief states (set of frozensets of state tuples)
    # A belief state is a set of possible states the agent could be in.
    visited_belief_sets = set()
    initial_belief_tuple_set = frozenset(state_to_tuple(s) for s in current_belief_set)
    visited_belief_sets.add(initial_belief_tuple_set)

    possible_actions = ['Up', 'Down', 'Left', 'Right']
    
    while queue_sensorless:
        if time.time() - start_time > time_limit:
            print(f"Sensorless Search Timeout ({time_limit}s)")
            return None
        
        action_plan, current_bs = queue_sensorless.popleft()
        
        # Check if ALL states in the current belief set are goal states
        if is_belief_state_goal(current_bs):
            print(f"Sensorless: Found plan with {len(action_plan)} actions. Belief set size at goal: {len(current_bs)}")
            return action_plan # This is a plan of actions
        
        for action_s in possible_actions:
            # Apply action to each state in the current belief set
            # The result is a new set of states (the next belief set)
            next_bs = apply_action_to_belief_states(current_bs, action_s)
            
            if not next_bs: # If action leads to no valid states (e.g., all states hit a wall)
                continue

            # Check if this new belief set has been visited
            next_bs_tuple_set = frozenset(state_to_tuple(s) for s in next_bs)
            if next_bs_tuple_set not in visited_belief_sets:
                visited_belief_sets.add(next_bs_tuple_set)
                new_plan_s = action_plan + [action_s]
                queue_sensorless.append((new_plan_s, next_bs))
                
    print("Sensorless: No plan found to make all belief states the goal state.")
    return None

# Solvability check for 8-puzzle (standard: goal is 1,2,3...8,0 with 0 at last pos)
# For a generic goal, the definition is more complex (relative parity).
# Here, our goal_state_fixed_global has 0 at the end.
def get_inversions(flat_state_no_zero):
    inversions = 0
    n = len(flat_state_no_zero)
    for i in range(n):
        for j in range(i + 1, n):
            if flat_state_no_zero[i] > flat_state_no_zero[j]:
                inversions += 1
    return inversions

def is_solvable(state_to_check): # Checks solvability against the fixed global goal
    # For GRID_SIZE = 3 (odd width grid):
    # Solvable if number of inversions is EVEN for the state AND for the goal, OR ODD for both.
    # Standard goal [1,2,3,4,5,6,7,8,0] has 0 inversions (even).
    # So, the state_to_check must also have an even number of inversions.
    
    flat_state = []
    for r in state_to_check: flat_state.extend(r)
    
    if 0 not in flat_state: return False # Invalid state for 8-puzzle

    # Standard check: if grid width is odd, solvability depends on parity of inversions.
    # If grid width is even, it also depends on row of blank tile (from bottom).
    # Our grid is 3x3 (odd).
    
    # Inversions for the state_to_check
    state_flat_no_zero = [tile for tile in flat_state if tile != 0]
    inversions_state = get_inversions(state_flat_no_zero)

    # Inversions for the goal_state (goal_state_fixed_global)
    goal_flat = []
    for r_goal in goal_state_fixed_global: goal_flat.extend(r_goal)
    goal_flat_no_zero = [tile for tile in goal_flat if tile != 0]
    inversions_goal = get_inversions(goal_flat_no_zero)

    if GRID_SIZE % 2 == 1: # Odd grid width (like 3x3)
        return (inversions_state % 2) == (inversions_goal % 2)
    else: # Even grid width (e.g. 4x4) - more complex, involves blank tile row
        # Find blank row from bottom (1-indexed) for state
        blank_row_state = -1
        for r_idx, row_s in enumerate(state_to_check):
            if 0 in row_s:
                blank_row_state = GRID_SIZE - r_idx
                break
        
        # Find blank row from bottom for goal
        blank_row_goal = -1
        for r_idx_g, row_g in enumerate(goal_state_fixed_global):
            if 0 in row_g:
                blank_row_goal = GRID_SIZE - r_idx_g
                break

        if blank_row_state == -1 or blank_row_goal == -1: return False # Should not happen

        # (inversions + blank_row_from_bottom) must have same parity for state and goal
        return ((inversions_state + blank_row_state) % 2) == \
               ((inversions_goal + blank_row_goal) % 2)


# Pygame UI Drawing and Interaction Logic
def algorithm_selection_popup(): # For Backtracking's sub-algorithm
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)

    title_font_popup = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_popup = title_font_popup.render("Select Sub-Algorithm for Backtracking", True, INFO_COLOR)
    title_rect_popup = title_surf_popup.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf_popup, title_rect_popup)

    # Algorithms suitable as sub-algorithms for backtracking
    # Exclude Backtracking itself, and Sensorless.
    algorithms_for_popup = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search')
    ]
    button_width_popup, button_height_popup = 150, 40
    button_margin_popup = 10
    columns_popup = 3
    start_x_popup = (POPUP_WIDTH - (columns_popup * button_width_popup + (columns_popup - 1) * button_margin_popup)) // 2
    start_y_popup = title_rect_popup.bottom + 30
    
    button_rects_popup = {}
    algo_buttons_popup = []

    for idx, (algo_id_p, algo_name_p) in enumerate(algorithms_for_popup):
        col_p = idx % columns_popup
        row_p = idx // columns_popup
        button_x_p = start_x_popup + col_p * (button_width_popup + button_margin_popup)
        button_y_p = start_y_popup + row_p * (button_height_popup + button_margin_popup)
        button_rect_p = pygame.Rect(button_x_p, button_y_p, button_width_popup, button_height_popup)
        button_rects_popup[algo_id_p] = button_rect_p
        algo_buttons_popup.append((algo_id_p, algo_name_p, button_rect_p))

    cancel_button_rect_popup = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    
    popup_rect_on_screen = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    selected_algorithm_name = None
    running_popup = True
    clock_popup = pygame.time.Clock()

    while running_popup:
        mouse_pos_screen = pygame.mouse.get_pos()
        # Convert mouse position to be relative to the popup surface
        mouse_pos_relative = (mouse_pos_screen[0] - popup_rect_on_screen.left, 
                              mouse_pos_screen[1] - popup_rect_on_screen.top)

        popup_surface.fill(INFO_BG) # Redraw background
        pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10) # Border
        popup_surface.blit(title_surf_popup, title_rect_popup) # Title

        # Draw algorithm buttons
        for algo_id_p, algo_name_p, btn_rect_p in algo_buttons_popup:
            is_hovered_p = btn_rect_p.collidepoint(mouse_pos_relative)
            btn_color_p = MENU_HOVER_COLOR if is_hovered_p else MENU_BUTTON_COLOR
            pygame.draw.rect(popup_surface, btn_color_p, btn_rect_p, border_radius=8)
            text_surf_p = BUTTON_FONT.render(algo_name_p, True, WHITE)
            text_rect_p = text_surf_p.get_rect(center=btn_rect_p.center)
            popup_surface.blit(text_surf_p, text_rect_p)

        # Draw Cancel button
        pygame.draw.rect(popup_surface, RED, cancel_button_rect_popup, border_radius=8)
        cancel_text_surf_p = BUTTON_FONT.render("Cancel", True, WHITE)
        cancel_text_rect_p = cancel_text_surf_p.get_rect(center=cancel_button_rect_popup.center)
        popup_surface.blit(cancel_text_surf_p, cancel_text_rect_p)

        screen.blit(popup_surface, popup_rect_on_screen) # Draw popup onto main screen
        pygame.display.flip() # Update display

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None # Or handle exit
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button_rect_popup.collidepoint(mouse_pos_relative):
                    return None # Cancelled
                for algo_id_p, _, btn_rect_p in algo_buttons_popup:
                    if btn_rect_p.collidepoint(mouse_pos_relative):
                        selected_algorithm_name = algo_id_p
                        running_popup = False # Exit popup loop
                        break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None # Cancelled

        clock_popup.tick(60)
    
    return selected_algorithm_name


scroll_y = 0 # For main algorithm menu scrolling
menu_surface = None # Surface for the scrollable menu
total_menu_height = 0 # Calculated height of all menu items

def draw_state(state_to_draw, x_pos, y_pos, title_str, is_current_anim_state=False, is_fixed_goal_display=False):
    title_font_ds = pygame.font.Font(None, 28) # Default font, size 28
    title_text_ds = title_font_ds.render(title_str, True, BLACK)
    title_x_ds = x_pos + (GRID_DISPLAY_WIDTH // 2 - title_text_ds.get_width() // 2)
    title_y_ds = y_pos - title_text_ds.get_height() - 5 # Position title above grid
    screen.blit(title_text_ds, (title_x_ds, title_y_ds))

    # Border for the grid
    pygame.draw.rect(screen, BLACK, (x_pos - 1, y_pos - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)

    for r_ds in range(GRID_SIZE):
        for c_ds in range(GRID_SIZE):
            cell_x_ds = x_pos + c_ds * CELL_SIZE
            cell_y_ds = y_pos + r_ds * CELL_SIZE
            cell_rect_ds = pygame.Rect(cell_x_ds, cell_y_ds, CELL_SIZE, CELL_SIZE)
            
            tile_val = -1 # Default for safety
            if state_to_draw and r_ds < len(state_to_draw) and c_ds < len(state_to_draw[r_ds]):
                 tile_val = state_to_draw[r_ds][c_ds]
            else: # Handle incomplete or malformed state gracefully
                pygame.draw.rect(screen, RED, cell_rect_ds.inflate(-6,-6), border_radius=8) # Error indication
                pygame.draw.rect(screen, BLACK, cell_rect_ds, 1)
                continue


            if tile_val != 0: # If not the empty tile
                # Color determination
                cell_fill_color = BLUE # Default for numbers
                if is_fixed_goal_display: # The "Goal State" panel
                    cell_fill_color = GREEN 
                elif is_current_anim_state and is_goal(state_to_draw): # If current state in animation IS the goal
                    cell_fill_color = GREEN
                
                pygame.draw.rect(screen, cell_fill_color, cell_rect_ds.inflate(-6, -6), border_radius=8)
                number_surf = FONT.render(str(tile_val), True, WHITE)
                screen.blit(number_surf, number_surf.get_rect(center=cell_rect_ds.center))
            else: # Empty tile (0)
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
            
            pygame.draw.rect(screen, BLACK, cell_rect_ds, 1) # Cell border


def show_popup(message_str, title_str="Info"):
    # Create a new surface for the popup
    popup_surface_sp = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface_sp.fill(INFO_BG) # Background color
    
    # Border for the popup
    border_rect_sp = popup_surface_sp.get_rect()
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, border_rect_sp, 4, border_radius=10)

    # Title
    title_font_sp = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_sp = title_font_sp.render(title_str, True, INFO_COLOR)
    title_rect_sp = title_surf_sp.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface_sp.blit(title_surf_sp, title_rect_sp)

    # Message text (multi-line handling)
    words_sp = message_str.split(' ')
    lines_sp = []
    current_line_sp = ""
    text_width_limit_sp = POPUP_WIDTH - 60 # Padding for text
    
    for word_sp in words_sp:
        # Handle explicit newlines within a "word"
        if "\n" in word_sp:
            parts = word_sp.split("\n")
            for i, part_sp in enumerate(parts):
                if not part_sp and i < len(parts) -1 : # Handle blank line from "\n\n" but not trailing empty part
                    lines_sp.append(current_line_sp)
                    current_line_sp = "" # Start new line for the blank line itself
                    lines_sp.append("") 
                    continue
                if not part_sp: continue # Skip empty parts

                test_line_part_sp = current_line_sp + (" " if current_line_sp and part_sp else "") + part_sp
                line_width_part_sp = INFO_FONT.size(test_line_part_sp)[0]
                
                if line_width_part_sp <= text_width_limit_sp:
                    current_line_sp = test_line_part_sp
                else:
                    lines_sp.append(current_line_sp)
                    current_line_sp = part_sp # This part starts a new line
                
                if i < len(parts) - 1: # If this part was before a \n
                    lines_sp.append(current_line_sp)
                    current_line_sp = ""
            continue # Move to next word_sp

        # Regular word processing
        test_line_sp = current_line_sp + (" " if current_line_sp else "") + word_sp
        line_width_sp = INFO_FONT.size(test_line_sp)[0]
        
        if line_width_sp <= text_width_limit_sp:
            current_line_sp = test_line_sp
        else:
            lines_sp.append(current_line_sp)
            current_line_sp = word_sp # Word starts a new line
    
    lines_sp.append(current_line_sp) # Add the last line

    line_height_sp = INFO_FONT.get_linesize()
    text_start_y_sp = title_rect_sp.bottom + 20
    
    for i, line_text_sp in enumerate(lines_sp):
        if text_start_y_sp + i * line_height_sp > POPUP_HEIGHT - 80: # Check if text overflows popup
            # Draw ellipsis and stop if text too long
            ellipsis_surf = INFO_FONT.render("...", True, BLACK)
            popup_surface_sp.blit(ellipsis_surf, ( (POPUP_WIDTH - ellipsis_surf.get_width()) // 2, text_start_y_sp + i * line_height_sp))
            break
        text_surf_sp = INFO_FONT.render(line_text_sp, True, BLACK)
        # Center each line of text
        text_rect_sp = text_surf_sp.get_rect(center=(POPUP_WIDTH // 2, text_start_y_sp + i * line_height_sp + line_height_sp // 2))
        popup_surface_sp.blit(text_surf_sp, text_rect_sp)

    # OK Button
    ok_button_rect_sp = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, ok_button_rect_sp, border_radius=8)
    ok_text_surf_sp = BUTTON_FONT.render("OK", True, WHITE)
    ok_text_rect_sp = ok_text_surf_sp.get_rect(center=ok_button_rect_sp.center)
    popup_surface_sp.blit(ok_text_surf_sp, ok_text_rect_sp)

    # Position popup on main screen and display
    popup_rect_on_screen_sp = popup_surface_sp.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(popup_surface_sp, popup_rect_on_screen_sp)
    pygame.display.flip()

    # Wait for user to click OK or press Enter/Escape
    waiting_for_ok = True
    while waiting_for_ok:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x_sp, mouse_y_sp = event.pos
                # Check click relative to popup's position on screen
                if ok_button_rect_sp.collidepoint(mouse_x_sp - popup_rect_on_screen_sp.left, 
                                                  mouse_y_sp - popup_rect_on_screen_sp.top):
                    waiting_for_ok = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    waiting_for_ok = False
        pygame.time.delay(20) # Small delay to reduce CPU usage

# Draw main algorithm selection menu (on the left)
def draw_menu(show_menu_flag, mouse_pos_dm, current_selected_algorithm_dm):
    global scroll_y, menu_surface, total_menu_height # Use globals for scroll state
    menu_elements_dict = {} # To return clickable areas

    if not show_menu_flag: # If menu is hidden, just draw the 'hamburger' open button
        menu_button_rect_dm = pygame.Rect(10, 10, 50, 40) # Position and size of open button
        pygame.draw.rect(screen, MENU_COLOR, menu_button_rect_dm, border_radius=5)
        # Draw hamburger icon lines
        bar_width_dm, bar_height_dm, space_dm = 30, 4, 7
        start_x_dm = menu_button_rect_dm.centerx - bar_width_dm // 2
        start_y_dm = menu_button_rect_dm.centery - (bar_height_dm * 3 + space_dm * 2) // 2 + bar_height_dm // 2
        for i in range(3):
            pygame.draw.rect(screen, WHITE, (start_x_dm, start_y_dm + i * (bar_height_dm + space_dm), bar_width_dm, bar_height_dm), border_radius=2)
        menu_elements_dict['open_button'] = menu_button_rect_dm
        return menu_elements_dict

    # If menu is shown:
    # List of algorithms for the menu
    algorithms_list_dm = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Backtracking', 'Backtracking'), # User's special algorithm
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Sensorless', 'Sensorless Plan')
    ]
    
    button_h_dm, padding_dm, button_margin_dm = 55, 10, 8
    # Calculate total height needed for all menu items
    total_menu_height = (len(algorithms_list_dm) * (button_h_dm + button_margin_dm)) - button_margin_dm + (2 * padding_dm)
    
    # Create or resize menu_surface if needed
    # Display height should be at least screen height for scrolling logic
    display_height_menu_surf = max(total_menu_height, HEIGHT) 
    if menu_surface is None or menu_surface.get_height() != display_height_menu_surf:
        menu_surface = pygame.Surface((MENU_WIDTH, display_height_menu_surf))

    menu_surface.fill(MENU_COLOR) # Background of the menu panel
    
    buttons_dict_dm = {} # Store rects of buttons for click detection
    y_pos_current_button = padding_dm # Start y for first button
    
    # Mouse position relative to the scrollable menu surface
    mouse_x_rel_dm, mouse_y_rel_dm = mouse_pos_dm[0], mouse_pos_dm[1] + scroll_y

    for algo_id_dm, algo_name_dm in algorithms_list_dm:
        button_rect_local_dm = pygame.Rect(padding_dm, y_pos_current_button, MENU_WIDTH - 2 * padding_dm, button_h_dm)
        
        is_hover_dm = button_rect_local_dm.collidepoint(mouse_x_rel_dm, mouse_y_rel_dm)
        is_selected_dm = (current_selected_algorithm_dm == algo_id_dm)
        
        button_color_dm = MENU_SELECTED_COLOR if is_selected_dm else \
                          (MENU_HOVER_COLOR if is_hover_dm else MENU_BUTTON_COLOR)
        pygame.draw.rect(menu_surface, button_color_dm, button_rect_local_dm, border_radius=5)
        
        text_surf_dm = BUTTON_FONT.render(algo_name_dm, True, WHITE)
        menu_surface.blit(text_surf_dm, text_surf_dm.get_rect(center=button_rect_local_dm.center))
        
        buttons_dict_dm[algo_id_dm] = button_rect_local_dm # Store local rect (relative to menu_surface)
        y_pos_current_button += button_h_dm + button_margin_dm

    # Blit the visible part of menu_surface to the screen
    visible_menu_area_rect = pygame.Rect(0, scroll_y, MENU_WIDTH, HEIGHT)
    screen.blit(menu_surface, (0,0), visible_menu_area_rect)

    # Close button for the menu (X)
    close_button_rect_dm = pygame.Rect(MENU_WIDTH - 40, 10, 30, 30) # Top-right of visible menu
    pygame.draw.rect(screen, RED, close_button_rect_dm, border_radius=5)
    cx_dm, cy_dm = close_button_rect_dm.center
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm - 7), (cx_dm + 7, cy_dm + 7), 3) # X mark
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm + 7), (cx_dm + 7, cy_dm - 7), 3)
    
    menu_elements_dict['close_button'] = close_button_rect_dm
    menu_elements_dict['buttons'] = buttons_dict_dm # Dict of {algo_id: local_rect}
    menu_elements_dict['menu_area'] = pygame.Rect(0, 0, MENU_WIDTH, HEIGHT) # Clickable area of menu on screen

    # Scrollbar if content exceeds screen height
    if total_menu_height > HEIGHT:
        scrollbar_track_height = HEIGHT - 2*padding_dm # available height for scrollbar track
        scrollbar_height_val = max(20, scrollbar_track_height * (HEIGHT / total_menu_height)) # Thumb height
        
        # Max y scroll position for the content
        max_scroll_y_content = total_menu_height - HEIGHT 
        scroll_ratio = scroll_y / max_scroll_y_content if max_scroll_y_content > 0 else 0
        
        # Max y for scrollbar thumb itself within its track
        scrollbar_track_y_start = padding_dm
        scrollbar_max_y_thumb = scrollbar_track_y_start + scrollbar_track_height - scrollbar_height_val
        
        scrollbar_y_pos_thumb = scrollbar_track_y_start + scroll_ratio * (scrollbar_track_height - scrollbar_height_val)
        scrollbar_y_pos_thumb = max(scrollbar_track_y_start, min(scrollbar_y_pos_thumb, scrollbar_max_y_thumb))


        scrollbar_rect_dm = pygame.Rect(MENU_WIDTH - 10, scrollbar_y_pos_thumb, 6, scrollbar_height_val)
        pygame.draw.rect(screen, GRAY, scrollbar_rect_dm, border_radius=3)
        menu_elements_dict['scrollbar_rect'] = scrollbar_rect_dm # For potential drag-scroll later

    return menu_elements_dict


def draw_grid_and_ui(current_anim_state_dgui, show_menu_dgui, current_algo_name_dgui, 
                     solve_times_dgui, last_solved_run_info_dgui, 
                     current_belief_state_size_dgui=None): # For Sensorless display
    
    screen.fill(WHITE) # Clear screen
    mouse_pos_dgui = pygame.mouse.get_pos()

    # Determine main content area based on menu visibility
    main_area_x_offset = MENU_WIDTH if show_menu_dgui else 0
    main_area_width_dgui = WIDTH - main_area_x_offset
    center_x_main_area = main_area_x_offset + main_area_width_dgui // 2

    # Positions for Initial and Goal state grids (top row)
    top_row_y_grids = GRID_PADDING + 40 # Extra space for titles
    grid_spacing_horizontal = GRID_PADDING * 1.5
    total_width_top_grids = 2 * GRID_DISPLAY_WIDTH + grid_spacing_horizontal
    start_x_top_grids = center_x_main_area - total_width_top_grids // 2
    
    initial_grid_x = start_x_top_grids
    goal_grid_x = start_x_top_grids + GRID_DISPLAY_WIDTH + grid_spacing_horizontal

    # Draw fixed initial and goal states
    # initial_state and goal_state are the global fixed ones.
    draw_state(initial_state, initial_grid_x, top_row_y_grids, "Initial State")
    draw_state(goal_state, goal_grid_x, top_row_y_grids, "Goal State", is_fixed_goal_display=True)

    # Positions for "Current State" grid and control buttons (bottom area)
    bottom_row_y_start = top_row_y_grids + GRID_DISPLAY_WIDTH + GRID_PADDING + 60 # Y for current state grid & info
    
    # Control buttons (Solve, Resets)
    button_width_ctrl, button_height_ctrl = 140, 45 # Slightly wider buttons
    buttons_start_x = main_area_x_offset + GRID_PADDING
    
    # Center buttons vertically relative to where current state grid might be, or just stack them
    buttons_mid_y_anchor = bottom_row_y_start + GRID_DISPLAY_WIDTH // 2 
    
    solve_button_y_pos = buttons_mid_y_anchor - button_height_ctrl * 2 - 16 # Stacked
    reset_solution_button_y_pos = solve_button_y_pos + button_height_ctrl + 8
    reset_all_button_y_pos = reset_solution_button_y_pos + button_height_ctrl + 8
    # "Reset Initial" button - its role is now similar to "Reset Solution"
    reset_initial_button_y_pos = reset_all_button_y_pos + button_height_ctrl + 8


    solve_button_rect_dgui = pygame.Rect(buttons_start_x, solve_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_solution_button_rect_dgui = pygame.Rect(buttons_start_x, reset_solution_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_all_button_rect_dgui = pygame.Rect(buttons_start_x, reset_all_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_initial_button_rect_dgui = pygame.Rect(buttons_start_x, reset_initial_button_y_pos, button_width_ctrl, button_height_ctrl)

    # Draw control buttons
    pygame.draw.rect(screen, RED, solve_button_rect_dgui, border_radius=5)
    solve_text_dgui = BUTTON_FONT.render("SOLVE", True, WHITE)
    screen.blit(solve_text_dgui, solve_text_dgui.get_rect(center=solve_button_rect_dgui.center))

    pygame.draw.rect(screen, BLUE, reset_solution_button_rect_dgui, border_radius=5)
    rs_text = BUTTON_FONT.render("Reset Solution", True, WHITE) # rs = reset solution
    screen.blit(rs_text, rs_text.get_rect(center=reset_solution_button_rect_dgui.center))

    pygame.draw.rect(screen, BLUE, reset_all_button_rect_dgui, border_radius=5)
    ra_text = BUTTON_FONT.render("Reset All", True, WHITE) # ra = reset all
    screen.blit(ra_text, ra_text.get_rect(center=reset_all_button_rect_dgui.center))
    
    pygame.draw.rect(screen, BLUE, reset_initial_button_rect_dgui, border_radius=5)
    ri_text = BUTTON_FONT.render("Reset Display", True, WHITE) # Changed name: Reset Initial -> Reset Display
    screen.blit(ri_text, ri_text.get_rect(center=reset_initial_button_rect_dgui.center))


    # "Current State" grid (where animation happens)
    current_state_grid_x = buttons_start_x + button_width_ctrl + GRID_PADDING * 1.5
    current_state_grid_y = bottom_row_y_start
    
    current_state_title = f"Current ({current_algo_name_dgui})"
    if current_algo_name_dgui == 'Sensorless' and current_belief_state_size_dgui is not None:
        current_state_title = f"Belief States: {current_belief_state_size_dgui}"
        # For sensorless, current_anim_state_dgui is one sample from belief set for visualization
    draw_state(current_anim_state_dgui, current_state_grid_x, current_state_grid_y, current_state_title, is_current_anim_state=True)


    # Info Area (for solve times, stats)
    info_area_x_pos = current_state_grid_x + GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5
    info_area_y_pos = bottom_row_y_start
    info_area_w = max(150, (main_area_x_offset + main_area_width_dgui) - info_area_x_pos - GRID_PADDING) # Width till screen edge
    info_area_h = GRID_DISPLAY_WIDTH # Match height of current_state grid
    info_area_rect_dgui = pygame.Rect(info_area_x_pos, info_area_y_pos, info_area_w, info_area_h)

    pygame.draw.rect(screen, INFO_BG, info_area_rect_dgui, border_radius=8) # Background
    pygame.draw.rect(screen, GRAY, info_area_rect_dgui, 2, border_radius=8) # Border
    
    info_pad_x_ia, info_pad_y_ia = 15, 10 # ia = info area
    line_h_ia = INFO_FONT.get_linesize() + 4
    current_info_y_draw = info_area_y_pos + info_pad_y_ia

    # Title for info area
    compare_title_surf_ia = TITLE_FONT.render("Solver Stats", True, BLACK)
    compare_title_x_ia = info_area_rect_dgui.centerx - compare_title_surf_ia.get_width() // 2
    screen.blit(compare_title_surf_ia, (compare_title_x_ia, current_info_y_draw))
    current_info_y_draw += compare_title_surf_ia.get_height() + 8

    if solve_times_dgui:
        # Sort by time, then by name if times are equal (for consistent order)
        sorted_times_list = sorted(solve_times_dgui.items(), key=lambda item: (item[1], item[0]))
        
        for algo_name_st, time_val_st in sorted_times_list:
            if current_info_y_draw + line_h_ia > info_area_y_pos + info_area_h - info_pad_y_ia: # Check bounds
                screen.blit(INFO_FONT.render("...", True, BLACK), (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
                break

            steps_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_steps")
            actions_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_actions") # For Sensorless
            reached_goal_st = last_solved_run_info_dgui.get(f"{algo_name_st}_reached_goal", None) # True, False, or None

            base_str_st = f"{algo_name_st}: {time_val_st:.3f}s"
            count_str_st = ""
            if steps_val_st is not None: count_str_st = f" ({steps_val_st} steps)"
            elif actions_val_st is not None: count_str_st = f" ({actions_val_st} actions)"
            else: count_str_st = " (--)" # No step/action count

            goal_indicator_st = ""
            if reached_goal_st is False : goal_indicator_st = " (Not Goal)"
            # If reached_goal_st is True or None (for algos not setting it), no indicator needed by default

            full_comp_str = base_str_st + count_str_st + goal_indicator_st
            
            # Truncate if too long for the info area
            max_text_width = info_area_w - 2 * info_pad_x_ia
            if INFO_FONT.size(full_comp_str)[0] > max_text_width:
                # Attempt to shorten: remove step count first
                shortened_str = base_str_st + goal_indicator_st
                if INFO_FONT.size(shortened_str)[0] <= max_text_width:
                    full_comp_str = shortened_str
                else: # Still too long, just use base string + goal
                    full_comp_str = base_str_st.split(":")[0] + ":" + goal_indicator_st # Algo name + goal
                    if INFO_FONT.size(full_comp_str)[0] > max_text_width: # Super short
                        full_comp_str = (base_str_st.split(":")[0][:10] + "...") if len(base_str_st.split(":")[0]) > 10 else base_str_st.split(":")[0]


            comp_surf_st = INFO_FONT.render(full_comp_str, True, BLACK)
            screen.blit(comp_surf_st, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
            current_info_y_draw += line_h_ia
    else:
        no_results_surf = INFO_FONT.render("(No results yet)", True, GRAY)
        screen.blit(no_results_surf, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))

    # Draw the algorithm selection menu (potentially overlaying other elements if open)
    menu_elements_dgui = draw_menu(show_menu_dgui, mouse_pos_dgui, current_algo_name_dgui)
    
    pygame.display.flip() # Update the full display

    # Return dictionary of clickable UI elements and their rects
    return {
        'solve_button': solve_button_rect_dgui,
        'reset_solution_button': reset_solution_button_rect_dgui,
        'reset_all_button': reset_all_button_rect_dgui,
        'reset_initial_button': reset_initial_button_rect_dgui, # Renamed to "Reset Display"
        'menu': menu_elements_dgui,
        # No need to return initial/goal grid rects for input, they are fixed.
    }


# Main game loop / application structure
def main():
    global scroll_y, initial_state, goal_state # These are now fixed after definition
                                               # but current_state_for_animation will change.
    
    current_state_for_animation = copy.deepcopy(initial_state) # State shown in "Current" panel, starts as initial
    
    solution_path_anim = None # Path for animation [(state1), (state2), ...]
    current_step_in_anim = 0
    
    is_solving_flag = False # True when an algorithm is running
    is_auto_animating_flag = False # True when solution is being animated step-by-step
    last_anim_step_time = 0

    show_algo_menu = False # Is the left-side algorithm menu visible
    current_selected_algorithm = 'A*' # Default algorithm

    # Store solve times and other info for comparison display
    all_solve_times = {} 
    last_run_solver_info = {} # e.g. { "A*_steps": 10, "A*_reached_goal": True }

    # selected_cell_for_input = None # REMOVED - initial/goal states are fixed
    
    game_clock = pygame.time.Clock()
    ui_elements_rects = {} # To store rects of buttons etc. from draw_grid_and_ui
    running_main_loop = True

    # Variables specific to Backtracking sub-algorithm selection
    # This needs to be available when the solve for backtracking is initiated
    backtracking_sub_algo_choice = None 

    while running_main_loop:
        mouse_pos_main = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_main_loop = False
                break

            # Mouse button down events
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                clicked_something_handled = False

                # 1. Algorithm Menu interactions (if shown)
                if show_algo_menu and not clicked_something_handled:
                    menu_data_main = ui_elements_rects.get('menu', {})
                    menu_area_main = menu_data_main.get('menu_area') # Rect of the visible menu panel
                    
                    if menu_area_main and menu_area_main.collidepoint(mouse_pos_main):
                        # Check close button for menu
                        close_btn_rect = menu_data_main.get('close_button')
                        if close_btn_rect and close_btn_rect.collidepoint(mouse_pos_main):
                            show_algo_menu = False
                            clicked_something_handled = True
                        
                        # Check algorithm selection buttons within menu
                        if not clicked_something_handled:
                            algo_buttons_local_rects = menu_data_main.get('buttons', {}) # {id: local_rect}
                            for algo_id, local_r in algo_buttons_local_rects.items():
                                # Convert local button rect (on menu_surface) to screen coordinates
                                screen_button_r = local_r.move(0, -scroll_y) 
                                if screen_button_r.collidepoint(mouse_pos_main):
                                    if current_selected_algorithm != algo_id:
                                        print(f"Algorithm selected: {algo_id}")
                                        current_selected_algorithm = algo_id
                                        # Reset animation/solve state when algo changes
                                        solution_path_anim = None
                                        current_step_in_anim = 0
                                        is_auto_animating_flag = False
                                        is_solving_flag = False # Stop any ongoing solve (if threaded, needs cancel)
                                        current_state_for_animation = copy.deepcopy(initial_state)
                                    show_algo_menu = False # Close menu after selection
                                    clicked_something_handled = True
                                    break
                        if not clicked_something_handled: # Click was inside menu but not on a button
                             clicked_something_handled = True # Consume click

                # 2. Open Menu button (if menu is hidden)
                if not show_algo_menu and not clicked_something_handled:
                    menu_data_main = ui_elements_rects.get('menu', {})
                    open_btn_rect = menu_data_main.get('open_button')
                    if open_btn_rect and open_btn_rect.collidepoint(mouse_pos_main):
                        show_algo_menu = True
                        scroll_y = 0 # Reset scroll when opening
                        clicked_something_handled = True
                
                # 3. Main control buttons (Solve, Resets)
                if not clicked_something_handled:
                    solve_btn = ui_elements_rects.get('solve_button')
                    if solve_btn and solve_btn.collidepoint(mouse_pos_main):
                        if not is_auto_animating_flag and not is_solving_flag:
                            # --- SOLVE ACTION ---
                            if current_selected_algorithm == 'Backtracking':
                                # Show popup to select sub-algorithm for Backtracking
                                print("Backtracking selected. Opening sub-algorithm selection popup...")
                                backtracking_sub_algo_choice = algorithm_selection_popup()
                                if backtracking_sub_algo_choice:
                                    print(f"Sub-algorithm for Backtracking: {backtracking_sub_algo_choice}. Starting solve...")
                                    is_solving_flag = True # Set flag to trigger solve logic below
                                else:
                                    print("Backtracking sub-algorithm selection cancelled.")
                            elif current_selected_algorithm == 'Sensorless':
                                # Sensorless search uses the initial_state as a template for belief states
                                # Or, if fully defined, it's the single state in initial belief set.
                                # is_valid_state_for_solve isn't strictly for sensorless templates
                                print(f"Starting Sensorless solve... Initial template: {initial_state}")
                                is_solving_flag = True
                            else: # For other algorithms
                                # Ensure initial_state is valid before solving (0-8, no duplicates)
                                # `initial_state` is fixed and valid, so this check is more for robustness
                                if is_valid_state_for_solve(initial_state):
                                    print(f"Starting solve with {current_selected_algorithm}...")
                                    is_solving_flag = True
                                else: # Should not happen with fixed valid initial_state
                                    show_popup("Fixed initial state is somehow invalid!", "Error")
                            
                            if is_solving_flag: # If any solve was initiated
                                solution_path_anim = None
                                current_step_in_anim = 0
                                is_auto_animating_flag = False
                                # current_state_for_animation is reset before animation if solution found
                        clicked_something_handled = True

                    reset_sol_btn = ui_elements_rects.get('reset_solution_button')
                    if not clicked_something_handled and reset_sol_btn and reset_sol_btn.collidepoint(mouse_pos_main):
                        print("Resetting solution state (current display).")
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        clicked_something_handled = True

                    reset_all_btn = ui_elements_rects.get('reset_all_button')
                    if not clicked_something_handled and reset_all_btn and reset_all_btn.collidepoint(mouse_pos_main):
                        print("Resetting all: current display and solve stats.")
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        all_solve_times.clear()
                        last_run_solver_info.clear()
                        clicked_something_handled = True
                    
                    reset_disp_btn = ui_elements_rects.get('reset_initial_button') # Renamed "Reset Display"
                    if not clicked_something_handled and reset_disp_btn and reset_disp_btn.collidepoint(mouse_pos_main):
                        print("Resetting current display to initial state.")
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        clicked_something_handled = True
                
                # Input for initial/goal states is REMOVED as they are fixed.
                # No need to handle clicks on those grids for input.

            # Mouse wheel for scrolling algorithm menu
            if event.type == pygame.MOUSEWHEEL and show_algo_menu:
                menu_data_main = ui_elements_rects.get('menu', {})
                menu_area_main = menu_data_main.get('menu_area')
                if menu_area_main and menu_area_main.collidepoint(mouse_pos_main) and total_menu_height > HEIGHT:
                    scroll_amount_mw = event.y * 35 # event.y is typically 1 or -1
                    max_scroll_val = max(0, total_menu_height - HEIGHT)
                    scroll_y = max(0, min(scroll_y - scroll_amount_mw, max_scroll_val))
            
            # Keydown events (e.g. for number input if it were enabled)
            # Since initial/goal state input is disabled, selected_cell_for_input is not used.
            # No KEYDOWN handling needed for cell input.

        if not running_main_loop: break # Exit if QUIT event occurred

        # --- Algorithm Solving Logic ---
        if is_solving_flag:
            is_solving_flag = False # Consume the flag, actual solve happens here
            solve_start_t = time.time()
            
            found_path_algo = None # For state-sequence algorithms
            found_action_plan_algo = None # For Sensorless (plan of actions)
            actual_start_state_for_anim = None # For Backtracking, this is the S_gen it solved from
            
            error_during_solve = False
            error_message_solve = ""

            try:
                state_to_solve_from = copy.deepcopy(initial_state) # Default for most algos

                if current_selected_algorithm == 'Backtracking':
                    if backtracking_sub_algo_choice: # Ensure sub-algo was chosen
                        found_path_algo, error_message_solve, actual_start_state_for_anim = \
                            backtracking_search(backtracking_sub_algo_choice) # Uses global goal_state
                        if found_path_algo:
                             current_state_for_animation = copy.deepcopy(actual_start_state_for_anim)
                        backtracking_sub_algo_choice = None # Reset choice
                    else: # Should not happen if logic is correct
                        error_message_solve = "Backtracking sub-algorithm not chosen."
                        error_during_solve = True
                
                elif current_selected_algorithm == 'Sensorless':
                    # Sensorless uses initial_state as the template for its initial belief set
                    found_action_plan_algo = sensorless_search(initial_state, time_limit=60)
                    if found_action_plan_algo:
                        # To visualize, execute plan on one sample state (e.g., the initial_state itself if fully defined)
                        sample_belief_states = generate_belief_states(initial_state)
                        vis_start_state = sample_belief_states[0] if sample_belief_states else initial_state
                        found_path_algo = execute_plan(vis_start_state, found_action_plan_algo)
                        current_state_for_animation = copy.deepcopy(vis_start_state)
                
                else: # Standard algorithms
                    current_state_for_animation = copy.deepcopy(state_to_solve_from) # Animation starts from initial_state
                    
                    algo_func_map = {
                        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 
                        'Greedy': greedy, 'IDA*': ida_star,
                        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
                        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
                        'Beam Search': beam_search, 'AND-OR': and_or_search
                    }
                    selected_algo_function = algo_func_map.get(current_selected_algorithm)

                    if selected_algo_function:
                        # Prepare arguments, especially time_limit if accepted
                        algo_args_list = [state_to_solve_from]
                        func_varnames = selected_algo_function.__code__.co_varnames[:selected_algo_function.__code__.co_argcount]
                        
                        default_time_limit = 30
                        if current_selected_algorithm in ['IDA*', 'Sensorless', 'Backtracking']: # Longer defaults for these
                            default_time_limit = 60
                        
                        if 'time_limit' in func_varnames:
                            algo_args_list.append(default_time_limit)
                        # Special case for DFS's max_depth if not taking time_limit
                        elif current_selected_algorithm == 'DFS' and 'max_depth' in func_varnames and 'time_limit' not in func_varnames:
                             algo_args_list.append(30) # Default max_depth for DFS

                        found_path_algo = selected_algo_function(*algo_args_list)
                    else:
                        error_message_solve = f"Algorithm '{current_selected_algorithm}' not found."
                        error_during_solve = True
            
            except Exception as e:
                error_message_solve = f"Error during {current_selected_algorithm} solve:\n{str(e)}"
                print(error_message_solve)
                traceback.print_exc()
                error_during_solve = True

            # --- Post-solve processing ---
            solve_duration_t = time.time() - solve_start_t
            all_solve_times[current_selected_algorithm] = solve_duration_t
            
            if error_during_solve:
                show_popup(error_message_solve if error_message_solve else "An unknown error occurred during solve.", "Solver Error")
            else:
                if found_path_algo and len(found_path_algo) > 0:
                    solution_path_anim = found_path_algo
                    current_step_in_anim = 0
                    is_auto_animating_flag = True
                    last_anim_step_time = time.time() # Start animation immediately

                    final_state_of_path = solution_path_anim[-1]
                    is_actually_goal_state = is_goal(final_state_of_path)
                    
                    num_steps_or_actions = len(solution_path_anim) -1
                    if current_selected_algorithm == 'Sensorless':
                        num_steps_or_actions = len(found_action_plan_algo) if found_action_plan_algo else 0
                        last_run_solver_info[f"{current_selected_algorithm}_actions"] = num_steps_or_actions
                        popup_msg = f"Sensorless plan found!\n{num_steps_or_actions} actions.\nTime: {solve_duration_t:.3f}s\n(Visualizing on one sample state)"
                    else:
                        last_run_solver_info[f"{current_selected_algorithm}_steps"] = num_steps_or_actions
                        popup_msg = f"{current_selected_algorithm} "
                        if is_actually_goal_state:
                            popup_msg += f"found solution!\n{num_steps_or_actions} steps."
                        else: # For algos like Hill Climbing that might not reach goal
                            popup_msg += f"finished.\n{num_steps_or_actions} steps (Not Goal)."
                        popup_msg += f"\nTime: {solve_duration_t:.3f}s"
                    
                    last_run_solver_info[f"{current_selected_algorithm}_reached_goal"] = is_actually_goal_state
                    show_popup(popup_msg, "Solve Complete" if is_actually_goal_state else "Search Finished")
                
                else: # No solution path found (or plan for sensorless)
                    last_run_solver_info[f"{current_selected_algorithm}_reached_goal"] = False
                    if f"{current_selected_algorithm}_steps" in last_run_solver_info:
                        del last_run_solver_info[f"{current_selected_algorithm}_steps"]
                    if f"{current_selected_algorithm}_actions" in last_run_solver_info:
                        del last_run_solver_info[f"{current_selected_algorithm}_actions"]

                    no_solution_msg = error_message_solve if error_message_solve else f"No solution found by {current_selected_algorithm}."
                    # Check for timeout (approximate)
                    used_time_limit = default_time_limit # Re-fetch default_time_limit used
                    if current_selected_algorithm in ['IDA*', 'Sensorless', 'Backtracking']: used_time_limit = 60
                    if solve_duration_t >= used_time_limit * 0.95 : # If close to time limit
                         no_solution_msg = f"{current_selected_algorithm} timed out after ~{used_time_limit}s."
                    
                    show_popup(no_solution_msg, "No Solution / Timeout")

        # --- Animation Step Logic ---
        if is_auto_animating_flag and solution_path_anim:
            current_time_anim = time.time()
            # Adjust animation delay based on path length
            anim_delay_val = 0.3 
            if len(solution_path_anim) > 30: anim_delay_val = 0.15
            if len(solution_path_anim) > 60: anim_delay_val = 0.08

            if current_time_anim - last_anim_step_time >= anim_delay_val:
                if current_step_in_anim < len(solution_path_anim) - 1:
                    current_step_in_anim += 1
                    current_state_for_animation = copy.deepcopy(solution_path_anim[current_step_in_anim])
                    last_anim_step_time = current_time_anim
                else: # Animation finished
                    is_auto_animating_flag = False
                    final_anim_state_is_goal = is_goal(current_state_for_animation)
                    print(f"Animation complete: {'Goal reached!' if final_anim_state_is_goal else 'Final state reached (Not Goal).'}")
        
        # --- Drawing UI ---
        belief_size_for_display = None
        if current_selected_algorithm == 'Sensorless':
            # For display, show size of current belief set if available, or initial one
            # This part is tricky as `current_belief_set` is not directly tracked in main for display.
            # For now, we can show the size of the *initial* belief set generated from `initial_state`.
            initial_bs_for_sensorless = generate_belief_states(initial_state)
            belief_size_for_display = len(initial_bs_for_sensorless) if initial_bs_for_sensorless else 0
            
        ui_elements_rects = draw_grid_and_ui(current_state_for_animation, show_algo_menu, 
                                             current_selected_algorithm, all_solve_times, 
                                             last_run_solver_info, belief_size_for_display)
        
        game_clock.tick(60) # Cap FPS

    pygame.quit()

if __name__ == "__main__":
    main()