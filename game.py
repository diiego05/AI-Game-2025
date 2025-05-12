import pygame
import sys
from collections import deque
import copy
import time
from queue import PriorityQueue
import traceback
import math
import random
import itertools # Needed for permutations in sensorless search

# --- Pygame Initialization and Constants ---
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
BLUE = (30, 144, 255) # Color for standard tiles
GREEN = (0, 255, 0)   # Color for goal state tiles / solved state
RED = (255, 0, 0)     # Color for errors, solve button, selected cell highlight
FONT = pygame.font.SysFont('Arial', 60) # Font for tile numbers
BUTTON_FONT = pygame.font.SysFont('Arial', 24) # Font for button text
INFO_FONT = pygame.font.SysFont('Arial', 18)   # Font for info panel text
TITLE_FONT = pygame.font.SysFont('Arial', 26, bold=True) # Font for grid titles and info panel title
MENU_COLOR = (50, 50, 50)           # Background color for the algorithm menu
MENU_BUTTON_COLOR = (70, 70, 70)    # Default color for menu buttons
MENU_HOVER_COLOR = (90, 90, 90)     # Color for menu buttons on hover
MENU_SELECTED_COLOR = pygame.Color('dodgerblue') # Color for the selected algorithm button
INFO_COLOR = (50, 50, 150)          # Border/accent color for popups and info panel
INFO_BG = (245, 245, 245)           # Background color for popups and info panel
POPUP_WIDTH = 600                   # Width of popup windows
POPUP_HEIGHT = 400                  # Height of popup windows

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LamVanDi-23110191 - 8 Puzzle Solver")

# --- Default Fixed States ---
# _FIXED_INITIAL_STATE_FLAT = [2,6,5,0,8,7,4,3,1] # A moderately hard state
_FIXED_INITIAL_STATE_FLAT = [1, 2, 3, 4, 0, 6, 7, 5, 8] # A simpler state
_FIXED_GOAL_STATE_FLAT = [1,2,3,4,5,6,7,8,0]

initial_state_fixed_global = [[_FIXED_INITIAL_STATE_FLAT[i*GRID_SIZE+j] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
goal_state_fixed_global = [[_FIXED_GOAL_STATE_FLAT[i*GRID_SIZE+j] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]

# --- Mutable Global States ---
# These are the states the user interacts with and the algorithms use
initial_state = copy.deepcopy(initial_state_fixed_global)
goal_state = copy.deepcopy(goal_state_fixed_global) # Goal is usually fixed, but mutable just in case

# --- Core Helper Functions ---

def find_empty(state):
    """Finds the row and column of the empty tile (0)."""
    if not isinstance(state, list) or len(state) != GRID_SIZE: return -1, -1
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE: return -1, -1
        for j in range(GRID_SIZE):
            if state[i][j] == 0:
                return i, j
    return -1, -1 # Should not happen for a valid state with a 0

def is_goal(state):
    """Checks if the given state matches the global goal state."""
    # Basic structure check
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            return False
    # Direct comparison with the global goal state
    return state == goal_state

def get_neighbors(state):
    """Generates valid neighbor states by moving the empty tile."""
    neighbors = []
    empty_i, empty_j = find_empty(state)
    if empty_i == -1: # No empty tile found (invalid state)
        # print("Warning: get_neighbors called on state with no empty tile.")
        return []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right relative to empty tile
    for di, dj in directions:
        new_i, new_j = empty_i + di, empty_j + dj # Position of tile to swap with empty
        # Check if the new position is within the grid boundaries
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
            # Create a deep copy to avoid modifying the original state
            new_state = copy.deepcopy(state)
            # Swap the empty tile with the adjacent tile
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    return neighbors

def state_to_tuple(state):
    """Converts a state (list of lists) to a tuple of tuples for hashing."""
    if not isinstance(state, list):
        # print(f"Warning: state_to_tuple received non-list type: {type(state)}")
        return None
    try:
        # Ensure all rows are lists before converting inner tuples
        if not all(isinstance(row, list) for row in state):
             # print(f"Warning: state_to_tuple received list with non-list rows: {state}")
             return None
        return tuple(tuple(row) for row in state)
    except TypeError as e:
        # print(f"Warning: TypeError during state_to_tuple: {e}, State: {state}")
        return None

def tuple_to_list(state_tuple):
    """Converts a state tuple back to a list of lists."""
    if not isinstance(state_tuple, tuple):
        # print(f"Warning: tuple_to_list received non-tuple type: {type(state_tuple)}")
        return None
    try:
        # Ensure all inner elements are tuples before converting to lists
        if not all(isinstance(row_tuple, tuple) for row_tuple in state_tuple):
            # print(f"Warning: tuple_to_list received tuple with non-tuple rows: {state_tuple}")
            return None
        return [list(row) for row in state_tuple]
    except TypeError as e:
        # print(f"Warning: TypeError during tuple_to_list: {e}, Tuple: {state_tuple}")
        return None

def apply_action_to_state(state_list, action):
    """
    Applies an action ('Up', 'Down', 'Left', 'Right') to a state list by moving the blank space.
    Returns a new state list or the original list if the move is invalid.
    Note: Action names refer to the direction the *blank space* moves.
    """
    if state_list is None:
        return None
    new_state = copy.deepcopy(state_list)
    empty_i, empty_j = find_empty(new_state)
    if empty_i == -1:
        # print("Warning: apply_action called on state with no empty tile.")
        return new_state # Return copy if no empty tile

    # Determine the coordinates of the tile to swap WITH the blank space
    swap_i, swap_j = empty_i, empty_j
    if action == 'Up': swap_i += 1 # Tile BELOW the blank moves Up
    elif action == 'Down': swap_i -= 1 # Tile ABOVE the blank moves Down
    elif action == 'Left': swap_j += 1 # Tile RIGHT of the blank moves Left
    elif action == 'Right': swap_j -= 1 # Tile LEFT of the blank moves Right
    else:
        # print(f"Warning: Invalid action '{action}' in apply_action_to_state.")
        return new_state # Return copy if action is invalid

    # Check if the tile to be swapped is within bounds
    if 0 <= swap_i < GRID_SIZE and 0 <= swap_j < GRID_SIZE:
        # Swap the blank space and the adjacent tile
        new_state[empty_i][empty_j], new_state[swap_i][swap_j] = new_state[swap_i][swap_j], new_state[empty_i][empty_j]
        return new_state
    else:
        # Action is invalid (tried to move blank off-board), return original state (copy)
        return new_state

def manhattan_distance(state):
    """Calculates the Manhattan distance heuristic for a given state."""
    distance = 0
    goal_pos = {}
    # Pre-calculate goal positions for faster lookup
    for r_goal in range(GRID_SIZE):
        for c_goal in range(GRID_SIZE):
            val = goal_state[r_goal][c_goal]
            if val != 0 : # Don't include the blank tile in distance calculation
                goal_pos[val] = (r_goal, c_goal)

    # Calculate distance for the current state
    if not isinstance(state, list): return float('inf') # Handle invalid input
    for r_curr in range(GRID_SIZE):
        if not isinstance(state[r_curr], list) or len(state[r_curr]) != GRID_SIZE: return float('inf')
        for c_curr in range(GRID_SIZE):
            tile = state[r_curr][c_curr]
            # Check if tile is valid (not None, not 0, and exists in goal)
            if tile is not None and tile != 0 and tile in goal_pos:
                goal_r, goal_c = goal_pos[tile]
                distance += abs(r_curr - goal_r) + abs(c_curr - goal_c)
            # Handle None? For standard A*, None shouldn't occur.
            # For Sensorless heuristic (if used), might need adjustment.
    return distance

def is_valid_state_for_solve(state_to_check):
    """Checks if a state is valid for standard (non-sensorless) algorithms."""
    flat_state = []
    try:
        # Basic structure check
        if not isinstance(state_to_check, list) or len(state_to_check) != GRID_SIZE: return False
        for row in state_to_check:
            if not isinstance(row, list) or len(row) != GRID_SIZE: return False
            # Check elements in the row
            for tile in row:
                if tile is None: return False # Standard algorithms don't accept None
                if not isinstance(tile, int): return False # Must be integer
                flat_state.append(tile)
    except TypeError:
        return False # Error during iteration

    # Check total number of elements
    if len(flat_state) != GRID_SIZE * GRID_SIZE:
        return False

    # Check if all numbers from 0 to N*N-1 are present exactly once
    expected_numbers = set(range(GRID_SIZE * GRID_SIZE))
    seen_numbers = set(flat_state)

    return seen_numbers == expected_numbers

def get_inversions(flat_state_no_zero):
    """Calculates the number of inversions in a flattened state (excluding 0)."""
    inversions = 0
    n = len(flat_state_no_zero)
    for i in range(n):
        for j in range(i + 1, n):
            # If a tile appears before a smaller tile (and neither is 0)
            if flat_state_no_zero[i] > flat_state_no_zero[j]:
                inversions += 1
    return inversions

def is_solvable(state_to_check):
    """
    Checks if a given 8-puzzle state is solvable relative to the fixed global goal.
    Returns False if the state contains None or is structurally invalid.
    """
    # 1. Basic validation (must be 3x3 list of lists of integers 0-8)
    if not isinstance(state_to_check, list) or len(state_to_check) != GRID_SIZE: return False
    flat_state = []
    has_blank = False
    for r_idx, r_val in enumerate(state_to_check):
        if not isinstance(r_val, list) or len(r_val) != GRID_SIZE: return False
        for c_idx, c_val in enumerate(r_val):
            if c_val is None: return False # Cannot determine solvability with unknowns
            if not isinstance(c_val, int) or c_val < 0 or c_val >= GRID_SIZE*GRID_SIZE:
                 return False # Must be integer within range
            flat_state.append(c_val)
            if c_val == 0: has_blank = True

    # 2. Check size and presence of blank tile (0)
    if len(flat_state) != GRID_SIZE*GRID_SIZE: return False
    if not has_blank: return False # Must contain the blank tile

    # 3. Check for duplicate numbers
    if len(set(flat_state)) != GRID_SIZE*GRID_SIZE: return False # Must have unique numbers 0-8

    # --- Solvability Calculation ---
    # Create flat list without the blank tile (0)
    state_flat_no_zero = [tile for tile in flat_state if tile != 0]
    inversions_state = get_inversions(state_flat_no_zero)

    # Calculate inversions for the fixed global goal state
    goal_flat = []
    for r_goal in goal_state_fixed_global: goal_flat.extend(r_goal)
    goal_flat_no_zero = [tile for tile in goal_flat if tile != 0]
    inversions_goal = get_inversions(goal_flat_no_zero)

    # Solvability rule for 3x3 grid (GRID_SIZE is odd)
    # The state is solvable iff the parity of its inversions matches the goal's.
    if GRID_SIZE % 2 == 1:
        return (inversions_state % 2) == (inversions_goal % 2)
    else:
        # Solvability rule for even width grids (like 4x4) - requires blank row position
        # Find blank row (from bottom, 1-indexed) for the state
        blank_row_state = -1
        for r_idx, row_s in enumerate(state_to_check):
            if 0 in row_s:
                blank_row_state = GRID_SIZE - r_idx # Row from bottom (1 to GRID_SIZE)
                break
        # Find blank row (from bottom, 1-indexed) for the goal state
        blank_row_goal = -1
        for r_idx_g, row_g in enumerate(goal_state_fixed_global):
            if 0 in row_g:
                blank_row_goal = GRID_SIZE - r_idx_g # Row from bottom
                break

        # This should not happen if validation passed, but safety check
        if blank_row_state == -1 or blank_row_goal == -1: return False

        # For even grid size: Solvable if ((inversions + blank_row) % 2) is same for both
        return ((inversions_state + blank_row_state) % 2) == \
               ((inversions_goal + blank_row_goal) % 2)

# --- Search Algorithms ---

def bfs(start_node_state, time_limit=30):
    """Breadth-First Search"""
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None # Invalid start state

    queue = deque([(start_node_state, [start_node_state])]) # Store (state, path_list)
    visited = {init_tuple} # Store visited state tuples

    while queue:
        if time.time() - start_time > time_limit:
            # print(f"BFS Timeout ({time_limit}s)")
            return None # Timeout
        current_s, path = queue.popleft()

        if is_goal(current_s):
            return path # Goal found

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor_s, path + [neighbor_s])) # Add neighbor and extended path
    return None # No solution found

def dfs(start_node_state, max_depth=30, time_limit=30):
    """Depth-First Search with depth limit and visited optimization."""
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    # Stack stores (state, path_list, depth)
    stack = [(start_node_state, [start_node_state], 0)]
    # Visited optimization: store visited state tuple and the minimum depth reached
    visited = {} # {state_tuple: min_depth}

    while stack:
        if time.time() - start_time > time_limit:
            # print(f"DFS Timeout ({time_limit}s)")
            return None # Timeout

        current_s, path, depth = stack.pop()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue # Should not happen if neighbors are valid

        # Pruning based on visited depth
        if current_tuple in visited and visited[current_tuple] <= depth:
            continue # Already found a path to this state at an equal or shallower depth

        visited[current_tuple] = depth # Update visited depth

        if is_goal(current_s):
            return path # Goal found

        # Depth limit check
        if depth >= max_depth:
            continue # Stop exploring this branch

        # Explore neighbors (reversed order for typical DFS stack behavior)
        neighbors = get_neighbors(current_s)
        for neighbor_s in reversed(neighbors):
            # No need to check visited here again, pruning check handles it
            stack.append((neighbor_s, path + [neighbor_s], depth + 1))

    return None # No solution found

def ids(start_node_state, max_depth_limit=30, time_limit=60): # Increased default time limit for IDS
    """Iterative Deepening Search"""
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    for depth_limit in range(max_depth_limit + 1):
        # Check global time limit before starting iteration
        if time.time() - start_time_global > time_limit:
            # print(f"IDS Global Timeout ({time_limit}s) before depth {depth_limit}")
            return None

        # Stack for DLS: (state, path_list, depth)
        stack = [(start_node_state, [start_node_state], 0)]
        # Visited set *within* the current DLS iteration to prevent cycles in this search
        visited_in_iteration = {init_tuple: 0} # {state_tuple: depth}

        while stack:
            # Check global time limit inside the loop
            if time.time() - start_time_global > time_limit:
                # print(f"IDS Global Timeout ({time_limit}s) during depth {depth_limit}")
                return None

            current_s, path, depth = stack.pop()

            if is_goal(current_s):
                return path # Goal found

            # Only explore neighbors if depth is less than the current limit
            if depth < depth_limit:
                neighbors = get_neighbors(current_s)
                for neighbor_s in reversed(neighbors): # Reverse for DFS stack order
                    neighbor_tuple = state_to_tuple(neighbor_s)
                    if neighbor_tuple is not None:
                        # If neighbor not visited in this iteration OR found at a shallower depth
                        if neighbor_tuple not in visited_in_iteration or visited_in_iteration[neighbor_tuple] > depth + 1:
                            visited_in_iteration[neighbor_tuple] = depth + 1
                            stack.append((neighbor_s, path + [neighbor_s], depth + 1))

    # print(f"IDS Failed after max depth {max_depth_limit}")
    return None # No solution found within max depth limit

def ucs(start_node_state, time_limit=30):
    """Uniform Cost Search"""
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    # Priority Queue stores: (cost, state, path_list)
    # Use path length as cost for 8-puzzle (each move costs 1)
    frontier = PriorityQueue()
    frontier.put((0, start_node_state, [start_node_state]))

    # Visited stores the minimum cost found so far to reach a state
    visited = {init_tuple: 0} # {state_tuple: min_cost}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            # print(f"UCS Timeout ({time_limit}s)")
            return None # Timeout

        cost, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        # Optimization: If we found a shorter path to this node already, skip
        if cost > visited.get(current_tuple, float('inf')):
             continue

        if is_goal(current_s):
            return path # Goal found

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            new_cost = cost + 1 # Cost of each step is 1

            # If neighbor is unvisited or we found a cheaper path to it
            if new_cost < visited.get(neighbor_tuple, float('inf')):
                visited[neighbor_tuple] = new_cost
                frontier.put((new_cost, neighbor_s, path + [neighbor_s]))
    return None # No solution found

def astar(start_node_state, time_limit=30):
    """A* Search using Manhattan Distance Heuristic"""
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    g_init = 0
    h_init = manhattan_distance(start_node_state)
    if h_init == float('inf'): return None # Invalid start state for heuristic
    f_init = g_init + h_init

    # Priority Queue stores: (f_score, g_score, state, path_list)
    frontier = PriorityQueue()
    frontier.put((f_init, g_init, start_node_state, [start_node_state]))

    # visited_g_scores stores the minimum g_score found so far for a state
    visited_g_scores = {init_tuple: g_init} # {state_tuple: min_g_score}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            # print(f"A* Timeout ({time_limit}s)")
            return None # Timeout

        f_score_curr, g_score_curr, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        # Optimization: If we already found a path with a lower or equal g_score, skip
        if g_score_curr > visited_g_scores.get(current_tuple, float('inf')):
            continue

        if is_goal(current_s):
            return path # Goal found

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            tentative_g_score = g_score_curr + 1 # Cost of move is 1

            # If this path to the neighbor is better than any previous path found
            if tentative_g_score < visited_g_scores.get(neighbor_tuple, float('inf')):
                visited_g_scores[neighbor_tuple] = tentative_g_score # Update the best g_score
                h_neighbor = manhattan_distance(neighbor_s)
                if h_neighbor == float('inf'): continue # Skip if neighbor invalid for heuristic
                f_neighbor = tentative_g_score + h_neighbor
                frontier.put((f_neighbor, tentative_g_score, neighbor_s, path + [neighbor_s]))
    return None # No solution found

def greedy(start_node_state, time_limit=30):
    """Greedy Best-First Search using Manhattan Distance"""
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    # Priority Queue stores: (heuristic_value, state, path_list)
    frontier = PriorityQueue()
    h_init = manhattan_distance(start_node_state)
    if h_init == float('inf'): return None # Invalid start state
    frontier.put((h_init, start_node_state, [start_node_state]))

    # Visited set to prevent cycles and redundant exploration
    visited = {init_tuple} # {state_tuple}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            # print(f"Greedy Timeout ({time_limit}s)")
            return None # Timeout

        h_val, current_s, path = frontier.get()

        if is_goal(current_s):
            return path # Goal found (might not be optimal)

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            # If neighbor is valid and not visited yet
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                h_neighbor = manhattan_distance(neighbor_s)
                if h_neighbor == float('inf'): continue # Skip invalid neighbor
                frontier.put((h_neighbor, neighbor_s, path + [neighbor_s]))

    return None # No solution found

# --- Corrected IDA* Implementation ---
def _search_ida_recursive(path_ida, g_score, threshold, start_time_ida, time_limit_ida):
    """
    Recursive helper for IDA*. Uses path-based cycle detection.
    Returns: (result_path or "Timeout" or None, next_threshold_candidate)
    """
    if time.time() - start_time_ida >= time_limit_ida:
        return "Timeout", float('inf') # Indicate timeout

    current_s_ida = path_ida[-1]

    h_ida = manhattan_distance(current_s_ida)
    if h_ida == float('inf'): # Handle invalid state during recursion
         return None, float('inf')
    f_score_ida = g_score + h_ida

    # Check if the current node's f-score exceeds the threshold
    if f_score_ida > threshold:
        return None, f_score_ida # Return f-score as candidate for next threshold

    # Check if the current state is the goal
    if is_goal(current_s_ida):
        return path_ida[:], threshold # Return copy of path

    min_new_threshold = float('inf') # Track min f-score > threshold

    # Create set of tuples in the current path for cycle checks
    # Handle potential None tuples if state conversion fails (shouldn't happen with valid states)
    current_path_tuples = {state_to_tuple(s) for s in path_ida if state_to_tuple(s) is not None}

    # Explore neighbors
    for neighbor_s_ida in get_neighbors(current_s_ida):
        neighbor_tuple_ida = state_to_tuple(neighbor_s_ida)
        if neighbor_tuple_ida is None: continue

        # --- Path-based Cycle Check ---
        if neighbor_tuple_ida in current_path_tuples:
            continue
        # --- End Cycle Check ---

        new_g_ida = g_score + 1

        path_ida.append(neighbor_s_ida) # Append neighbor before recursing

        result_ida, recursive_threshold_ida = _search_ida_recursive(
            path_ida, new_g_ida, threshold, start_time_ida, time_limit_ida
        )

        path_ida.pop() # Remove neighbor after returning (backtrack)

        # Handle results
        if result_ida == "Timeout": return "Timeout", float('inf')
        if result_ida is not None: return result_ida, threshold # Solution found

        min_new_threshold = min(min_new_threshold, recursive_threshold_ida)

    return None, min_new_threshold # No solution in this branch

def ida_star(start_node_state, time_limit=60):
    """ Iterative Deepening A* Search. """
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    initial_h = manhattan_distance(start_node_state)
    if initial_h == float('inf'): return None # Cannot start if heuristic invalid
    threshold = initial_h

    while True:
        if time.time() - start_time_global >= time_limit:
            # print(f"IDA* Global Timeout ({time_limit}s reached)")
            return None

        current_path = [start_node_state] # Start path for this iteration

        result, new_threshold_candidate = _search_ida_recursive(
            current_path, 0, threshold, start_time_global, time_limit
        )

        if result == "Timeout": return None
        if result is not None: return result # Solution found

        if new_threshold_candidate == float('inf'):
            # print(f"IDA* Exhausted search space")
            return None # Search space exhausted

        # Pruning edge case: if threshold doesn't increase, abort
        if new_threshold_candidate <= threshold:
             # print(f"IDA* Threshold did not increase ({threshold} -> {new_threshold_candidate}), likely stuck. Aborting.")
             return None

        threshold = new_threshold_candidate
        # print(f"IDA* Increasing threshold to: {threshold}") # Optional debug
# --- End Corrected IDA* ---


# --- Hill Climbing Variants ---
def simple_hill_climbing(start_node_state, time_limit=30): # First-choice Hill Climbing
    """ Simple Hill Climbing (takes first better neighbor). """
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path # Start state invalid

    while True:
        if time.time() - start_time > time_limit:
            # print("Simple Hill Climbing Timeout")
            return path # Return path found so far

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        moved = False
        # Optional: Shuffle neighbors for stochastic first choice
        # random.shuffle(neighbors)
        for neighbor_s in neighbors:
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor == float('inf'): continue # Skip invalid neighbor
            if h_neighbor < current_h: # Found a better neighbor
                current_s = neighbor_s
                current_h = h_neighbor
                path.append(current_s)
                moved = True
                break # Take the first better neighbor found

        if not moved: # No better neighbor found, reached local optimum/plateau
            # print("Simple Hill Climbing: Reached local optimum/plateau.")
            return path

    return path # Should be unreachable for valid puzzle

def steepest_hill_climbing(start_node_state, time_limit=30): # Steepest Ascent
    """ Steepest Ascent Hill Climbing (takes the best neighbor). """
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path # Invalid start

    while True:
        if time.time() - start_time > time_limit:
            # print("Steepest Hill Climbing Timeout")
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        best_next_s = None
        best_next_h = current_h # Initialize with current heuristic value

        # Find the neighbor with the lowest heuristic value
        for neighbor_s in neighbors:
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor == float('inf'): continue # Skip invalid
            if h_neighbor < best_next_h: # Found a neighbor strictly better
                best_next_h = h_neighbor
                best_next_s = neighbor_s

        # If no neighbor has a strictly lower heuristic value
        if best_next_s is None: # Note: >= check was removed to match standard steepest ascent
            # print("Steepest Hill Climbing: Reached local optimum/plateau.")
            return path

        # Move to the best neighbor found
        current_s = best_next_s
        current_h = best_next_h
        path.append(current_s)

    return path

def random_hill_climbing(start_node_state, time_limit=30, max_iter_no_improve=500): # Stochastic Hill Climbing
    """ Stochastic Hill Climbing (randomly picks a neighbor, moves if better/equal). """
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path # Invalid start
    iter_no_improve = 0

    while True:
        if time.time() - start_time > time_limit:
            # print("Stochastic Hill Climbing Timeout")
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        # Randomly select one neighbor to evaluate
        random_neighbor = random.choice(neighbors)
        neighbor_h = manhattan_distance(random_neighbor)
        if neighbor_h == float('inf'): continue # Skip if random choice is invalid

        # Move to the neighbor if it's better or equal
        if neighbor_h <= current_h:
            if neighbor_h < current_h:
                iter_no_improve = 0 # Reset counter if improved
            else:
                iter_no_improve += 1 # Increment if moved sideways

            current_s = random_neighbor
            current_h = neighbor_h
            path.append(current_s)
        else: # Neighbor is worse
            iter_no_improve += 1

        # Stop if no improvement for a while (stuck on plateau or optimum)
        if iter_no_improve >= max_iter_no_improve:
            # print(f"Stochastic Hill Climbing: Stopped after {max_iter_no_improve} iterations without improvement.")
            return path
    return path

def simulated_annealing(start_node_state, initial_temp=1000, cooling_rate=0.99, min_temp=0.1, time_limit=30):
    """ Simulated Annealing algorithm. """
    start_time = time.time()
    current_s = start_node_state
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return [start_node_state] # Return path with invalid start
    path = [current_s] # Path stores the sequence of *accepted* states

    best_s_so_far = current_s # Track the best state encountered globally
    best_h_so_far = current_h

    temp = initial_temp

    while temp > min_temp:
        if time.time() - start_time > time_limit:
            # print("Simulated Annealing Timeout")
            return path # Return the actual path taken up to timeout

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break # Stop if no neighbors

        # Randomly select a neighbor
        next_s = random.choice(neighbors)
        next_h = manhattan_distance(next_s)
        if next_h == float('inf'): continue # Skip if neighbor invalid

        delta_h = next_h - current_h # Difference in heuristic (cost)

        # Decide whether to move to the neighbor
        if delta_h < 0: # Neighbor is better, always move
            current_s = next_s
            current_h = next_h
            path.append(current_s) # Add accepted state to path
            # Update best found so far if this is the new best
            if current_h < best_h_so_far:
                best_s_so_far = current_s
                best_h_so_far = current_h
        else: # Neighbor is worse or equal
            # Accept worse move with a probability based on temperature
            if temp > 0: # Avoid division by zero
                acceptance_prob = math.exp(-delta_h / temp)
                if random.random() < acceptance_prob:
                    current_s = next_s
                    current_h = next_h
                    path.append(current_s) # Add accepted state to path

        # Cool down the temperature
        temp *= cooling_rate

    # print(f"Simulated Annealing Finished. Temp: {temp:.4f}, Best H: {best_h_so_far}")
    return path # Return the path taken

# --- Beam Search ---
def beam_search(start_node_state, beam_width=5, time_limit=30):
    """ Beam Search algorithm. """
    start_time = time.time()
    if is_goal(start_node_state): return [start_node_state]

    initial_h = manhattan_distance(start_node_state)
    if initial_h == float('inf'): return None # Cannot start

    # Beam stores tuples: (state_list, path_list_of_states, heuristic_value)
    beam = [(start_node_state, [start_node_state], initial_h)]
    # visited_tuples_global prevents revisiting states across beam expansions
    visited_tuples_global = {state_to_tuple(start_node_state)}
    best_goal_path_found = None # Store the best goal path found so far

    max_iterations = 100 # Limit beam search depth/iterations

    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            # print(f"Beam Search Timeout ({time_limit}s)")
            return best_goal_path_found # Return best goal found, or None

        # Stores candidates for the next beam: (state_list, path_list_of_states, heuristic_value)
        next_level_candidates = []

        # Expand nodes currently in the beam
        for s_beam, path_beam, _ in beam:
            for neighbor_s_beam in get_neighbors(s_beam):
                neighbor_tuple_beam = state_to_tuple(neighbor_s_beam)
                if neighbor_tuple_beam is None: continue

                # Check if visited globally before adding to candidates
                if neighbor_tuple_beam not in visited_tuples_global:
                    new_path_beam = path_beam + [neighbor_s_beam]

                    # Check if this neighbor is the goal
                    if is_goal(neighbor_s_beam):
                        if best_goal_path_found is None or len(new_path_beam) < len(best_goal_path_found):
                            best_goal_path_found = new_path_beam
                        # Continue search within beam width

                    h_neighbor_beam = manhattan_distance(neighbor_s_beam)
                    if h_neighbor_beam == float('inf'): continue # Skip invalid

                    next_level_candidates.append((neighbor_s_beam, new_path_beam, h_neighbor_beam))
                    # Mark as visited *when adding* to candidates
                    visited_tuples_global.add(neighbor_tuple_beam)

        if not next_level_candidates:
            # print("Beam Search: No candidates generated.")
            break # No more states to expand

        # Sort candidates based on heuristic value (lower is better)
        next_level_candidates.sort(key=lambda x: x[2])

        # Select the top 'beam_width' candidates for the next beam
        beam = next_level_candidates[:beam_width]

        if not beam:
            # print("Beam Search: Beam became empty.")
            break # Beam is empty, search terminates

    # print(f"Beam Search Finished. Best Goal Path Length: {len(best_goal_path_found) if best_goal_path_found else 'None'}")
    return best_goal_path_found

# --- AND-OR Search (Simplified DFS interpretation for 8-puzzle) ---
def _and_or_recursive(state_ao, path_ao, visited_ao_tuples, start_time_ao, time_limit_ao, depth_ao, max_depth_ao=50):
    """ Recursive helper for AND-OR (DFS interpretation) """
    state_tuple_ao = state_to_tuple(state_ao)
    if state_tuple_ao is None: return "Fail", None # Invalid state

    if time.time() - start_time_ao > time_limit_ao: return "Timeout", None
    if depth_ao > max_depth_ao: return "Fail", None # Depth limit reached
    if is_goal(state_ao): return "Solved", path_ao[:] # Return copy

    # Cycle detection for current path
    if state_tuple_ao in visited_ao_tuples: return "Fail", None

    visited_ao_tuples.add(state_tuple_ao) # Mark visited for this path

    # Explore neighbors (OR branches)
    for neighbor_ao in get_neighbors(state_ao):
        # Pass a copy of visited set for independent branch exploration
        status_ao, solution_path_ao = _and_or_recursive(
            neighbor_ao, path_ao + [neighbor_ao],
            visited_ao_tuples.copy(), # Pass copy
            start_time_ao, time_limit_ao, depth_ao + 1, max_depth_ao
        )
        if status_ao == "Timeout": return "Timeout", None
        if status_ao == "Solved": return "Solved", solution_path_ao # Return first solution found

    # If none of the OR branches led to a solution
    # visited_ao_tuples.remove(state_tuple_ao) # No need to remove due to using copy
    return "Fail", None

def and_or_search(start_node_state, time_limit=30, max_depth=50):
    """ AND-OR Search (interpreted as DFS for 8-puzzle pathfinding). """
    start_time = time.time()
    initial_visited = set() # Visited set for the initial call
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    status, solution_path = _and_or_recursive(
        start_node_state, [start_node_state], # Initial path
        initial_visited, start_time, time_limit, 0, max_depth # Initial depth 0
    )

    if status == "Solved": return solution_path
    # else: print(f"AND-OR Search Failed (Status: {status})")
    return None
# --- End AND-OR Search ---

# --- Backtracking (Meta-Algorithm) ---
def backtracking_search(selected_sub_algorithm_name, max_attempts=50, time_limit_overall=60):
    """ Tries to find a solvable start state and solve it using a chosen sub-algorithm. """
    start_time_overall = time.time()

    algo_map = {
        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 'Greedy': greedy,
        'IDA*': ida_star,
        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
        'Beam Search': beam_search, 'AND-OR': and_or_search,
        # Allow Sensorless/Unknown as sub-algorithm
        'Sensorless': sensorless_search,
        'Unknown': sensorless_search,
    }
    sub_algo_func = algo_map.get(selected_sub_algorithm_name)
    if not sub_algo_func:
        # print(f"Error: Backtracking sub-algorithm '{selected_sub_algorithm_name}' not found.")
        return None, f"Sub-algorithm '{selected_sub_algorithm_name}' not supported.", None, None

    # print(f"Backtracking: Starting search for a solvable state for '{selected_sub_algorithm_name}'...")
    time_limit_per_sub_solve = max(1.0, time_limit_overall / max_attempts if max_attempts > 0 else time_limit_overall / 5)
    time_limit_per_sub_solve = min(time_limit_per_sub_solve, 15 if selected_sub_algorithm_name not in ['IDA*', 'Sensorless', 'Unknown', 'IDS'] else 30)

    for attempts in range(max_attempts):
        if time.time() - start_time_overall > time_limit_overall:
            return None, "Backtracking: Global timeout.", None, None

        # Generate a new random state candidate
        nums = list(range(GRID_SIZE * GRID_SIZE))
        random.shuffle(nums)
        current_attempt_2d_start = [nums[i*GRID_SIZE:(i+1)*GRID_SIZE] for i in range(GRID_SIZE)]

        # Ensure it's solvable (no need to check if already tried, generation is random)
        if not is_solvable(current_attempt_2d_start):
            continue # Generate another if not solvable

        # print(f"Backtracking Attempt {attempts+1}/{max_attempts}: Trying state {state_to_tuple(current_attempt_2d_start)} with {selected_sub_algorithm_name}")

        # Prepare arguments for the sub-algorithm
        algo_params = [current_attempt_2d_start] # First arg is always the state
        sub_algo_func_varnames = sub_algo_func.__code__.co_varnames[:sub_algo_func.__code__.co_argcount]

        # Add time_limit if the function accepts it
        if 'time_limit' in sub_algo_func_varnames:
            algo_params.append(time_limit_per_sub_solve)
        # Add other specific params if needed
        elif selected_sub_algorithm_name == 'DFS' and 'max_depth' in sub_algo_func_varnames:
             algo_params.append(30)
        elif selected_sub_algorithm_name == 'IDS' and 'max_depth_limit' in sub_algo_func_varnames:
             algo_params.append(30)
        elif selected_sub_algorithm_name == 'Stochastic Hill' and 'max_iter_no_improve' in sub_algo_func_varnames:
             algo_params.append(500)
        elif selected_sub_algorithm_name == 'Beam Search' and 'beam_width' in sub_algo_func_varnames:
             algo_params.append(5)

        # Call the sub-algorithm
        path_from_sub_algo = None
        action_plan_from_sub_algo = None
        belief_size_sub = 0
        try:
            if selected_sub_algorithm_name in ['Sensorless', 'Unknown']:
                action_plan_from_sub_algo, belief_size_sub = sub_algo_func(*algo_params)
                if action_plan_from_sub_algo is not None:
                     path_from_sub_algo = execute_plan(current_attempt_2d_start, action_plan_from_sub_algo)
            else:
                path_from_sub_algo = sub_algo_func(*algo_params)

        except Exception as e:
            # print(f"Backtracking: Error calling sub-algorithm {selected_sub_algorithm_name}: {e}")
            # traceback.print_exc() # Optional full traceback
            continue # Try next attempt

        # Check if the sub-algorithm succeeded
        if path_from_sub_algo and len(path_from_sub_algo) > 0:
            # For Sensorless/Unknown, plan existence means success
            sub_success = False
            if selected_sub_algorithm_name in ['Sensorless', 'Unknown']:
                sub_success = (action_plan_from_sub_algo is not None)
            else: # For others, check if goal is reached
                sub_success = is_goal(path_from_sub_algo[-1])

            if sub_success:
                # print(f"Backtracking SUCCESS: {selected_sub_algorithm_name} solved state {state_to_tuple(current_attempt_2d_start)}")
                # Return path, no error, the successful start state, and the action plan if applicable
                return path_from_sub_algo, None, current_attempt_2d_start, action_plan_from_sub_algo
        # else: print(f"Backtracking Info: {selected_sub_algorithm_name} failed for state {state_to_tuple(current_attempt_2d_start)}")

    # After all attempts
    msg = f"Backtracking: No solvable state found and solved by {selected_sub_algorithm_name} after {attempts+1} attempts."
    if time.time() - start_time_overall > time_limit_overall:
        msg = "Backtracking: Global timeout. " + msg
    # print(msg)
    return None, msg, None, None
# --- End Backtracking ---

# ----- Start: Functions for Sensorless Search / Unknown Env -----
def generate_belief_states(partial_state_template):
    """
    Generates all possible, solvable, complete states based on a template
    where None represents unknown cells.
    Returns a list of complete state lists. Returns empty list on error or if none are solvable.
    """
    flat_state_template = []
    unknown_positions_indices_flat = []
    known_numbers_set = set()
    is_template_with_unknowns = False

    # Validate input format
    if not isinstance(partial_state_template, list) or len(partial_state_template) != GRID_SIZE:
        # print("Error: Invalid partial state format (not list or wrong rows).")
        return []

    # Flatten the template, track knowns/unknowns
    for r_idx, r_val in enumerate(partial_state_template):
        if not isinstance(r_val, list) or len(r_val) != GRID_SIZE:
            # print(f"Error: Invalid partial state format (row {r_idx} not list or wrong cols).")
            return []
        for c_idx, tile in enumerate(r_val):
            if tile is None:
                is_template_with_unknowns = True
                unknown_positions_indices_flat.append(r_idx * GRID_SIZE + c_idx)
            elif isinstance(tile, int):
                if not (0 <= tile < GRID_SIZE * GRID_SIZE):
                    # print(f"Error: Tile value {tile} out of range at ({r_idx},{c_idx}).")
                    return []
                if tile in known_numbers_set:
                    # print(f"Error: Duplicate known tile value {tile} found.")
                    return []
                known_numbers_set.add(tile)
            else: # Tile is not None and not int
                 # print(f"Error: Invalid tile type '{type(tile)}' at ({r_idx},{c_idx}) in partial state.")
                 return []
            flat_state_template.append(tile)

    if len(flat_state_template) != GRID_SIZE * GRID_SIZE:
        # print("Error: Partial state flattened size mismatch.")
        return []

    # Case 1: No unknowns in the template
    if not is_template_with_unknowns:
        # Check if it's a complete, valid state
        if len(known_numbers_set) != GRID_SIZE * GRID_SIZE:
             # print("Error: Fully specified state has incorrect number of unique tiles.")
             return []
        # Check solvability
        if is_solvable(partial_state_template):
            # print("Info: No unknowns, returning the single valid state.")
            return [copy.deepcopy(partial_state_template)]
        else:
            # print("Info: No unknowns, but the state is unsolvable.")
            return []

    # Case 2: Template has unknowns
    all_possible_tiles = list(range(GRID_SIZE * GRID_SIZE))
    missing_numbers_to_fill = [num for num in all_possible_tiles if num not in known_numbers_set]

    # Check consistency
    if len(missing_numbers_to_fill) != len(unknown_positions_indices_flat):
        # print("Error: Mismatch between number of unknowns and missing numbers.")
        return []

    belief_states_generated = []
    # print(f"Info: Generating permutations for {len(unknown_positions_indices_flat)} unknowns with numbers {missing_numbers_to_fill}")

    # Iterate through permutations of missing numbers to fill unknown spots
    for perm_fill_nums in itertools.permutations(missing_numbers_to_fill):
        new_flat_state_filled = list(flat_state_template)
        # Fill in the unknowns based on the current permutation
        for i, unknown_idx in enumerate(unknown_positions_indices_flat):
            new_flat_state_filled[unknown_idx] = perm_fill_nums[i]

        # Convert flat list back to 2D grid
        state_2d_filled = []
        for r_idx_fill in range(GRID_SIZE):
            row_start = r_idx_fill * GRID_SIZE
            state_2d_filled.append(new_flat_state_filled[row_start : row_start + GRID_SIZE])

        # Check if the generated complete state is solvable
        if is_solvable(state_2d_filled):
            belief_states_generated.append(state_2d_filled)

    # print(f"Info: Generated {len(belief_states_generated)} solvable belief states.")
    return belief_states_generated

def is_belief_state_goal(list_of_belief_states):
    """Checks if all states in the belief set are goal states."""
    if not list_of_belief_states:
        return False # An empty belief state is not the goal
    for state_b in list_of_belief_states:
        if not is_goal(state_b): # Uses the global is_goal check
            return False
    return True

def apply_action_to_belief_states(list_of_belief_states, action_str):
    """
    Applies an action to each state in the belief set.
    Returns a list of resulting unique state lists.
    """
    new_belief_states_set = set() # Use a set of tuples to handle uniqueness
    for state_b in list_of_belief_states:
        next_s_b_list = apply_action_to_state(state_b, action_str)
        if next_s_b_list: # Make sure a valid state was returned
            next_s_b_tuple = state_to_tuple(next_s_b_list)
            if next_s_b_tuple: # Ensure tuple conversion worked
                 new_belief_states_set.add(next_s_b_tuple)

    # Convert the set of tuples back to a list of lists, handling potential None tuples
    return [tuple_to_list(s_tuple) for s_tuple in new_belief_states_set if s_tuple is not None]

def sensorless_search(start_belief_state_template, time_limit=60):
    """
    Performs a Sensorless Search (BFS on belief states). Also used for "Unknown Env".
    Input: A partial state template (list of lists with None for unknowns).
    Output: (action_plan, belief_state_size) or (None, size/0) if no plan found or error/timeout.
    """
    start_time = time.time()
    # print("Sensorless/Unknown: Generating initial belief states...")
    initial_belief_set_lists = generate_belief_states(start_belief_state_template)

    if not initial_belief_set_lists:
        # print("Sensorless/Unknown: Could not generate any valid initial belief states from the template.")
        return None, 0

    initial_belief_set_size = len(initial_belief_set_lists)
    # print(f"Sensorless/Unknown: Initial belief set size: {initial_belief_set_size}")

    # Convert initial belief set to tuples for the queue and visited set
    try:
        initial_belief_tuples = {state_to_tuple(s) for s in initial_belief_set_lists}
        if None in initial_belief_tuples:
             # print("Sensorless/Unknown Error: None tuple found in initial belief set.")
             return None, initial_belief_set_size # Return size found so far
    except Exception as e:
        # print(f"Sensorless/Unknown Error converting initial belief states to tuples: {e}")
        return None, initial_belief_set_size

    # Queue stores tuples: (action_plan_list, belief_state_set_of_tuples)
    queue_sensorless = deque([ ( [], initial_belief_tuples ) ])

    # Visited stores frozensets of belief state tuples
    visited_belief_sets = {frozenset(initial_belief_tuples)}

    possible_actions = ['Up', 'Down', 'Left', 'Right']
    max_plan_length = 30 # Heuristic limit to prevent very long searches
    nodes_explored = 0
    last_processed_bs_size = initial_belief_set_size # Keep track of size

    # print("Sensorless/Unknown: Starting BFS on belief states...")

    while queue_sensorless:
        nodes_explored += 1
        # if nodes_explored % 500 == 0: print(f"Sensorless/Unknown: Explored {nodes_explored} belief states...")

        if time.time() - start_time > time_limit:
            # print(f"Sensorless/Unknown: Time limit ({time_limit}s) exceeded.")
            # Return size of current belief state being processed when timeout occurs
            current_bs_size_on_timeout = len(queue_sensorless[0][1]) if queue_sensorless else last_processed_bs_size
            return None, current_bs_size_on_timeout

        action_plan, current_bs_tuples = queue_sensorless.popleft()
        last_processed_bs_size = len(current_bs_tuples) # Update last size

        if len(action_plan) > max_plan_length: continue

        # Convert current belief state tuples back to lists for goal checking
        current_bs_lists = [tuple_to_list(s) for s in current_bs_tuples if s is not None]
        if not current_bs_lists or len(current_bs_lists) != len(current_bs_tuples):
            # print("Sensorless/Unknown Warning: Issue converting current belief state tuples to lists.")
            continue # Skip if conversion failed

        if is_belief_state_goal(current_bs_lists):
            # print(f"Sensorless/Unknown: Goal reached! Plan length: {len(action_plan)}. Explored {nodes_explored} belief states.")
            return action_plan, len(current_bs_lists) # Return plan and final size

        # Explore actions
        for action_s in possible_actions:
            # Apply action needs list format
            next_bs_lists = apply_action_to_belief_states(current_bs_lists, action_s)

            if not next_bs_lists: continue # Action resulted in no valid states

            # Convert result back to set of tuples for processing and visited check
            try:
                next_bs_tuples = {state_to_tuple(s) for s in next_bs_lists}
                if None in next_bs_tuples: continue # Skip if conversion failed for any state
                if not next_bs_tuples: continue # Skip if empty set resulted

                next_bs_frozenset = frozenset(next_bs_tuples)
            except Exception as e:
                # print(f"Sensorless/Unknown Warning: Error converting next belief states to tuples/frozenset: {e}")
                continue # Skip this action if conversion fails

            if next_bs_frozenset not in visited_belief_sets:
                visited_belief_sets.add(next_bs_frozenset)
                new_plan_s = action_plan + [action_s]
                queue_sensorless.append((new_plan_s, next_bs_tuples))

    # print(f"Sensorless/Unknown: Search finished after exploring {nodes_explored} belief states. No goal path found.")
    return None, last_processed_bs_size # Return size of last state processed if no solution
# ----- End: Sensorless Search Implementation -----

# --- Helper for visualizing plans ---
def execute_plan(start_state, action_plan):
    """ Executes an action plan from a start state for visualization. Returns list of states or None on error."""
    if start_state is None or not isinstance(start_state, list): return None

    current_state = copy.deepcopy(start_state)
    state_sequence = [current_state] # Include the start state

    if not action_plan: # If the plan is empty, return just the start state
        return state_sequence

    for action in action_plan:
        next_state = apply_action_to_state(current_state, action)
        if next_state is None: # Should not happen with current apply_action logic, but safeguard
             # print(f"Warning: execute_plan encountered None state after action '{action}'. Stopping.")
             return state_sequence # Return path up to the error
        current_state = next_state # Update current state for the next action
        state_sequence.append(copy.deepcopy(current_state)) # Add the state after the action

    return state_sequence
# --- End Plan Execution Helper ---

# --- UI and Drawing Functions ---

def algorithm_selection_popup():
    """Displays a popup window for selecting a sub-algorithm for Backtracking."""
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)

    title_font_popup = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_popup = title_font_popup.render("Select Sub-Algorithm for Backtracking", True, INFO_COLOR)
    title_rect_popup = title_surf_popup.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf_popup, title_rect_popup)

    # Algorithms available as sub-algorithms (exclude Backtracking itself)
    algorithms_for_popup = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Sensorless', 'Sensorless Plan'), # Allow sensorless as sub-algo
        ('Unknown', 'Unknown Env')         # Allow Unknown as sub-algo
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
        # Convert mouse position relative to the popup surface
        mouse_pos_relative = (mouse_pos_screen[0] - popup_rect_on_screen.left,
                              mouse_pos_screen[1] - popup_rect_on_screen.top)

        popup_surface.fill(INFO_BG)
        pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
        popup_surface.blit(title_surf_popup, title_rect_popup)

        # Draw algorithm buttons
        for algo_id_p, algo_name_p, btn_rect_p in algo_buttons_popup:
            is_hovered_p = btn_rect_p.collidepoint(mouse_pos_relative)
            btn_color_p = MENU_HOVER_COLOR if is_hovered_p else MENU_BUTTON_COLOR
            pygame.draw.rect(popup_surface, btn_color_p, btn_rect_p, border_radius=8)
            text_surf_p = BUTTON_FONT.render(algo_name_p, True, WHITE)
            text_rect_p = text_surf_p.get_rect(center=btn_rect_p.center)
            popup_surface.blit(text_surf_p, text_rect_p)

        # Draw Cancel button
        is_hovered_cancel = cancel_button_rect_popup.collidepoint(mouse_pos_relative)
        cancel_color = MENU_HOVER_COLOR if is_hovered_cancel else RED
        pygame.draw.rect(popup_surface, cancel_color, cancel_button_rect_popup, border_radius=8)
        cancel_text_surf_p = BUTTON_FONT.render("Cancel", True, WHITE)
        cancel_text_rect_p = cancel_text_surf_p.get_rect(center=cancel_button_rect_popup.center)
        popup_surface.blit(cancel_text_surf_p, cancel_text_rect_p)

        screen.blit(popup_surface, popup_rect_on_screen)
        pygame.display.flip()

        # Event handling for the popup
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() # Quit the whole application if popup is closed
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check Cancel button click
                if cancel_button_rect_popup.collidepoint(mouse_pos_relative):
                    return None # Return None if cancelled
                # Check Algorithm button clicks
                for algo_id_p, _, btn_rect_p in algo_buttons_popup:
                    if btn_rect_p.collidepoint(mouse_pos_relative):
                        selected_algorithm_name = algo_id_p
                        running_popup = False # Exit loop
                        break # Stop checking other buttons
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None # Return None if Escape is pressed

        clock_popup.tick(60) # Limit FPS for the popup loop

    return selected_algorithm_name # Return the chosen algorithm ID

# --- Scroll and Menu Surface Variables ---
scroll_y = 0
menu_surface = None
total_menu_height = 0

def draw_state(state_to_draw, x_pos, y_pos, title_str, is_current_anim_state=False, is_fixed_goal_display=False, is_editable=False, selected_cell_coords=None):
    """Draws a single 3x3 puzzle state grid."""
    title_font_ds = pygame.font.Font(None, 28) # Use default font
    title_text_ds = title_font_ds.render(title_str, True, BLACK)
    title_x_ds = x_pos + (GRID_DISPLAY_WIDTH // 2 - title_text_ds.get_width() // 2)
    title_y_ds = y_pos - title_text_ds.get_height() - 5 # Position title above grid
    screen.blit(title_text_ds, (title_x_ds, title_y_ds))

    # Draw border around the grid
    pygame.draw.rect(screen, BLACK, (x_pos - 1, y_pos - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)

    # Check if state_to_draw is valid before iterating
    is_valid_structure = isinstance(state_to_draw, list) and len(state_to_draw) == GRID_SIZE and \
                         all(isinstance(row, list) and len(row) == GRID_SIZE for row in state_to_draw)

    for r_ds in range(GRID_SIZE):
        for c_ds in range(GRID_SIZE):
            cell_x_ds = x_pos + c_ds * CELL_SIZE
            cell_y_ds = y_pos + r_ds * CELL_SIZE
            cell_rect_ds = pygame.Rect(cell_x_ds, cell_y_ds, CELL_SIZE, CELL_SIZE)

            tile_val = None
            if is_valid_structure:
                tile_val = state_to_draw[r_ds][c_ds]
            # else: Optionally draw error indicator if structure wrong

            # Determine cell appearance based on tile value
            if tile_val is None: # Represents unknown/empty cell during input
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
                # Draw '?'
                q_font = pygame.font.SysFont('Arial', 40)
                q_surf = q_font.render("?", True, BLACK)
                screen.blit(q_surf, q_surf.get_rect(center=cell_rect_ds.center))
            elif tile_val == 0: # Represents the blank tile
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
            else: # Represents a numbered tile (1-8)
                # Determine fill color
                cell_fill_color = BLUE
                if is_fixed_goal_display:
                    cell_fill_color = GREEN # Goal state tiles are green
                elif is_current_anim_state and is_valid_structure and is_goal(state_to_draw):
                    cell_fill_color = GREEN # Current anim state green if it's the goal

                pygame.draw.rect(screen, cell_fill_color, cell_rect_ds.inflate(-6, -6), border_radius=8)
                # Draw the number
                try:
                    number_surf = FONT.render(str(tile_val), True, WHITE)
                    screen.blit(number_surf, number_surf.get_rect(center=cell_rect_ds.center))
                except ValueError: # Handle unexpected non-numeric tile_val
                     pygame.draw.rect(screen, RED, cell_rect_ds.inflate(-10,-10)) # Error indicator

            # Draw cell border
            pygame.draw.rect(screen, BLACK, cell_rect_ds, 1)

            # Highlight if editable and selected
            if is_editable and selected_cell_coords == (r_ds, c_ds):
                 pygame.draw.rect(screen, RED, cell_rect_ds, 3) # Red border for selected cell

def show_popup(message_str, title_str="Info"):
    """Displays a modal popup message box."""
    popup_surface_sp = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface_sp.fill(INFO_BG)
    border_rect_sp = popup_surface_sp.get_rect()
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, border_rect_sp, 4, border_radius=10)

    # Draw Title
    title_font_sp = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_sp = title_font_sp.render(title_str, True, INFO_COLOR)
    title_rect_sp = title_surf_sp.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface_sp.blit(title_surf_sp, title_rect_sp)

    # --- Text Wrapping ---
    words_sp = message_str.replace('\n', ' \n ').split(' ') # Handle explicit newlines
    lines_sp = []
    current_line_sp = ""
    text_width_limit_sp = POPUP_WIDTH - 60 # Padding on sides

    for word_sp in words_sp:
        if word_sp == '\n': # Handle explicit newline marker
            lines_sp.append(current_line_sp)
            current_line_sp = ""
            continue
        # Handle words containing internal newlines (split earlier)
        test_line_sp = current_line_sp + (" " if current_line_sp else "") + word_sp
        line_width_sp, _ = INFO_FONT.size(test_line_sp)

        if line_width_sp <= text_width_limit_sp:
            current_line_sp = test_line_sp
        else: # Word doesn't fit, start new line
            lines_sp.append(current_line_sp)
            current_line_sp = word_sp
    lines_sp.append(current_line_sp) # Add the last line
    # --- End Text Wrapping ---

    # Draw the wrapped text lines
    line_height_sp = INFO_FONT.get_linesize()
    text_start_y_sp = title_rect_sp.bottom + 20
    max_lines_display = (POPUP_HEIGHT - text_start_y_sp - 70) // line_height_sp # Max lines before OK button

    for i, line_text_sp in enumerate(lines_sp):
        if i >= max_lines_display:
            # Indicate truncation if too many lines
            ellipsis_surf = INFO_FONT.render("...", True, BLACK)
            popup_surface_sp.blit(ellipsis_surf, ( (POPUP_WIDTH - ellipsis_surf.get_width()) // 2, text_start_y_sp + i * line_height_sp))
            break
        text_surf_sp = INFO_FONT.render(line_text_sp, True, BLACK)
        # Center each line horizontally
        text_rect_sp = text_surf_sp.get_rect(centerx=POPUP_WIDTH // 2, top=text_start_y_sp + i * line_height_sp)
        popup_surface_sp.blit(text_surf_sp, text_rect_sp)

    # Draw OK Button
    ok_button_rect_sp = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, ok_button_rect_sp, border_radius=8)
    ok_text_surf_sp = BUTTON_FONT.render("OK", True, WHITE)
    ok_text_rect_sp = ok_text_surf_sp.get_rect(center=ok_button_rect_sp.center)
    popup_surface_sp.blit(ok_text_surf_sp, ok_text_rect_sp)

    # Display the popup and wait for OK
    popup_rect_on_screen_sp = popup_surface_sp.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(popup_surface_sp, popup_rect_on_screen_sp)
    pygame.display.flip()

    waiting_for_ok = True
    while waiting_for_ok:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Get mouse position relative to the popup window
                mouse_x_sp, mouse_y_sp = event.pos
                mouse_rel_x = mouse_x_sp - popup_rect_on_screen_sp.left
                mouse_rel_y = mouse_y_sp - popup_rect_on_screen_sp.top
                if ok_button_rect_sp.collidepoint(mouse_rel_x, mouse_rel_y):
                    waiting_for_ok = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    waiting_for_ok = False
        pygame.time.delay(20) # Small delay to prevent high CPU usage

def draw_menu(show_menu_flag, mouse_pos_dm, current_selected_algorithm_dm):
    """Draws the algorithm selection menu on the left."""
    global scroll_y, menu_surface, total_menu_height
    menu_elements_dict = {} # To return clickable element rects

    # Button to open the menu if it's closed
    if not show_menu_flag:
        menu_button_rect_dm = pygame.Rect(10, 10, 50, 40) # Position and size
        pygame.draw.rect(screen, MENU_COLOR, menu_button_rect_dm, border_radius=5)
        # Draw hamburger icon
        bar_width_dm, bar_height_dm, space_dm = 30, 4, 7
        start_x_dm = menu_button_rect_dm.centerx - bar_width_dm // 2
        start_y_dm = menu_button_rect_dm.centery - (bar_height_dm * 3 + space_dm * 2) // 2 + bar_height_dm // 2
        for i in range(3):
            pygame.draw.rect(screen, WHITE, (start_x_dm, start_y_dm + i * (bar_height_dm + space_dm), bar_width_dm, bar_height_dm), border_radius=2)
        menu_elements_dict['open_button'] = menu_button_rect_dm
        return menu_elements_dict # Only return the open button

    # --- If menu is open ---
    # Define algorithm list (kept both Sensorless and Unknown as requested)
    algorithms_list_dm = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Backtracking', 'Backtracking'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Sensorless', 'Sensorless Plan'), # Kept this
        ('Unknown', 'Unknown Env')         # Added this
    ]

    button_h_dm, padding_dm, button_margin_dm = 55, 10, 8 # Layout parameters
    # Calculate total height needed for all buttons
    total_menu_height = (len(algorithms_list_dm) * (button_h_dm + button_margin_dm)) - button_margin_dm + (2 * padding_dm)

    # Create or resize the off-screen surface for the menu content if needed
    display_height_menu_surf = max(total_menu_height, HEIGHT) # Surface must be at least screen height
    if menu_surface is None or menu_surface.get_height() != display_height_menu_surf:
        menu_surface = pygame.Surface((MENU_WIDTH, display_height_menu_surf))

    menu_surface.fill(MENU_COLOR) # Background

    # Draw algorithm buttons onto the menu_surface
    buttons_dict_dm = {} # Store rects relative to menu_surface
    y_pos_current_button = padding_dm
    # Calculate mouse position relative to the scrollable menu surface
    mouse_x_rel_dm, mouse_y_rel_dm = mouse_pos_dm[0], mouse_pos_dm[1] + scroll_y

    for algo_id_dm, algo_name_dm in algorithms_list_dm:
        button_rect_local_dm = pygame.Rect(padding_dm, y_pos_current_button, MENU_WIDTH - 2 * padding_dm, button_h_dm)

        # Check for hover and selection state
        is_hover_dm = button_rect_local_dm.collidepoint(mouse_x_rel_dm, mouse_y_rel_dm)
        is_selected_dm = (current_selected_algorithm_dm == algo_id_dm)

        # Determine button color
        button_color_dm = MENU_SELECTED_COLOR if is_selected_dm else \
                          (MENU_HOVER_COLOR if is_hover_dm else MENU_BUTTON_COLOR)
        pygame.draw.rect(menu_surface, button_color_dm, button_rect_local_dm, border_radius=5)

        # Draw button text
        text_surf_dm = BUTTON_FONT.render(algo_name_dm, True, WHITE)
        text_rect_dm = text_surf_dm.get_rect(center=button_rect_local_dm.center)
        menu_surface.blit(text_surf_dm, text_rect_dm)

        buttons_dict_dm[algo_id_dm] = button_rect_local_dm # Store local rect
        y_pos_current_button += button_h_dm + button_margin_dm

    # Blit the visible portion of the menu_surface onto the main screen
    visible_menu_area_rect = pygame.Rect(0, scroll_y, MENU_WIDTH, HEIGHT)
    screen.blit(menu_surface, (0,0), visible_menu_area_rect)

    # Draw the close button directly onto the screen (over the menu)
    close_button_rect_dm = pygame.Rect(MENU_WIDTH - 40, 10, 30, 30)
    pygame.draw.rect(screen, RED, close_button_rect_dm, border_radius=5)
    # Draw 'X' icon
    cx_dm, cy_dm = close_button_rect_dm.center
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm - 7), (cx_dm + 7, cy_dm + 7), 3)
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm + 7), (cx_dm + 7, cy_dm - 7), 3)

    # Store elements for interaction handling in main loop
    menu_elements_dict['close_button'] = close_button_rect_dm
    menu_elements_dict['buttons'] = buttons_dict_dm # These are local rects on menu_surface
    menu_elements_dict['menu_area'] = pygame.Rect(0, 0, MENU_WIDTH, HEIGHT) # Screen area occupied by menu

    # Draw scrollbar if content exceeds screen height
    if total_menu_height > HEIGHT:
        scrollbar_track_height = HEIGHT - 2*padding_dm # Height available for scrollbar track
        # Calculate scrollbar thumb height proportionally
        scrollbar_height_val = max(20, scrollbar_track_height * (HEIGHT / total_menu_height))

        # Calculate scrollbar thumb position
        max_scroll_y_content = total_menu_height - HEIGHT # Max value for scroll_y
        scroll_ratio = scroll_y / max_scroll_y_content if max_scroll_y_content > 0 else 0

        scrollbar_track_y_start = padding_dm
        # Max Y position the top of the thumb can reach
        scrollbar_max_y_thumb = scrollbar_track_y_start + scrollbar_track_height - scrollbar_height_val

        # Calculate Y position for the top of the thumb
        scrollbar_y_pos_thumb = scrollbar_track_y_start + scroll_ratio * (scrollbar_track_height - scrollbar_height_val)
        # Clamp thumb position within the track bounds
        scrollbar_y_pos_thumb = max(scrollbar_track_y_start, min(scrollbar_y_pos_thumb, scrollbar_max_y_thumb))

        scrollbar_rect_dm = pygame.Rect(MENU_WIDTH - 10, scrollbar_y_pos_thumb, 6, scrollbar_height_val)
        pygame.draw.rect(screen, GRAY, scrollbar_rect_dm, border_radius=3)
        # Note: Scrollbar interaction logic (dragging) is not implemented here, only mouse wheel scroll
        menu_elements_dict['scrollbar_rect'] = scrollbar_rect_dm # For visual reference

    return menu_elements_dict

def draw_grid_and_ui(current_anim_state_dgui, show_menu_dgui, current_algo_name_dgui,
                     solve_times_dgui, last_solved_run_info_dgui,
                     current_belief_state_size_dgui=None, selected_cell_for_input_coords=None):
    """Draws the main game screen including grids, buttons, info panel, and menu."""

    screen.fill(WHITE) # Clear screen
    mouse_pos_dgui = pygame.mouse.get_pos()

    # Determine layout based on whether the menu is shown
    main_area_x_offset = MENU_WIDTH if show_menu_dgui else 0
    main_area_width_dgui = WIDTH - main_area_x_offset
    center_x_main_area = main_area_x_offset + main_area_width_dgui // 2

    # --- Top Row: Initial and Goal Grids ---
    top_row_y_grids = GRID_PADDING + 40 # Y position for top grids
    grid_spacing_horizontal = GRID_PADDING * 1.5
    total_width_top_grids = 2 * GRID_DISPLAY_WIDTH + grid_spacing_horizontal
    start_x_top_grids = center_x_main_area - total_width_top_grids // 2

    initial_grid_x = start_x_top_grids
    goal_grid_x = start_x_top_grids + GRID_DISPLAY_WIDTH + grid_spacing_horizontal

    # Draw Initial State Grid (Editable)
    draw_state(initial_state, initial_grid_x, top_row_y_grids, "Initial State",
               is_editable=True, selected_cell_coords=selected_cell_for_input_coords)
    # Draw Goal State Grid (Fixed Display)
    draw_state(goal_state_fixed_global, goal_grid_x, top_row_y_grids, "Goal State", is_fixed_goal_display=True) # Always show fixed goal

    # --- Bottom Row: Controls, Current State, Info Panel ---
    bottom_row_y_start = top_row_y_grids + GRID_DISPLAY_WIDTH + GRID_PADDING + 60 # Y position for bottom elements

    # --- Control Buttons ---
    button_width_ctrl, button_height_ctrl = 140, 45
    buttons_start_x = main_area_x_offset + GRID_PADDING # X position relative to main area start
    buttons_mid_y_anchor = bottom_row_y_start + GRID_DISPLAY_WIDTH // 2 # Vertical anchor point

    # Calculate Y positions for buttons relative to anchor
    solve_button_y_pos = buttons_mid_y_anchor - button_height_ctrl * 2 - 16
    reset_solution_button_y_pos = solve_button_y_pos + button_height_ctrl + 8
    reset_all_button_y_pos = reset_solution_button_y_pos + button_height_ctrl + 8
    reset_initial_button_y_pos = reset_all_button_y_pos + button_height_ctrl + 8

    # Create Rects for buttons
    solve_button_rect_dgui = pygame.Rect(buttons_start_x, solve_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_solution_button_rect_dgui = pygame.Rect(buttons_start_x, reset_solution_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_all_button_rect_dgui = pygame.Rect(buttons_start_x, reset_all_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_initial_button_rect_dgui = pygame.Rect(buttons_start_x, reset_initial_button_y_pos, button_width_ctrl, button_height_ctrl)

    # Draw Buttons
    pygame.draw.rect(screen, RED, solve_button_rect_dgui, border_radius=5)
    solve_text_dgui = BUTTON_FONT.render("SOLVE", True, WHITE)
    screen.blit(solve_text_dgui, solve_text_dgui.get_rect(center=solve_button_rect_dgui.center))

    pygame.draw.rect(screen, BLUE, reset_solution_button_rect_dgui, border_radius=5)
    rs_text = BUTTON_FONT.render("Reset Solution", True, WHITE)
    screen.blit(rs_text, rs_text.get_rect(center=reset_solution_button_rect_dgui.center))

    pygame.draw.rect(screen, BLUE, reset_all_button_rect_dgui, border_radius=5)
    ra_text = BUTTON_FONT.render("Reset All", True, WHITE)
    screen.blit(ra_text, ra_text.get_rect(center=reset_all_button_rect_dgui.center))

    pygame.draw.rect(screen, BLUE, reset_initial_button_rect_dgui, border_radius=5)
    ri_text = BUTTON_FONT.render("Reset Display", True, WHITE)
    screen.blit(ri_text, ri_text.get_rect(center=reset_initial_button_rect_dgui.center))

    # --- Current State Grid ---
    current_state_grid_x = buttons_start_x + button_width_ctrl + GRID_PADDING * 1.5
    current_state_grid_y = bottom_row_y_start

    # Determine title for the current state grid based on algorithm
    current_state_title = f"Current ({current_algo_name_dgui})"
    # Check for both Sensorless and Unknown
    if current_algo_name_dgui in ['Sensorless', 'Unknown']:
        if current_belief_state_size_dgui is not None and current_belief_state_size_dgui >= 0:
             current_state_title = f"Belief States: {current_belief_state_size_dgui}"
        else:
             current_state_title = "Belief State (Template)"

    # Draw the current animation state
    draw_state(current_anim_state_dgui, current_state_grid_x, current_state_grid_y, current_state_title, is_current_anim_state=True)

    # --- Information Panel ---
    info_area_x_pos = current_state_grid_x + GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5
    info_area_y_pos = bottom_row_y_start
    # Calculate width to fill remaining space
    info_area_w = max(150, (main_area_x_offset + main_area_width_dgui) - info_area_x_pos - GRID_PADDING)
    info_area_h = GRID_DISPLAY_WIDTH # Match height of the current grid
    info_area_rect_dgui = pygame.Rect(info_area_x_pos, info_area_y_pos, info_area_w, info_area_h)

    # Draw info panel background and border
    pygame.draw.rect(screen, INFO_BG, info_area_rect_dgui, border_radius=8)
    pygame.draw.rect(screen, GRAY, info_area_rect_dgui, 2, border_radius=8)

    # Draw content inside info panel
    info_pad_x_ia, info_pad_y_ia = 15, 10 # Padding inside the panel
    line_h_ia = INFO_FONT.get_linesize() + 4 # Line height for stats
    current_info_y_draw = info_area_y_pos + info_pad_y_ia # Start Y for drawing text

    # Draw Info Panel Title
    compare_title_surf_ia = TITLE_FONT.render("Solver Stats", True, BLACK)
    compare_title_x_ia = info_area_rect_dgui.centerx - compare_title_surf_ia.get_width() // 2
    screen.blit(compare_title_surf_ia, (compare_title_x_ia, current_info_y_draw))
    current_info_y_draw += compare_title_surf_ia.get_height() + 8

    # Draw Solver Statistics (if available)
    if solve_times_dgui:
        # Sort results alphabetically by algorithm name for consistent order
        sorted_times_list = sorted(solve_times_dgui.items(), key=lambda item: item[0])

        for algo_name_st, time_val_st in sorted_times_list:
            # Check if drawing exceeds panel height
            if current_info_y_draw + line_h_ia > info_area_y_pos + info_area_h - info_pad_y_ia:
                screen.blit(INFO_FONT.render("...", True, BLACK), (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
                break # Stop drawing if no more space

            # Get steps or actions count and goal status
            actions_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_actions")
            steps_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_steps")
            reached_goal_st = last_solved_run_info_dgui.get(f"{algo_name_st}_reached_goal", None)

            # Format the count string (prefer actions if available)
            count_str_st = ""
            if actions_val_st is not None: count_str_st = f" ({actions_val_st} actions)"
            elif steps_val_st is not None: count_str_st = f" ({steps_val_st} steps)"
            else: count_str_st = " (--)" # Placeholder if no count

            # Format goal indicator
            goal_indicator_st = ""
            if reached_goal_st is False : goal_indicator_st = " (Not Goal)"
            # Add indicator if plan search failed specifically for Sensorless/Unknown
            elif algo_name_st in ['Sensorless', 'Unknown'] and reached_goal_st is None:
                 goal_indicator_st = " (Plan Fail)"

            # Assemble the full string
            base_str_st = f"{algo_name_st}: {time_val_st:.3f}s"
            full_comp_str = base_str_st + count_str_st + goal_indicator_st

            # --- Text Truncation ---
            max_text_width = info_area_w - 2 * info_pad_x_ia
            if INFO_FONT.size(full_comp_str)[0] > max_text_width:
                shortened_str = base_str_st + goal_indicator_st
                if INFO_FONT.size(shortened_str)[0] <= max_text_width:
                    full_comp_str = shortened_str
                else:
                    name_part = base_str_st.split(":")[0]
                    max_name_len = 10
                    if len(name_part) > max_name_len: name_part = name_part[:max_name_len] + "..."
                    full_comp_str = name_part + ": " + goal_indicator_st
            # --- End Truncation ---

            # Draw the stats line
            comp_surf_st = INFO_FONT.render(full_comp_str, True, BLACK)
            screen.blit(comp_surf_st, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
            current_info_y_draw += line_h_ia
    else:
        # Display message if no results yet
        no_results_surf = INFO_FONT.render("(No results yet)", True, GRAY)
        screen.blit(no_results_surf, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))

    # --- Draw the Menu ---
    menu_elements_dgui = draw_menu(show_menu_dgui, mouse_pos_dgui, current_algo_name_dgui)

    # --- Final Display Update ---
    pygame.display.flip() # Update the full screen

    # --- Return dictionary of UI element rects for interaction ---
    initial_grid_rect_on_screen = pygame.Rect(initial_grid_x, top_row_y_grids, GRID_DISPLAY_WIDTH, GRID_DISPLAY_WIDTH)

    return {
        'solve_button': solve_button_rect_dgui,
        'reset_solution_button': reset_solution_button_rect_dgui,
        'reset_all_button': reset_all_button_rect_dgui,
        'reset_initial_button': reset_initial_button_rect_dgui,
        'menu': menu_elements_dgui, # Contains menu button rects
        'initial_grid_area': initial_grid_rect_on_screen # Rect for the editable initial grid
    }

# --- Main Game Loop ---
def main():
    global scroll_y, initial_state, goal_state # Allow modification of global states

    # --- State Variables ---
    current_state_for_animation = copy.deepcopy(initial_state) # State shown in the 'Current' grid
    solution_path_anim = None   # List of states (or None) from the solver for animation
    action_plan_anim = None     # Stores the list of actions (for sensorless/unknown)
    current_step_in_anim = 0    # Index for animation step
    is_solving_flag = False     # True when an algorithm is actively running
    is_auto_animating_flag = False # True when solution path is being animated
    last_anim_step_time = 0     # Timestamp for animation timing
    show_algo_menu = False      # Whether the left menu is visible
    current_selected_algorithm = 'A*' # Default algorithm
    all_solve_times = {}        # Stores {algo_name: time}
    last_run_solver_info = {}   # Stores {algo_name_key: value} like steps, actions, reached_goal
    game_clock = pygame.time.Clock() # Pygame clock for FPS control
    ui_elements_rects = {}      # Stores rects of buttons, grids, etc., returned by draw_grid_and_ui
    running_main_loop = True    # Main loop flag
    backtracking_sub_algo_choice = None # Stores the sub-algorithm chosen for Backtracking
    current_sensorless_belief_size_for_display = None # Stores belief state size for UI
    selected_cell_for_input_coords = None # (row, col) of the initial_state cell selected, or None

    # --- Main Loop ---
    while running_main_loop:
        mouse_pos_main = pygame.mouse.get_pos() # Get mouse position once per frame

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_main_loop = False
                break # Exit event loop

            # --- Mouse Button Down ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left mouse button
                clicked_something_handled = False # Flag to prevent multiple actions per click

                # 1. Check Click on Initial State Grid (if editable)
                initial_grid_area_rect = ui_elements_rects.get('initial_grid_area')
                if initial_grid_area_rect and initial_grid_area_rect.collidepoint(mouse_pos_main):
                    # Calculate clicked cell within the grid
                    grid_offset_x = initial_grid_area_rect.left
                    grid_offset_y = initial_grid_area_rect.top
                    clicked_col = (mouse_pos_main[0] - grid_offset_x) // CELL_SIZE
                    clicked_row = (mouse_pos_main[1] - grid_offset_y) // CELL_SIZE

                    # Check if click was inside a valid cell
                    if 0 <= clicked_row < GRID_SIZE and 0 <= clicked_col < GRID_SIZE:
                        selected_cell_for_input_coords = (clicked_row, clicked_col)
                        # print(f"Selected cell: {selected_cell_for_input_coords}") # Debug
                    else:
                        selected_cell_for_input_coords = None # Clicked padding, deselect
                    clicked_something_handled = True # Click inside grid area handled
                # Deselect cell if clicking outside the initial grid AND not on the open menu
                elif selected_cell_for_input_coords:
                     is_click_on_menu = False
                     menu_area = ui_elements_rects.get('menu', {}).get('menu_area')
                     if menu_area and show_algo_menu and menu_area.collidepoint(mouse_pos_main):
                         is_click_on_menu = True
                     if not is_click_on_menu:
                         selected_cell_for_input_coords = None

                # 2. Check Click on Open Menu Elements
                if show_algo_menu and not clicked_something_handled:
                    menu_data_main = ui_elements_rects.get('menu', {})
                    menu_area_main = menu_data_main.get('menu_area')

                    if menu_area_main and menu_area_main.collidepoint(mouse_pos_main):
                        # Check Close Button
                        close_btn_rect = menu_data_main.get('close_button')
                        if close_btn_rect and close_btn_rect.collidepoint(mouse_pos_main):
                            show_algo_menu = False
                            clicked_something_handled = True

                        # Check Algorithm Buttons (if close button not clicked)
                        if not clicked_something_handled:
                            algo_buttons_local_rects = menu_data_main.get('buttons', {})
                            # Calculate mouse Y relative to scroll position
                            mouse_y_relative_to_scroll = mouse_pos_main[1] + scroll_y
                            for algo_id, local_r in algo_buttons_local_rects.items():
                                # Check collision using local rect and relative mouse Y
                                if local_r.collidepoint(mouse_pos_main[0], mouse_y_relative_to_scroll):
                                    if current_selected_algorithm != algo_id:
                                        # print(f"Algorithm selected: {algo_id}")
                                        current_selected_algorithm = algo_id
                                        # Reset state when algorithm changes
                                        solution_path_anim = None
                                        action_plan_anim = None
                                        current_step_in_anim = 0
                                        is_auto_animating_flag = False
                                        is_solving_flag = False
                                        current_state_for_animation = copy.deepcopy(initial_state)
                                        current_sensorless_belief_size_for_display = None
                                    show_algo_menu = False # Close menu after selection
                                    clicked_something_handled = True
                                    break # Stop checking other buttons

                        # If click was inside menu area but not on close or algo button, still handle it
                        if not clicked_something_handled:
                             clicked_something_handled = True # Consume click

                # 3. Check Click on Menu Open Button (if menu closed)
                if not show_algo_menu and not clicked_something_handled:
                    menu_data_main = ui_elements_rects.get('menu', {})
                    open_btn_rect = menu_data_main.get('open_button')
                    if open_btn_rect and open_btn_rect.collidepoint(mouse_pos_main):
                        show_algo_menu = True
                        scroll_y = 0 # Reset scroll on open
                        clicked_something_handled = True

                # 4. Check Click on Control Buttons (Solve, Reset, etc.)
                if not clicked_something_handled:
                    solve_btn = ui_elements_rects.get('solve_button')
                    reset_sol_btn = ui_elements_rects.get('reset_solution_button')
                    reset_all_btn = ui_elements_rects.get('reset_all_button')
                    reset_disp_btn = ui_elements_rects.get('reset_initial_button')

                    # --- SOLVE Button ---
                    if solve_btn and solve_btn.collidepoint(mouse_pos_main):
                        if not is_auto_animating_flag and not is_solving_flag:
                            should_start_solving = False # Flag to trigger solving below
                            # --- Pre-Solve Checks ---
                            if current_selected_algorithm == 'Backtracking':
                                backtracking_sub_algo_choice = algorithm_selection_popup()
                                if backtracking_sub_algo_choice: # Only proceed if user selected algo
                                    should_start_solving = True
                            # Allow Sensorless/Unknown without full validation
                            elif current_selected_algorithm in ['Sensorless', 'Unknown']:
                                # Basic check: must be a list of lists of correct size
                                template_valid = isinstance(initial_state, list) and len(initial_state) == GRID_SIZE and \
                                                 all(isinstance(r, list) and len(r)==GRID_SIZE for r in initial_state)
                                if template_valid:
                                     should_start_solving = True
                                else:
                                     show_popup("Initial state structure is invalid (must be 3x3 list of lists, can contain None for Unknown/Sensorless).", "Invalid Template Structure")
                            else: # Standard algorithms need full validation
                                if is_valid_state_for_solve(initial_state):
                                    if is_solvable(initial_state):
                                        should_start_solving = True
                                    else:
                                        show_popup("The current initial state is not solvable relative to the goal state.", "Unsolvable State")
                                else:
                                    show_popup("Initial state is not valid (must be 3x3 grid with numbers 0-8 exactly once) for standard algorithms.", "Invalid Initial State")
                            # --- End Pre-Solve Checks ---

                            # If checks passed, set flag to start solving in the logic section below
                            if should_start_solving:
                                is_solving_flag = True
                                solution_path_anim = None # Clear previous solution
                                action_plan_anim = None
                                current_step_in_anim = 0
                                is_auto_animating_flag = False
                                current_sensorless_belief_size_for_display = None # Reset belief display
                        clicked_something_handled = True

                    # --- RESET SOLUTION Button ---
                    elif reset_sol_btn and reset_sol_btn.collidepoint(mouse_pos_main):
                        # print("Reset Solution clicked")
                        current_state_for_animation = copy.deepcopy(initial_state) # Reset display to initial
                        solution_path_anim = None
                        action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        current_sensorless_belief_size_for_display = None
                        clicked_something_handled = True

                    # --- RESET ALL Button ---
                    elif reset_all_btn and reset_all_btn.collidepoint(mouse_pos_main):
                        # print("Reset All clicked")
                        initial_state = copy.deepcopy(initial_state_fixed_global) # Reset initial to fixed default
                        current_state_for_animation = copy.deepcopy(initial_state) # Reset display
                        solution_path_anim = None
                        action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        all_solve_times.clear()         # Clear solve times
                        last_run_solver_info.clear()    # Clear run info
                        current_sensorless_belief_size_for_display = None
                        selected_cell_for_input_coords = None # Deselect cell
                        clicked_something_handled = True

                    # --- RESET DISPLAY Button ---
                    elif reset_disp_btn and reset_disp_btn.collidepoint(mouse_pos_main):
                        # print("Reset Display clicked")
                        # Only resets the animation/display grid to match the current initial_state
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None # Clear animation path
                        action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False # Stop any solving/animation
                        is_auto_animating_flag = False
                        # Keep solve times and run info
                        clicked_something_handled = True

            # --- Mouse Wheel Scroll (for Menu) ---
            elif event.type == pygame.MOUSEWHEEL and show_algo_menu:
                menu_data_main = ui_elements_rects.get('menu', {})
                menu_area_main = menu_data_main.get('menu_area')
                # Check if mouse is over the menu area and scrolling is possible
                if menu_area_main and menu_area_main.collidepoint(mouse_pos_main) and total_menu_height > HEIGHT:
                    scroll_amount_mw = event.y * 35 # Adjust sensitivity
                    max_scroll_val = max(0, total_menu_height - HEIGHT) # Calculate max scroll offset
                    # Update scroll_y, clamping between 0 and max_scroll_val
                    scroll_y = max(0, min(scroll_y - scroll_amount_mw, max_scroll_val))

            # --- Keyboard Input (for Initial State Grid) ---
            elif event.type == pygame.KEYDOWN:
                if selected_cell_for_input_coords: # Check if a cell is selected
                    r, c = selected_cell_for_input_coords

                    if pygame.K_0 <= event.key <= pygame.K_8:
                        num = event.key - pygame.K_0
                        # Prevent entering a number already present elsewhere (unless it's 0)
                        can_place = True
                        if num != 0:
                            for row_idx, row_val in enumerate(initial_state):
                                for col_idx, cell_val in enumerate(row_val):
                                     if cell_val == num and (row_idx != r or col_idx != c):
                                         can_place = False; break
                                if not can_place: break

                        if can_place:
                             initial_state[r][c] = num
                             current_state_for_animation = copy.deepcopy(initial_state) # Update display
                             # Clear solution/belief size as state changed
                             solution_path_anim = None
                             action_plan_anim = None
                             current_sensorless_belief_size_for_display = None
                        # else: Optional feedback about duplicate

                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        # Allow setting to None (unknown)
                        initial_state[r][c] = None
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        action_plan_anim = None
                        current_sensorless_belief_size_for_display = None
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        selected_cell_for_input_coords = None # Deselect cell

        # --- End Event Handling for frame ---
        if not running_main_loop: break # Exit main loop if QUIT event received

        # --- Algorithm Solving Logic (Triggered by is_solving_flag) ---
        if is_solving_flag:
            is_solving_flag = False # Consume the flag
            solve_start_t = time.time()

            # Reset result variables
            found_path_algo = None
            found_action_plan_algo = None
            actual_start_state_for_anim = None # Used by Backtracking
            belief_size_at_end = 0 # Used by Sensorless/Unknown

            error_during_solve = False
            error_message_solve = ""

            try:
                # Use a deepcopy of the current initial_state for the solver
                state_to_solve_from = copy.deepcopy(initial_state)
                # Reset animation display to the state being solved
                current_state_for_animation = copy.deepcopy(state_to_solve_from)

                # print(f"--- Starting Solve: {current_selected_algorithm} ---")
                # --- Call Selected Algorithm ---
                if current_selected_algorithm == 'Backtracking':
                    if backtracking_sub_algo_choice:
                        # Backtracking returns path, error msg, start state, optional plan
                        found_path_algo, error_message_solve, actual_start_state_for_anim, found_action_plan_algo = \
                            backtracking_search(backtracking_sub_algo_choice)

                        # If backtracking found a state and solved it, update the main initial state
                        if found_path_algo and actual_start_state_for_anim:
                             initial_state = copy.deepcopy(actual_start_state_for_anim)
                             current_state_for_animation = copy.deepcopy(actual_start_state_for_anim)
                             # Store action plan if the sub-algo was sensorless/unknown
                             action_plan_anim = found_action_plan_algo # Will be None otherwise
                             current_selected_algorithm = backtracking_sub_algo_choice # Reflect sub-algo used for stats
                        # else: Backtracking failed, error_message_solve might be set

                        backtracking_sub_algo_choice = None # Reset choice
                    else:
                        error_message_solve = "Backtracking sub-algorithm not chosen."
                        error_during_solve = True

                elif current_selected_algorithm in ['Sensorless', 'Unknown']:
                    # Call sensorless search with the (potentially partial) state
                    found_action_plan_algo, belief_size_at_end = sensorless_search(state_to_solve_from, time_limit=60)
                    current_sensorless_belief_size_for_display = belief_size_at_end # Update UI variable

                    if found_action_plan_algo is not None: # Plan found
                        # print(f"{current_selected_algorithm}: Plan found. Generating visualization path.")
                        action_plan_anim = found_action_plan_algo # Store the plan

                        # Generate a sample state for visualization
                        sample_belief_states = generate_belief_states(state_to_solve_from)
                        vis_start_state = state_to_solve_from # Fallback to template
                        if sample_belief_states:
                             vis_start_state = sample_belief_states[0] # Use first valid belief state
                        # else: Handle case where no valid states generated from template

                        # Execute plan to get state sequence for animation
                        found_path_algo = execute_plan(vis_start_state, found_action_plan_algo)
                        # Start animation from the sample state used for visualization
                        current_state_for_animation = copy.deepcopy(vis_start_state)
                    # else: No plan found, found_path_algo remains None

                else: # Standard algorithms expecting a complete, valid state
                    algo_func_map = {
                        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar,
                        'Greedy': greedy, 'IDA*': ida_star,
                        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
                        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
                        'Beam Search': beam_search, 'AND-OR': and_or_search
                    }
                    selected_algo_function = algo_func_map.get(current_selected_algorithm)

                    if selected_algo_function:
                         # Prepare arguments (state + optional time limit, etc.)
                        algo_args_list = [state_to_solve_from]
                        func_varnames = selected_algo_function.__code__.co_varnames[:selected_algo_function.__code__.co_argcount]
                        default_time_limit = 30
                        if current_selected_algorithm in ['IDA*', 'IDS']: default_time_limit = 60

                        # Add time limit if accepted
                        if 'time_limit' in func_varnames:
                            algo_args_list.append(default_time_limit)
                        # Special args for specific algorithms
                        elif current_selected_algorithm == 'DFS' and 'max_depth' in func_varnames:
                            algo_args_list.append(30) # Default max depth
                        elif current_selected_algorithm == 'IDS' and 'max_depth_limit' in func_varnames:
                             algo_args_list.append(30) # Default depth limit for IDS
                        elif current_selected_algorithm == 'Stochastic Hill' and 'max_iter_no_improve' in func_varnames:
                             algo_args_list.append(500) # Default iter limit
                        elif current_selected_algorithm == 'Beam Search' and 'beam_width' in func_varnames:
                             algo_args_list.append(5) # Default beam width

                        # Call the function
                        found_path_algo = selected_algo_function(*algo_args_list)
                        action_plan_anim = None # No action plan for these standard algorithms
                    else:
                        error_message_solve = f"Algorithm '{current_selected_algorithm}' function not found in map."
                        error_during_solve = True

            except Exception as e:
                error_message_solve = f"Runtime Error during {current_selected_algorithm} solve:\n{traceback.format_exc()}"
                error_during_solve = True
                # Ensure these are None if error occurred during solve
                action_plan_anim = None
                found_path_algo = None

            # --- Post-Solve Processing & Info Update ---
            solve_duration_t = time.time() - solve_start_t
            all_solve_times[current_selected_algorithm] = solve_duration_t # Store time
            # print(f"--- Solve Finished: {current_selected_algorithm} took {solve_duration_t:.4f}s ---")

            if error_during_solve:
                show_popup(error_message_solve if error_message_solve else "An unknown error occurred during solve.", "Solver Runtime Error")
                # Clear potentially misleading info if error occurred
                last_run_solver_info.pop(f"{current_selected_algorithm}_reached_goal", None)
                last_run_solver_info.pop(f"{current_selected_algorithm}_steps", None)
                last_run_solver_info.pop(f"{current_selected_algorithm}_actions", None)
            else:
                # Check if a result was found (either state path or action plan)
                result_found = (found_path_algo and len(found_path_algo) > 0) or \
                               (action_plan_anim is not None)

                if result_found:
                    solution_path_anim = found_path_algo # Path for animation (even if from plan)
                    current_step_in_anim = 0
                    is_auto_animating_flag = True # Start animation
                    last_anim_step_time = time.time()

                    # Determine success, steps/actions, and popup message
                    is_actually_goal_state = False # Default to false
                    num_steps_or_actions = 0
                    popup_msg = ""

                    if current_selected_algorithm in ['Sensorless', 'Unknown']:
                        if action_plan_anim is not None:
                            num_steps_or_actions = len(action_plan_anim)
                            last_run_solver_info[f"{current_selected_algorithm}_actions"] = num_steps_or_actions
                            is_actually_goal_state = True # Plan guarantees goal
                            popup_msg = f"{current_selected_algorithm}: Plan found!\n{num_steps_or_actions} actions."
                            if current_selected_algorithm == 'Unknown':
                                popup_msg += "\n(Visualizing on one sample state)"
                        else: # Plan search failed
                            is_actually_goal_state = None # Use None to indicate plan failure
                            popup_msg = f"{current_selected_algorithm}: Plan search failed."
                            last_run_solver_info[f"{current_selected_algorithm}_reached_goal"] = None # Indicate plan fail

                    else: # Standard algorithms or Backtracking with standard sub-algo
                        if solution_path_anim:
                             final_state_of_path = solution_path_anim[-1]
                             is_actually_goal_state = is_goal(final_state_of_path)
                             num_steps_or_actions = len(solution_path_anim) - 1 # Path includes start
                             last_run_solver_info[f"{current_selected_algorithm}_steps"] = num_steps_or_actions
                             popup_msg = f"{current_selected_algorithm} "
                             if is_actually_goal_state:
                                 popup_msg += f"found solution!\n{num_steps_or_actions} steps."
                             else: # e.g., Hill Climbing stopped short
                                 popup_msg += f"finished.\n{num_steps_or_actions} steps (Not Goal)."
                        else: # Should not happen if result_found is True, but safeguard
                             is_actually_goal_state = False
                             popup_msg = f"{current_selected_algorithm}: Path invalid after solve."

                    # Record goal status unless already set to None for plan failure
                    if is_actually_goal_state is not None:
                         last_run_solver_info[f"{current_selected_algorithm}_reached_goal"] = is_actually_goal_state

                    # Add time and show popup
                    popup_msg += f"\nTime: {solve_duration_t:.3f}s"
                    popup_title = "Solve Complete" if is_actually_goal_state is True else \
                                  ("Plan Search Failed" if is_actually_goal_state is None else "Search Finished")
                    show_popup(popup_msg, popup_title)

                else: # No solution path or plan found
                    last_run_solver_info[f"{current_selected_algorithm}_reached_goal"] = False
                    # Clear any potential stale steps/actions counts
                    last_run_solver_info.pop(f"{current_selected_algorithm}_steps", None)
                    last_run_solver_info.pop(f"{current_selected_algorithm}_actions", None)

                    no_solution_msg = error_message_solve if error_message_solve else f"No solution found by {current_selected_algorithm}."

                    # Check for timeout based on typical limits
                    used_time_limit_for_msg = 30
                    if current_selected_algorithm in ['IDA*', 'Sensorless', 'Unknown', 'Backtracking', 'IDS']: used_time_limit_for_msg = 60
                    if solve_duration_t >= used_time_limit_for_msg * 0.98 : # Approx check
                         no_solution_msg = f"{current_selected_algorithm} timed out after ~{used_time_limit_for_msg}s."

                    show_popup(no_solution_msg, "No Solution / Timeout")

        # --- Animation Step ---
        if is_auto_animating_flag and solution_path_anim: # Animate using the state path
            current_time_anim = time.time()
            # Adjust animation speed based on path length
            anim_delay_val = 0.3
            if len(solution_path_anim) > 30: anim_delay_val = 0.15
            if len(solution_path_anim) > 60: anim_delay_val = 0.08

            if current_time_anim - last_anim_step_time >= anim_delay_val:
                if current_step_in_anim < len(solution_path_anim) - 1:
                    current_step_in_anim += 1
                    # Ensure the state being animated is valid
                    next_anim_state = solution_path_anim[current_step_in_anim]
                    if next_anim_state: # Check if state is not None or invalid structure
                        current_state_for_animation = copy.deepcopy(next_anim_state)
                    else: # Invalid state in path, stop animation
                        is_auto_animating_flag = False
                    last_anim_step_time = current_time_anim
                else: 
                    is_auto_animating_flag = False
               
        belief_size_for_display = current_sensorless_belief_size_for_display
        if current_selected_algorithm in ['Sensorless', 'Unknown'] and belief_size_for_display is None and not is_solving_flag and not is_auto_animating_flag:
             try:

                 temp_initial_bs = generate_belief_states(initial_state)
                 belief_size_for_display = len(temp_initial_bs) if temp_initial_bs is not None else 0
             except Exception:
                 belief_size_for_display = 0 

    
        ui_elements_rects = draw_grid_and_ui(
            current_state_for_animation, show_algo_menu,
            current_selected_algorithm, all_solve_times,
            last_run_solver_info,
            belief_size_for_display, 
            selected_cell_for_input_coords
        )

        game_clock.tick(60) 

    pygame.quit() 


if __name__ == "__main__":
    try:
        main()
    except Exception as main_error:
        print("\n--- UNHANDLED ERROR IN MAIN LOOP ---")
        traceback.print_exc()
        print("------------------------------------")
        pygame.quit() 
        sys.exit(1)   

