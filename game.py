import pygame
import sys
from collections import deque, defaultdict # Thêm defaultdict
import copy
import time
from queue import PriorityQueue
import traceback
import math
import random
import itertools

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

_FIXED_INITIAL_STATE_FLAT = [2,6,5,0,8,7,4,3,1]
_FIXED_GOAL_STATE_FLAT = [1,2,3,4,5,6,7,8,0]

initial_state_fixed_global = [[_FIXED_INITIAL_STATE_FLAT[i*GRID_SIZE+j] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
goal_state_fixed_global = [[_FIXED_GOAL_STATE_FLAT[i*GRID_SIZE+j] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]

initial_state = copy.deepcopy(initial_state_fixed_global)
# Goal state remains fixed based on the global definition
# goal_state = copy.deepcopy(goal_state_fixed_global) # Not needed as goal_state_fixed_global is used directly

# --- Helper Functions (State Manipulation, Validation, etc.) ---

def draw_state(state_to_draw, x_pos, y_pos, title_str, is_current_anim_state=False, is_fixed_goal_display=False, is_editable=False, selected_cell_coords=None):
    """Draws a single 3x3 puzzle state grid."""
    title_font_ds = pygame.font.Font(None, 28) # Use default font
    title_text_ds = title_font_ds.render(title_str, True, BLACK)
    title_x_ds = x_pos + (GRID_DISPLAY_WIDTH // 2 - title_text_ds.get_width() // 2)
    title_y_ds = y_pos - title_text_ds.get_height() - 5 # Position title above grid
    screen.blit(title_text_ds, (title_x_ds, title_y_ds))

    pygame.draw.rect(screen, BLACK, (x_pos - 1, y_pos - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)

    is_valid_structure = isinstance(state_to_draw, list) and len(state_to_draw) == GRID_SIZE and \
                         all(isinstance(row, list) and len(row) == GRID_SIZE for row in state_to_draw)

    for r_ds in range(GRID_SIZE):
        for c_ds in range(GRID_SIZE):
            cell_x_ds = x_pos + c_ds * CELL_SIZE
            cell_y_ds = y_pos + r_ds * CELL_SIZE
            cell_rect_ds = pygame.Rect(cell_x_ds, cell_y_ds, CELL_SIZE, CELL_SIZE)

            tile_val = None
            if is_valid_structure:
                try:
                    tile_val = state_to_draw[r_ds][c_ds]
                except IndexError:
                     tile_val = None # Should not happen with valid structure

            # --- Drawing logic for cells ---
            cell_bg_color = GRAY # Default for empty/None
            text_color = BLACK
            is_empty_tile = False

            if tile_val is None:
                # Draw '?' for None/unknown cells during editing
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
                q_font = pygame.font.SysFont('Arial', 40)
                q_surf = q_font.render("?", True, BLACK)
                screen.blit(q_surf, q_surf.get_rect(center=cell_rect_ds.center))
            elif tile_val == 0:
                # Draw empty tile (0) distinctly
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
                is_empty_tile = True
            else:
                # Draw numbered tiles
                cell_fill_color = BLUE # Default tile color
                text_color = WHITE
                # Highlight goal state or current state if it's the goal
                if is_fixed_goal_display:
                    cell_fill_color = GREEN
                elif is_current_anim_state and is_valid_structure and state_to_draw == goal_state_fixed_global: # Check against fixed goal
                    cell_fill_color = GREEN

                pygame.draw.rect(screen, cell_fill_color, cell_rect_ds.inflate(-6, -6), border_radius=8)
                try:
                    # Ensure tile_val is treated as string for rendering
                    number_surf = FONT.render(str(tile_val), True, text_color)
                    screen.blit(number_surf, number_surf.get_rect(center=cell_rect_ds.center))
                except (ValueError, TypeError):
                     # Draw red square if rendering fails (shouldn't happen with int)
                     pygame.draw.rect(screen, RED, cell_rect_ds.inflate(-10,-10))

            # Draw cell border
            pygame.draw.rect(screen, BLACK, cell_rect_ds, 1)

            # Highlight selected cell during editing
            if is_editable and selected_cell_coords == (r_ds, c_ds):
                 pygame.draw.rect(screen, RED, cell_rect_ds, 3) # Thicker red border


def find_empty(state):
    """Finds the row and column of the empty tile (0)."""
    if not isinstance(state, list) or len(state) != GRID_SIZE: return -1, -1
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE: return -1, -1
        for j in range(GRID_SIZE):
            try:
                # Ensure comparison is with integer 0
                if isinstance(state[i][j], int) and state[i][j] == 0:
                    return i, j
            except (TypeError, IndexError):
                # Handle potential errors if state structure is incorrect
                continue
    return -1, -1 # Return invalid coords if 0 is not found

def is_goal(state):
    """Checks if the given state is the goal state."""
    # Check basic structure first
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            return False
    # Ensure it's a valid, complete state before comparing
    # Note: is_valid_state_for_solve checks for None, which is important
    if not is_valid_state_for_solve(state):
        return False
    # Compare element-wise with the fixed global goal state
    return state == goal_state_fixed_global

def get_neighbors(state):
    """Generates neighbor states by moving the empty tile."""
    neighbors = []
    empty_i, empty_j = find_empty(state)
    if empty_i == -1:
        # This indicates an invalid state (no empty tile)
        # print("Warning: get_neighbors called on state without an empty tile.")
        return []

    # Directions: (di, dj) pairs represent where the *tile* would be if the empty space moved there
    # E.g., (-1, 0) means the tile above the empty space moves down into the empty space.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right tile moves relative to empty space

    for di, dj in directions:
        tile_i, tile_j = empty_i + di, empty_j + dj # Coords of the tile to potentially swap with empty

        # Check if the tile coordinates are within the grid bounds
        if 0 <= tile_i < GRID_SIZE and 0 <= tile_j < GRID_SIZE:
            new_state = copy.deepcopy(state)
            # Swap the empty tile (at empty_i, empty_j) with the adjacent tile (at tile_i, tile_j)
            new_state[empty_i][empty_j], new_state[tile_i][tile_j] = new_state[tile_i][tile_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    return neighbors

# --- NEW: Get valid actions (directions the *blank tile* can move) ---
def get_valid_actions(state):
    """Returns a list of valid actions ('Up', 'Down', 'Left', 'Right') for the blank tile."""
    actions = []
    empty_i, empty_j = find_empty(state)
    if empty_i == -1:
        # print("Warning: get_valid_actions called on state without an empty tile.")
        return []
    # Check potential moves *for the blank tile*
    # Action 'Up' means blank moves to row empty_i - 1
    if empty_i > 0: actions.append('Up')
    # Action 'Down' means blank moves to row empty_i + 1
    if empty_i < GRID_SIZE - 1: actions.append('Down')
    # Action 'Left' means blank moves to column empty_j - 1
    if empty_j > 0: actions.append('Left')
    # Action 'Right' means blank moves to column empty_j + 1
    if empty_j < GRID_SIZE - 1: actions.append('Right')
    return actions

# --- UPDATED: Apply action (move blank tile) ---
def apply_action_to_state(state_list, action):
    """
    Applies an action (moving the blank tile) to a state.
    Action is a string: 'Up', 'Down', 'Left', 'Right' representing the blank's move direction.
    Returns the new state list, or None if the action is invalid or fails.
    """
    # Validate input state first
    if state_list is None: return None # Cannot apply action to None
    # Q-Learning needs a complete state to work with
    if not is_valid_state_for_solve(state_list):
        # print(f"Warning: apply_action_to_state called with invalid/incomplete state: {state_list}")
        # For Q-learning, treat invalid states as dead ends maybe? Or just fail.
        return None # Indicate failure for invalid input

    new_state = copy.deepcopy(state_list)
    empty_i, empty_j = find_empty(new_state)
    if empty_i == -1:
        # print("Warning: apply_action_to_state called on state without empty tile (post-copy).")
        return None # Should not happen if is_valid_state_for_solve passed

    # Determine the coordinates of the *tile* to swap with the blank based on the blank's move direction
    tile_to_swap_i, tile_to_swap_j = -1, -1
    if action == 'Up' and empty_i > 0: # Blank moves up, swaps with tile above it
        tile_to_swap_i, tile_to_swap_j = empty_i - 1, empty_j
    elif action == 'Down' and empty_i < GRID_SIZE - 1: # Blank moves down
        tile_to_swap_i, tile_to_swap_j = empty_i + 1, empty_j
    elif action == 'Left' and empty_j > 0: # Blank moves left
        tile_to_swap_i, tile_to_swap_j = empty_i, empty_j - 1
    elif action == 'Right' and empty_j < GRID_SIZE - 1: # Blank moves right
        tile_to_swap_i, tile_to_swap_j = empty_i, empty_j + 1
    else:
        # The action provided is not possible from the current blank position
        # print(f"Warning: Invalid action '{action}' requested for blank at ({empty_i},{empty_j}).")
        # Return the *original copied state* to indicate no change, or None?
        # For Q-learning, returning the same state might be confusing. Let's return None for invalid move.
        return None # Indicate action was invalid/impossible

    # Check if tile coordinates are valid (redundant check if logic above is correct, but safe)
    if 0 <= tile_to_swap_i < GRID_SIZE and 0 <= tile_to_swap_j < GRID_SIZE:
        # Perform the swap
        new_state[empty_i][empty_j], new_state[tile_to_swap_i][tile_to_swap_j] = \
            new_state[tile_to_swap_i][tile_to_swap_j], new_state[empty_i][empty_j]
        return new_state
    else:
        # This case should technically not be reached if the initial action check is correct
        # print(f"Error: Logic flaw in apply_action_to_state for action '{action}'.")
        return None # Indicate internal error


def state_to_tuple(state):
    """Converts a 2D list state to a tuple of tuples for hashing."""
    if not isinstance(state, list):
        # print("state_to_tuple: Input is not a list.")
        return None
    try:
        # Basic structure check
        if len(state) != GRID_SIZE:
             # print(f"state_to_tuple: Incorrect number of rows {len(state)}.")
             return None
        tuple_rows = []
        for r in range(GRID_SIZE):
            if not isinstance(state[r], list) or len(state[r]) != GRID_SIZE:
                 # print(f"state_to_tuple: Invalid row structure at index {r}.")
                 return None
            # Ensure elements are suitable (int or None)
            # Q-Learning *requires* complete states, so check for None here.
            row_tuple_content = []
            for c in range(GRID_SIZE):
                 tile = state[r][c]
                 if not isinstance(tile, int): # Must be int for Q-learning state representation
                     # print(f"state_to_tuple: Non-integer {tile} at ({r},{c}).")
                     return None
                 row_tuple_content.append(tile)
            tuple_rows.append(tuple(row_tuple_content))
        return tuple(tuple_rows)
    except (TypeError, IndexError) as e:
        # print(f"state_to_tuple: Exception occurred - {e}")
        return None


def tuple_to_list(state_tuple):
    """Converts a tuple state back to a 2D list."""
    if not isinstance(state_tuple, tuple) or len(state_tuple) != GRID_SIZE:
        return None
    try:
        new_list = []
        for row_tuple in state_tuple:
            if not isinstance(row_tuple, tuple) or len(row_tuple) != GRID_SIZE:
                return None
            # Convert row tuple back to list
            new_list.append(list(row_tuple))
        # Final check: ensure all elements are integers
        if not all(isinstance(tile, int) for row in new_list for tile in row):
            return None
        return new_list
    except (TypeError, IndexError):
        return None


def manhattan_distance(state):
    """Calculates the Manhattan distance heuristic."""
    distance = 0
    goal_pos = {}
    # Precompute goal positions {tile_value: (row, col)}
    for r_goal in range(GRID_SIZE):
        for c_goal in range(GRID_SIZE):
            val = goal_state_fixed_global[r_goal][c_goal]
            if val != 0 : # We don't count the blank tile's distance
                goal_pos[val] = (r_goal, c_goal)

    # Check if state is valid list structure
    if not isinstance(state, list): return float('inf')

    for r_curr in range(GRID_SIZE):
        # Check row structure
        if not isinstance(state[r_curr], list) or len(state[r_curr]) != GRID_SIZE: return float('inf')
        for c_curr in range(GRID_SIZE):
            try:
                tile = state[r_curr][c_curr]
                # Heuristic needs a complete, valid state (all integers 0-8)
                if tile is None or not isinstance(tile, int): return float('inf')
                if tile != 0: # Calculate distance only for numbered tiles
                    if tile in goal_pos:
                        goal_r, goal_c = goal_pos[tile]
                        distance += abs(r_curr - goal_r) + abs(c_curr - goal_c)
                    else:
                        # This means the tile value is invalid (e.g., 9 in a 3x3 grid)
                        return float('inf') # Invalid state for heuristic
            except (IndexError, TypeError):
                 return float('inf') # Error accessing state element

    return distance


def is_valid_state_for_solve(state_to_check):
    """Checks if the state is a valid 3x3 grid with numbers 0-8 exactly once."""
    flat_state = []
    try:
        # Check outer list structure
        if not isinstance(state_to_check, list) or len(state_to_check) != GRID_SIZE:
             # print("is_valid_state_for_solve: Not list or wrong number of rows.")
             return False
        for r in range(GRID_SIZE):
            row = state_to_check[r]
            # Check inner list structure
            if not isinstance(row, list) or len(row) != GRID_SIZE:
                 # print(f"is_valid_state_for_solve: Row {r} not list or wrong size.")
                 return False
            for c in range(GRID_SIZE):
                tile = row[c]
                # Check tile type - MUST be integer for solving algorithms (except maybe Sensorless template)
                # Allow 0, but disallow None for standard solvers
                if not isinstance(tile, int):
                    # print(f"is_valid_state_for_solve: Tile at ({r},{c}) is not an integer: {tile} ({type(tile)})")
                    return False
                flat_state.append(tile)
    except (TypeError, IndexError) as e:
        # print(f"is_valid_state_for_solve: Exception during check: {e}")
        return False # Error during checks means invalid

    # Check total number of elements
    if len(flat_state) != GRID_SIZE * GRID_SIZE:
        # print(f"is_valid_state_for_solve: Incorrect number of elements: {len(flat_state)}")
        return False # Should not happen if grid size checks pass

    # Check if numbers 0 to N^2-1 are present exactly once
    expected_numbers = set(range(GRID_SIZE * GRID_SIZE)) # e.g., {0, 1, 2, 3, 4, 5, 6, 7, 8}
    seen_numbers = set(flat_state)

    if seen_numbers != expected_numbers:
        # print(f"is_valid_state_for_solve: Numbers mismatch. Seen: {seen_numbers}, Expected: {expected_numbers}")
        return False

    return True # Passed all checks


def get_inversions(flat_state_no_zero):
    """Calculates the number of inversions in a flattened state (excluding 0)."""
    inversions = 0
    n = len(flat_state_no_zero)
    for i in range(n):
        for j in range(i + 1, n):
            # Ensure we are comparing actual numbers
            if flat_state_no_zero[i] > flat_state_no_zero[j]:
                inversions += 1
    return inversions

def is_solvable(state_to_check):
    """Checks if a given 8-puzzle state is solvable relative to the fixed goal."""
    # First, ensure the state itself is valid (complete, correct numbers)
    if not is_valid_state_for_solve(state_to_check):
        # print("is_solvable: Input state is not valid for solving.")
        return False

    # Flatten the state and remove the blank tile (0)
    flat_state = [tile for row in state_to_check for tile in row]
    state_flat_no_zero = [tile for tile in flat_state if tile != 0]
    inversions_state = get_inversions(state_flat_no_zero)

    # Do the same for the fixed global goal state
    goal_flat = [tile for row in goal_state_fixed_global for tile in row]
    goal_flat_no_zero = [tile for tile in goal_flat if tile != 0]
    inversions_goal = get_inversions(goal_flat_no_zero)

    # Solvability rule for N x N puzzle:
    grid_width = GRID_SIZE

    if grid_width % 2 == 1: # Odd grid width (e.g., 3x3)
        # Solvable if the number of inversions has the same parity (both even or both odd)
        return (inversions_state % 2) == (inversions_goal % 2)
    else: # Even grid width (e.g., 4x4) - Requires blank row position
        # Find blank row (0-indexed from top)
        blank_row_state = -1
        for r_idx, row_s in enumerate(state_to_check):
            if 0 in row_s:
                blank_row_state = r_idx
                break
        # Calculate row position from the bottom (1-indexed)
        blank_row_from_bottom_state = grid_width - blank_row_state

        # Find blank row for the goal state
        blank_row_goal = -1
        for r_idx_g, row_g in enumerate(goal_state_fixed_global):
            if 0 in row_g:
                blank_row_goal = r_idx_g
                break
        blank_row_from_bottom_goal = grid_width - blank_row_goal

        # Solvability condition for even grids:
        # Parity of (inversions + blank_row_from_bottom) must match
        return ((inversions_state + blank_row_from_bottom_state) % 2) == \
               ((inversions_goal + blank_row_from_bottom_goal) % 2)


# --- Search Algorithms (Existing ones: BFS, DFS, IDS, UCS, A*, Greedy, IDA*, Hill Climbing variants, Beam, AND-OR)---
# ... (Keep the existing implementations of these algorithms) ...
def bfs(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    queue = deque([(start_node_state, [start_node_state])])
    visited = {init_tuple}

    while queue:
        if time.time() - start_time > time_limit:
            # print("BFS Timeout")
            return None # Timeout
        current_s, path = queue.popleft()

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor_s, path + [neighbor_s]))
    return None # No solution found

def dfs(start_node_state, max_depth=30, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    stack = [(start_node_state, [start_node_state], 0)] # state, path, depth
    visited = {} # Store tuple -> min_depth_reached

    while stack:
        if time.time() - start_time > time_limit:
            # print("DFS Timeout")
            return None # Timeout

        current_s, path, depth = stack.pop()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue # Should not happen

        # Pruning based on depth visited
        if current_tuple in visited and visited[current_tuple] <= depth:
            continue
        visited[current_tuple] = depth

        if is_goal(current_s):
            return path

        if depth >= max_depth:
            continue # Depth limit reached

        # Add neighbors in reverse order for intuitive DFS exploration
        neighbors = get_neighbors(current_s)
        for neighbor_s in reversed(neighbors):
            stack.append((neighbor_s, path + [neighbor_s], depth + 1))

    return None # No solution found

def ids(start_node_state, max_depth_limit=30, time_limit=60):
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    for depth_limit in range(max_depth_limit + 1):
        # Check global time limit at the start of each DLS iteration
        if time.time() - start_time_global > time_limit:
            # print("IDS Global Timeout")
            return None

        stack = [(start_node_state, [start_node_state], 0)] # state, path, depth
        # Visited set specific to the current depth-limited search (DLS)
        visited_in_iteration = {init_tuple: 0} # state_tuple -> depth

        while stack:
            # Check global time limit inside the DLS loop
            if time.time() - start_time_global > time_limit:
                 # print("IDS Inner Timeout")
                 return None

            current_s, path, depth = stack.pop()
            # No need to re-tuple here if neighbours generate valid states

            if is_goal(current_s):
                return path

            # Explore neighbors only if below the current depth limit
            if depth < depth_limit:
                neighbors = get_neighbors(current_s)
                for neighbor_s in reversed(neighbors): # Consistent DFS order
                    neighbor_tuple = state_to_tuple(neighbor_s)
                    if neighbor_tuple is not None:
                        # Add to stack only if not visited in this iteration or found via a shorter path
                        if neighbor_tuple not in visited_in_iteration or visited_in_iteration[neighbor_tuple] > depth + 1:
                             visited_in_iteration[neighbor_tuple] = depth + 1
                             stack.append((neighbor_s, path + [neighbor_s], depth + 1))

    return None # No solution found within max_depth_limit

def ucs(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    # Priority queue stores (cost, state, path)
    frontier = PriorityQueue()
    frontier.put((0, start_node_state, [start_node_state]))

    # Visited dictionary stores {state_tuple: cost}
    visited = {init_tuple: 0}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            # print("UCS Timeout")
            return None # Timeout

        cost, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        # Optimization: If we pull a state with higher cost than already found, skip
        if cost > visited.get(current_tuple, float('inf')):
             continue

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            new_cost = cost + 1 # Assuming uniform cost

            # If this path to the neighbor is cheaper than any previous path
            if new_cost < visited.get(neighbor_tuple, float('inf')):
                visited[neighbor_tuple] = new_cost
                frontier.put((new_cost, neighbor_s, path + [neighbor_s]))

    return None # No solution found

def astar(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    # Calculate initial heuristic
    h_init = manhattan_distance(start_node_state)
    if h_init == float('inf'): return None # Cannot solve if heuristic is invalid
    g_init = 0
    f_init = g_init + h_init

    # Priority queue stores (f_score, g_score, state, path)
    frontier = PriorityQueue()
    frontier.put((f_init, g_init, start_node_state, [start_node_state]))

    # visited_g_scores stores {state_tuple: g_score}
    visited_g_scores = {init_tuple: g_init}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
             # print("A* Timeout")
             return None # Timeout

        f_score_curr, g_score_curr, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        # Optimization: If we found a shorter path to this state already, skip
        if g_score_curr > visited_g_scores.get(current_tuple, float('inf')):
            continue

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            tentative_g_score = g_score_curr + 1 # Cost of edge is 1

            # If this path to the neighbor is better than any previous one found
            if tentative_g_score < visited_g_scores.get(neighbor_tuple, float('inf')):
                visited_g_scores[neighbor_tuple] = tentative_g_score
                h_neighbor = manhattan_distance(neighbor_s)
                if h_neighbor == float('inf'): continue # Skip if heuristic fails

                f_neighbor = tentative_g_score + h_neighbor
                frontier.put((f_neighbor, tentative_g_score, neighbor_s, path + [neighbor_s]))

    return None # No solution found

def greedy(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    # Priority queue stores (h_score, state, path)
    frontier = PriorityQueue()
    h_init = manhattan_distance(start_node_state)
    if h_init == float('inf'): return None
    frontier.put((h_init, start_node_state, [start_node_state]))

    # Visited set stores state_tuples
    visited = {init_tuple}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            # print("Greedy Timeout")
            return None # Timeout

        h_val, current_s, path = frontier.get()
        # No need to check if current is goal here if we check neighbors?
        # Check goal *before* exploring neighbors is standard
        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            # Add to frontier only if not visited
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple) # Mark visited when adding
                h_neighbor = manhattan_distance(neighbor_s)
                if h_neighbor == float('inf'): continue # Skip invalid
                frontier.put((h_neighbor, neighbor_s, path + [neighbor_s]))

    return None # No solution found

# --- Corrected IDA* Implementation ---
# Need the recursive helper function
def _search_ida_recursive(path_ida, g_score, threshold, start_time_ida, time_limit_ida, visited_in_path):
    """Recursive helper for IDA*."""
    if time.time() - start_time_ida >= time_limit_ida:
        return "Timeout", float('inf')

    current_s_ida = path_ida[-1]
    current_tuple_ida = state_to_tuple(current_s_ida)

    h_ida = manhattan_distance(current_s_ida)
    if h_ida == float('inf'):
         return None, float('inf') # Treat as dead end
    f_score_ida = g_score + h_ida

    if f_score_ida > threshold:
        return None, f_score_ida # Return the f-score exceeding threshold

    if is_goal(current_s_ida):
        return path_ida[:], threshold # Found solution

    min_new_threshold = float('inf') # Track min f-score > threshold

    # Explore neighbors
    for neighbor_s_ida in get_neighbors(current_s_ida):
        neighbor_tuple_ida = state_to_tuple(neighbor_s_ida)
        if neighbor_tuple_ida is None: continue

        # Avoid cycles within the current path search
        if neighbor_tuple_ida in visited_in_path:
            continue

        new_g_ida = g_score + 1

        # Recurse
        path_ida.append(neighbor_s_ida)
        visited_in_path.add(neighbor_tuple_ida)

        result_ida, recursive_threshold_ida = _search_ida_recursive(
            path_ida, new_g_ida, threshold, start_time_ida, time_limit_ida, visited_in_path
        )

        # Backtrack
        path_ida.pop()
        visited_in_path.remove(neighbor_tuple_ida)

        # Handle results
        if result_ida == "Timeout": return "Timeout", float('inf')
        if result_ida is not None: return result_ida, threshold # Propagate solution

        # Update min threshold for next iteration if this branch failed
        min_new_threshold = min(min_new_threshold, recursive_threshold_ida)

    return None, min_new_threshold # Return min f-score > threshold found

def ida_star(start_node_state, time_limit=60):
    """Iterative Deepening A* Search."""
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    initial_h = manhattan_distance(start_node_state)
    if initial_h == float('inf'): return None
    threshold = initial_h

    while True:
        if time.time() - start_time_global >= time_limit:
            # print("IDA* Global Timeout")
            return None

        current_path = [start_node_state]
        visited_path_tuples = {init_tuple} # Track nodes in current path

        result, new_threshold_candidate = _search_ida_recursive(
            current_path, 0, threshold, start_time_global, time_limit, visited_path_tuples
        )

        if result == "Timeout": return None
        if result is not None: return result # Solution found

        if new_threshold_candidate == float('inf'):
            return None # No solution exists
        if new_threshold_candidate <= threshold:
             # Safety check, should not happen
             # print("IDA* Warning: Threshold did not increase.")
             return None

        threshold = new_threshold_candidate
        # print(f"IDA* Increasing threshold to: {threshold}")


# --- Hill Climbing Variants ---
def simple_hill_climbing(start_node_state, time_limit=30):
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path # Return path up to invalid state

    while True: # Loop until stuck or goal
        if time.time() - start_time > time_limit:
            # print("Simple Hill Climbing Timeout")
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break # Should not happen if not goal

        moved = False
        # Optional: Shuffle neighbors to explore different paths on plateaus
        shuffled_neighbors = list(neighbors)
        random.shuffle(shuffled_neighbors)

        # Find the *first* neighbor that is better
        for neighbor_s in shuffled_neighbors:
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor == float('inf'): continue # Skip invalid

            if h_neighbor < current_h:
                current_s = neighbor_s
                current_h = h_neighbor
                path.append(current_s)
                moved = True
                break # Take the first improvement found

        # If no better neighbor was found, we are stuck
        if not moved:
            return path

    return path # Fallback (shouldn't be reached)

def steepest_hill_climbing(start_node_state, time_limit=30):
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path

    while True:
        if time.time() - start_time > time_limit:
            # print("Steepest Hill Climbing Timeout")
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        best_next_s = None
        best_next_h = current_h # Start assuming no improvement

        # Find the *best* neighbor among all valid neighbors
        for neighbor_s in neighbors:
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor == float('inf'): continue

            if h_neighbor < best_next_h: # Found a new best
                best_next_h = h_neighbor
                best_next_s = neighbor_s
            # Handle ties: current logic takes the last one found with the best h.
            # Could randomize tie-breaking.

        # If no neighbor is strictly better, we are stuck
        if best_next_s is None or best_next_h >= current_h:
            return path # Local optimum or plateau

        # Move to the best neighbor
        current_s = best_next_s
        current_h = best_next_h
        path.append(current_s)

    return path # Fallback

def random_hill_climbing(start_node_state, time_limit=30, max_iter_no_improve=500):
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path
    iter_no_improve = 0

    while True:
        if time.time() - start_time > time_limit:
             # print("Random Hill Climbing Timeout")
             return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        # Select a random neighbor
        random_neighbor = random.choice(neighbors)
        neighbor_h = manhattan_distance(random_neighbor)
        if neighbor_h == float('inf'): continue # Skip if random choice invalid

        # Move if neighbor is better or equal (allows sideways moves)
        if neighbor_h <= current_h:
            if neighbor_h < current_h:
                iter_no_improve = 0 # Reset counter on improvement
            else:
                iter_no_improve += 1 # Sideways move

            current_s = random_neighbor
            current_h = neighbor_h
            path.append(current_s)
        else:
            # Neighbor is worse, stay put
            iter_no_improve += 1

        # Stop if no improvement for too long
        if iter_no_improve >= max_iter_no_improve:
            # print(f"RHC stopped after {max_iter_no_improve} iters no improve.")
            return path
    return path

def simulated_annealing(start_node_state, initial_temp=20.0, cooling_rate=0.95, min_temp=0.1, time_limit=15):
    start_time = time.time()
    current_s = start_node_state
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return [start_node_state] # Cannot start if invalid

    path = [current_s]
    best_s_so_far = copy.deepcopy(current_s)
    best_h_so_far = current_h
    best_path_so_far = [copy.deepcopy(current_s)]

    temp = initial_temp
    iteration = 0
    no_improve_count = 0 # Counts iterations without finding a new *best_s_so_far*
    max_no_improve_restart = 500 # If best_s_so_far doesn't improve for this many iterations, consider restart
    max_iterations = 5000

    while temp > min_temp and iteration < max_iterations:
        if time.time() - start_time > time_limit:
            # print("SA Timeout: Returning best path found so far.")
            break # Exit main loop on timeout

        if is_goal(current_s):
            # print("SA: Goal reached during annealing!")
            # Update best path if current path to goal is better
            if current_h < best_h_so_far: # Should be true if current_s is goal
                 best_s_so_far = copy.deepcopy(current_s)
                 best_h_so_far = current_h
                 best_path_so_far = path[:]
            break # Exit main loop if goal reached

        neighbors = get_neighbors(current_s)
        if not neighbors: break # Stuck

        # Evaluate neighbors and select one (can be simplified for random choice)
        # For 8-puzzle, all neighbors are usually equally easy to generate
        chosen_neighbor_s = random.choice(neighbors)
        chosen_neighbor_h = manhattan_distance(chosen_neighbor_s)

        if chosen_neighbor_h == float('inf'): # Skip invalid neighbor
            iteration += 1
            no_improve_count +=1
            continue

        delta_h = chosen_neighbor_h - current_h

        if delta_h < 0: # Always accept better state
            current_s = chosen_neighbor_s
            current_h = chosen_neighbor_h
            path.append(copy.deepcopy(current_s))
            if current_h < best_h_so_far:
                best_s_so_far = copy.deepcopy(current_s)
                best_h_so_far = current_h
                best_path_so_far = path[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif temp > 0 and random.random() < math.exp(-delta_h / temp): # Accept worse state with probability
            current_s = chosen_neighbor_s
            current_h = chosen_neighbor_h
            path.append(copy.deepcopy(current_s))
            no_improve_count += 1 # Didn't improve best_s_so_far
        else: # Did not move
            no_improve_count += 1

        # Optional: Restart from best_s_so_far if stuck in a local optimum for too long
        if no_improve_count >= max_no_improve_restart:
            # print(f"SA Restarting from best_s_so_far at iteration {iteration}")
            current_s = copy.deepcopy(best_s_so_far) # Go back to overall best state
            current_h = best_h_so_far
            path = best_path_so_far[:] # Reset current path to the best path found
            # temp = initial_temp * 0.8 # Optionally reset temperature partially
            no_improve_count = 0


        temp *= cooling_rate # Geometric cooling
        iteration += 1

        # Debug output
        # if iteration % 100 == 0:
        #     print(f"SA Iteration {iteration}: temp={temp:.2f}, current_h={current_h}, best_h={best_h_so_far}, path_len={len(path)}")

    # After loop finishes (timeout, min_temp, max_iter, or goal)
    # Attempt to complete path from best_s_so_far to goal using A* if not already goal
    if not is_goal(best_s_so_far):
        # print("SA: Attempting A* completion from best state found.")
        remaining_time_for_astar = max(1, time_limit - (time.time() - start_time)) # Use remaining time for A*
        if is_valid_state_for_solve(best_s_so_far): # Ensure best_s_so_far is valid before A*
            astar_completion_path = astar(best_s_so_far, time_limit=remaining_time_for_astar)
            if astar_completion_path and len(astar_completion_path) > 0:
                # Combine SA path up to best_s_so_far with A* completion
                # Ensure no duplication of best_s_so_far
                if best_path_so_far and best_path_so_far[-1] == astar_completion_path[0]:
                    return best_path_so_far[:-1] + astar_completion_path
                else: # This case implies best_path_so_far was empty or ended differently
                    return best_path_so_far + astar_completion_path # May need adjustment
        # If A* fails or best_s_so_far invalid, return the best path found by SA alone
    return best_path_so_far


# --- Beam Search ---
def beam_search(start_node_state, beam_width=5, time_limit=30):
    start_time = time.time()
    # Basic validation
    if not is_valid_state_for_solve(start_node_state): return None
    if is_goal(start_node_state): return [start_node_state]

    initial_h = manhattan_distance(start_node_state)
    if initial_h == float('inf'): return None

    # Beam stores tuples: (state, path, heuristic_value)
    beam = [(start_node_state, [start_node_state], initial_h)]
    visited_tuples_global = {state_to_tuple(start_node_state)} # Track all visited states
    best_goal_path_found = None # Store best goal path

    max_iterations = 100 # Limit iterations

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            # print("Beam Search Timeout")
            return best_goal_path_found # Return best goal found so far

        next_level_candidates = []

        # Generate successors for states in beam
        for s_beam, path_beam, _ in beam:
            for neighbor_s_beam in get_neighbors(s_beam):
                neighbor_tuple_beam = state_to_tuple(neighbor_s_beam)
                if neighbor_tuple_beam is None: continue

                if neighbor_tuple_beam not in visited_tuples_global:
                    visited_tuples_global.add(neighbor_tuple_beam)
                    new_path_beam = path_beam + [neighbor_s_beam]

                    # Check for goal
                    if is_goal(neighbor_s_beam):
                        if best_goal_path_found is None or len(new_path_beam) < len(best_goal_path_found):
                            best_goal_path_found = new_path_beam
                            # print(f"Beam search found goal path length {len(new_path_beam)}") # Optional debug

                    h_neighbor_beam = manhattan_distance(neighbor_s_beam)
                    if h_neighbor_beam == float('inf'): continue

                    next_level_candidates.append((neighbor_s_beam, new_path_beam, h_neighbor_beam))

        if not next_level_candidates: break # No new states

        # Sort candidates by heuristic (best first)
        next_level_candidates.sort(key=lambda x: x[2])

        # Select top 'beam_width' for next beam
        beam = next_level_candidates[:beam_width]

        if not beam: break # Beam empty

    return best_goal_path_found # Return best goal path found, or None


# --- AND-OR Search (Simplified as DFS with path cycle check) ---
def _and_or_recursive(state_ao, path_ao, visited_ao_tuples, start_time_ao, time_limit_ao, depth_ao, max_depth_ao=50):
    state_tuple_ao = state_to_tuple(state_ao)
    if state_tuple_ao is None: return "Fail", None

    # Base Cases
    if time.time() - start_time_ao > time_limit_ao: return "Timeout", None
    if depth_ao > max_depth_ao: return "Fail", None
    if is_goal(state_ao): return "Solved", path_ao[:]

    # Cycle detection (within current path)
    if state_tuple_ao in visited_ao_tuples:
        return "Fail", None

    visited_ao_tuples.add(state_tuple_ao) # Add to current path visited set

    # OR node logic (try neighbors)
    for neighbor_ao in get_neighbors(state_ao):
        # Recursive call for the neighbor
        status_ao, solution_path_ao = _and_or_recursive(
            neighbor_ao, path_ao + [neighbor_ao],
            visited_ao_tuples, # Pass same set for path cycle check
            start_time_ao, time_limit_ao, depth_ao + 1, max_depth_ao
        )

        # Check result
        if status_ao == "Timeout":
             visited_ao_tuples.remove(state_tuple_ao) # Backtrack visited
             return "Timeout", None
        if status_ao == "Solved":
             visited_ao_tuples.remove(state_tuple_ao) # Backtrack visited
             return "Solved", solution_path_ao

    # If all branches failed, backtrack
    visited_ao_tuples.remove(state_tuple_ao)
    return "Fail", None

def and_or_search(start_node_state, time_limit=30, max_depth=50):
    """AND-OR Search (simplified as DFS with path cycle check)."""
    start_time = time.time()
    initial_visited_path = set() # Set for path cycle check
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    initial_visited_path.add(init_tuple) # Add start node

    status, solution_path = _and_or_recursive(
        start_node_state, [start_node_state],
        initial_visited_path,
        start_time, time_limit, 0, max_depth
    )

    if status == "Solved":
        return solution_path
    # else: print(f"AND-OR Search Status: {status}")
    return None


# --- Backtracking (Meta-Algorithm) ---
def backtracking_search(selected_sub_algorithm_name, max_attempts=50, time_limit_overall=60):
    """
    Meta-algorithm: Generates random solvable states and applies a sub-algorithm.
    """
    start_time_overall = time.time()

    # Map algorithm names to functions
    algo_map = {
        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 'Greedy': greedy,
        'IDA*': ida_star,
        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
        'Beam Search': beam_search, 'AND-OR': and_or_search,
        'Sensorless': sensorless_search,
        'Unknown': sensorless_search,
        'Q-Learning': q_learning_train_and_solve,
        'GA': genetic_algorithm_solve, # Add GA
    }
    sub_algo_func = algo_map.get(selected_sub_algorithm_name)
    if not sub_algo_func:
        return None, f"Sub-algorithm '{selected_sub_algorithm_name}' not supported for Backtracking.", None, None

    # Determine Time Limit Per Attempt
    slow_algos = ['IDA*', 'Sensorless', 'Unknown', 'IDS', 'Q-Learning', 'GA'] # Added GA
    default_time_per_attempt = 15
    slow_time_per_attempt = 60
    if max_attempts <= 0: max_attempts = 1
    base_time_limit_per_sub_solve = time_limit_overall / max_attempts
    min_time_per_attempt = 1.0 if selected_sub_algorithm_name not in slow_algos else 10.0
    max_time_per_attempt = default_time_per_attempt if selected_sub_algorithm_name not in slow_algos else slow_time_per_attempt
    time_limit_per_sub_solve = max(min_time_per_attempt, base_time_limit_per_sub_solve)
    time_limit_per_sub_solve = min(time_limit_per_sub_solve, max_time_per_attempt)

    # print(f"Backtracking using {selected_sub_algorithm_name} with time limit/attempt: {time_limit_per_sub_solve:.2f}s")

    for attempts_count in range(max_attempts): # Renamed to avoid conflict
        if time.time() - start_time_overall > time_limit_overall:
            # print("Backtracking: Global timeout.")
            return None, "Backtracking: Global timeout.", None, None

        # Generate Random Solvable State
        current_attempt_2d_start = None
        while True:
             nums = list(range(GRID_SIZE * GRID_SIZE))
             random.shuffle(nums)
             state_try = [nums[i*GRID_SIZE:(i+1)*GRID_SIZE] for i in range(GRID_SIZE)]
             if is_solvable(state_try): # Check against fixed goal
                  current_attempt_2d_start = state_try
                  break
        # print(f"Attempt {attempts_count+1}: Trying solvable state: {current_attempt_2d_start}")

        # Prepare and Run Sub-Algorithm
        algo_params = [current_attempt_2d_start]
        sub_algo_func_varnames = sub_algo_func.__code__.co_varnames[:sub_algo_func.__code__.co_argcount]

        if 'time_limit' in sub_algo_func_varnames:
            algo_params.append(time_limit_per_sub_solve)
        elif selected_sub_algorithm_name == 'DFS' and 'max_depth' in sub_algo_func_varnames:
            algo_params.append(30)
        elif selected_sub_algorithm_name == 'IDS' and 'max_depth_limit' in sub_algo_func_varnames:
             algo_params.append(30)
        elif selected_sub_algorithm_name == 'Stochastic Hill' and 'max_iter_no_improve' in sub_algo_func_varnames:
             algo_params.append(500)
        elif selected_sub_algorithm_name == 'Beam Search' and 'beam_width' in sub_algo_func_varnames:
             algo_params.append(5)
        # Q-Learning & GA take time_limit, handled above

        path_from_sub_algo = None
        action_plan_from_sub_algo = None
        belief_size_sub = 0

        try:
            if selected_sub_algorithm_name in ['Sensorless', 'Unknown']:
                action_plan_from_sub_algo, belief_size_sub = sub_algo_func(*algo_params)
                if action_plan_from_sub_algo is not None:
                    path_from_sub_algo = execute_plan(current_attempt_2d_start, action_plan_from_sub_algo)
                    if path_from_sub_algo is None: action_plan_from_sub_algo = None
            else: # Standard solvers including Q-Learning and GA
                path_from_sub_algo = sub_algo_func(*algo_params)

        except Exception as e:
            print(f"Error in sub-algorithm '{selected_sub_algorithm_name}': {e}")
            # traceback.print_exc()
            continue

        # Check if Sub-Algorithm Succeeded
        if path_from_sub_algo and len(path_from_sub_algo) > 0:
            sub_success = False
            if selected_sub_algorithm_name in ['Sensorless', 'Unknown']:
                sub_success = (action_plan_from_sub_algo is not None)
            else:
                 if isinstance(path_from_sub_algo[-1], list) and is_valid_state_for_solve(path_from_sub_algo[-1]): # Check validity before is_goal
                    sub_success = is_goal(path_from_sub_algo[-1])

            if sub_success:
                return path_from_sub_algo, None, current_attempt_2d_start, action_plan_from_sub_algo

    msg = f"Backtracking: No solution found by {selected_sub_algorithm_name} after {max_attempts} attempts." # Use max_attempts
    if time.time() - start_time_overall > time_limit_overall:
        msg = "Backtracking: Global timeout likely occurred. " + msg
    return None, msg, None, None


# ----- Sensorless Search / Unknown Env Functions -----
def generate_belief_states(partial_state_template):
    flat_state_template = []
    unknown_positions_indices_flat = []
    known_numbers_set = set()
    is_template_with_unknowns = False

    if not isinstance(partial_state_template, list) or len(partial_state_template) != GRID_SIZE:
        return []

    for r_idx, r_val in enumerate(partial_state_template):
        if not isinstance(r_val, list) or len(r_val) != GRID_SIZE:
            return []
        for c_idx, tile in enumerate(r_val):
            flat_state_template.append(tile)
            if tile is None:
                is_template_with_unknowns = True
                unknown_positions_indices_flat.append(r_idx * GRID_SIZE + c_idx)
            elif isinstance(tile, int):
                if not (0 <= tile < GRID_SIZE * GRID_SIZE):
                    return []
                if tile in known_numbers_set:
                    return []
                known_numbers_set.add(tile)
            else:
                 return []

    if len(flat_state_template) != GRID_SIZE * GRID_SIZE:
        return []

    if not is_template_with_unknowns:
        if is_valid_state_for_solve(partial_state_template) and is_solvable(partial_state_template):
            return [copy.deepcopy(partial_state_template)]
        else:
            return []

    all_possible_tiles = list(range(GRID_SIZE * GRID_SIZE))
    missing_numbers_to_fill = [num for num in all_possible_tiles if num not in known_numbers_set]

    if len(missing_numbers_to_fill) != len(unknown_positions_indices_flat):
        return []

    belief_states_generated = []
    for perm_fill_nums in itertools.permutations(missing_numbers_to_fill):
        new_flat_state_filled = list(flat_state_template)
        for i, unknown_idx in enumerate(unknown_positions_indices_flat):
            new_flat_state_filled[unknown_idx] = perm_fill_nums[i]

        state_2d_filled = []
        valid_2d = True
        if len(new_flat_state_filled) == GRID_SIZE * GRID_SIZE:
            for r_idx_fill in range(GRID_SIZE):
                row_start = r_idx_fill * GRID_SIZE
                state_2d_filled.append(new_flat_state_filled[row_start : row_start + GRID_SIZE])
        else: valid_2d = False

        if valid_2d and is_valid_state_for_solve(state_2d_filled):
             if is_solvable(state_2d_filled):
                belief_states_generated.append(state_2d_filled)
    return belief_states_generated


def is_belief_state_goal(list_of_belief_states):
    if not list_of_belief_states: return False
    for state_b in list_of_belief_states:
        if not is_goal(state_b): return False
    return True

def apply_action_to_belief_states(list_of_belief_states, action_str):
    new_belief_states_set = set()
    for state_b in list_of_belief_states:
        next_s_b_list = apply_action_to_state(state_b, action_str)
        if next_s_b_list:
            next_s_b_tuple = state_to_tuple(next_s_b_list)
            if next_s_b_tuple:
                 new_belief_states_set.add(next_s_b_tuple)
    return [tuple_to_list(s_tuple) for s_tuple in new_belief_states_set if s_tuple is not None]


def sensorless_search(start_belief_state_template, time_limit=60):
    start_time = time.time()
    initial_belief_set_lists = generate_belief_states(start_belief_state_template)

    if not initial_belief_set_lists:
        return None, 0
    initial_belief_set_size = len(initial_belief_set_lists)

    try:
        initial_belief_tuples = {state_to_tuple(s) for s in initial_belief_set_lists}
        if None in initial_belief_tuples: return None, initial_belief_set_size
    except Exception as e: return None, initial_belief_set_size

    queue_sensorless = deque([ ( [], initial_belief_tuples ) ])
    visited_belief_sets = {frozenset(initial_belief_tuples)}

    possible_actions = ['Up', 'Down', 'Left', 'Right']
    max_plan_length = 30
    nodes_explored = 0
    last_processed_bs_size = initial_belief_set_size

    while queue_sensorless:
        nodes_explored += 1
        if time.time() - start_time > time_limit:
            current_bs_size_on_timeout = len(queue_sensorless[0][1]) if queue_sensorless else last_processed_bs_size
            return None, current_bs_size_on_timeout

        action_plan, current_bs_tuples = queue_sensorless.popleft()
        last_processed_bs_size = len(current_bs_tuples)

        if len(action_plan) > max_plan_length: continue

        current_bs_lists = [tuple_to_list(s) for s in current_bs_tuples if s is not None]
        if not current_bs_lists or len(current_bs_lists) != len(current_bs_tuples): continue

        if is_belief_state_goal(current_bs_lists):
            return action_plan, len(current_bs_lists)

        for action_s in possible_actions:
            next_bs_lists = apply_action_to_belief_states(current_bs_lists, action_s)
            if not next_bs_lists: continue
            try:
                next_bs_tuples = {state_to_tuple(s) for s in next_bs_lists}
                if None in next_bs_tuples or not next_bs_tuples: continue
                next_bs_frozenset = frozenset(next_bs_tuples)
            except Exception: continue

            if next_bs_frozenset not in visited_belief_sets:
                visited_belief_sets.add(next_bs_frozenset)
                new_plan_s = action_plan + [action_s]
                queue_sensorless.append((new_plan_s, next_bs_tuples))
    return None, last_processed_bs_size


# --- Helper for visualizing plans ---
def execute_plan(start_state, action_plan):
    if start_state is None or not isinstance(start_state, list): return None
    if not is_valid_state_for_solve(start_state):
        # print(f"Execute plan called with invalid start state: {start_state}")
        return [start_state] # Return path containing only invalid start

    current_state = copy.deepcopy(start_state)
    state_sequence = [current_state]
    if not action_plan: return state_sequence

    for action in action_plan:
        next_state = apply_action_to_state(current_state, action)
        if next_state is None:
             # print(f"Error during plan execution: Action '{action}' failed on state {current_state}")
             return state_sequence
        current_state = next_state
        state_sequence.append(copy.deepcopy(current_state))
    return state_sequence


# --- AC3 Helper Functions ---
def find_first_empty_cell(state):
    if not isinstance(state, list): return None
    for r in range(GRID_SIZE):
        if not isinstance(state[r], list): return None
        for c in range(GRID_SIZE):
            try:
                if state[r][c] is None:
                    return r, c
            except IndexError: return None
    return None

def get_used_numbers(state):
    used = set()
    if not isinstance(state, list): return used
    for r in range(GRID_SIZE):
        if not isinstance(state[r], list): continue
        for c in range(GRID_SIZE):
            try:
                tile = state[r][c]
                if tile is not None and isinstance(tile, int):
                    used.add(tile)
            except IndexError: continue
    return used

def ac3_fill_state(current_state_template):
    state_copy = copy.deepcopy(current_state_template)
    find_result = find_first_empty_cell(state_copy)

    if find_result is None:
        if is_valid_state_for_solve(state_copy) and is_solvable(state_copy):
            return state_copy
        else:
            return None

    r, c = find_result
    used_nums = get_used_numbers(state_copy)
    all_nums = set(range(GRID_SIZE * GRID_SIZE))
    available_nums = list(all_nums - used_nums)

    for num in available_nums:
        state_copy[r][c] = num
        result = ac3_fill_state(state_copy)
        if result is not None:
            return result
        state_copy[r][c] = None
    return None


# --- Q-Learning Implementation ---
def q_learning_train_and_solve(start_node_state, time_limit=60):
    start_time = time.time()
    goal_state_tuple = state_to_tuple(goal_state_fixed_global)

    if not is_valid_state_for_solve(start_node_state):
        print("Q-Learning Error: Start state is invalid/incomplete.")
        return None
    if not is_solvable(start_node_state):
        print("Q-Learning Error: Start state is unsolvable.")
        return None
    if goal_state_tuple is None:
        print("Q-Learning Error: Goal state is invalid.")
        return None

    start_node_tuple = state_to_tuple(start_node_state)
    if start_node_tuple is None: return None

    learning_rate = 0.1
    discount_factor = 0.95
    epsilon_start = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05
    num_episodes = 1500
    max_steps_per_episode = 300

    q_table = defaultdict(lambda: defaultdict(float))
    epsilon = epsilon_start
    print(f"Q-Learning: Starting training ({num_episodes} episodes, time limit {time_limit}s)...")

    training_episodes_completed = 0
    for episode in range(num_episodes):
        if time.time() - start_time > time_limit:
            print(f"Q-Learning: Training timeout during episode {episode+1}/{num_episodes}.")
            break
        training_episodes_completed = episode + 1
        current_state_list = copy.deepcopy(start_node_state)

        for step in range(max_steps_per_episode):
            current_state_tuple = state_to_tuple(current_state_list)
            if current_state_tuple is None: break
            if current_state_tuple == goal_state_tuple: break

            valid_actions = get_valid_actions(current_state_list)
            if not valid_actions: break

            chosen_action = None
            if random.random() < epsilon:
                chosen_action = random.choice(valid_actions)
            else:
                current_q_values = q_table[current_state_tuple]
                valid_q_values = {act: current_q_values.get(act, 0.0) for act in valid_actions}
                if valid_q_values:
                    max_q = max(valid_q_values.values())
                    best_actions = [act for act, q in valid_q_values.items() if q == max_q]
                    chosen_action = random.choice(best_actions)
                else:
                    chosen_action = random.choice(valid_actions)
            if chosen_action is None: break

            next_state_list = apply_action_to_state(current_state_list, chosen_action)
            if next_state_list is None:
                continue

            next_state_tuple = state_to_tuple(next_state_list)
            if next_state_tuple is None: continue

            reward = -1
            is_terminal = False
            if next_state_tuple == goal_state_tuple:
                reward = 100
                is_terminal = True

            old_q_value = q_table[current_state_tuple][chosen_action]
            next_q_state_values = q_table[next_state_tuple]
            max_q_next = 0.0
            if not is_terminal and next_q_state_values:
                max_q_next = max(next_q_state_values.values())

            new_q_value = old_q_value + learning_rate * (reward + discount_factor * max_q_next - old_q_value)
            q_table[current_state_tuple][chosen_action] = new_q_value
            current_state_list = next_state_list
            if is_terminal: break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode + 1) % (num_episodes // 10 or 1) == 0:
             print(f"  Q-Learn Episode {episode+1}/{num_episodes}. Epsilon={epsilon:.3f}. Q-table size={len(q_table)}")

    time_after_train = time.time()
    print(f"Q-Learning: Training finished ({training_episodes_completed} episodes). Time: {time_after_train - start_time:.3f}s. Q-table size: {len(q_table)} states.")

    print("Q-Learning: Extracting path...")
    path = [start_node_state]
    current_state_list = copy.deepcopy(start_node_state)
    max_path_steps = max(500, max_steps_per_episode * 2)

    for step in range(max_path_steps):
        if time.time() - start_time > time_limit:
            print("Q-Learning: Path extraction timeout.")
            return path

        current_state_tuple = state_to_tuple(current_state_list)
        if current_state_tuple is None:
            print("Q-Learning Error: Invalid state tuple during path extraction.")
            return path

        if current_state_tuple == goal_state_tuple:
            print(f"Q-Learning: Path found in {len(path)-1} steps.")
            return path

        valid_actions = get_valid_actions(current_state_list)
        if not valid_actions:
            print("Q-Learning Warning: Stuck during path extraction (no valid actions).")
            return path

        current_q_values = q_table[current_state_tuple]
        valid_q_action_values = {act: current_q_values.get(act, -float('inf')) for act in valid_actions}

        if not valid_q_action_values or all(q == -float('inf') for q in valid_q_action_values.values()):
             # print(f"Q-Learning Warning: No known positive Q-values for state {current_state_tuple}. Choosing random valid action.")
             best_action = random.choice(valid_actions)
        else:
            max_q_val = -float('inf') # Changed variable name to avoid conflict
            best_actions_list = []    # Changed variable name
            for act, q_val_act in valid_q_action_values.items(): # Changed variable name
                 if q_val_act > max_q_val:
                      max_q_val = q_val_act
                      best_actions_list = [act]
                 elif q_val_act == max_q_val:
                      best_actions_list.append(act)
            if not best_actions_list:
                 print("Q-Learning Error: Could not find best action despite having Q-values.")
                 return path
            best_action = random.choice(best_actions_list)

        next_state_list = apply_action_to_state(current_state_list, best_action)
        if next_state_list is None:
            print(f"Q-Learning Error: Applying best action '{best_action}' failed during extraction.")
            return path

        path.append(next_state_list)
        current_state_list = next_state_list

    print(f"Q-Learning: Path extraction reached max steps ({max_path_steps}) without finding goal.")
    return path

# --- Genetic Algorithm Implementation ---
# Helper structure for individuals in GA
class GAIndividual:
    def __init__(self, moves):
        self.moves = list(moves) # Sequence of action strings
        self.fitness = 0.0

    def __lt__(self, other): # For sorting by fitness
        return self.fitness < other.fitness # Note: sorte reverse=True for max fitness

    def __repr__(self):
        return f"Ind(fit={self.fitness:.2f}, moves_len={len(self.moves)}: {self.moves[:10]}{'...' if len(self.moves)>10 else ''})"

# GA Configuration Constants
GA_POPULATION_SIZE = 100
GA_NUM_GENERATIONS = 100
GA_MAX_INITIAL_MOVES = 35
GA_MAX_MOVES_PER_INDIVIDUAL = 70
GA_MUTATION_RATE = 0.1
GA_POINT_MUTATION_RATE = 0.05
GA_CROSSOVER_RATE = 0.8
GA_TOURNAMENT_SIZE = 5
GA_ELITISM_COUNT = 5

def _ga_calculate_fitness(individual, start_state_list, goal_state_tuple_ga):
    """Calculates fitness for an individual (sequence of moves).
    Returns: (fitness_value, list_of_moves_to_goal_if_any_or_None)
    """
    current_s_list = copy.deepcopy(start_state_list)
    path_taken_len = 0
    goal_reached_at_step = -1
    actual_moves_to_goal = None # Will store the truncated moves if goal is reached

    if not is_valid_state_for_solve(current_s_list):
        return 0.0, None

    if not individual.moves: # Handle empty move sequence
        md = manhattan_distance(current_s_list)
        if md == float('inf'): return 0.0, None
        max_md_approx = 30.0
        fitness = (max_md_approx - md) / max_md_approx if max_md_approx > 0 else 0.5
        return max(0.01, fitness), None


    for i, move_action in enumerate(individual.moves):
        next_s_list = apply_action_to_state(current_s_list, move_action)
        if next_s_list is None: # Invalid move
            md_before_fail = manhattan_distance(current_s_list)
            if md_before_fail == float('inf'): return 0.0, None
            return 1.0 / (1.0 + md_before_fail + (len(individual.moves) - i) * 5.0), None

        current_s_list = next_s_list
        path_taken_len += 1
        current_s_tuple = state_to_tuple(current_s_list)

        if current_s_tuple is None:
            return 0.0, None

        if current_s_tuple == goal_state_tuple_ga:
            goal_reached_at_step = path_taken_len
            actual_moves_to_goal = individual.moves[:path_taken_len] # Truncate moves
            break
    
    if not is_valid_state_for_solve(current_s_list):
         return 0.0, None

    if goal_reached_at_step != -1 and actual_moves_to_goal is not None:
        # Higher fitness for shorter paths to goal
        return 1000.0 + (GA_MAX_MOVES_PER_INDIVIDUAL - goal_reached_at_step), actual_moves_to_goal
    else:
        md = manhattan_distance(current_s_list)
        if md == float('inf'): return 0.0, None
        max_md_approx = 30.0
        fitness = (max_md_approx - md) / max_md_approx if max_md_approx > 0 else 0.5
        fitness -= (len(individual.moves) / GA_MAX_MOVES_PER_INDIVIDUAL if GA_MAX_MOVES_PER_INDIVIDUAL > 0 else 0) * 0.1
        return max(0.01, fitness), None


def _ga_initialize_population(start_state_list, population_size, max_initial_moves):
    population = []
    for _ in range(population_size):
        current_s_temp = copy.deepcopy(start_state_list)
        moves_sequence = []
        # Ensure max_initial_moves is at least 1 for randint
        actual_max_initial_moves = max(1, max_initial_moves)
        num_moves_for_this_individual = random.randint(max(1, actual_max_initial_moves//2), actual_max_initial_moves)

        for _ in range(num_moves_for_this_individual):
            if not is_valid_state_for_solve(current_s_temp):
                break
            valid_actions = get_valid_actions(current_s_temp)
            if not valid_actions: break

            chosen_action = random.choice(valid_actions)
            next_s_temp = apply_action_to_state(current_s_temp, chosen_action)
            if next_s_temp is None: break

            moves_sequence.append(chosen_action)
            current_s_temp = next_s_temp

            if len(moves_sequence) >= GA_MAX_MOVES_PER_INDIVIDUAL: break
        population.append(GAIndividual(moves_sequence))
    return population

def _ga_tournament_selection(population, tournament_size):
    selected_parents = []
    # Ensure tournament_size is not larger than population size
    actual_tournament_size = min(tournament_size, len(population))
    if actual_tournament_size == 0: return [] # Cannot select if population or tournament size is 0

    for _ in range(len(population)):
        tournament = random.sample(population, actual_tournament_size)
        winner = max(tournament, key=lambda ind: ind.fitness)
        selected_parents.append(winner)
    return selected_parents

def _ga_crossover(parent1, parent2):
    if random.random() < GA_CROSSOVER_RATE:
        len1, len2 = len(parent1.moves), len(parent2.moves)
        if min(len1, len2) < 1: # If one parent is empty, crossover is tricky
            # Return copies, or one combined with other's part
            child1_moves = parent1.moves[:] + parent2.moves[len2//2:]
            child2_moves = parent2.moves[:] + parent1.moves[len1//2:]
        elif min(len1, len2) < 2 : # If one parent has only 1 move
            point1 = 0 if len1 == 1 else random.randint(1, len1 -1)
            point2 = 0 if len2 == 1 else random.randint(1, len2 -1)
            child1_moves = parent1.moves[:point1] + parent2.moves[point2:]
            child2_moves = parent2.moves[:point2] + parent1.moves[point1:]
        else: # Both parents have at least 2 moves
            point1 = random.randint(1, len1 -1)
            point2 = random.randint(1, len2 -1)
            child1_moves = parent1.moves[:point1] + parent2.moves[point2:]
            child2_moves = parent2.moves[:point2] + parent1.moves[point1:]

        child1_moves = child1_moves[:GA_MAX_MOVES_PER_INDIVIDUAL]
        child2_moves = child2_moves[:GA_MAX_MOVES_PER_INDIVIDUAL]
        return GAIndividual(child1_moves), GAIndividual(child2_moves)
    else:
        return GAIndividual(parent1.moves[:]), GAIndividual(parent2.moves[:])


def _ga_mutate(individual, point_mutation_rate, start_state_for_validation):
    mutated_flag = False
    if random.random() < GA_MUTATION_RATE:
        new_moves = list(individual.moves)

        for i in range(len(new_moves)):
            if random.random() < point_mutation_rate:
                possible_actions = ['Up', 'Down', 'Left', 'Right']
                current_move = new_moves[i]
                possible_new_moves = [m for m in possible_actions if m != current_move]
                if possible_new_moves:
                    new_moves[i] = random.choice(possible_new_moves)
                    mutated_flag = True

        if random.random() < 0.2:
            if random.random() < 0.5 and len(new_moves) < GA_MAX_MOVES_PER_INDIVIDUAL:
                if len(new_moves) > 0: # Only insert if there are existing moves to determine position
                    add_pos = random.randint(0, len(new_moves))
                    random_action = random.choice(['Up', 'Down', 'Left', 'Right'])
                    new_moves.insert(add_pos, random_action)
                    mutated_flag = True
                elif len(new_moves) == 0: # If empty, add one based on start state
                     valid_initial_actions = get_valid_actions(start_state_for_validation)
                     if valid_initial_actions:
                         new_moves.append(random.choice(valid_initial_actions))
                         mutated_flag = True

            elif len(new_moves) > 1:
                remove_pos = random.randint(0, len(new_moves) - 1)
                del new_moves[remove_pos]
                mutated_flag = True

        if mutated_flag:
            individual.moves = new_moves[:GA_MAX_MOVES_PER_INDIVIDUAL]

        if not individual.moves and mutated_flag :
            valid_initial_actions = get_valid_actions(start_state_for_validation)
            if valid_initial_actions:
                individual.moves = [random.choice(valid_initial_actions)]
    return individual


def genetic_algorithm_solve(start_node_state, time_limit=60):
    start_time_ga = time.time()
    if not is_valid_state_for_solve(start_node_state):
        print("GA Error: Start state is invalid.")
        return None
    if not is_solvable(start_node_state):
        print("GA Error: Start state is unsolvable.")
        return None

    goal_tuple_ga = state_to_tuple(goal_state_fixed_global)
    if goal_tuple_ga is None:
        print("GA Error: Goal state is invalid.")
        return None

    population = _ga_initialize_population(start_node_state, GA_POPULATION_SIZE, GA_MAX_INITIAL_MOVES)
    if not population:
        print("GA Error: Failed to initialize population.")
        return None

    # best_solution_individual stores an individual whose .moves are *exactly* the path to goal
    best_solution_individual = None
    best_fitness_overall = -float('inf')
    # best_individual_ever stores the individual from population with highest raw fitness
    best_individual_ever = population[0] if population else GAIndividual([])


    print(f"GA: Starting ({GA_NUM_GENERATIONS} generations, pop_size={GA_POPULATION_SIZE}, time_limit={time_limit}s)...")
    completed_generations = 0
    for generation in range(GA_NUM_GENERATIONS):
        completed_generations = generation + 1
        if time.time() - start_time_ga > time_limit:
            print(f"GA: Timeout during generation {generation + 1}/{GA_NUM_GENERATIONS}.")
            break

        for ind in population:
            # _ga_calculate_fitness now returns (fitness_value, moves_to_goal_if_any)
            fitness_val, moves_if_goal = _ga_calculate_fitness(ind, start_node_state, goal_tuple_ga)
            ind.fitness = fitness_val # Store raw fitness for sorting and selection

            if ind.fitness > best_fitness_overall:
                best_fitness_overall = ind.fitness
                best_individual_ever = copy.deepcopy(ind) # Stores the full chromosome individual

            if moves_if_goal is not None: # Goal was reached by this individual
                # The fitness_val is already the score for reaching the goal
                if best_solution_individual is None or fitness_val > best_solution_individual.fitness:
                    # Create/update best_solution_individual with the *truncated* moves
                    best_solution_individual = GAIndividual(moves_if_goal)
                    best_solution_individual.fitness = fitness_val # Store its achieving fitness
                    # print(f"GA Gen {generation+1}: New best goal solution! Fitness: {best_solution_individual.fitness:.2f}, Moves: {len(best_solution_individual.moves)}")

        # Early termination if a good enough goal-reaching solution is found
        if best_solution_individual is not None and \
           best_solution_individual.fitness >= 1000.0 + (GA_MAX_MOVES_PER_INDIVIDUAL - (GA_MAX_MOVES_PER_INDIVIDUAL * 0.5) ): # e.g. path length < half of max moves
             # print(f"GA Gen {generation+1}: Good solution found ({len(best_solution_individual.moves)} moves). Terminating early.")
             break

        new_population = []
        if not population:
            print("GA Warning: Population empty mid-generation.")
            break

        population.sort(key=lambda i: i.fitness, reverse=True)
        new_population.extend(population[:GA_ELITISM_COUNT])

        if not new_population and GA_ELITISM_COUNT > 0 and population:
             new_population.extend(population[:1])

        parents_for_offspring_pool = _ga_tournament_selection(population, GA_TOURNAMENT_SIZE)
        if not parents_for_offspring_pool:
            parents_for_offspring_pool = population[:max(1, len(population)//2)]

        attempt = 0
        max_attempts_to_fill_pop = GA_POPULATION_SIZE * 3 # Increased attempts
        while len(new_population) < GA_POPULATION_SIZE and attempt < max_attempts_to_fill_pop:
            if not parents_for_offspring_pool: break

            # Ensure there are at least two parents to choose from if possible
            if len(parents_for_offspring_pool) < 2 and len(parents_for_offspring_pool) > 0:
                 p1 = parents_for_offspring_pool[0]
                 p2 = parents_for_offspring_pool[0] # Use same parent if only one
            elif len(parents_for_offspring_pool) >= 2:
                p1_idx, p2_idx = random.sample(range(len(parents_for_offspring_pool)), 2)
                p1 = parents_for_offspring_pool[p1_idx]
                p2 = parents_for_offspring_pool[p2_idx]
            else: # No parents, shouldn't happen if fallback above works
                break

            child1, child2 = _ga_crossover(p1, p2)
            child1 = _ga_mutate(child1, GA_POINT_MUTATION_RATE, start_node_state)
            child2 = _ga_mutate(child2, GA_POINT_MUTATION_RATE, start_node_state)

            if len(new_population) < GA_POPULATION_SIZE: new_population.append(child1)
            if len(new_population) < GA_POPULATION_SIZE: new_population.append(child2)
            attempt +=1

        if len(new_population) < GA_POPULATION_SIZE and population: # Fill remaining with random from old pop
            needed = GA_POPULATION_SIZE - len(new_population)
            new_population.extend(random.sample(population, min(needed, len(population))))


        population = new_population
        if not population:
            print("GA Error: Population became empty. Terminating.")
            break

        if (generation + 1) % (GA_NUM_GENERATIONS // 10 or 1) == 0:
            current_best_gen_fitness = population[0].fitness if population else -1
            print(f"  GA Gen {generation+1}/{GA_NUM_GENERATIONS}. Best pop fit: {current_best_gen_fitness:.2f}. Overall best raw: {best_fitness_overall:.2f}. Pop: {len(population)}")
            if best_solution_individual:
                print(f"    Best goal sol moves: {len(best_solution_individual.moves)}, fitness: {best_solution_individual.fitness:.2f}")

    # --- After all generations ---
    time_after_ga = time.time()
    print(f"GA: Finished ({completed_generations} gens). Time: {time_after_ga - start_time_ga:.3f}s.")
    print(f"GA: Best fitness overall during run: {best_fitness_overall:.2f}")
    if best_individual_ever:
        print(f"GA: Best individual_ever moves (full chromosome): {best_individual_ever.moves[:20]}{'...' if len(best_individual_ever.moves) > 20 else ''}")

    final_solution_moves = None
    source_of_moves = "None"

    if best_solution_individual: # This individual's .moves are already truncated to goal
        print(f"GA: A goal-reaching solution was recorded (fitness {best_solution_individual.fitness:.2f}) with {len(best_solution_individual.moves)} moves.")
        final_solution_moves = best_solution_individual.moves
        source_of_moves = "best_solution_individual (truncated to goal)"
    elif best_individual_ever and best_individual_ever.moves :
        print(f"GA: No explicit goal-reaching solution recorded. Falling back to best_individual_ever (fitness {best_individual_ever.fitness:.2f}) with {len(best_individual_ever.moves)} moves.")
        final_solution_moves = best_individual_ever.moves # Use full chromosome here
        source_of_moves = "best_individual_ever (fallback - full chromosome)"
    else:
        print("GA: No valid move sequence found by GA at all.")
        return None # Return None if no path can be formed

    if final_solution_moves:
        # Ensure final_solution_moves is not empty before executing
        if not final_solution_moves:
            print(f"GA: Final selected moves from '{source_of_moves}' are empty. Cannot execute.")
            # Return a path containing only the start state if moves are empty
            if is_valid_state_for_solve(start_node_state): return [start_node_state]
            return None

        print(f"GA: Attempting to execute plan from '{source_of_moves}' with {len(final_solution_moves)} moves.")
        solution_path = execute_plan(start_node_state, final_solution_moves)

        if solution_path and len(solution_path) > 0 and solution_path[-1] is not None and is_valid_state_for_solve(solution_path[-1]):
            if is_goal(solution_path[-1]):
                print(f"GA: Path from '{source_of_moves}' (len {len(solution_path)-1}) successfully leads to goal: {solution_path[-1]}")
            else:
                # This case should be less frequent now for 'best_solution_individual'
                print(f"GA: Path from '{source_of_moves}' (len {len(solution_path)-1}) does NOT lead to goal. Final state: {solution_path[-1]}, MD: {manhattan_distance(solution_path[-1])}")
        elif solution_path and len(solution_path) > 0 and solution_path[-1] is None:
             print(f"GA: Path from '{source_of_moves}' resulted in a None state at the end. Moves: {final_solution_moves}")
        elif not solution_path: # execute_plan itself returned None
            print(f"GA: Error executing plan from '{source_of_moves}'. Moves: {final_solution_moves}")
            return None
        return solution_path
    else:
        print("GA: No move sequence selected for execution.")
        # Return a path containing only the start state if no moves can be executed
        if is_valid_state_for_solve(start_node_state):
            return [start_node_state]
        return None

# --- UI and Drawing Functions ---

def solver_selection_popup():
    """Popup for selecting sub-solver after AC3."""
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)

    title_font_popup = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_popup = title_font_popup.render("AC3: Select Solver Algorithm", True, INFO_COLOR)
    title_rect_popup = title_surf_popup.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf_popup, title_rect_popup)

    solver_algorithms = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Q-Learning', 'Q-Learning'), ('GA', 'Genetic Algorithm'),
    ]
    button_width_popup, button_height_popup = 150, 40
    button_margin_popup = 10
    columns_popup = 3
    num_rows_popup = (len(solver_algorithms) + columns_popup - 1) // columns_popup
    start_x_popup = (POPUP_WIDTH - (columns_popup * button_width_popup + (columns_popup - 1) * button_margin_popup)) // 2
    start_y_popup = title_rect_popup.bottom + 30

    button_rects_popup = {}
    algo_buttons_popup = []

    for idx, (algo_id_p, algo_name_p) in enumerate(solver_algorithms):
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
        mouse_pos_relative = (mouse_pos_screen[0] - popup_rect_on_screen.left,
                              mouse_pos_screen[1] - popup_rect_on_screen.top)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button_rect_popup.collidepoint(mouse_pos_relative): return None
                for algo_id_p, _, btn_rect_p in algo_buttons_popup:
                    if btn_rect_p.collidepoint(mouse_pos_relative):
                        selected_algorithm_name = algo_id_p; running_popup = False; break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return None

        popup_surface.fill(INFO_BG)
        pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
        popup_surface.blit(title_surf_popup, title_rect_popup)

        for algo_id_p, algo_name_p, btn_rect_p in algo_buttons_popup:
            is_hovered_p = btn_rect_p.collidepoint(mouse_pos_relative)
            btn_color_p = MENU_HOVER_COLOR if is_hovered_p else MENU_BUTTON_COLOR
            pygame.draw.rect(popup_surface, btn_color_p, btn_rect_p, border_radius=8)
            text_surf_p = BUTTON_FONT.render(algo_name_p, True, WHITE)
            popup_surface.blit(text_surf_p, text_surf_p.get_rect(center=btn_rect_p.center))

        is_hovered_cancel = cancel_button_rect_popup.collidepoint(mouse_pos_relative)
        cancel_color = MENU_HOVER_COLOR if is_hovered_cancel else RED
        pygame.draw.rect(popup_surface, cancel_color, cancel_button_rect_popup, border_radius=8)
        cancel_text_surf_p = BUTTON_FONT.render("Cancel", True, WHITE)
        popup_surface.blit(cancel_text_surf_p, cancel_text_surf_p.get_rect(center=cancel_button_rect_popup.center))

        screen.blit(popup_surface, popup_rect_on_screen)
        pygame.display.flip()
        clock_popup.tick(60)

    return selected_algorithm_name


def backtracking_selection_popup():
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)

    title_font_popup = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_popup = title_font_popup.render("Select Backtracking Sub-Algorithm", True, INFO_COLOR)
    title_rect_popup = title_surf_popup.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf_popup, title_rect_popup)

    sub_algorithms = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Sensorless', 'Sensorless Plan'),
        ('Unknown', 'Unknown Env'),
        ('Q-Learning', 'Q-Learning'), ('GA', 'Genetic Algorithm'),
    ]
    button_width_popup, button_height_popup = 150, 40
    button_margin_popup = 10
    columns_popup = 3
    num_rows_popup = (len(sub_algorithms) + columns_popup - 1) // columns_popup
    start_x_popup = (POPUP_WIDTH - (columns_popup * button_width_popup + (columns_popup - 1) * button_margin_popup)) // 2
    start_y_popup = title_rect_popup.bottom + 30

    button_rects_popup = {}
    algo_buttons_popup = []

    for idx, (algo_id_p, algo_name_p) in enumerate(sub_algorithms):
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
        mouse_pos_relative = (mouse_pos_screen[0] - popup_rect_on_screen.left,
                              mouse_pos_screen[1] - popup_rect_on_screen.top)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button_rect_popup.collidepoint(mouse_pos_relative): return None
                for algo_id_p, _, btn_rect_p in algo_buttons_popup:
                    if btn_rect_p.collidepoint(mouse_pos_relative):
                        selected_algorithm_name = algo_id_p; running_popup = False; break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return None

        popup_surface.fill(INFO_BG)
        pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
        popup_surface.blit(title_surf_popup, title_rect_popup)

        for algo_id_p, algo_name_p, btn_rect_p in algo_buttons_popup:
            is_hovered_p = btn_rect_p.collidepoint(mouse_pos_relative)
            btn_color_p = MENU_HOVER_COLOR if is_hovered_p else MENU_BUTTON_COLOR
            pygame.draw.rect(popup_surface, btn_color_p, btn_rect_p, border_radius=8)
            text_surf_p = BUTTON_FONT.render(algo_name_p, True, WHITE)
            popup_surface.blit(text_surf_p, text_surf_p.get_rect(center=btn_rect_p.center))

        is_hovered_cancel = cancel_button_rect_popup.collidepoint(mouse_pos_relative)
        cancel_color = MENU_HOVER_COLOR if is_hovered_cancel else RED
        pygame.draw.rect(popup_surface, cancel_color, cancel_button_rect_popup, border_radius=8)
        cancel_text_surf_p = BUTTON_FONT.render("Cancel", True, WHITE)
        popup_surface.blit(cancel_text_surf_p, cancel_text_surf_p.get_rect(center=cancel_button_rect_popup.center))

        screen.blit(popup_surface, popup_rect_on_screen)
        pygame.display.flip()
        clock_popup.tick(60)

    return selected_algorithm_name


scroll_y = 0
menu_surface = None
total_menu_height = 0

def draw_menu(show_menu_flag, mouse_pos_dm, current_selected_algorithm_dm):
    global scroll_y, menu_surface, total_menu_height
    menu_elements_dict = {}

    if not show_menu_flag:
        menu_button_rect_dm = pygame.Rect(10, 10, 50, 40)
        pygame.draw.rect(screen, MENU_COLOR, menu_button_rect_dm, border_radius=5)
        bar_width_dm, bar_height_dm, space_dm = 30, 4, 7
        start_x_dm = menu_button_rect_dm.centerx - bar_width_dm // 2
        start_y_dm = menu_button_rect_dm.centery - (bar_height_dm * 3 + space_dm * 2) // 2 + bar_height_dm // 2
        for i in range(3):
            pygame.draw.rect(screen, WHITE, (start_x_dm, start_y_dm + i * (bar_height_dm + space_dm), bar_width_dm, bar_height_dm), border_radius=2)
        menu_elements_dict['open_button'] = menu_button_rect_dm
        return menu_elements_dict

    algorithms_list_dm = [
        ('AC3', 'AC3 Preprocessing'),
        ('Backtracking', 'Backtracking Search'),
        ('Sensorless', 'Sensorless Plan'),
        ('Unknown', 'Unknown Env'),
        ('---', '--- Solvers ---'),
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Q-Learning', 'Q-Learning'), ('GA', 'Genetic Algorithm'),
    ]

    button_h_dm, padding_dm, button_margin_dm = 55, 10, 8
    total_menu_height = (len(algorithms_list_dm) * (button_h_dm + button_margin_dm)) - button_margin_dm + (2 * padding_dm)
    display_height_menu_surf = max(total_menu_height, HEIGHT)
    if menu_surface is None or menu_surface.get_height() != display_height_menu_surf:
        menu_surface = pygame.Surface((MENU_WIDTH, display_height_menu_surf))

    menu_surface.fill(MENU_COLOR)
    buttons_dict_dm = {}
    y_pos_current_button = padding_dm
    mouse_x_rel_dm, mouse_y_rel_dm = mouse_pos_dm[0], mouse_pos_dm[1] + scroll_y

    for algo_id_dm, algo_name_dm in algorithms_list_dm:
        button_rect_local_dm = pygame.Rect(padding_dm, y_pos_current_button, MENU_WIDTH - 2 * padding_dm, button_h_dm)

        if algo_id_dm == '---':
            pygame.draw.line(menu_surface, GRAY, (button_rect_local_dm.left + 5, button_rect_local_dm.centery),
                                                  (button_rect_local_dm.right - 5, button_rect_local_dm.centery), 1)
            text_surf_dm = INFO_FONT.render(algo_name_dm, True, GRAY)
            menu_surface.blit(text_surf_dm, text_surf_dm.get_rect(center=button_rect_local_dm.center))
        else:
            is_hover_dm = button_rect_local_dm.collidepoint(mouse_x_rel_dm, mouse_y_rel_dm)
            is_selected_dm = (current_selected_algorithm_dm == algo_id_dm)
            button_color_dm = MENU_SELECTED_COLOR if is_selected_dm else \
                              (MENU_HOVER_COLOR if is_hover_dm else MENU_BUTTON_COLOR)
            pygame.draw.rect(menu_surface, button_color_dm, button_rect_local_dm, border_radius=5)
            text_surf_dm = BUTTON_FONT.render(algo_name_dm, True, WHITE)
            menu_surface.blit(text_surf_dm, text_surf_dm.get_rect(center=button_rect_local_dm.center))
            buttons_dict_dm[algo_id_dm] = button_rect_local_dm
        y_pos_current_button += button_h_dm + button_margin_dm

    visible_menu_area_rect = pygame.Rect(0, scroll_y, MENU_WIDTH, HEIGHT)
    screen.blit(menu_surface, (0,0), visible_menu_area_rect)

    close_button_rect_dm = pygame.Rect(MENU_WIDTH - 40, 10, 30, 30)
    pygame.draw.rect(screen, RED, close_button_rect_dm, border_radius=5)
    cx_dm, cy_dm = close_button_rect_dm.center
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm - 7), (cx_dm + 7, cy_dm + 7), 3)
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm + 7), (cx_dm + 7, cy_dm - 7), 3)

    menu_elements_dict['close_button'] = close_button_rect_dm
    screen_buttons_dict = {}
    for algo_id, rect_local in buttons_dict_dm.items():
         screen_buttons_dict[algo_id] = rect_local.move(0, -scroll_y)
    menu_elements_dict['buttons'] = screen_buttons_dict
    menu_elements_dict['menu_area'] = pygame.Rect(0, 0, MENU_WIDTH, HEIGHT)

    if total_menu_height > HEIGHT:
        scrollbar_track_height = HEIGHT - 2*padding_dm
        scrollbar_height_val = max(20, scrollbar_track_height * (HEIGHT / total_menu_height))
        max_scroll_y_content = total_menu_height - HEIGHT
        scroll_ratio = scroll_y / max_scroll_y_content if max_scroll_y_content > 0 else 0
        scrollbar_track_y_start = padding_dm
        scrollbar_y_pos_thumb = scrollbar_track_y_start + scroll_ratio * (scrollbar_track_height - scrollbar_height_val)
        scrollbar_rect_dm = pygame.Rect(MENU_WIDTH - 10, scrollbar_y_pos_thumb, 6, scrollbar_height_val)
        pygame.draw.rect(screen, GRAY, scrollbar_rect_dm, border_radius=3)
        menu_elements_dict['scrollbar_rect'] = scrollbar_rect_dm
    return menu_elements_dict


def show_popup(message_str, title_str="Info"):
    popup_surface_sp = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface_sp.fill(INFO_BG)
    border_rect_sp = popup_surface_sp.get_rect()
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, border_rect_sp, 4, border_radius=10)

    title_font_sp = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_sp = title_font_sp.render(title_str, True, INFO_COLOR)
    title_rect_sp = title_surf_sp.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface_sp.blit(title_surf_sp, title_rect_sp)

    words_sp = message_str.replace('\n', ' \n ').split(' ')
    lines_sp = []
    current_line_sp = ""
    text_width_limit_sp = POPUP_WIDTH - 60
    line_height_sp = INFO_FONT.get_linesize()
    max_lines_display = (POPUP_HEIGHT - title_rect_sp.bottom - 70) // line_height_sp

    for word_sp in words_sp:
        if word_sp == '\n':
            lines_sp.append(current_line_sp)
            current_line_sp = ""
            continue
        test_line_sp = current_line_sp + (" " if current_line_sp else "") + word_sp
        line_width_sp, _ = INFO_FONT.size(test_line_sp)
        if line_width_sp <= text_width_limit_sp:
            current_line_sp = test_line_sp
        else:
            lines_sp.append(current_line_sp)
            current_line_sp = word_sp
    lines_sp.append(current_line_sp)

    text_start_y_sp = title_rect_sp.bottom + 20
    for i, line_text_sp in enumerate(lines_sp):
        if i >= max_lines_display:
            ellipsis_surf = INFO_FONT.render("...", True, BLACK)
            popup_surface_sp.blit(ellipsis_surf, ( (POPUP_WIDTH - ellipsis_surf.get_width()) // 2, text_start_y_sp + i * line_height_sp))
            break
        text_surf_sp = INFO_FONT.render(line_text_sp, True, BLACK)
        text_rect_sp = text_surf_sp.get_rect(centerx=POPUP_WIDTH // 2, top=text_start_y_sp + i * line_height_sp)
        popup_surface_sp.blit(text_surf_sp, text_rect_sp)

    ok_button_rect_sp = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, ok_button_rect_sp, border_radius=8)
    ok_text_surf_sp = BUTTON_FONT.render("OK", True, WHITE)
    ok_text_rect_sp = ok_text_surf_sp.get_rect(center=ok_button_rect_sp.center)
    popup_surface_sp.blit(ok_text_surf_sp, ok_text_rect_sp)

    popup_rect_on_screen_sp = popup_surface_sp.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(popup_surface_sp, popup_rect_on_screen_sp)
    pygame.display.flip()

    waiting_for_ok = True
    while waiting_for_ok:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos_screen = event.pos
                mouse_rel_x = mouse_pos_screen[0] - popup_rect_on_screen_sp.left
                mouse_rel_y = mouse_pos_screen[1] - popup_rect_on_screen_sp.top
                if ok_button_rect_sp.collidepoint(mouse_rel_x, mouse_rel_y):
                    waiting_for_ok = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    waiting_for_ok = False
        pygame.time.delay(20)


def draw_grid_and_ui(current_anim_state_dgui, show_menu_dgui, current_algo_name_dgui,
                     solver_after_ac3_dgui,
                     solve_times_dgui,
                     last_solved_run_info_dgui,
                     current_belief_state_size_dgui=None,
                     selected_cell_for_input_coords=None):
    screen.fill(WHITE)
    mouse_pos_dgui = pygame.mouse.get_pos()

    main_area_x_offset = MENU_WIDTH if show_menu_dgui else 0
    main_area_width_dgui = WIDTH - main_area_x_offset
    center_x_main_area = main_area_x_offset + main_area_width_dgui // 2

    top_row_y_grids = GRID_PADDING + 40
    grid_spacing_horizontal = GRID_PADDING * 1.5
    total_width_top_grids = 2 * GRID_DISPLAY_WIDTH + grid_spacing_horizontal
    start_x_top_grids = center_x_main_area - total_width_top_grids // 2

    initial_grid_x = start_x_top_grids
    goal_grid_x = start_x_top_grids + GRID_DISPLAY_WIDTH + grid_spacing_horizontal

    draw_state(initial_state, initial_grid_x, top_row_y_grids, "Initial State (Editable)",
               is_editable=True, selected_cell_coords=selected_cell_for_input_coords)
    draw_state(goal_state_fixed_global, goal_grid_x, top_row_y_grids, "Goal State",
               is_fixed_goal_display=True)

    bottom_row_y_start = top_row_y_grids + GRID_DISPLAY_WIDTH + GRID_PADDING + 60
    button_width_ctrl, button_height_ctrl = 140, 45
    buttons_start_x = main_area_x_offset + GRID_PADDING
    buttons_mid_y_anchor = bottom_row_y_start + GRID_DISPLAY_WIDTH // 2

    solve_button_y_pos = buttons_mid_y_anchor - button_height_ctrl * 2 - 16
    reset_solution_button_y_pos = solve_button_y_pos + button_height_ctrl + 8
    reset_all_button_y_pos = reset_solution_button_y_pos + button_height_ctrl + 8
    reset_initial_button_y_pos = reset_all_button_y_pos + button_height_ctrl + 8

    solve_button_rect_dgui = pygame.Rect(buttons_start_x, solve_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_solution_button_rect_dgui = pygame.Rect(buttons_start_x, reset_solution_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_all_button_rect_dgui = pygame.Rect(buttons_start_x, reset_all_button_y_pos, button_width_ctrl, button_height_ctrl)
    reset_initial_button_rect_dgui = pygame.Rect(buttons_start_x, reset_initial_button_y_pos, button_width_ctrl, button_height_ctrl)

    pygame.draw.rect(screen, RED, solve_button_rect_dgui, border_radius=5)
    screen.blit(BUTTON_FONT.render("SOLVE", True, WHITE), solve_button_rect_dgui.inflate(-20, -10))

    pygame.draw.rect(screen, BLUE, reset_solution_button_rect_dgui, border_radius=5)
    rs_text = BUTTON_FONT.render("Reset Solution", True, WHITE)
    screen.blit(rs_text, rs_text.get_rect(center=reset_solution_button_rect_dgui.center))

    pygame.draw.rect(screen, BLUE, reset_all_button_rect_dgui, border_radius=5)
    ra_text = BUTTON_FONT.render("Reset All", True, WHITE)
    screen.blit(ra_text, ra_text.get_rect(center=reset_all_button_rect_dgui.center))

    pygame.draw.rect(screen, BLUE, reset_initial_button_rect_dgui, border_radius=5)
    ri_text = BUTTON_FONT.render("Reset Display", True, WHITE)
    screen.blit(ri_text, ri_text.get_rect(center=reset_initial_button_rect_dgui.center))

    current_state_grid_x = buttons_start_x + button_width_ctrl + GRID_PADDING * 1.5
    current_state_grid_y = bottom_row_y_start
    current_state_title = f"Current ({current_algo_name_dgui}"
    if current_algo_name_dgui == 'AC3' and solver_after_ac3_dgui:
        current_state_title += f" -> {solver_after_ac3_dgui})"
    elif current_algo_name_dgui in ['Sensorless', 'Unknown']:
         if current_belief_state_size_dgui is not None and current_belief_state_size_dgui >= 0:
             current_state_title = f"Belief States: {current_belief_state_size_dgui}"
         else:
             current_state_title = "Belief State (Template)"
    elif current_algo_name_dgui.startswith("BT("):
         current_state_title = f"Current ({current_algo_name_dgui})"
    else:
        current_state_title += ")"
    draw_state(current_anim_state_dgui, current_state_grid_x, current_state_grid_y, current_state_title, is_current_anim_state=True)

    info_area_x_pos = current_state_grid_x + GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5
    info_area_y_pos = bottom_row_y_start
    info_area_w = max(150, (main_area_x_offset + main_area_width_dgui) - info_area_x_pos - GRID_PADDING)
    info_area_h = GRID_DISPLAY_WIDTH
    info_area_rect_dgui = pygame.Rect(info_area_x_pos, info_area_y_pos, info_area_w, info_area_h)

    pygame.draw.rect(screen, INFO_BG, info_area_rect_dgui, border_radius=8)
    pygame.draw.rect(screen, GRAY, info_area_rect_dgui, 2, border_radius=8)

    info_pad_x_ia, info_pad_y_ia = 15, 10
    line_h_ia = INFO_FONT.get_linesize() + 4
    current_info_y_draw = info_area_y_pos + info_pad_y_ia

    compare_title_surf_ia = TITLE_FONT.render("Solver Stats", True, BLACK)
    compare_title_x_ia = info_area_rect_dgui.centerx - compare_title_surf_ia.get_width() // 2
    screen.blit(compare_title_surf_ia, (compare_title_x_ia, current_info_y_draw))
    current_info_y_draw += compare_title_surf_ia.get_height() + 8

    if solve_times_dgui:
        sorted_times_list = sorted(solve_times_dgui.items(), key=lambda item: item[0])
        for algo_name_st, time_val_st in sorted_times_list:
            if current_info_y_draw + line_h_ia > info_area_y_pos + info_area_h - info_pad_y_ia:
                screen.blit(INFO_FONT.render("...", True, BLACK), (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
                break

            actions_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_actions")
            steps_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_steps")
            reached_goal_st = last_solved_run_info_dgui.get(f"{algo_name_st}_reached_goal", None)

            count_str_st = ""
            if actions_val_st is not None: count_str_st = f" ({actions_val_st} actions)"
            elif steps_val_st is not None: count_str_st = f" ({steps_val_st} steps)"

            goal_indicator_st = ""
            if reached_goal_st is False : goal_indicator_st = " (Not Goal)"
            elif (algo_name_st.startswith('Sensorless') or algo_name_st.startswith('Unknown') or \
                  (algo_name_st.startswith('BT(') and ('Sensorless' in algo_name_st or 'Unknown' in algo_name_st))) and reached_goal_st is None: # Adjusted BT check
                 goal_indicator_st = " (Plan Fail)"
            elif reached_goal_st is None and algo_name_st not in ['AC3', 'Backtracking']:
                 goal_indicator_st = " (?)"


            base_str_st = f"{algo_name_st}: {time_val_st:.3f}s"
            full_comp_str = base_str_st + count_str_st + goal_indicator_st

            max_text_width = info_area_w - 2 * info_pad_x_ia
            if INFO_FONT.size(full_comp_str)[0] > max_text_width:
                shortened_str = base_str_st + goal_indicator_st
                if INFO_FONT.size(shortened_str)[0] <= max_text_width:
                    full_comp_str = shortened_str
                else:
                    max_name_len = 18
                    name_part = algo_name_st
                    if len(name_part) > max_name_len: name_part = name_part[:max_name_len-3] + "..."
                    full_comp_str = f"{name_part}: {time_val_st:.3f}s{goal_indicator_st}"

            comp_surf_st = INFO_FONT.render(full_comp_str, True, BLACK)
            screen.blit(comp_surf_st, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
            current_info_y_draw += line_h_ia
    else:
        no_results_surf = INFO_FONT.render("(No results yet)", True, GRAY)
        screen.blit(no_results_surf, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))

    menu_elements_dgui = draw_menu(show_menu_dgui, mouse_pos_dgui, current_algo_name_dgui)
    initial_grid_rect_on_screen = pygame.Rect(initial_grid_x, top_row_y_grids, GRID_DISPLAY_WIDTH, GRID_DISPLAY_WIDTH)
    return {
        'solve_button': solve_button_rect_dgui,
        'reset_solution_button': reset_solution_button_rect_dgui,
        'reset_all_button': reset_all_button_rect_dgui,
        'reset_initial_button': reset_initial_button_rect_dgui,
        'menu': menu_elements_dgui,
        'initial_grid_area': initial_grid_rect_on_screen
    }


# --- Main Game Loop ---
def main():
    global scroll_y, initial_state

    current_state_for_animation = copy.deepcopy(initial_state)
    solution_path_anim = None
    action_plan_anim = None
    current_step_in_anim = 0
    is_solving_flag = False
    is_auto_animating_flag = False
    last_anim_step_time = 0
    show_algo_menu = False
    current_selected_algorithm = 'A*'
    solver_algorithm_to_run = None
    solver_name_for_stats = None
    all_solve_times = {}
    last_run_solver_info = {}
    game_clock = pygame.time.Clock()
    ui_elements_rects = {}
    running_main_loop = True
    backtracking_sub_algo_choice = None
    current_sensorless_belief_size_for_display = None
    selected_cell_for_input_coords = None

    while running_main_loop:
        mouse_pos_main = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_main_loop = False; break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_something_handled = False

                initial_grid_area_rect = ui_elements_rects.get('initial_grid_area')
                if initial_grid_area_rect and initial_grid_area_rect.collidepoint(mouse_pos_main):
                    grid_offset_x = initial_grid_area_rect.left
                    grid_offset_y = initial_grid_area_rect.top
                    clicked_col = (mouse_pos_main[0] - grid_offset_x) // CELL_SIZE
                    clicked_row = (mouse_pos_main[1] - grid_offset_y) // CELL_SIZE
                    if 0 <= clicked_row < GRID_SIZE and 0 <= clicked_col < GRID_SIZE:
                        selected_cell_for_input_coords = (clicked_row, clicked_col)
                    else:
                        selected_cell_for_input_coords = None
                    clicked_something_handled = True
                elif selected_cell_for_input_coords:
                     menu_area = ui_elements_rects.get('menu', {}).get('menu_area')
                     if not show_algo_menu or not (menu_area and menu_area.collidepoint(mouse_pos_main)):
                         selected_cell_for_input_coords = None

                if show_algo_menu and not clicked_something_handled:
                    menu_data = ui_elements_rects.get('menu', {})
                    menu_area = menu_data.get('menu_area')
                    if menu_area and menu_area.collidepoint(mouse_pos_main):
                        close_btn_rect = menu_data.get('close_button')
                        if close_btn_rect and close_btn_rect.collidepoint(mouse_pos_main):
                            show_algo_menu = False; clicked_something_handled = True

                        if not clicked_something_handled:
                            algo_buttons = menu_data.get('buttons', {})
                            for algo_id, btn_rect_screen in algo_buttons.items():
                                if algo_id != '---' and btn_rect_screen.collidepoint(mouse_pos_main):
                                    if current_selected_algorithm != algo_id:
                                        current_selected_algorithm = algo_id
                                        solver_algorithm_to_run = None
                                        solver_name_for_stats = None
                                        solution_path_anim = None; action_plan_anim = None
                                        current_step_in_anim = 0; is_auto_animating_flag = False
                                        is_solving_flag = False
                                        current_sensorless_belief_size_for_display = None
                                    show_algo_menu = False
                                    clicked_something_handled = True; break
                        if not clicked_something_handled:
                             clicked_something_handled = True

                elif not show_algo_menu and not clicked_something_handled:
                    menu_data = ui_elements_rects.get('menu', {})
                    open_btn_rect = menu_data.get('open_button')
                    if open_btn_rect and open_btn_rect.collidepoint(mouse_pos_main):
                        show_algo_menu = True; scroll_y = 0; clicked_something_handled = True

                if not clicked_something_handled:
                    solve_btn = ui_elements_rects.get('solve_button')
                    reset_sol_btn = ui_elements_rects.get('reset_solution_button')
                    reset_all_btn = ui_elements_rects.get('reset_all_button')
                    reset_disp_btn = ui_elements_rects.get('reset_initial_button')

                    if solve_btn and solve_btn.collidepoint(mouse_pos_main):
                        if not is_auto_animating_flag and not is_solving_flag:
                            should_start_solving = False
                            solver_algorithm_to_run = None
                            solver_name_for_stats = None
                            temp_state_for_solving = copy.deepcopy(initial_state)

                            if current_selected_algorithm == 'AC3':
                                has_empty_cells_ac3 = any(cell is None for row in temp_state_for_solving for cell in row)
                                ac3_processed_state = None
                                if has_empty_cells_ac3:
                                    filled_state_ac3 = ac3_fill_state(temp_state_for_solving)
                                    if filled_state_ac3:
                                        initial_state = filled_state_ac3
                                        ac3_processed_state = filled_state_ac3
                                        current_state_for_animation = copy.deepcopy(initial_state)
                                        show_popup("AC3: State auto-completed and validated.", "AC3 Success")
                                    else:
                                        show_popup("AC3: Could not find a solvable completion.", "AC3 Failed")
                                else:
                                    if is_valid_state_for_solve(temp_state_for_solving) and is_solvable(temp_state_for_solving):
                                        ac3_processed_state = temp_state_for_solving
                                    else:
                                         show_popup("AC3: Initial state is complete but invalid or unsolvable.", "AC3 Validation Failed")
                                if ac3_processed_state is not None:
                                    solver_choice = solver_selection_popup()
                                    if solver_choice:
                                        solver_algorithm_to_run = solver_choice
                                        solver_name_for_stats = f"AC3->{solver_choice}"
                                        should_start_solving = True
                            elif current_selected_algorithm == 'Backtracking':
                                sub_choice = backtracking_selection_popup()
                                if sub_choice:
                                    backtracking_sub_algo_choice = sub_choice
                                    solver_algorithm_to_run = 'Backtracking'
                                    should_start_solving = True
                            elif current_selected_algorithm in ['Sensorless', 'Unknown']:
                                is_valid_template = isinstance(temp_state_for_solving, list) and \
                                                    len(temp_state_for_solving) == GRID_SIZE and \
                                                    all(isinstance(r, list) and len(r)==GRID_SIZE for r in temp_state_for_solving) and \
                                                    all(isinstance(t, int) or t is None for r in temp_state_for_solving for t in r)
                                if is_valid_template:
                                    solver_algorithm_to_run = current_selected_algorithm
                                    solver_name_for_stats = current_selected_algorithm
                                    should_start_solving = True
                                else:
                                     show_popup("Initial state structure is invalid for Unknown/Sensorless.", "Invalid Template")
                            else:
                                if not is_valid_state_for_solve(temp_state_for_solving):
                                     show_popup(f"Initial state must be complete and valid (0-8) for {current_selected_algorithm}.", "Invalid/Incomplete State")
                                elif not is_solvable(temp_state_for_solving):
                                     show_popup(f"The current initial state is not solvable.", "Unsolvable State")
                                else:
                                    solver_algorithm_to_run = current_selected_algorithm
                                    solver_name_for_stats = current_selected_algorithm
                                    should_start_solving = True

                            if should_start_solving and solver_algorithm_to_run:
                                is_solving_flag = True
                                solution_path_anim = None; action_plan_anim = None
                                current_step_in_anim = 0; is_auto_animating_flag = False
                                current_sensorless_belief_size_for_display = None
                        clicked_something_handled = True
                    elif reset_sol_btn and reset_sol_btn.collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None; action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False; is_auto_animating_flag = False
                        current_sensorless_belief_size_for_display = None
                        clicked_something_handled = True
                    elif reset_all_btn and reset_all_btn.collidepoint(mouse_pos_main):
                        initial_state = copy.deepcopy(initial_state_fixed_global)
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None; action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False; is_auto_animating_flag = False
                        all_solve_times.clear()
                        last_run_solver_info.clear()
                        current_sensorless_belief_size_for_display = None
                        selected_cell_for_input_coords = None
                        clicked_something_handled = True
                    elif reset_disp_btn and reset_disp_btn.collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None; action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False; is_auto_animating_flag = False
                        current_sensorless_belief_size_for_display = None
                        clicked_something_handled = True

            elif event.type == pygame.MOUSEWHEEL and show_algo_menu:
                menu_data = ui_elements_rects.get('menu', {})
                menu_area = menu_data.get('menu_area')
                if menu_area and menu_area.collidepoint(mouse_pos_main) and total_menu_height > HEIGHT:
                    scroll_amount_mw = event.y * 35
                    max_scroll_val = max(0, total_menu_height - HEIGHT)
                    scroll_y = max(0, min(scroll_y - scroll_amount_mw, max_scroll_val))

            elif event.type == pygame.KEYDOWN:
                if not is_solving_flag and not is_auto_animating_flag and selected_cell_for_input_coords:
                    r, c = selected_cell_for_input_coords
                    if pygame.K_0 <= event.key <= pygame.K_8:
                        num = event.key - pygame.K_0
                        can_place = True
                        if num != 0:
                            for row_idx, row_val in enumerate(initial_state):
                                for col_idx, cell_val in enumerate(row_val):
                                    if cell_val == num and (row_idx != r or col_idx != c):
                                        can_place = False; break
                                if not can_place: break
                        if can_place:
                            initial_state[r][c] = num
                            current_state_for_animation = copy.deepcopy(initial_state)
                            solution_path_anim = None; action_plan_anim = None; current_step_in_anim = 0
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        if initial_state[r][c] is not None:
                             initial_state[r][c] = None
                             current_state_for_animation = copy.deepcopy(initial_state)
                             solution_path_anim = None; action_plan_anim = None; current_step_in_anim = 0
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN or event.key == pygame.K_TAB:
                         selected_cell_for_input_coords = None
                    elif event.key == pygame.K_UP:
                         selected_cell_for_input_coords = ((r - 1 + GRID_SIZE) % GRID_SIZE, c)
                    elif event.key == pygame.K_DOWN:
                         selected_cell_for_input_coords = ((r + 1) % GRID_SIZE, c)
                    elif event.key == pygame.K_LEFT:
                         selected_cell_for_input_coords = (r, (c - 1 + GRID_SIZE) % GRID_SIZE)
                    elif event.key == pygame.K_RIGHT:
                         selected_cell_for_input_coords = (r, (c + 1) % GRID_SIZE)

        if not running_main_loop: break

        if is_solving_flag:
            is_solving_flag = False
            solve_start_t = time.time()
            found_path_algo = None
            found_action_plan_algo = None
            actual_start_state_for_anim = None
            belief_size_at_end = 0
            error_during_solve = False
            error_message_solve = ""

            if solver_name_for_stats is None: solver_name_for_stats = solver_algorithm_to_run

            algo_func_map = {
                'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar,
                'Greedy': greedy, 'IDA*': ida_star,
                'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
                'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
                'Beam Search': beam_search, 'AND-OR': and_or_search,
                'Sensorless': sensorless_search, 'Unknown': sensorless_search,
                'Q-Learning': q_learning_train_and_solve,
                'GA': genetic_algorithm_solve,
            }

            try:
                state_to_solve_from = copy.deepcopy(initial_state)
                current_state_for_animation = copy.deepcopy(state_to_solve_from)

                if solver_algorithm_to_run == 'Backtracking':
                    if backtracking_sub_algo_choice:
                        found_path_algo, error_message_solve, actual_start_state_for_anim, found_action_plan_algo = \
                            backtracking_search(backtracking_sub_algo_choice)
                        if found_path_algo and actual_start_state_for_anim:
                            initial_state = copy.deepcopy(actual_start_state_for_anim)
                            current_state_for_animation = copy.deepcopy(actual_start_state_for_anim)
                            action_plan_anim = found_action_plan_algo
                            solver_name_for_stats = f"BT({backtracking_sub_algo_choice})"
                        else:
                            solver_name_for_stats = f"BT({backtracking_sub_algo_choice})-Fail"
                        backtracking_sub_algo_choice = None
                    else:
                        error_message_solve = "Backtracking sub-algorithm choice missing."; error_during_solve = True
                elif solver_algorithm_to_run in ['Sensorless', 'Unknown']:
                    found_action_plan_algo, belief_size_at_end = sensorless_search(state_to_solve_from, time_limit=60)
                    current_sensorless_belief_size_for_display = belief_size_at_end
                    if found_action_plan_algo is not None:
                        action_plan_anim = found_action_plan_algo
                        sample_belief_states = generate_belief_states(state_to_solve_from)
                        vis_start_state = state_to_solve_from
                        if sample_belief_states: vis_start_state = sample_belief_states[0]
                        if is_valid_state_for_solve(vis_start_state):
                             found_path_algo = execute_plan(vis_start_state, found_action_plan_algo)
                        else: found_path_algo = None
                        current_state_for_animation = copy.deepcopy(vis_start_state)
                else:
                    selected_algo_function = algo_func_map.get(solver_algorithm_to_run)
                    if selected_algo_function:
                        algo_args_list = [state_to_solve_from]
                        func_varnames = selected_algo_function.__code__.co_varnames[:selected_algo_function.__code__.co_argcount]

                        default_time_limit = 30
                        if solver_algorithm_to_run in ['IDA*', 'IDS', 'Sensorless', 'Unknown', 'Q-Learning', 'GA']:
                            default_time_limit = 60

                        if 'time_limit' in func_varnames:
                            algo_args_list.append(default_time_limit)
                        elif solver_algorithm_to_run == 'DFS' and 'max_depth' in func_varnames: algo_args_list.append(30)
                        elif solver_algorithm_to_run == 'IDS' and 'max_depth_limit' in func_varnames: algo_args_list.append(30)
                        elif solver_algorithm_to_run == 'Stochastic Hill' and 'max_iter_no_improve' in func_varnames: algo_args_list.append(500)
                        elif solver_algorithm_to_run == 'Beam Search' and 'beam_width' in func_varnames: algo_args_list.append(5)

                        found_path_algo = selected_algo_function(*algo_args_list)
                        action_plan_anim = None
                    else:
                         error_message_solve = f"Algorithm function for '{solver_algorithm_to_run}' not found."; error_during_solve = True
            except Exception as e:
                error_message_solve = f"Runtime Error during {solver_name_for_stats} solve:\n{traceback.format_exc()}"
                error_during_solve = True
                action_plan_anim = None; found_path_algo = None

            solve_duration_t = time.time() - solve_start_t
            if solver_name_for_stats:
                 all_solve_times[solver_name_for_stats] = solve_duration_t
            else:
                 temp_solver_name = f"UnknownRun@{solve_start_t:.0f}" # Make it a bit more readable
                 all_solve_times[temp_solver_name] = solve_duration_t
                 solver_name_for_stats = temp_solver_name


            if error_during_solve:
                show_popup(error_message_solve if error_message_solve else "An unknown error occurred.", "Solver Error")
                last_run_solver_info.pop(f"{solver_name_for_stats}_reached_goal", None)
                last_run_solver_info.pop(f"{solver_name_for_stats}_steps", None)
                last_run_solver_info.pop(f"{solver_name_for_stats}_actions", None)
            else:
                result_found = (action_plan_anim is not None) or \
                               (found_path_algo and len(found_path_algo) > 0)
                if result_found:
                    solution_path_anim = found_path_algo
                    current_step_in_anim = 0
                    is_auto_animating_flag = (solution_path_anim is not None and len(solution_path_anim) > 1)
                    last_anim_step_time = time.time()

                    is_actually_goal_state = False
                    num_steps_or_actions = 0
                    popup_msg = ""
                    is_plan_based = solver_algorithm_to_run in ['Sensorless', 'Unknown'] or \
                                    (solver_name_for_stats and (solver_name_for_stats.startswith('BT(Sensorless') or \
                                                                solver_name_for_stats.startswith('BT(Unknown')))

                    if is_plan_based:
                         if action_plan_anim is not None:
                            num_steps_or_actions = len(action_plan_anim)
                            last_run_solver_info[f"{solver_name_for_stats}_actions"] = num_steps_or_actions
                            is_actually_goal_state = True
                            popup_msg = f"{solver_name_for_stats}: Plan found!\n{num_steps_or_actions} actions."
                            if solver_algorithm_to_run in ['Sensorless', 'Unknown']:
                                popup_msg += "\n(Visualizing plan on one sample state)"
                         else:
                            is_actually_goal_state = None
                            popup_msg = f"{solver_name_for_stats}: Plan search failed."
                            last_run_solver_info[f"{solver_name_for_stats}_reached_goal"] = None
                    else: # Path-based (BFS, A*, Q-Learning, GA, Hill Climb, etc.)
                        if solution_path_anim and len(solution_path_anim) > 0:
                             final_state_of_path = solution_path_anim[-1]
                             # Check if final state is actually the goal and valid
                             if isinstance(final_state_of_path, list) and is_valid_state_for_solve(final_state_of_path):
                                 is_actually_goal_state = is_goal(final_state_of_path)
                             else: # Path format error, or invalid final state
                                 is_actually_goal_state = False
                                 # print(f"Warning: final state in path for {solver_name_for_stats} is invalid or not list: {final_state_of_path}")


                             num_steps_or_actions = max(0, len(solution_path_anim) - 1)
                             last_run_solver_info[f"{solver_name_for_stats}_steps"] = num_steps_or_actions
                             popup_msg = f"{solver_name_for_stats} "
                             if is_actually_goal_state:
                                 popup_msg += f"found solution!\n{num_steps_or_actions} steps."
                             else:
                                 popup_msg += f"finished.\n{num_steps_or_actions} steps (Not Goal)."
                        else:
                             is_actually_goal_state = False
                             popup_msg = f"{solver_name_for_stats}: No valid path returned."

                    if is_actually_goal_state is not None:
                         last_run_solver_info[f"{solver_name_for_stats}_reached_goal"] = is_actually_goal_state

                    popup_msg += f"\nTime: {solve_duration_t:.3f}s"
                    popup_title = "Solve Complete" if is_actually_goal_state is True else \
                                  ("Plan Search Failed" if is_actually_goal_state is None else "Search Finished")
                    show_popup(popup_msg, popup_title)
                else:
                    last_run_solver_info[f"{solver_name_for_stats}_reached_goal"] = False
                    last_run_solver_info.pop(f"{solver_name_for_stats}_steps", None)
                    last_run_solver_info.pop(f"{solver_name_for_stats}_actions", None)
                    no_solution_msg = error_message_solve if error_message_solve else f"No solution/path found by {solver_name_for_stats}."
                    used_time_limit_for_msg = 30
                    if solver_algorithm_to_run in ['IDA*', 'IDS', 'Sensorless', 'Unknown', 'Backtracking', 'Q-Learning', 'GA']:
                        used_time_limit_for_msg = 60
                    if solve_duration_t >= used_time_limit_for_msg * 0.98 :
                         no_solution_msg = f"{solver_name_for_stats} timed out after ~{used_time_limit_for_msg}s."
                    show_popup(no_solution_msg, "No Solution / Timeout")

            solver_algorithm_to_run = None
            solver_name_for_stats = None
            backtracking_sub_algo_choice = None


        if is_auto_animating_flag and solution_path_anim:
            current_time_anim = time.time()
            anim_delay_val = 0.3
            if len(solution_path_anim) > 30: anim_delay_val = 0.15
            if len(solution_path_anim) > 60: anim_delay_val = 0.08

            if current_time_anim - last_anim_step_time >= anim_delay_val:
                if current_step_in_anim < len(solution_path_anim) - 1:
                    current_step_in_anim += 1
                    next_anim_state = solution_path_anim[current_step_in_anim]
                    if next_anim_state and isinstance(next_anim_state, list) and len(next_anim_state) == GRID_SIZE:
                        current_state_for_animation = copy.deepcopy(next_anim_state)
                    else:
                        is_auto_animating_flag = False
                    last_anim_step_time = current_time_anim
                else:
                    is_auto_animating_flag = False


        belief_size_for_display = current_sensorless_belief_size_for_display
        active_algo_display_name = solver_name_for_stats if solver_name_for_stats else current_selected_algorithm

        if active_algo_display_name in ['Sensorless', 'Unknown'] and belief_size_for_display is None and not is_solving_flag and not is_auto_animating_flag:
             try:
                 if isinstance(initial_state, list):
                     temp_initial_bs = generate_belief_states(initial_state)
                     belief_size_for_display = len(temp_initial_bs) if temp_initial_bs is not None else 0
                 else: belief_size_for_display = 0
             except Exception: belief_size_for_display = 0


        ui_elements_rects = draw_grid_and_ui(
            current_state_for_animation, show_algo_menu,
            current_selected_algorithm,
            solver_name_for_stats,
            all_solve_times,
            last_run_solver_info,
            belief_size_for_display,
            selected_cell_for_input_coords
        )

        pygame.display.flip()
        game_clock.tick(60)

    pygame.quit()


# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as main_error:
        print("\n--- UNHANDLED ERROR IN MAIN LOOP ---")
        traceback.print_exc()
        print("------------------------------------")
        pygame.quit()
        sys.exit(1)

