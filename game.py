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
goal_state = copy.deepcopy(goal_state_fixed_global) # goal_state is fixed, derived from fixed global

# KHÔI PHỤC HÀM draw_state
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

            if tile_val is None:
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
                q_font = pygame.font.SysFont('Arial', 40)
                q_surf = q_font.render("?", True, BLACK)
                screen.blit(q_surf, q_surf.get_rect(center=cell_rect_ds.center))
            elif tile_val == 0:
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
            else:
                cell_fill_color = BLUE
                if is_fixed_goal_display:
                    cell_fill_color = GREEN
                elif is_current_anim_state and is_valid_structure and is_goal(state_to_draw):
                    cell_fill_color = GREEN

                pygame.draw.rect(screen, cell_fill_color, cell_rect_ds.inflate(-6, -6), border_radius=8)
                try:
                    number_surf = FONT.render(str(tile_val), True, WHITE)
                    screen.blit(number_surf, number_surf.get_rect(center=cell_rect_ds.center))
                except (ValueError, TypeError):
                     pygame.draw.rect(screen, RED, cell_rect_ds.inflate(-10,-10))

            pygame.draw.rect(screen, BLACK, cell_rect_ds, 1)

            if is_editable and selected_cell_coords == (r_ds, c_ds):
                 pygame.draw.rect(screen, RED, cell_rect_ds, 3)


def find_empty(state):
    if not isinstance(state, list) or len(state) != GRID_SIZE: return -1, -1
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE: return -1, -1
        for j in range(GRID_SIZE):
            try:
                if isinstance(state[i][j], int) and state[i][j] == 0:
                    return i, j
            except (TypeError, IndexError):
                continue
    return -1, -1

def is_goal(state):
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            return False
    if not is_valid_state_for_solve(state):
        return False
    return state == goal_state_fixed_global # Compare with the fixed global goal

def get_neighbors(state):
    neighbors = []
    empty_i, empty_j = find_empty(state)
    if empty_i == -1:
        return []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
        for r in range(GRID_SIZE):
            if not isinstance(state[r], list) or len(state[r]) != GRID_SIZE: return None
            for c in range(GRID_SIZE):
                 if state[r][c] is not None and not isinstance(state[r][c], int): return None
        return tuple(tuple(row) for row in state)
    except (TypeError, IndexError):
        return None


def tuple_to_list(state_tuple):
    if not isinstance(state_tuple, tuple) or len(state_tuple) != GRID_SIZE:
        return None
    try:
        new_list = []
        for row_tuple in state_tuple:
            if not isinstance(row_tuple, tuple) or len(row_tuple) != GRID_SIZE:
                return None
            new_list.append(list(row_tuple))
        return new_list
    except (TypeError, IndexError):
        return None


def apply_action_to_state(state_list, action):
    if state_list is None or not is_valid_state_for_solve(state_list):
        return None # Or return state_list based on desired behavior for invalid input
    new_state = copy.deepcopy(state_list)
    empty_i, empty_j = find_empty(new_state)
    if empty_i == -1:
        return new_state

    swap_i, swap_j = empty_i, empty_j
    if action == 'Up': swap_i += 1
    elif action == 'Down': swap_i -= 1
    elif action == 'Left': swap_j += 1
    elif action == 'Right': swap_j -= 1
    else:
        return new_state

    if 0 <= swap_i < GRID_SIZE and 0 <= swap_j < GRID_SIZE:
        new_state[empty_i][empty_j], new_state[swap_i][swap_j] = new_state[swap_i][swap_j], new_state[empty_i][empty_j]
        return new_state
    else:
        return new_state


def manhattan_distance(state):
    distance = 0
    goal_pos = {}
    for r_goal in range(GRID_SIZE):
        for c_goal in range(GRID_SIZE):
            val = goal_state_fixed_global[r_goal][c_goal]
            if val != 0 :
                goal_pos[val] = (r_goal, c_goal)

    if not isinstance(state, list): return float('inf')
    for r_curr in range(GRID_SIZE):
        if not isinstance(state[r_curr], list) or len(state[r_curr]) != GRID_SIZE: return float('inf')
        for c_curr in range(GRID_SIZE):
            tile = state[r_curr][c_curr]
            if tile is None: return float('inf') # Heuristic needs complete state
            if tile != 0 and tile in goal_pos:
                goal_r, goal_c = goal_pos[tile]
                distance += abs(r_curr - goal_r) + abs(c_curr - goal_c)
    return distance


def is_valid_state_for_solve(state_to_check):
    flat_state = []
    try:
        if not isinstance(state_to_check, list) or len(state_to_check) != GRID_SIZE: return False
        for row in state_to_check:
            if not isinstance(row, list) or len(row) != GRID_SIZE: return False
            for tile in row:
                if tile is None: return False
                if not isinstance(tile, int): return False
                flat_state.append(tile)
    except TypeError:
        return False

    if len(flat_state) != GRID_SIZE * GRID_SIZE:
        return False

    expected_numbers = set(range(GRID_SIZE * GRID_SIZE))
    seen_numbers = set(flat_state)

    return seen_numbers == expected_numbers

def get_inversions(flat_state_no_zero):
    inversions = 0
    n = len(flat_state_no_zero)
    for i in range(n):
        for j in range(i + 1, n):
            if flat_state_no_zero[i] > flat_state_no_zero[j]:
                inversions += 1
    return inversions

def is_solvable(state_to_check):
    if not is_valid_state_for_solve(state_to_check):
        return False

    flat_state = [tile for row in state_to_check for tile in row]

    state_flat_no_zero = [tile for tile in flat_state if tile != 0]
    inversions_state = get_inversions(state_flat_no_zero)

    goal_flat = [tile for row in goal_state_fixed_global for tile in row]
    goal_flat_no_zero = [tile for tile in goal_flat if tile != 0]
    inversions_goal = get_inversions(goal_flat_no_zero)

    if GRID_SIZE % 2 == 1:
        return (inversions_state % 2) == (inversions_goal % 2)
    else:
        blank_row_state = -1
        for r_idx, row_s in enumerate(state_to_check):
            if 0 in row_s:
                blank_row_state = r_idx
                break
        blank_row_from_bottom_state = GRID_SIZE - blank_row_state

        blank_row_goal = -1
        for r_idx_g, row_g in enumerate(goal_state_fixed_global):
            if 0 in row_g:
                blank_row_goal = r_idx_g
                break
        blank_row_from_bottom_goal = GRID_SIZE - blank_row_goal

        return ((inversions_state + blank_row_from_bottom_state) % 2) == \
               ((inversions_goal + blank_row_from_bottom_goal) % 2)

# --- Search Algorithms ---

def bfs(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    queue = deque([(start_node_state, [start_node_state])])
    visited = {init_tuple}

    while queue:
        if time.time() - start_time > time_limit:
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
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    stack = [(start_node_state, [start_node_state], 0)]
    visited = {}

    while stack:
        if time.time() - start_time > time_limit:
            return None

        current_s, path, depth = stack.pop()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        if current_tuple in visited and visited[current_tuple] <= depth:
            continue

        visited[current_tuple] = depth

        if is_goal(current_s):
            return path

        if depth >= max_depth:
            continue

        neighbors = get_neighbors(current_s)
        for neighbor_s in reversed(neighbors):
            stack.append((neighbor_s, path + [neighbor_s], depth + 1))

    return None

def ids(start_node_state, max_depth_limit=30, time_limit=60):
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    for depth_limit in range(max_depth_limit + 1):
        if time.time() - start_time_global > time_limit:
            return None

        stack = [(start_node_state, [start_node_state], 0)]
        visited_in_iteration = {init_tuple: 0}

        while stack:
            if time.time() - start_time_global > time_limit:
                return None

            current_s, path, depth = stack.pop()

            if is_goal(current_s):
                return path

            if depth < depth_limit:
                neighbors = get_neighbors(current_s)
                for neighbor_s in reversed(neighbors):
                    neighbor_tuple = state_to_tuple(neighbor_s)
                    if neighbor_tuple is not None:
                        if neighbor_tuple not in visited_in_iteration or visited_in_iteration[neighbor_tuple] > depth + 1:
                            visited_in_iteration[neighbor_tuple] = depth + 1
                            stack.append((neighbor_s, path + [neighbor_s], depth + 1))

    return None

def ucs(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    frontier = PriorityQueue()
    frontier.put((0, start_node_state, [start_node_state]))

    visited = {init_tuple: 0}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            return None

        cost, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        if cost > visited.get(current_tuple, float('inf')):
             continue

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            new_cost = cost + 1

            if new_cost < visited.get(neighbor_tuple, float('inf')):
                visited[neighbor_tuple] = new_cost
                frontier.put((new_cost, neighbor_s, path + [neighbor_s]))
    return None

def astar(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    g_init = 0
    h_init = manhattan_distance(start_node_state)
    if h_init == float('inf'): return None
    f_init = g_init + h_init

    frontier = PriorityQueue()
    frontier.put((f_init, g_init, start_node_state, [start_node_state]))

    visited_g_scores = {init_tuple: g_init}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            return None

        f_score_curr, g_score_curr, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s)

        if current_tuple is None: continue

        if g_score_curr > visited_g_scores.get(current_tuple, float('inf')):
            continue

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            tentative_g_score = g_score_curr + 1

            if tentative_g_score < visited_g_scores.get(neighbor_tuple, float('inf')):
                visited_g_scores[neighbor_tuple] = tentative_g_score
                h_neighbor = manhattan_distance(neighbor_s)
                if h_neighbor == float('inf'): continue
                f_neighbor = tentative_g_score + h_neighbor
                frontier.put((f_neighbor, tentative_g_score, neighbor_s, path + [neighbor_s]))
    return None

def greedy(start_node_state, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    frontier = PriorityQueue()
    h_init = manhattan_distance(start_node_state)
    if h_init == float('inf'): return None
    frontier.put((h_init, start_node_state, [start_node_state]))

    visited = {init_tuple}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            return None

        h_val, current_s, path = frontier.get()

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                h_neighbor = manhattan_distance(neighbor_s)
                if h_neighbor == float('inf'): continue
                frontier.put((h_neighbor, neighbor_s, path + [neighbor_s]))

    return None

# --- Corrected IDA* Implementation ---
def _search_ida_recursive(path_ida, g_score, threshold, start_time_ida, time_limit_ida):
    if time.time() - start_time_ida >= time_limit_ida:
        return "Timeout", float('inf')

    current_s_ida = path_ida[-1]

    h_ida = manhattan_distance(current_s_ida)
    if h_ida == float('inf'):
         return None, float('inf')
    f_score_ida = g_score + h_ida

    if f_score_ida > threshold:
        return None, f_score_ida

    if is_goal(current_s_ida):
        return path_ida[:], threshold

    min_new_threshold = float('inf')

    current_path_tuples = {state_to_tuple(s) for s in path_ida if state_to_tuple(s) is not None}

    for neighbor_s_ida in get_neighbors(current_s_ida):
        neighbor_tuple_ida = state_to_tuple(neighbor_s_ida)
        if neighbor_tuple_ida is None: continue

        if neighbor_tuple_ida in current_path_tuples:
            continue

        new_g_ida = g_score + 1

        path_ida.append(neighbor_s_ida)

        result_ida, recursive_threshold_ida = _search_ida_recursive(
            path_ida, new_g_ida, threshold, start_time_ida, time_limit_ida
        )

        path_ida.pop()

        if result_ida == "Timeout": return "Timeout", float('inf')
        if result_ida is not None: return result_ida, threshold

        min_new_threshold = min(min_new_threshold, recursive_threshold_ida)

    return None, min_new_threshold

def ida_star(start_node_state, time_limit=60):
    start_time_global = time.time()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    initial_h = manhattan_distance(start_node_state)
    if initial_h == float('inf'): return None
    threshold = initial_h

    while True:
        if time.time() - start_time_global >= time_limit:
            return None

        current_path = [start_node_state]

        result, new_threshold_candidate = _search_ida_recursive(
            current_path, 0, threshold, start_time_global, time_limit
        )

        if result == "Timeout": return None
        if result is not None: return result

        if new_threshold_candidate == float('inf'):
            return None

        if new_threshold_candidate <= threshold:
             return None

        threshold = new_threshold_candidate

# --- Hill Climbing Variants ---
def simple_hill_climbing(start_node_state, time_limit=30):
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path

    while True:
        if time.time() - start_time > time_limit:
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        moved = False

        for neighbor_s in neighbors:
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor == float('inf'): continue
            if h_neighbor < current_h:
                current_s = neighbor_s
                current_h = h_neighbor
                path.append(current_s)
                moved = True
                break

        if not moved:
            return path

    return path

def steepest_hill_climbing(start_node_state, time_limit=30):
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path

    while True:
        if time.time() - start_time > time_limit:
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        best_next_s = None
        best_next_h = current_h

        for neighbor_s in neighbors:
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor == float('inf'): continue
            if h_neighbor < best_next_h:
                best_next_h = h_neighbor
                best_next_s = neighbor_s

        if best_next_s is None:
            return path

        current_s = best_next_s
        current_h = best_next_h
        path.append(current_s)

    return path

def random_hill_climbing(start_node_state, time_limit=30, max_iter_no_improve=500):
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return path
    iter_no_improve = 0

    while True:
        if time.time() - start_time > time_limit:
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        random_neighbor = random.choice(neighbors)
        neighbor_h = manhattan_distance(random_neighbor)
        if neighbor_h == float('inf'): continue

        if neighbor_h <= current_h:
            if neighbor_h < current_h:
                iter_no_improve = 0
            else:
                iter_no_improve += 1

            current_s = random_neighbor
            current_h = neighbor_h
            path.append(current_s)
        else:
            iter_no_improve += 1

        if iter_no_improve >= max_iter_no_improve:
            return path
    return path

def simulated_annealing(start_node_state, initial_temp=1000, cooling_rate=0.99, min_temp=0.1, time_limit=30):
    start_time = time.time()
    current_s = start_node_state
    current_h = manhattan_distance(current_s)
    if current_h == float('inf'): return [start_node_state]
    path = [current_s]

    best_s_so_far = current_s
    best_h_so_far = current_h

    temp = initial_temp

    while temp > min_temp:
        if time.time() - start_time > time_limit:
            return path

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break

        next_s = random.choice(neighbors)
        next_h = manhattan_distance(next_s)
        if next_h == float('inf'): continue

        delta_h = next_h - current_h

        if delta_h < 0:
            current_s = next_s
            current_h = next_h
            path.append(current_s)
            if current_h < best_h_so_far:
                best_s_so_far = current_s
                best_h_so_far = current_h
        else:
            if temp > 0:
                acceptance_prob = math.exp(-delta_h / temp)
                if random.random() < acceptance_prob:
                    current_s = next_s
                    current_h = next_h
                    path.append(current_s)

        temp *= cooling_rate

    return path

# --- Beam Search ---
def beam_search(start_node_state, beam_width=5, time_limit=30):
    start_time = time.time()
    if is_goal(start_node_state): return [start_node_state]

    initial_h = manhattan_distance(start_node_state)
    if initial_h == float('inf'): return None

    beam = [(start_node_state, [start_node_state], initial_h)]
    visited_tuples_global = {state_to_tuple(start_node_state)}
    best_goal_path_found = None

    max_iterations = 100

    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            return best_goal_path_found

        next_level_candidates = []

        for s_beam, path_beam, _ in beam:
            for neighbor_s_beam in get_neighbors(s_beam):
                neighbor_tuple_beam = state_to_tuple(neighbor_s_beam)
                if neighbor_tuple_beam is None: continue

                if neighbor_tuple_beam not in visited_tuples_global:
                    new_path_beam = path_beam + [neighbor_s_beam]

                    if is_goal(neighbor_s_beam):
                        if best_goal_path_found is None or len(new_path_beam) < len(best_goal_path_found):
                            best_goal_path_found = new_path_beam

                    h_neighbor_beam = manhattan_distance(neighbor_s_beam)
                    if h_neighbor_beam == float('inf'): continue

                    next_level_candidates.append((neighbor_s_beam, new_path_beam, h_neighbor_beam))
                    visited_tuples_global.add(neighbor_tuple_beam)

        if not next_level_candidates:
            break

        next_level_candidates.sort(key=lambda x: x[2])

        beam = next_level_candidates[:beam_width]

        if not beam:
            break

    return best_goal_path_found

# --- AND-OR Search (Simplified DFS interpretation for 8-puzzle) ---
def _and_or_recursive(state_ao, path_ao, visited_ao_tuples, start_time_ao, time_limit_ao, depth_ao, max_depth_ao=50):
    state_tuple_ao = state_to_tuple(state_ao)
    if state_tuple_ao is None: return "Fail", None

    if time.time() - start_time_ao > time_limit_ao: return "Timeout", None
    if depth_ao > max_depth_ao: return "Fail", None
    if is_goal(state_ao): return "Solved", path_ao[:]

    if state_tuple_ao in visited_ao_tuples: return "Fail", None

    visited_ao_tuples.add(state_tuple_ao)

    for neighbor_ao in get_neighbors(state_ao):
        status_ao, solution_path_ao = _and_or_recursive(
            neighbor_ao, path_ao + [neighbor_ao],
            visited_ao_tuples.copy(),
            start_time_ao, time_limit_ao, depth_ao + 1, max_depth_ao
        )
        if status_ao == "Timeout": return "Timeout", None
        if status_ao == "Solved": return "Solved", solution_path_ao

    return "Fail", None

def and_or_search(start_node_state, time_limit=30, max_depth=50):
    start_time = time.time()
    initial_visited = set()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    status, solution_path = _and_or_recursive(
        start_node_state, [start_node_state],
        initial_visited, start_time, time_limit, 0, max_depth
    )

    if status == "Solved": return solution_path
    return None

# --- Backtracking (Meta-Algorithm) ---
def backtracking_search(selected_sub_algorithm_name, max_attempts=50, time_limit_overall=60):
    start_time_overall = time.time()

    algo_map = {
        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 'Greedy': greedy,
        'IDA*': ida_star,
        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
        'Beam Search': beam_search, 'AND-OR': and_or_search,
        'Sensorless': sensorless_search,
        'Unknown': sensorless_search,
    }
    sub_algo_func = algo_map.get(selected_sub_algorithm_name)
    if not sub_algo_func:
        return None, f"Sub-algorithm '{selected_sub_algorithm_name}' not supported for Backtracking.", None, None

    time_limit_per_sub_solve = max(1.0, time_limit_overall / max_attempts if max_attempts > 0 else time_limit_overall / 5)
    time_limit_per_sub_solve = min(time_limit_per_sub_solve, 15 if selected_sub_algorithm_name not in ['IDA*', 'Sensorless', 'Unknown', 'IDS'] else 30)

    for attempts in range(max_attempts):
        if time.time() - start_time_overall > time_limit_overall:
            return None, "Backtracking: Global timeout.", None, None

        nums = list(range(GRID_SIZE * GRID_SIZE))
        random.shuffle(nums)
        current_attempt_2d_start = [nums[i*GRID_SIZE:(i+1)*GRID_SIZE] for i in range(GRID_SIZE)]

        if not is_solvable(current_attempt_2d_start):
            continue

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
            continue

        if path_from_sub_algo and len(path_from_sub_algo) > 0:
            sub_success = False
            if selected_sub_algorithm_name in ['Sensorless', 'Unknown']:
                sub_success = (action_plan_from_sub_algo is not None)
            else:
                 if isinstance(path_from_sub_algo[-1], list):
                    sub_success = is_goal(path_from_sub_algo[-1])
                 else: sub_success = False

            if sub_success:
                return path_from_sub_algo, None, current_attempt_2d_start, action_plan_from_sub_algo

    msg = f"Backtracking: No solvable state found by {selected_sub_algorithm_name} after {attempts+1} attempts."
    if time.time() - start_time_overall > time_limit_overall:
        msg = "Backtracking: Global timeout. " + msg
    return None, msg, None, None


# ----- Start: Functions for Sensorless Search / Unknown Env -----
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
            flat_state_template.append(tile)

    if len(flat_state_template) != GRID_SIZE * GRID_SIZE:
        return []

    if not is_template_with_unknowns:
        if len(known_numbers_set) != GRID_SIZE * GRID_SIZE:
             return []
        if is_solvable(partial_state_template):
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
        for r_idx_fill in range(GRID_SIZE):
            row_start = r_idx_fill * GRID_SIZE
            state_2d_filled.append(new_flat_state_filled[row_start : row_start + GRID_SIZE])

        if is_solvable(state_2d_filled):
            belief_states_generated.append(state_2d_filled)

    return belief_states_generated

def is_belief_state_goal(list_of_belief_states):
    if not list_of_belief_states:
        return False
    for state_b in list_of_belief_states:
        if not is_goal(state_b):
            return False
    return True

def apply_action_to_belief_states(list_of_belief_states, action_str):
    new_belief_states_set = set()
    for state_b in list_of_belief_states:
        # This function might be called with states that aren't yet "valid_for_solve"
        # (e.g. during sensorless search if a state becomes invalid temporarily due to action)
        # So, apply_action_to_state needs to handle potentially invalid states robustly.
        # Assuming apply_action_to_state returns a new state or the original if move is bad.
        temp_state = copy.deepcopy(state_b) # Make a working copy
        empty_i, empty_j = find_empty(temp_state)
        if empty_i == -1: # No empty tile, cannot apply action
            next_s_b_list = temp_state # Or None, depending on desired strictness
        else:
            swap_i, swap_j = empty_i, empty_j
            if action_str == 'Up': swap_i += 1
            elif action_str == 'Down': swap_i -= 1
            elif action_str == 'Left': swap_j += 1
            elif action_str == 'Right': swap_j -= 1

            if 0 <= swap_i < GRID_SIZE and 0 <= swap_j < GRID_SIZE:
                temp_state[empty_i][empty_j], temp_state[swap_i][swap_j] = temp_state[swap_i][swap_j], temp_state[empty_i][empty_j]
                next_s_b_list = temp_state
            else:
                next_s_b_list = temp_state # Invalid move, return original (copy)

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
        if None in initial_belief_tuples:
             return None, initial_belief_set_size
    except Exception as e:
        return None, initial_belief_set_size

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
        if not current_bs_lists or len(current_bs_lists) != len(current_bs_tuples):
            continue

        if is_belief_state_goal(current_bs_lists):
            return action_plan, len(current_bs_lists)

        for action_s in possible_actions:
            next_bs_lists = apply_action_to_belief_states(current_bs_lists, action_s)

            if not next_bs_lists: continue

            try:
                next_bs_tuples = {state_to_tuple(s) for s in next_bs_lists}
                if None in next_bs_tuples: continue
                if not next_bs_tuples: continue

                next_bs_frozenset = frozenset(next_bs_tuples)
            except Exception as e:
                continue

            if next_bs_frozenset not in visited_belief_sets:
                visited_belief_sets.add(next_bs_frozenset)
                new_plan_s = action_plan + [action_s]
                queue_sensorless.append((new_plan_s, next_bs_tuples))

    return None, last_processed_bs_size

# --- Helper for visualizing plans ---
def execute_plan(start_state, action_plan):
    if start_state is None or not isinstance(start_state, list): return None
    if not is_valid_state_for_solve(start_state): return [start_state] # Path is just the invalid start

    current_state = copy.deepcopy(start_state)
    state_sequence = [current_state]

    if not action_plan:
        return state_sequence

    for action in action_plan:
        next_state = apply_action_to_state(current_state, action) # Uses the more robust apply_action
        if next_state is None: # Should not happen if start_state was valid and actions are good
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
            except IndexError:
                return None
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
            except IndexError:
                continue
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
    # random.shuffle(available_nums) # Can be enabled for varied attempts

    for num in available_nums:
        state_copy[r][c] = num
        result = ac3_fill_state(state_copy)
        if result is not None:
            return result
        state_copy[r][c] = None

    return None


# --- UI and Drawing Functions ---

def solver_selection_popup():
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
    ]
    button_width_popup, button_height_popup = 150, 40
    button_margin_popup = 10
    columns_popup = 3
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

        popup_surface.fill(INFO_BG)
        pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
        popup_surface.blit(title_surf_popup, title_rect_popup)

        for algo_id_p, algo_name_p, btn_rect_p in algo_buttons_popup:
            is_hovered_p = btn_rect_p.collidepoint(mouse_pos_relative)
            btn_color_p = MENU_HOVER_COLOR if is_hovered_p else MENU_BUTTON_COLOR
            pygame.draw.rect(popup_surface, btn_color_p, btn_rect_p, border_radius=8)
            text_surf_p = BUTTON_FONT.render(algo_name_p, True, WHITE)
            text_rect_p = text_surf_p.get_rect(center=btn_rect_p.center)
            popup_surface.blit(text_surf_p, text_rect_p)

        is_hovered_cancel = cancel_button_rect_popup.collidepoint(mouse_pos_relative)
        cancel_color = MENU_HOVER_COLOR if is_hovered_cancel else RED
        pygame.draw.rect(popup_surface, cancel_color, cancel_button_rect_popup, border_radius=8)
        cancel_text_surf_p = BUTTON_FONT.render("Cancel", True, WHITE)
        cancel_text_rect_p = cancel_text_surf_p.get_rect(center=cancel_button_rect_popup.center)
        popup_surface.blit(cancel_text_surf_p, cancel_text_rect_p)

        screen.blit(popup_surface, popup_rect_on_screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button_rect_popup.collidepoint(mouse_pos_relative):
                    return None
                for algo_id_p, _, btn_rect_p in algo_buttons_popup:
                    if btn_rect_p.collidepoint(mouse_pos_relative):
                        selected_algorithm_name = algo_id_p
                        running_popup = False
                        break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None

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
        ('Unknown', 'Unknown Env')
    ]
    button_width_popup, button_height_popup = 150, 40
    button_margin_popup = 10
    columns_popup = 3
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

        popup_surface.fill(INFO_BG)
        pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
        popup_surface.blit(title_surf_popup, title_rect_popup)

        for algo_id_p, algo_name_p, btn_rect_p in algo_buttons_popup:
            is_hovered_p = btn_rect_p.collidepoint(mouse_pos_relative)
            btn_color_p = MENU_HOVER_COLOR if is_hovered_p else MENU_BUTTON_COLOR
            pygame.draw.rect(popup_surface, btn_color_p, btn_rect_p, border_radius=8)
            text_surf_p = BUTTON_FONT.render(algo_name_p, True, WHITE)
            text_rect_p = text_surf_p.get_rect(center=btn_rect_p.center)
            popup_surface.blit(text_surf_p, text_rect_p)

        is_hovered_cancel = cancel_button_rect_popup.collidepoint(mouse_pos_relative)
        cancel_color = MENU_HOVER_COLOR if is_hovered_cancel else RED
        pygame.draw.rect(popup_surface, cancel_color, cancel_button_rect_popup, border_radius=8)
        cancel_text_surf_p = BUTTON_FONT.render("Cancel", True, WHITE)
        cancel_text_rect_p = cancel_text_surf_p.get_rect(center=cancel_button_rect_popup.center)
        popup_surface.blit(cancel_text_surf_p, cancel_text_rect_p)

        screen.blit(popup_surface, popup_rect_on_screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button_rect_popup.collidepoint(mouse_pos_relative):
                    return None
                for algo_id_p, _, btn_rect_p in algo_buttons_popup:
                    if btn_rect_p.collidepoint(mouse_pos_relative):
                        selected_algorithm_name = algo_id_p
                        running_popup = False
                        break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None

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

        is_separator = (algo_id_dm == '---')
        if is_separator:
            pygame.draw.line(menu_surface, GRAY, (button_rect_local_dm.left + 5, button_rect_local_dm.centery),
                                                  (button_rect_local_dm.right - 5, button_rect_local_dm.centery), 1)
            text_surf_dm = INFO_FONT.render(algo_name_dm, True, GRAY)
            text_rect_dm = text_surf_dm.get_rect(center=button_rect_local_dm.center)
            menu_surface.blit(text_surf_dm, text_rect_dm)

        else:
            is_hover_dm = button_rect_local_dm.collidepoint(mouse_x_rel_dm, mouse_y_rel_dm)
            is_selected_dm = (current_selected_algorithm_dm == algo_id_dm)

            button_color_dm = MENU_SELECTED_COLOR if is_selected_dm else \
                              (MENU_HOVER_COLOR if is_hover_dm else MENU_BUTTON_COLOR)
            pygame.draw.rect(menu_surface, button_color_dm, button_rect_local_dm, border_radius=5)

            text_surf_dm = BUTTON_FONT.render(algo_name_dm, True, WHITE)
            text_rect_dm = text_surf_dm.get_rect(center=button_rect_local_dm.center)
            menu_surface.blit(text_surf_dm, text_rect_dm)

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
    menu_elements_dict['buttons'] = buttons_dict_dm
    menu_elements_dict['menu_area'] = pygame.Rect(0, 0, MENU_WIDTH, HEIGHT)

    if total_menu_height > HEIGHT:
        scrollbar_track_height = HEIGHT - 2*padding_dm
        scrollbar_height_val = max(20, scrollbar_track_height * (HEIGHT / total_menu_height))

        max_scroll_y_content = total_menu_height - HEIGHT
        scroll_ratio = scroll_y / max_scroll_y_content if max_scroll_y_content > 0 else 0

        scrollbar_track_y_start = padding_dm
        scrollbar_max_y_thumb = scrollbar_track_y_start + scrollbar_track_height - scrollbar_height_val

        scrollbar_y_pos_thumb = scrollbar_track_y_start + scroll_ratio * (scrollbar_track_height - scrollbar_height_val)
        scrollbar_y_pos_thumb = max(scrollbar_track_y_start, min(scrollbar_y_pos_thumb, scrollbar_max_y_thumb))

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

    line_height_sp = INFO_FONT.get_linesize()
    text_start_y_sp = title_rect_sp.bottom + 20
    max_lines_display = (POPUP_HEIGHT - text_start_y_sp - 70) // line_height_sp

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
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x_sp, mouse_y_sp = event.pos
                mouse_rel_x = mouse_x_sp - popup_rect_on_screen_sp.left
                mouse_rel_y = mouse_y_sp - popup_rect_on_screen_sp.top
                if ok_button_rect_sp.collidepoint(mouse_rel_x, mouse_rel_y):
                    waiting_for_ok = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    waiting_for_ok = False
        pygame.time.delay(20)

def draw_grid_and_ui(current_anim_state_dgui, show_menu_dgui, current_algo_name_dgui,
                     solver_after_ac3_dgui,
                     solve_times_dgui, last_solved_run_info_dgui,
                     current_belief_state_size_dgui=None, selected_cell_for_input_coords=None):

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

    draw_state(initial_state, initial_grid_x, top_row_y_grids, "Initial State",
               is_editable=True, selected_cell_coords=selected_cell_for_input_coords)
    draw_state(goal_state_fixed_global, goal_grid_x, top_row_y_grids, "Goal State", is_fixed_goal_display=True)

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
            else: count_str_st = " (--)"

            goal_indicator_st = ""
            if reached_goal_st is False : goal_indicator_st = " (Not Goal)"
            elif (algo_name_st.startswith('Sensorless') or algo_name_st.startswith('Unknown') or \
                  algo_name_st.startswith('BT(Sensorless') or algo_name_st.startswith('BT(Unknown')) and reached_goal_st is None:
                 goal_indicator_st = " (Plan Fail)"


            base_str_st = f"{algo_name_st}: {time_val_st:.3f}s"
            full_comp_str = base_str_st + count_str_st + goal_indicator_st

            max_text_width = info_area_w - 2 * info_pad_x_ia
            if INFO_FONT.size(full_comp_str)[0] > max_text_width:
                shortened_str = base_str_st + goal_indicator_st
                if INFO_FONT.size(shortened_str)[0] <= max_text_width:
                    full_comp_str = shortened_str
                else:
                    name_part = base_str_st.split(":")[0]
                    max_name_len = 15 if '->' in algo_name_st or 'BT(' in algo_name_st else 10
                    if len(name_part) > max_name_len: name_part = name_part[:max_name_len] + "..."
                    full_comp_str = name_part + ": " + goal_indicator_st

            comp_surf_st = INFO_FONT.render(full_comp_str, True, BLACK)
            screen.blit(comp_surf_st, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
            current_info_y_draw += line_h_ia
    else:
        no_results_surf = INFO_FONT.render("(No results yet)", True, GRAY)
        screen.blit(no_results_surf, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))

    menu_elements_dgui = draw_menu(show_menu_dgui, mouse_pos_dgui, current_algo_name_dgui)

    pygame.display.flip()

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
    global scroll_y, initial_state, goal_state

    current_state_for_animation = copy.deepcopy(initial_state)
    solution_path_anim = None
    action_plan_anim = None
    current_step_in_anim = 0
    is_solving_flag = False
    is_auto_animating_flag = False
    last_anim_step_time = 0
    show_algo_menu = False
    current_selected_algorithm = 'A*'
    solver_algorithm_to_run = None # Stores the actual solver (e.g., A* after AC3)
    all_solve_times = {}
    last_run_solver_info = {}
    game_clock = pygame.time.Clock()
    ui_elements_rects = {}
    running_main_loop = True
    backtracking_sub_algo_choice = None # For Backtracking meta-algorithm
    current_sensorless_belief_size_for_display = None
    selected_cell_for_input_coords = None

    while running_main_loop:
        mouse_pos_main = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_main_loop = False
                break

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
                     is_click_on_menu = False
                     menu_area = ui_elements_rects.get('menu', {}).get('menu_area')
                     if menu_area and show_algo_menu and menu_area.collidepoint(mouse_pos_main):
                         is_click_on_menu = True
                     if not is_click_on_menu:
                         selected_cell_for_input_coords = None

                if show_algo_menu and not clicked_something_handled:
                    menu_data_main = ui_elements_rects.get('menu', {})
                    menu_area_main = menu_data_main.get('menu_area')

                    if menu_area_main and menu_area_main.collidepoint(mouse_pos_main):
                        close_btn_rect = menu_data_main.get('close_button')
                        if close_btn_rect and close_btn_rect.collidepoint(mouse_pos_main):
                            show_algo_menu = False
                            clicked_something_handled = True

                        if not clicked_something_handled:
                            algo_buttons_local_rects = menu_data_main.get('buttons', {})
                            mouse_y_relative_to_scroll = mouse_pos_main[1] + scroll_y
                            for algo_id, local_r in algo_buttons_local_rects.items():
                                if algo_id != '---' and local_r.collidepoint(mouse_pos_main[0], mouse_y_relative_to_scroll):
                                    if current_selected_algorithm != algo_id:
                                        current_selected_algorithm = algo_id
                                        solver_algorithm_to_run = None
                                        solution_path_anim = None
                                        action_plan_anim = None
                                        current_step_in_anim = 0
                                        is_auto_animating_flag = False
                                        is_solving_flag = False
                                        current_sensorless_belief_size_for_display = None
                                    show_algo_menu = False
                                    clicked_something_handled = True
                                    break

                        if not clicked_something_handled:
                             clicked_something_handled = True

                if not show_algo_menu and not clicked_something_handled:
                    menu_data_main = ui_elements_rects.get('menu', {})
                    open_btn_rect = menu_data_main.get('open_button')
                    if open_btn_rect and open_btn_rect.collidepoint(mouse_pos_main):
                        show_algo_menu = True
                        scroll_y = 0
                        clicked_something_handled = True

                if not clicked_something_handled:
                    solve_btn = ui_elements_rects.get('solve_button')
                    reset_sol_btn = ui_elements_rects.get('reset_solution_button')
                    reset_all_btn = ui_elements_rects.get('reset_all_button')
                    reset_disp_btn = ui_elements_rects.get('reset_initial_button')

                    if solve_btn and solve_btn.collidepoint(mouse_pos_main):
                        if not is_auto_animating_flag and not is_solving_flag:
                            should_start_solving = False
                            solver_algorithm_to_run = None # Reset before checks
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
                                        should_start_solving = True

                            elif current_selected_algorithm == 'Backtracking':
                                sub_choice = backtracking_selection_popup()
                                if sub_choice:
                                    backtracking_sub_algo_choice = sub_choice
                                    solver_algorithm_to_run = 'Backtracking'
                                    should_start_solving = True

                            elif current_selected_algorithm in ['Sensorless', 'Unknown']:
                                if isinstance(temp_state_for_solving, list) and len(temp_state_for_solving) == GRID_SIZE and \
                                   all(isinstance(r, list) and len(r)==GRID_SIZE for r in temp_state_for_solving):
                                    solver_algorithm_to_run = current_selected_algorithm
                                    should_start_solving = True
                                else:
                                     show_popup("Initial state structure is invalid for Unknown/Sensorless.", "Invalid Template")

                            else: # Standard Solvers
                                if any(cell is None for row in temp_state_for_solving for cell in row):
                                     show_popup(f"Initial state must be complete for {current_selected_algorithm}.\nUse AC3 Preprocessing to fill.", "Incomplete State")
                                elif not is_valid_state_for_solve(temp_state_for_solving):
                                    show_popup(f"Initial state is not valid for {current_selected_algorithm}.", "Invalid State")
                                elif not is_solvable(temp_state_for_solving):
                                     show_popup(f"The current initial state is not solvable.", "Unsolvable State")
                                else:
                                    solver_algorithm_to_run = current_selected_algorithm
                                    should_start_solving = True

                            if should_start_solving and solver_algorithm_to_run:
                                is_solving_flag = True
                                solution_path_anim = None
                                action_plan_anim = None
                                current_step_in_anim = 0
                                is_auto_animating_flag = False
                                current_sensorless_belief_size_for_display = None
                        clicked_something_handled = True


                    elif reset_sol_btn and reset_sol_btn.collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        current_sensorless_belief_size_for_display = None
                        solver_algorithm_to_run = None
                        clicked_something_handled = True

                    elif reset_all_btn and reset_all_btn.collidepoint(mouse_pos_main):
                        initial_state = copy.deepcopy(initial_state_fixed_global)
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        all_solve_times.clear()
                        last_run_solver_info.clear()
                        current_sensorless_belief_size_for_display = None
                        selected_cell_for_input_coords = None
                        solver_algorithm_to_run = None
                        clicked_something_handled = True

                    elif reset_disp_btn and reset_disp_btn.collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        action_plan_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        solver_algorithm_to_run = None
                        clicked_something_handled = True


            elif event.type == pygame.MOUSEWHEEL and show_algo_menu:
                menu_data_main = ui_elements_rects.get('menu', {})
                menu_area_main = menu_data_main.get('menu_area')
                if menu_area_main and menu_area_main.collidepoint(mouse_pos_main) and total_menu_height > HEIGHT:
                    scroll_amount_mw = event.y * 35
                    max_scroll_val = max(0, total_menu_height - HEIGHT)
                    scroll_y = max(0, min(scroll_y - scroll_amount_mw, max_scroll_val))

            elif event.type == pygame.KEYDOWN:
                if not is_solving_flag and not is_auto_animating_flag and selected_cell_for_input_coords:
                    r, c = selected_cell_for_input_coords

                    if pygame.K_0 <= event.key <= pygame.K_8:
                        num = event.key - pygame.K_0
                        can_place = True
                        # Check for duplicates (0 can be anywhere, others unique)
                        if num != 0:
                            for row_idx, row_val in enumerate(initial_state):
                                for col_idx, cell_val in enumerate(row_val):
                                    if cell_val == num and (row_idx != r or col_idx != c):
                                        can_place = False; break
                                if not can_place: break

                        if can_place:
                            initial_state[r][c] = num
                            current_state_for_animation = copy.deepcopy(initial_state)
                            solution_path_anim = None
                            action_plan_anim = None
                            current_step_in_anim = 0

                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        if initial_state[r][c] is not None:
                             initial_state[r][c] = None
                             current_state_for_animation = copy.deepcopy(initial_state)
                             solution_path_anim = None
                             action_plan_anim = None
                             current_step_in_anim = 0

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

            algo_name_for_stats = solver_algorithm_to_run
            if current_selected_algorithm == 'AC3' and solver_algorithm_to_run:
                algo_name_for_stats = f"AC3->{solver_algorithm_to_run}"
            elif current_selected_algorithm == 'Backtracking': # Name set later if successful
                 algo_name_for_stats = 'Backtracking' # Default if sub-algo not chosen or fails
            elif solver_algorithm_to_run:
                 algo_name_for_stats = solver_algorithm_to_run


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
                            algo_name_for_stats = f"BT({backtracking_sub_algo_choice})"
                        # else: algo_name_for_stats remains 'Backtracking'
                        backtracking_sub_algo_choice = None
                    else:
                        error_message_solve = "Backtracking sub-algorithm choice missing."
                        error_during_solve = True

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
                        else:
                             found_path_algo = None
                             # show_popup(f"{solver_algorithm_to_run}: Plan found, but cannot visualize invalid sample.", "Visualization Warning")
                        current_state_for_animation = copy.deepcopy(vis_start_state)

                else:
                    algo_func_map = {
                        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar,
                        'Greedy': greedy, 'IDA*': ida_star,
                        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
                        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
                        'Beam Search': beam_search, 'AND-OR': and_or_search
                    }
                    selected_algo_function = algo_func_map.get(solver_algorithm_to_run)

                    if selected_algo_function:
                        algo_args_list = [state_to_solve_from]
                        func_varnames = selected_algo_function.__code__.co_varnames[:selected_algo_function.__code__.co_argcount]
                        default_time_limit = 30
                        if solver_algorithm_to_run in ['IDA*', 'IDS']: default_time_limit = 60

                        if 'time_limit' in func_varnames:
                            algo_args_list.append(default_time_limit)
                        elif solver_algorithm_to_run == 'DFS' and 'max_depth' in func_varnames:
                            algo_args_list.append(30)
                        elif solver_algorithm_to_run == 'IDS' and 'max_depth_limit' in func_varnames:
                             algo_args_list.append(30)
                        elif solver_algorithm_to_run == 'Stochastic Hill' and 'max_iter_no_improve' in func_varnames:
                             algo_args_list.append(500)
                        elif solver_algorithm_to_run == 'Beam Search' and 'beam_width' in func_varnames:
                             algo_args_list.append(5)

                        found_path_algo = selected_algo_function(*algo_args_list)
                        action_plan_anim = None
                    else:
                         error_message_solve = f"Algorithm '{solver_algorithm_to_run}' function not found."
                         error_during_solve = True

            except Exception as e:
                error_message_solve = f"Runtime Error during {algo_name_for_stats} solve:\n{traceback.format_exc()}"
                error_during_solve = True
                action_plan_anim = None
                found_path_algo = None

            solve_duration_t = time.time() - solve_start_t
            all_solve_times[algo_name_for_stats] = solve_duration_t


            if error_during_solve:
                show_popup(error_message_solve if error_message_solve else "An unknown error occurred.", "Solver Error")
                last_run_solver_info.pop(f"{algo_name_for_stats}_reached_goal", None)
                last_run_solver_info.pop(f"{algo_name_for_stats}_steps", None)
                last_run_solver_info.pop(f"{algo_name_for_stats}_actions", None)
            else:
                result_found = (action_plan_anim is not None) or \
                               (found_path_algo and len(found_path_algo) > 0)

                if result_found:
                    solution_path_anim = found_path_algo
                    current_step_in_anim = 0
                    is_auto_animating_flag = (solution_path_anim is not None)
                    last_anim_step_time = time.time()

                    is_actually_goal_state = False
                    num_steps_or_actions = 0
                    popup_msg = ""

                    is_plan_based = solver_algorithm_to_run in ['Sensorless', 'Unknown'] or \
                                    algo_name_for_stats.startswith('BT(Sensorless') or \
                                    algo_name_for_stats.startswith('BT(Unknown')

                    if is_plan_based:
                         if action_plan_anim is not None:
                            num_steps_or_actions = len(action_plan_anim)
                            last_run_solver_info[f"{algo_name_for_stats}_actions"] = num_steps_or_actions
                            is_actually_goal_state = True
                            popup_msg = f"{algo_name_for_stats}: Plan found!\n{num_steps_or_actions} actions."
                            if 'Unknown' in algo_name_for_stats:
                                popup_msg += "\n(Visualizing on one sample state)"
                         else:
                            is_actually_goal_state = None
                            popup_msg = f"{algo_name_for_stats}: Plan search failed."
                            last_run_solver_info[f"{algo_name_for_stats}_reached_goal"] = None

                    else:
                        if solution_path_anim:
                             final_state_of_path = solution_path_anim[-1]
                             if isinstance(final_state_of_path, list):
                                 is_actually_goal_state = is_goal(final_state_of_path)
                             else:
                                 is_actually_goal_state = False
                             num_steps_or_actions = len(solution_path_anim) - 1
                             last_run_solver_info[f"{algo_name_for_stats}_steps"] = num_steps_or_actions
                             popup_msg = f"{algo_name_for_stats} "
                             if is_actually_goal_state:
                                 popup_msg += f"found solution!\n{num_steps_or_actions} steps."
                             else:
                                 popup_msg += f"finished.\n{num_steps_or_actions} steps (Not Goal)."
                        else:
                             is_actually_goal_state = False
                             popup_msg = f"{algo_name_for_stats}: No valid path found."

                    if is_actually_goal_state is not None:
                         last_run_solver_info[f"{algo_name_for_stats}_reached_goal"] = is_actually_goal_state

                    popup_msg += f"\nTime: {solve_duration_t:.3f}s"
                    popup_title = "Solve Complete" if is_actually_goal_state is True else \
                                  ("Plan Search Failed" if is_actually_goal_state is None else "Search Finished")
                    show_popup(popup_msg, popup_title)

                else:
                    last_run_solver_info[f"{algo_name_for_stats}_reached_goal"] = False
                    last_run_solver_info.pop(f"{algo_name_for_stats}_steps", None)
                    last_run_solver_info.pop(f"{algo_name_for_stats}_actions", None)

                    no_solution_msg = error_message_solve if error_message_solve else f"No solution by {algo_name_for_stats}."

                    used_time_limit_for_msg = 30
                    check_algo_id_for_timeout = solver_algorithm_to_run
                    if check_algo_id_for_timeout in ['IDA*', 'IDS', 'Sensorless', 'Unknown', 'Backtracking']:
                        used_time_limit_for_msg = 60

                    if solve_duration_t >= used_time_limit_for_msg * 0.98 :
                         no_solution_msg = f"{algo_name_for_stats} timed out after ~{used_time_limit_for_msg}s."

                    show_popup(no_solution_msg, "No Solution / Timeout")

            solver_algorithm_to_run = None
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
                    if next_anim_state and isinstance(next_anim_state, list):
                        current_state_for_animation = copy.deepcopy(next_anim_state)
                    else:
                        is_auto_animating_flag = False
                    last_anim_step_time = current_time_anim
                else:
                    is_auto_animating_flag = False


        belief_size_for_display = current_sensorless_belief_size_for_display
        active_algo_display_name = solver_algorithm_to_run if solver_algorithm_to_run else current_selected_algorithm
        if active_algo_display_name in ['Sensorless', 'Unknown'] and belief_size_for_display is None and not is_solving_flag and not is_auto_animating_flag:
             try:
                 if isinstance(initial_state, list) and len(initial_state) == GRID_SIZE and \
                    all(isinstance(r, list) and len(r)==GRID_SIZE for r in initial_state):
                     temp_initial_bs = generate_belief_states(initial_state)
                     belief_size_for_display = len(temp_initial_bs) if temp_initial_bs is not None else 0
                 else:
                     belief_size_for_display = 0
             except Exception:
                 belief_size_for_display = 0


        ui_elements_rects = draw_grid_and_ui(
            current_state_for_animation, show_algo_menu,
            current_selected_algorithm,
            solver_algorithm_to_run, 
            all_solve_times,
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