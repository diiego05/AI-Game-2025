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
goal_state = copy.deepcopy(goal_state_fixed_global)


def find_empty(state):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] == 0:
                return i, j
    return -1, -1
def is_goal(state):
    
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        print(f"Invalid state format: {state}")
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            print(f"Invalid row {i} in state: {state[i]}")
            return False
    if state == goal_state:
        print(f"Goal state reached: {state}")
    else:
        print(f"State {state} is not goal: {goal_state}")
    return state == goal_state

def get_neighbors(state):
    
    neighbors = []
    empty_i, empty_j = find_empty(state)
    if empty_i == -1:
        print(f"No empty tile found in state: {state}")
        return []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, xuống, trái, phải
    for di, dj in directions:
        new_i, new_j = empty_i + di, empty_j + dj
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
            new_state = copy.deepcopy(state)
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    print(f"Neighbors of {state}: {neighbors}")
    return neighbors

def state_to_tuple(state):
    if not isinstance(state, list):
        return None
    try:
        return tuple(tuple(row) for row in state)
    except TypeError:
        return None

def tuple_to_list(state_tuple):
    if not isinstance(state_tuple, tuple):
        return None
    try:
        return [list(row) for row in state_tuple]
    except TypeError:
        return None

def apply_action_to_state(state_list, action):
    if state_list is None:
        return None
    new_state = copy.deepcopy(state_list)
    empty_i, empty_j = find_empty(new_state)
    if empty_i == -1:
        return new_state
    di, dj = 0, 0
    if action == 'Up':    di = -1
    elif action == 'Down':  di = 1
    elif action == 'Left':  dj = -1
    elif action == 'Right': dj = 1
    else:
        return new_state

    new_i, new_j = empty_i + di, empty_j + dj
    if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
        new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
        return new_state
    return state_list


def manhattan_distance(state):
    
    distance = 0
    goal_pos = {}
    for r_goal in range(GRID_SIZE):
        for c_goal in range(GRID_SIZE):
            val = goal_state[r_goal][c_goal]
            if val != 0:
                goal_pos[val] = (r_goal, c_goal)

    print(f"Goal positions: {goal_pos}")
    for r_curr in range(GRID_SIZE):
        for c_curr in range(GRID_SIZE):
            tile = state[r_curr][c_curr]
            if tile is not None and tile != 0 and tile in goal_pos:
                goal_r, goal_c = goal_pos[tile]
                tile_distance = abs(r_curr - goal_r) + abs(c_curr - goal_c)
                distance += tile_distance
                print(f"Tile {tile} at ({r_curr},{c_curr}), goal at ({goal_r},{goal_c}), distance: {tile_distance}")
            elif tile is None:
                print(f"Warning: None found in state at ({r_curr},{c_curr})")
    print(f"Manhattan distance for {state}: {distance}")
    return distance

def is_valid_state_for_solve(state_to_check):
    flat_state = []
    try:
        for row in state_to_check:
            for tile in row:
                if tile is None: return False
                flat_state.append(tile)
    except TypeError:
        return False
        
    if len(flat_state) != GRID_SIZE * GRID_SIZE:
        return False
    
    seen_numbers = set()
    for tile in flat_state:
        if not isinstance(tile, int) or not (0 <= tile < GRID_SIZE * GRID_SIZE):
            return False
        if tile in seen_numbers:
            return False
        seen_numbers.add(tile)
    return len(seen_numbers) == GRID_SIZE * GRID_SIZE


def execute_plan(start_state, action_plan):
    if not action_plan:
        return [start_state]
    current_state = copy.deepcopy(start_state)
    state_sequence = [current_state]
    for action in action_plan:
        next_state = apply_action_to_state(current_state, action)
        current_state = next_state
        state_sequence.append(copy.deepcopy(current_state))
    return state_sequence

def get_inversions(flat_state_no_zero):
    inversions = 0
    n = len(flat_state_no_zero)
    for i in range(n):
        for j in range(i + 1, n):
            if flat_state_no_zero[i] > flat_state_no_zero[j]:
                inversions += 1
    return inversions

def is_solvable(state_to_check):
    flat_state = []
    for r_idx, r_val in enumerate(state_to_check): 
        if not isinstance(r_val, list) or len(r_val) != GRID_SIZE: return False
        for c_idx, c_val in enumerate(r_val):
            if c_val is None : return False
            if not isinstance(c_val, int) or c_val < 0 or c_val >= GRID_SIZE*GRID_SIZE :
                 return False
            flat_state.append(c_val)
            
    if len(flat_state) != GRID_SIZE*GRID_SIZE: return False
    if 0 not in flat_state: return False 

    state_flat_no_zero = [tile for tile in flat_state if tile != 0]
    if len(set(state_flat_no_zero)) != GRID_SIZE*GRID_SIZE -1 : return False

    inversions_state = get_inversions(state_flat_no_zero)

    goal_flat = []
    for r_goal in goal_state_fixed_global: goal_flat.extend(r_goal)
    goal_flat_no_zero = [tile for tile in goal_flat if tile != 0]
    inversions_goal = get_inversions(goal_flat_no_zero)

    if GRID_SIZE % 2 == 1:
        return (inversions_state % 2) == (inversions_goal % 2)
    else:
        blank_row_state = -1
        for r_idx, row_s in enumerate(state_to_check):
            if 0 in row_s:
                blank_row_state = GRID_SIZE - r_idx
                break
        
        blank_row_goal = -1
        for r_idx_g, row_g in enumerate(goal_state_fixed_global):
            if 0 in row_g:
                blank_row_goal = GRID_SIZE - r_idx_g
                break

        if blank_row_state == -1 or blank_row_goal == -1: return False

        return ((inversions_state + blank_row_state) % 2) == \
               ((inversions_goal + blank_row_goal) % 2)


def bfs(start_node_state, time_limit=30):
    start_time = time.time()
    queue = deque([(start_node_state, [start_node_state])])
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None
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


def ids(start_node_state, max_depth_limit=30, time_limit=30):
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
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None
    
    frontier.put((0, start_node_state, [start_node_state]))
    visited = {init_tuple: 0}

    while not frontier.empty():
        if time.time() - start_time > time_limit:
            return None

        cost, current_s, path = frontier.get()
        current_tuple = state_to_tuple(current_s) 

        if current_tuple is None: continue 

        if cost > visited[current_tuple]:
            continue

        if is_goal(current_s):
            return path

        for neighbor_s in get_neighbors(current_s):
            neighbor_tuple = state_to_tuple(neighbor_s)
            if neighbor_tuple is None: continue

            new_cost = cost + 1 
            
            if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                visited[neighbor_tuple] = new_cost
                frontier.put((new_cost, neighbor_s, path + [neighbor_s]))
    return None

def astar(start_node_state, time_limit=30):
    start_time = time.time()
    frontier = PriorityQueue() 
    
    g_init = 0
    h_init = manhattan_distance(start_node_state)
    f_init = g_init + h_init
    
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

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
                f_neighbor = tentative_g_score + h_neighbor
                frontier.put((f_neighbor, tentative_g_score, neighbor_s, path + [neighbor_s]))
    return None

def greedy(start_node_state, time_limit=30): 
    start_time = time.time()
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None: return None

    frontier.put((manhattan_distance(start_node_state), start_node_state, [start_node_state]))
    visited = {init_tuple} 

    while not frontier.empty():
        if time.time() - start_time > time_limit:
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

def heuristic_misplaced(state_list, goal_state_list):
    """
    Hàm heuristic tính số ô sai vị trí so với trạng thái đích.
    Args:
        state_list: Trạng thái hiện tại (danh sách 2D).
        goal_state_list: Trạng thái đích (danh sách 2D).
    Returns:
        int: Số ô sai vị trí (không tính ô trống 0).
    """
    misplaced = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if state_list[r][c] != 0 and state_list[r][c] != goal_state_list[r][c]:
                misplaced += 1
    print(f"Misplaced heuristic for {state_list}: {misplaced}")
    return misplaced
    
def ida_star(start_node_state, time_limit=60):
    """
    Thuật toán IDA* để giải bài toán 8-puzzle.
    Args:
        start_node_state: Trạng thái ban đầu (2D).
        time_limit: Giới hạn thời gian (giây).
    Returns:
        list: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy.
    """
    start_time_global = time.time()
    print(f"Starting IDA* with initial state: {start_node_state}")
    print(f"Goal state: {goal_state}")

    def search(current_state_list, g_cost, threshold, current_path_list, visited_in_path):
        """
        Hàm đệ quy cho tìm kiếm IDA*.
        Args:
            current_state_list: Trạng thái hiện tại.
            g_cost: Chi phí từ trạng thái ban đầu đến hiện tại.
            threshold: Ngưỡng f-score hiện tại.
            current_path_list: Đường đi hiện tại.
            visited_in_path: Tập hợp trạng thái đã thăm trong đường đi hiện tại.
        Returns:
            tuple: (ngưỡng mới hoặc f-cost, đường đi nếu tìm thấy hoặc None).
        """
        if time.time() - start_time_global >= time_limit:
            print("IDA* timeout")
            return float('inf'), None

        current_state_tuple = tuple(map(tuple, current_state_list))
        if current_state_tuple in visited_in_path:
            print(f"Cycle detected at state: {current_state_list}")
            return float('inf'), None

        f_cost = g_cost + heuristic_misplaced(current_state_list, goal_state)
        print(f"Exploring state: {current_state_list}, g: {g_cost}, h: {f_cost - g_cost}, f: {f_cost}, threshold: {threshold}")

        if f_cost > threshold:
            print(f"f-cost {f_cost} exceeds threshold {threshold}")
            return f_cost, None

        if is_goal(current_state_list):
            print(f"Goal found with path: {current_path_list + [current_state_list]}")
            return f_cost, current_path_list + [current_state_list]

        min_next_threshold = float('inf')
        visited_in_path.add(current_state_tuple)

        for neighbor_list in get_neighbors(current_state_list):
            new_threshold, result_path = search(
                neighbor_list, g_cost + 1, threshold, current_path_list + [current_state_list], visited_in_path.copy()
            )
            if result_path:
                print(f"Solution found: {result_path}")
                return new_threshold, result_path
            min_next_threshold = min(min_next_threshold, new_threshold)

        print(f"Returning min threshold: {min_next_threshold}")
        return min_next_threshold, None

    threshold = heuristic_misplaced(start_node_state, goal_state)
    print(f"Initial threshold: {threshold}")

    while True:
        if time.time() - start_time_global >= time_limit:
            print("IDA* global timeout")
            return None

        print(f"New iteration with threshold: {threshold}")
        new_threshold, result_path = search(start_node_state, 0, threshold, [], set())
        if result_path:
            print(f"Solution path: {result_path}")
            return result_path
        if new_threshold == float('inf'):
            print("No solution found (infinite threshold)")
            return None
        threshold = new_threshold
        print(f"Updating threshold to: {threshold}")


def simple_hill_climbing(start_node_state, time_limit=30): 
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)

    while True:
        if time.time() - start_time > time_limit:
            return path 

        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors: break 

        best_neighbor_found = None
        moved = False
        for neighbor_s in neighbors: 
            h_neighbor = manhattan_distance(neighbor_s)
            if h_neighbor < current_h:
                best_neighbor_found = neighbor_s
                current_h = h_neighbor 
                moved = True
                break 

        if not moved or best_neighbor_found is None: 
            return path 

        current_s = best_neighbor_found
        path.append(current_s)


def steepest_hill_climbing(start_node_state, time_limit=30): 
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)

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
            if h_neighbor < best_next_h: 
                best_next_h = h_neighbor
                best_next_s = neighbor_s
        
        if best_next_s is None: 
            return path 

        current_s = best_next_s
        current_h = best_next_h
        path.append(current_s)

def random_hill_climbing(start_node_state, time_limit=30, max_iter_no_improve=500): 
    start_time = time.time()
    current_s = start_node_state
    path = [current_s]
    current_h = manhattan_distance(current_s)
    iter_no_improve = 0

    while True:
        if time.time() - start_time > time_limit:
            return path
        
        if is_goal(current_s):
            return path

        neighbors = get_neighbors(current_s)
        if not neighbors:
            break
        
        random_neighbor = random.choice(neighbors)
        neighbor_h = manhattan_distance(random_neighbor)

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


def beam_search(start_node_state, beam_width=5, time_limit=30):
    start_time = time.time()
    if is_goal(start_node_state): return [start_node_state]

    h_init = manhattan_distance(start_node_state)
    beam = [(start_node_state, [start_node_state], h_init)]
    visited_tuples_global = {state_to_tuple(start_node_state)}
    best_goal_path_found = None

    while beam: 
        if time.time() - start_time > time_limit:
            return best_goal_path_found 

        next_level_candidates = [] 

        for s_beam, path_beam, _ in beam:
            for neighbor_s_beam in get_neighbors(s_beam):
                neighbor_tuple_beam = state_to_tuple(neighbor_s_beam)
                if neighbor_tuple_beam is None: 
                    continue 

                new_path_beam = path_beam + [neighbor_s_beam]
                
                if is_goal(neighbor_s_beam):
                    if best_goal_path_found is None or len(new_path_beam) < len(best_goal_path_found):
                        best_goal_path_found = new_path_beam

                h_neighbor_beam = manhattan_distance(neighbor_s_beam)
                if neighbor_tuple_beam not in visited_tuples_global:
                     next_level_candidates.append((neighbor_s_beam, new_path_beam, h_neighbor_beam))
        
        if not next_level_candidates: 
            break

        next_level_candidates.sort(key=lambda x: x[2])
        
        new_beam = []
        temp_added_to_new_beam_tuples = set() 
        for cand_s, cand_path, cand_h in next_level_candidates:
            if len(new_beam) >= beam_width: break 

            cand_tuple = state_to_tuple(cand_s)
            if cand_tuple not in visited_tuples_global and cand_tuple not in temp_added_to_new_beam_tuples :
                 new_beam.append((cand_s, cand_path, cand_h))
                 visited_tuples_global.add(cand_tuple) 
                 temp_added_to_new_beam_tuples.add(cand_tuple)
        beam = new_beam
        
    return best_goal_path_found


def _and_or_recursive(state_ao, path_ao, solved_states_ao, unsolved_states_ao, start_time_ao, time_limit_ao, depth_ao, max_depth_ao=50):
    state_tuple_ao = state_to_tuple(state_ao)
    if state_tuple_ao is None: return "UNSOLVED", None

    if time.time() - start_time_ao > time_limit_ao:
        return "Timeout", None
    
    if depth_ao > max_depth_ao: 
        return "UNSOLVED", None 

    if is_goal(state_ao):
        solved_states_ao.add(state_tuple_ao)
        return "SOLVED", path_ao
    
    if state_tuple_ao in solved_states_ao: 
        return "SOLVED", path_ao 

    if state_tuple_ao in unsolved_states_ao: 
        return "UNSOLVED", None

    unsolved_states_ao.add(state_tuple_ao) 

    for neighbor_ao in get_neighbors(state_ao):
        neighbor_tuple_ao = state_to_tuple(neighbor_ao)
        if neighbor_tuple_ao is None: continue

        if neighbor_tuple_ao not in solved_states_ao and neighbor_tuple_ao not in unsolved_states_ao:
            status_ao, solution_path_ao = _and_or_recursive(
                neighbor_ao, path_ao + [neighbor_ao], 
                solved_states_ao, unsolved_states_ao, 
                start_time_ao, time_limit_ao, depth_ao + 1, max_depth_ao
            )

            if status_ao == "Timeout":
                return "Timeout", None
            if status_ao == "SOLVED":
                solved_states_ao.add(state_tuple_ao)
                if state_tuple_ao in unsolved_states_ao: 
                    unsolved_states_ao.remove(state_tuple_ao)
                return "SOLVED", solution_path_ao 
    
    return "UNSOLVED", None


def and_or_search(start_node_state, time_limit=30): 
    start_time = time.time()
    solved_states = set()    
    unsolved_states = set()  

    init_tuple = state_to_tuple(start_node_state)
    if init_tuple is None:
        return None

    status, solution_path = _and_or_recursive(
        start_node_state, [start_node_state], 
        solved_states, unsolved_states, 
        start_time, time_limit, 0
    )

    if status == "SOLVED":
        return solution_path
    elif status == "Timeout":
        return None
    else: 
        return None

def backtracking_search(selected_sub_algorithm_name, max_attempts=50, time_limit_overall=60):
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

    base_attempt_flat = [0, 1, 2, 3, 4, 5, 6, 7, 8] 
    states_to_try_q.append(list(base_attempt_flat)) 
    tried_start_state_tuples.add(tuple(base_attempt_flat))
    
    generated_neighbors_of_base = False
    time_limit_per_sub_solve = max(1.0, time_limit_overall / max_attempts if max_attempts > 0 else time_limit_overall / 5) 
    time_limit_per_sub_solve = min(time_limit_per_sub_solve, 15) 

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
        
        if not is_solvable(current_attempt_2d_start):
            continue 
        
        algo_params = [current_attempt_2d_start] 
        sub_algo_func_varnames = sub_algo_func.__code__.co_varnames[:sub_algo_func.__code__.co_argcount]
        
        if 'time_limit' in sub_algo_func_varnames:
            if selected_sub_algorithm_name == 'IDA*': 
                 algo_params.append(max(time_limit_per_sub_solve, 30)) 
            else:
                algo_params.append(time_limit_per_sub_solve)
        elif selected_sub_algorithm_name == 'DFS': 
            algo_params.append(30) 
            if 'time_limit' in sub_algo_func_varnames: 
                 algo_params.append(time_limit_per_sub_solve)

        path_from_sub_algo = None
        try:
            path_from_sub_algo = sub_algo_func(*algo_params)
        except Exception as e:
            traceback.print_exc() 
        
        if path_from_sub_algo and len(path_from_sub_algo) > 0:
            if is_goal(path_from_sub_algo[-1]):
                solution_found_path = path_from_sub_algo
                actual_start_state_of_solution = current_attempt_2d_start
                break 
        
        if not solution_found_path and not states_to_try_q and attempts < max_attempts:
            if not generated_neighbors_of_base:
                parent_for_new_states = base_attempt_flat 
                newly_added_count = 0
                for i in range(len(parent_for_new_states)):
                    for j in range(i + 1, len(parent_for_new_states)):
                        if len(states_to_try_q) > max_attempts * 2: break 
                        
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
            else: 
                newly_added_count = 0
                for _ in range(min(20, max_attempts - attempts)): 
                    if len(states_to_try_q) > max_attempts * 2: break

                    random_flat_list = list(range(GRID_SIZE * GRID_SIZE))
                    random.shuffle(random_flat_list)
                    random_tuple_to_check = tuple(random_flat_list)

                    if random_tuple_to_check not in tried_start_state_tuples:
                        states_to_try_q.append(random_flat_list)
                        tried_start_state_tuples.add(random_tuple_to_check)
                        newly_added_count +=1
    
    if solution_found_path:
        return solution_found_path, None, actual_start_state_of_solution
    else:
        msg = f"Backtracking: No solution found after {attempts} attempts with {selected_sub_algorithm_name}."
        if time.time() - start_time_overall > time_limit_overall:
            msg = "Backtracking: Global timeout. " + msg
        return None, msg, None

def generate_belief_states(partial_state_template):
    flat_state_template = []
    is_template_with_unknowns = False
    for r_idx, r_val in enumerate(partial_state_template):
        if not isinstance(r_val, list) or len(r_val) != GRID_SIZE:
            return []
        for tile in r_val:
            if not isinstance(tile, int) and tile is not None: # Allow None, but not other non-ints
                return []
            flat_state_template.append(tile)
            if tile is None: is_template_with_unknowns = True # Use None for unknown
    
    if len(flat_state_template) != GRID_SIZE * GRID_SIZE:
        return []

    if not is_template_with_unknowns:
        if is_solvable(partial_state_template):
            return [partial_state_template]
        else:
            return []

    known_numbers_set = set()
    for tile_val in flat_state_template:
        if tile_val is not None:
            if tile_val in known_numbers_set:
                return []
            known_numbers_set.add(tile_val)

    all_possible_tiles = list(range(GRID_SIZE * GRID_SIZE))
    missing_numbers_to_fill = [num for num in all_possible_tiles if num not in known_numbers_set]
    
    unknown_positions_indices_flat = [i for i, tile_val in enumerate(flat_state_template) if tile_val is None]
    
    belief_states_generated = []
    if len(missing_numbers_to_fill) != len(unknown_positions_indices_flat):
        return [] 

    for perm_fill_nums in itertools.permutations(missing_numbers_to_fill):
        new_flat_state_filled = list(flat_state_template) 
        for i, unknown_idx in enumerate(unknown_positions_indices_flat):
            new_flat_state_filled[unknown_idx] = perm_fill_nums[i]
        
        state_2d_filled = []
        for r_idx in range(0, len(new_flat_state_filled), GRID_SIZE):
            state_2d_filled.append(new_flat_state_filled[r_idx : r_idx + GRID_SIZE])
        
        if is_solvable(state_2d_filled):
            belief_states_generated.append(state_2d_filled)
            
    return belief_states_generated


def is_belief_state_goal(list_of_belief_states):
    if not list_of_belief_states: return False
    for state_b in list_of_belief_states:
        if not is_goal(state_b):
            return False
    return True

def apply_action_to_belief_states(list_of_belief_states, action_str):
    new_belief_states_set = set()
    for state_b in list_of_belief_states:
        next_s_b = apply_action_to_state(state_b, action_str)
        if next_s_b:
            next_s_b_tuple = state_to_tuple(next_s_b)
            if next_s_b_tuple:
                 new_belief_states_set.add(next_s_b_tuple)
    return [tuple_to_list(s_tuple) for s_tuple in new_belief_states_set]


def sensorless_search(start_belief_state_template, time_limit=60):
    start_time = time.time()
    current_belief_set = generate_belief_states(start_belief_state_template)
    if not current_belief_set:
        return None, 0
    
    initial_belief_set_size = len(current_belief_set)
    queue_sensorless = deque([ ( [], current_belief_set ) ])
    visited_belief_sets = set()
    
    try:
        initial_valid_tuples = [s_tuple for s_tuple in (state_to_tuple(s) for s in current_belief_set) if s_tuple is not None]
        if len(initial_valid_tuples) != len(current_belief_set):
            pass
        initial_belief_tuple_set = frozenset(initial_valid_tuples)
        visited_belief_sets.add(initial_belief_tuple_set)
    except Exception as e:
        return None, initial_belief_set_size

    possible_actions = ['Up', 'Down', 'Left', 'Right']
    max_plan_length = 30

    while queue_sensorless:
        if time.time() - start_time > time_limit:
            return None, len(queue_sensorless[0][1]) if queue_sensorless else 0

        action_plan, current_bs = queue_sensorless.popleft()
        
        if len(action_plan) > max_plan_length:
            continue

        if is_belief_state_goal(current_bs):
            return action_plan, len(current_bs)
        
        for action_s in possible_actions:
            next_bs = apply_action_to_belief_states(current_bs, action_s)
            if not next_bs:
                continue

            try:
                valid_next_bs_tuples = [s_tuple for s_tuple in (state_to_tuple(s) for s in next_bs) if s_tuple is not None]
                if len(valid_next_bs_tuples) != len(next_bs):
                     pass
                if not valid_next_bs_tuples: 
                    continue
                next_bs_tuple_set = frozenset(valid_next_bs_tuples)
            except Exception as e:
                continue

            if next_bs_tuple_set not in visited_belief_sets:
                visited_belief_sets.add(next_bs_tuple_set)
                new_plan_s = action_plan + [action_s]
                queue_sensorless.append((new_plan_s, next_bs))
                
    return None, 0

def algorithm_selection_popup():
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)

    title_font_popup = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_popup = title_font_popup.render("Select Sub-Algorithm for Backtracking", True, INFO_COLOR)
    title_rect_popup = title_surf_popup.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf_popup, title_rect_popup)

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

        pygame.draw.rect(popup_surface, RED, cancel_button_rect_popup, border_radius=8)
        cancel_text_surf_p = BUTTON_FONT.render("Cancel", True, WHITE)
        cancel_text_rect_p = cancel_text_surf_p.get_rect(center=cancel_button_rect_popup.center)
        popup_surface.blit(cancel_text_surf_p, cancel_text_rect_p)

        screen.blit(popup_surface, popup_rect_on_screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
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

def draw_state(state_to_draw, x_pos, y_pos, title_str, is_current_anim_state=False, is_fixed_goal_display=False, is_editable=False, selected_cell_coords=None):
    title_font_ds = pygame.font.Font(None, 28)
    title_text_ds = title_font_ds.render(title_str, True, BLACK)
    title_x_ds = x_pos + (GRID_DISPLAY_WIDTH // 2 - title_text_ds.get_width() // 2)
    title_y_ds = y_pos - title_text_ds.get_height() - 5
    screen.blit(title_text_ds, (title_x_ds, title_y_ds))

    pygame.draw.rect(screen, BLACK, (x_pos - 1, y_pos - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)

    for r_ds in range(GRID_SIZE):
        for c_ds in range(GRID_SIZE):
            cell_x_ds = x_pos + c_ds * CELL_SIZE
            cell_y_ds = y_pos + r_ds * CELL_SIZE
            cell_rect_ds = pygame.Rect(cell_x_ds, cell_y_ds, CELL_SIZE, CELL_SIZE)
            
            tile_val = None 
            if state_to_draw and r_ds < len(state_to_draw) and isinstance(state_to_draw[r_ds], list) and c_ds < len(state_to_draw[r_ds]):
                 tile_val = state_to_draw[r_ds][c_ds]
            else:
                pygame.draw.rect(screen, RED, cell_rect_ds.inflate(-6,-6), border_radius=8)
                pygame.draw.rect(screen, BLACK, cell_rect_ds, 1)
                continue

            if tile_val is None: 
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
                # No question mark, just empty gray for None
            elif tile_val != 0:
                cell_fill_color = BLUE
                if is_fixed_goal_display:
                    cell_fill_color = GREEN
                elif is_current_anim_state and is_goal(state_to_draw):
                    cell_fill_color = GREEN
                
                pygame.draw.rect(screen, cell_fill_color, cell_rect_ds.inflate(-6, -6), border_radius=8)
                number_surf = FONT.render(str(tile_val), True, WHITE)
                screen.blit(number_surf, number_surf.get_rect(center=cell_rect_ds.center))
            else: # tile_val is 0 (empty)
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
            
            pygame.draw.rect(screen, BLACK, cell_rect_ds, 1)

            if is_editable and selected_cell_coords == (r_ds, c_ds):
                 pygame.draw.rect(screen, RED, cell_rect_ds, 3) # Highlight selected cell

def show_popup(message_str, title_str="Info"):
    popup_surface_sp = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface_sp.fill(INFO_BG)
    border_rect_sp = popup_surface_sp.get_rect()
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, border_rect_sp, 4, border_radius=10)

    title_font_sp = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf_sp = title_font_sp.render(title_str, True, INFO_COLOR)
    title_rect_sp = title_surf_sp.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface_sp.blit(title_surf_sp, title_rect_sp)

    words_sp = message_str.split(' ')
    lines_sp = []
    current_line_sp = ""
    text_width_limit_sp = POPUP_WIDTH - 60
    
    for word_sp in words_sp:
        if "\n" in word_sp:
            parts = word_sp.split("\n")
            for i, part_sp in enumerate(parts):
                if not part_sp and i < len(parts) -1 :
                    lines_sp.append(current_line_sp)
                    current_line_sp = ""
                    lines_sp.append("")
                    continue
                if not part_sp: continue

                test_line_part_sp = current_line_sp + (" " if current_line_sp and part_sp else "") + part_sp
                line_width_part_sp = INFO_FONT.size(test_line_part_sp)[0]
                
                if line_width_part_sp <= text_width_limit_sp:
                    current_line_sp = test_line_part_sp
                else:
                    lines_sp.append(current_line_sp)
                    current_line_sp = part_sp
                
                if i < len(parts) - 1:
                    lines_sp.append(current_line_sp)
                    current_line_sp = ""
            continue

        test_line_sp = current_line_sp + (" " if current_line_sp else "") + word_sp
        line_width_sp = INFO_FONT.size(test_line_sp)[0]
        
        if line_width_sp <= text_width_limit_sp:
            current_line_sp = test_line_sp
        else:
            lines_sp.append(current_line_sp)
            current_line_sp = word_sp
    
    lines_sp.append(current_line_sp)

    line_height_sp = INFO_FONT.get_linesize()
    text_start_y_sp = title_rect_sp.bottom + 20
    
    for i, line_text_sp in enumerate(lines_sp):
        if text_start_y_sp + i * line_height_sp > POPUP_HEIGHT - 80:
            ellipsis_surf = INFO_FONT.render("...", True, BLACK)
            popup_surface_sp.blit(ellipsis_surf, ( (POPUP_WIDTH - ellipsis_surf.get_width()) // 2, text_start_y_sp + i * line_height_sp))
            break
        text_surf_sp = INFO_FONT.render(line_text_sp, True, BLACK)
        text_rect_sp = text_surf_sp.get_rect(center=(POPUP_WIDTH // 2, text_start_y_sp + i * line_height_sp + line_height_sp // 2))
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
                if ok_button_rect_sp.collidepoint(mouse_x_sp - popup_rect_on_screen_sp.left, 
                                                  mouse_y_sp - popup_rect_on_screen_sp.top):
                    waiting_for_ok = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    waiting_for_ok = False
        pygame.time.delay(20)

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
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Backtracking', 'Backtracking'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Sensorless', 'Sensorless Plan')
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

def draw_grid_and_ui(current_anim_state_dgui, show_menu_dgui, current_algo_name_dgui, 
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

    draw_state(initial_state, initial_grid_x, top_row_y_grids, "Initial State", is_editable=True, selected_cell_coords=selected_cell_for_input_coords)
    draw_state(goal_state, goal_grid_x, top_row_y_grids, "Goal State", is_fixed_goal_display=True)

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
    
    current_state_title = f"Current ({current_algo_name_dgui})"
    if current_algo_name_dgui == 'Sensorless' and current_belief_state_size_dgui is not None:
        current_state_title = f"Belief States: {current_belief_state_size_dgui}"
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
        sorted_times_list = sorted(solve_times_dgui.items(), key=lambda item: (item[1], item[0]))
        
        for algo_name_st, time_val_st in sorted_times_list:
            if current_info_y_draw + line_h_ia > info_area_y_pos + info_area_h - info_pad_y_ia:
                screen.blit(INFO_FONT.render("...", True, BLACK), (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
                break

            steps_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_steps")
            actions_val_st = last_solved_run_info_dgui.get(f"{algo_name_st}_actions")
            reached_goal_st = last_solved_run_info_dgui.get(f"{algo_name_st}_reached_goal", None)

            base_str_st = f"{algo_name_st}: {time_val_st:.3f}s"
            count_str_st = ""
            if steps_val_st is not None: count_str_st = f" ({steps_val_st} steps)"
            elif actions_val_st is not None: count_str_st = f" ({actions_val_st} actions)"
            else: count_str_st = " (--)"

            goal_indicator_st = ""
            if reached_goal_st is False : goal_indicator_st = " (Not Goal)"
            
            full_comp_str = base_str_st + count_str_st + goal_indicator_st
            
            max_text_width = info_area_w - 2 * info_pad_x_ia
            if INFO_FONT.size(full_comp_str)[0] > max_text_width:
                shortened_str = base_str_st + goal_indicator_st
                if INFO_FONT.size(shortened_str)[0] <= max_text_width:
                    full_comp_str = shortened_str
                else:
                    full_comp_str = base_str_st.split(":")[0] + ":" + goal_indicator_st
                    if INFO_FONT.size(full_comp_str)[0] > max_text_width:
                        full_comp_str = (base_str_st.split(":")[0][:10] + "...") if len(base_str_st.split(":")[0]) > 10 else base_str_st.split(":")[0]

            comp_surf_st = INFO_FONT.render(full_comp_str, True, BLACK)
            screen.blit(comp_surf_st, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))
            current_info_y_draw += line_h_ia
    else:
        no_results_surf = INFO_FONT.render("(No results yet)", True, GRAY)
        screen.blit(no_results_surf, (info_area_x_pos + info_pad_x_ia, current_info_y_draw))

    menu_elements_dgui = draw_menu(show_menu_dgui, mouse_pos_dgui, current_algo_name_dgui)
    pygame.display.flip()

    # Store rects of grids for click detection
    initial_grid_rect_on_screen = pygame.Rect(initial_grid_x, top_row_y_grids, GRID_DISPLAY_WIDTH, GRID_DISPLAY_WIDTH)

    return {
        'solve_button': solve_button_rect_dgui,
        'reset_solution_button': reset_solution_button_rect_dgui,
        'reset_all_button': reset_all_button_rect_dgui,
        'reset_initial_button': reset_initial_button_rect_dgui,
        'menu': menu_elements_dgui,
        'initial_grid_area': initial_grid_rect_on_screen 
    }

def main():
    global scroll_y, initial_state, goal_state
    current_state_for_animation = copy.deepcopy(initial_state)
    solution_path_anim = None
    current_step_in_anim = 0
    is_solving_flag = False
    is_auto_animating_flag = False
    last_anim_step_time = 0
    show_algo_menu = False
    current_selected_algorithm = 'A*'
    all_solve_times = {}
    last_run_solver_info = {}
    game_clock = pygame.time.Clock()
    ui_elements_rects = {}
    running_main_loop = True
    backtracking_sub_algo_choice = None
    current_sensorless_belief_size_for_display = None
    selected_cell_for_input_coords = None # (row, col) of the initial_state grid cell selected for input

    while running_main_loop:
        mouse_pos_main = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_main_loop = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_something_handled = False
                # Handle clicks on initial_state grid cells if editable
                initial_grid_area_rect = ui_elements_rects.get('initial_grid_area')
                if initial_grid_area_rect and initial_grid_area_rect.collidepoint(mouse_pos_main):
                    # Click is within the initial state grid area
                    grid_offset_x = initial_grid_area_rect.left
                    grid_offset_y = initial_grid_area_rect.top
                    
                    clicked_col = (mouse_pos_main[0] - grid_offset_x) // CELL_SIZE
                    clicked_row = (mouse_pos_main[1] - grid_offset_y) // CELL_SIZE

                    if 0 <= clicked_row < GRID_SIZE and 0 <= clicked_col < GRID_SIZE:
                        selected_cell_for_input_coords = (clicked_row, clicked_col)
                        clicked_something_handled = True
                    else:
                        selected_cell_for_input_coords = None # Clicked padding, deselect
                elif not (menu_surface and menu_surface.get_rect().collidepoint(mouse_pos_main) and show_algo_menu):
                    # Clicked outside editable grid and not on open menu, deselect
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
                            for algo_id, local_r in algo_buttons_local_rects.items():
                                screen_button_r = local_r.move(0, -scroll_y)
                                if screen_button_r.collidepoint(mouse_pos_main):
                                    if current_selected_algorithm != algo_id:
                                        current_selected_algorithm = algo_id
                                        solution_path_anim = None
                                        current_step_in_anim = 0
                                        is_auto_animating_flag = False
                                        is_solving_flag = False
                                        current_state_for_animation = copy.deepcopy(initial_state)
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
                    if solve_btn and solve_btn.collidepoint(mouse_pos_main):
                        if not is_auto_animating_flag and not is_solving_flag:
                            if current_selected_algorithm == 'Backtracking':
                                backtracking_sub_algo_choice = algorithm_selection_popup()
                                if backtracking_sub_algo_choice:
                                    is_solving_flag = True
                            elif current_selected_algorithm == 'Sensorless':
                                is_solving_flag = True # No validation for Sensorless initial state
                            else:
                                if is_valid_state_for_solve(initial_state):
                                    if is_solvable(initial_state):
                                        is_solving_flag = True
                                    else:
                                        show_popup(f"The current initial state ({initial_state}) is not solvable relative to the goal state.", "Unsolvable State")
                                else:
                                    show_popup(f"Initial state ({initial_state}) is not fully defined (must be numbers 0-8, no duplicates) for standard algorithms.", "Invalid Initial State")
                            
                            if is_solving_flag:
                                solution_path_anim = None
                                current_step_in_anim = 0
                                is_auto_animating_flag = False
                                current_sensorless_belief_size_for_display = None
                        clicked_something_handled = True

                    reset_sol_btn = ui_elements_rects.get('reset_solution_button')
                    if not clicked_something_handled and reset_sol_btn and reset_sol_btn.collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        current_sensorless_belief_size_for_display = None
                        clicked_something_handled = True

                    reset_all_btn = ui_elements_rects.get('reset_all_button')
                    if not clicked_something_handled and reset_all_btn and reset_all_btn.collidepoint(mouse_pos_main):
                        initial_state = copy.deepcopy(initial_state_fixed_global)
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        all_solve_times.clear()
                        last_run_solver_info.clear()
                        current_sensorless_belief_size_for_display = None
                        selected_cell_for_input_coords = None
                        clicked_something_handled = True
                    
                    reset_disp_btn = ui_elements_rects.get('reset_initial_button')
                    if not clicked_something_handled and reset_disp_btn and reset_disp_btn.collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None
                        current_step_in_anim = 0
                        is_solving_flag = False
                        is_auto_animating_flag = False
                        clicked_something_handled = True
                
            if event.type == pygame.MOUSEWHEEL and show_algo_menu:
                menu_data_main = ui_elements_rects.get('menu', {})
                menu_area_main = menu_data_main.get('menu_area')
                if menu_area_main and menu_area_main.collidepoint(mouse_pos_main) and total_menu_height > HEIGHT:
                    scroll_amount_mw = event.y * 35
                    max_scroll_val = max(0, total_menu_height - HEIGHT)
                    scroll_y = max(0, min(scroll_y - scroll_amount_mw, max_scroll_val))
            
            if event.type == pygame.KEYDOWN:
                if selected_cell_for_input_coords:
                    r, c = selected_cell_for_input_coords
                    if pygame.K_0 <= event.key <= pygame.K_8:
                        num = event.key - pygame.K_0
                        initial_state[r][c] = num
                        current_state_for_animation = copy.deepcopy(initial_state) # Update display
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        initial_state[r][c] = None # Use None for empty/unknown
                        current_state_for_animation = copy.deepcopy(initial_state)
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        selected_cell_for_input_coords = None # Deselect on Esc/Enter
                
        if not running_main_loop: break

        if is_solving_flag:
            is_solving_flag = False
            solve_start_t = time.time()
            
            found_path_algo = None
            found_action_plan_algo = None
            actual_start_state_for_anim = None
            belief_size_at_end_sensorless = 0
            
            error_during_solve = False
            error_message_solve = ""

            try:
                state_to_solve_from = copy.deepcopy(initial_state)

                if current_selected_algorithm == 'Backtracking':
                    if backtracking_sub_algo_choice:
                        found_path_algo, error_message_solve, actual_start_state_for_anim = \
                            backtracking_search(backtracking_sub_algo_choice)
                        if found_path_algo:
                             current_state_for_animation = copy.deepcopy(actual_start_state_for_anim)
                             initial_state = copy.deepcopy(actual_start_state_for_anim)
                        backtracking_sub_algo_choice = None
                    else:
                        error_message_solve = "Backtracking sub-algorithm not chosen."
                        error_during_solve = True
                
                elif current_selected_algorithm == 'Sensorless':
                    found_action_plan_algo, belief_size_at_end_sensorless = sensorless_search(state_to_solve_from, time_limit=60)
                    current_sensorless_belief_size_for_display = belief_size_at_end_sensorless
                    if found_action_plan_algo:
                        sample_belief_states = generate_belief_states(state_to_solve_from)
                        vis_start_state = state_to_solve_from
                        if sample_belief_states:
                             vis_start_state = sample_belief_states[0]
                        
                        found_path_algo = execute_plan(vis_start_state, found_action_plan_algo)
                        current_state_for_animation = copy.deepcopy(vis_start_state)
                
                else:
                    current_state_for_animation = copy.deepcopy(state_to_solve_from)
                    algo_func_map = {
                        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 
                        'Greedy': greedy, 'IDA*': ida_star,
                        'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
                        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing,
                        'Beam Search': beam_search, 'AND-OR': and_or_search
                    }
                    selected_algo_function = algo_func_map.get(current_selected_algorithm)

                    if selected_algo_function:
                        algo_args_list = [state_to_solve_from]
                        func_varnames = selected_algo_function.__code__.co_varnames[:selected_algo_function.__code__.co_argcount]
                        default_time_limit = 30
                        if current_selected_algorithm in ['IDA*']: default_time_limit = 60
                        
                        if 'time_limit' in func_varnames:
                            algo_args_list.append(default_time_limit)
                        elif current_selected_algorithm == 'DFS' and 'max_depth' in func_varnames and 'time_limit' not in func_varnames:
                             algo_args_list.append(30)

                        found_path_algo = selected_algo_function(*algo_args_list)
                    else:
                        error_message_solve = f"Algorithm '{current_selected_algorithm}' not found."
                        error_during_solve = True
            
            except Exception as e:
                error_message_solve = f"Error during {current_selected_algorithm} solve:\n{str(e)}"
                traceback.print_exc()
                error_during_solve = True

            solve_duration_t = time.time() - solve_start_t
            all_solve_times[current_selected_algorithm] = solve_duration_t
            
            if error_during_solve:
                show_popup(error_message_solve if error_message_solve else "An unknown error occurred during solve.", "Solver Error")
            else:
                if found_path_algo and len(found_path_algo) > 0:
                    solution_path_anim = found_path_algo
                    current_step_in_anim = 0
                    is_auto_animating_flag = True
                    last_anim_step_time = time.time()

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
                        else:
                            popup_msg += f"finished.\n{num_steps_or_actions} steps (Not Goal)."
                        popup_msg += f"\nTime: {solve_duration_t:.3f}s"
                    
                    last_run_solver_info[f"{current_selected_algorithm}_reached_goal"] = is_actually_goal_state
                    show_popup(popup_msg, "Solve Complete" if is_actually_goal_state else "Search Finished")
                
                else:
                    last_run_solver_info[f"{current_selected_algorithm}_reached_goal"] = False
                    if f"{current_selected_algorithm}_steps" in last_run_solver_info:
                        del last_run_solver_info[f"{current_selected_algorithm}_steps"]
                    if f"{current_selected_algorithm}_actions" in last_run_solver_info:
                        del last_run_solver_info[f"{current_selected_algorithm}_actions"]

                    no_solution_msg = error_message_solve if error_message_solve else f"No solution found by {current_selected_algorithm}."
                    
                    used_time_limit_for_msg = 30 
                    if current_selected_algorithm in ['IDA*', 'Sensorless', 'Backtracking']: used_time_limit_for_msg = 60
                    
                    if solve_duration_t >= used_time_limit_for_msg * 0.95 :
                         no_solution_msg = f"{current_selected_algorithm} timed out after ~{used_time_limit_for_msg}s."
                    
                    show_popup(no_solution_msg, "No Solution / Timeout")

        if is_auto_animating_flag and solution_path_anim:
            current_time_anim = time.time()
            anim_delay_val = 0.3 
            if len(solution_path_anim) > 30: anim_delay_val = 0.15
            if len(solution_path_anim) > 60: anim_delay_val = 0.08

            if current_time_anim - last_anim_step_time >= anim_delay_val:
                if current_step_in_anim < len(solution_path_anim) - 1:
                    current_step_in_anim += 1
                    current_state_for_animation = copy.deepcopy(solution_path_anim[current_step_in_anim])
                    last_anim_step_time = current_time_anim
                else:
                    is_auto_animating_flag = False
                    final_anim_state_is_goal = is_goal(current_state_for_animation)
        
        belief_size_for_display = current_sensorless_belief_size_for_display
        if current_selected_algorithm == 'Sensorless' and belief_size_for_display is None:
            temp_initial_bs = generate_belief_states(initial_state)
            belief_size_for_display = len(temp_initial_bs) if temp_initial_bs else 0
            
        ui_elements_rects = draw_grid_and_ui(current_state_for_animation, show_algo_menu, 
                                             current_selected_algorithm, all_solve_times, 
                                             last_run_solver_info, belief_size_for_display,
                                             selected_cell_for_input_coords) # Pass selected cell to draw function
        
        game_clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()