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

# Trạng thái ban đầu và đích
initial_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
goal_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]

# Các hàm tiện ích
def find_empty(state):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] == 0:
                return i, j
    return -1, -1

def is_goal(state):
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            return False
    return state == goal_state

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
        return new_state
    di, dj = 0, 0
    if action == 'Up':
        di = -1
    elif action == 'Down':
        di = 1
    elif action == 'Left':
        dj = -1
    elif action == 'Right':
        dj = 1
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
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            val = goal_state[r][c]
            if val != 0 and val != -1:
                goal_pos[val] = (r, c)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tile = state[i][j]
            if tile != 0 and tile != -1 and tile in goal_pos:
                goal_r, goal_c = goal_pos[tile]
                distance += abs(goal_r - i) + abs(goal_c - j)
    return distance

def is_valid_state(state):
    flat_state = [tile for row in state for tile in row]
    valid_numbers = set(range(9))
    state_numbers = set(tile for tile in flat_state if tile != -1)
    return state_numbers.issubset(valid_numbers) and len(state_numbers) <= len(flat_state) and -1 not in state_numbers

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

# Các thuật toán tìm kiếm (giữ nguyên từ mã gốc, chỉ liệt kê để tham khảo)
def bfs(initial_state, time_limit=30):
    start_time = time.time()
    queue = deque([(initial_state, [initial_state])])
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        return None
    visited = {init_tuple}
    while queue:
        if time.time() - start_time > time_limit:
            print("BFS Timeout")
            return None
        current_state, path = queue.popleft()
        if is_goal(current_state):
            return path
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor, path + [neighbor]))
    return None

def dfs(initial_state, max_depth=30, time_limit=30):
    start_time = time.time()
    stack = [(initial_state, [initial_state], 0)]
    visited = {}
    while stack:
        if time.time() - start_time > time_limit:
            print("DFS Timeout")
            return None
        current_state, path, depth = stack.pop()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None:
            continue
        if current_tuple in visited and visited[current_tuple] <= depth:
            continue
        if depth > max_depth:
            continue
        visited[current_tuple] = depth
        if is_goal(current_state):
            return path
        neighbors = get_neighbors(current_state)
        for neighbor in reversed(neighbors):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is not None and (neighbor_tuple not in visited or visited[neighbor_tuple] > depth + 1):
                stack.append((neighbor, path + [neighbor], depth + 1))
    return None

def ids(initial_state, max_depth_limit=30, time_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        return None
    for depth_limit in range(max_depth_limit + 1):
        if time.time() - start_time > time_limit:
            print("IDS Global Timeout")
            return None
        stack = [(initial_state, [initial_state], 0)]
        visited_in_iteration = {init_tuple: 0}
        while stack:
            if time.time() - start_time > time_limit:
                print("IDS Iteration Timeout")
                return None
            current_state, path, depth = stack.pop()
            if is_goal(current_state):
                return path
            if depth < depth_limit:
                neighbors = get_neighbors(current_state)
                for neighbor in reversed(neighbors):
                    neighbor_tuple = state_to_tuple(neighbor)
                    if neighbor_tuple is not None and (neighbor_tuple not in visited_in_iteration or visited_in_iteration[neighbor_tuple] > depth + 1):
                        visited_in_iteration[neighbor_tuple] = depth + 1
                        stack.append((neighbor, path + [neighbor], depth + 1))
    return None

def ucs(initial_state, time_limit=30):
    start_time = time.time()
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        return None
    frontier.put((0, initial_state, [initial_state]))
    visited = {init_tuple: 0}
    while not frontier.empty():
        if time.time() - start_time > time_limit:
            print("UCS Timeout")
            return None
        cost, current_state, path = frontier.get()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None:
            continue
        if is_goal(current_state):
            return path
        if current_tuple in visited and cost > visited[current_tuple]:
            continue
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is None:
                continue
            new_cost = cost + 1
            if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                visited[neighbor_tuple] = new_cost
                frontier.put((new_cost, neighbor, path + [neighbor]))
    return None

def astar(initial_state, time_limit=30):
    start_time = time.time()
    frontier = PriorityQueue()
    g_init = 0
    h_init = manhattan_distance(initial_state)
    f_init = g_init + h_init
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        return None
    frontier.put((f_init, g_init, initial_state, [initial_state]))
    visited = {init_tuple: g_init}
    while not frontier.empty():
        if time.time() - start_time > time_limit:
            print("A* Timeout")
            return None
        f_score, g_score, current_state, path = frontier.get()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None:
            continue
        if is_goal(current_state):
            return path
        if current_tuple in visited and g_score > visited[current_tuple]:
            continue
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is None:
                continue
            tentative_g = g_score + 1
            if neighbor_tuple not in visited or tentative_g < visited[neighbor_tuple]:
                visited[neighbor_tuple] = tentative_g
                h = manhattan_distance(neighbor)
                f = tentative_g + h
                frontier.put((f, tentative_g, neighbor, path + [neighbor]))
    return None

def greedy(initial_state, time_limit=30):
    start_time = time.time()
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        return None
    frontier.put((manhattan_distance(initial_state), initial_state, [initial_state]))
    visited = {init_tuple}
    while not frontier.empty():
        if time.time() - start_time > time_limit:
            print("Greedy Timeout")
            return None
        h_val, current_state, path = frontier.get()
        if is_goal(current_state):
            return path
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is None or neighbor_tuple in visited:
                continue
            visited.add(neighbor_tuple)
            h = manhattan_distance(neighbor)
            frontier.put((h, neighbor, path + [neighbor]))
    return None

def ida_star(initial_state, time_limit=60):
    start_time = time.time()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        return None
    threshold = manhattan_distance(initial_state)
    path = [initial_state]
    while True:
        if time.time() - start_time >= time_limit:
            print("IDA* Global Timeout")
            return None
        visited_in_iteration = {init_tuple: 0}
        result, new_threshold = search_ida(path, 0, threshold, visited_in_iteration, start_time, time_limit)
        if result == "Timeout":
            print("IDA* Timeout during search")
            return None
        if result is not None:
            return result
        if new_threshold == float('inf'):
            print("IDA*: Search exhausted")
            return None
        threshold = new_threshold

def search_ida(path, g, threshold, visited_in_iteration, start_time, time_limit):
    current_state = path[-1]
    h = manhattan_distance(current_state)
    f = g + h
    if f > threshold:
        return None, f
    if is_goal(current_state):
        return path, threshold
    if time.time() - start_time >= time_limit:
        return "Timeout", float('inf')
    min_new_threshold = float('inf')
    for neighbor in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor)
        if neighbor_tuple is None:
            continue
        new_g = g + 1
        if neighbor_tuple not in visited_in_iteration or new_g < visited_in_iteration[neighbor_tuple]:
            visited_in_iteration[neighbor_tuple] = new_g
            path.append(neighbor)
            result, recursive_threshold = search_ida(path, new_g, threshold, visited_in_iteration, start_time, time_limit)
            path.pop()
            if result == "Timeout":
                return "Timeout", float('inf')
            if result is not None:
                return result, threshold
            min_new_threshold = min(min_new_threshold, recursive_threshold)
    return None, min_new_threshold

def simple_hill_climbing(initial_state, time_limit=30):
    start_time = time.time()
    current_state = initial_state
    path = [current_state]
    current_h = manhattan_distance(current_state)
    while True:
        if time.time() - start_time > time_limit:
            print("Simple Hill Climb Timeout/Stuck")
            return path
        if is_goal(current_state):
            return path
        neighbors = get_neighbors(current_state)
        best_neighbor = None
        found_better = False
        for neighbor in neighbors:
            h = manhattan_distance(neighbor)
            if h < current_h:
                best_neighbor = neighbor
                current_h = h
                found_better = True
                break
        if not found_better:
            print("Simple Hill Climb: Local optimum")
            return path
        current_state = best_neighbor
        path.append(current_state)

def steepest_hill_climbing(initial_state, time_limit=30):
    start_time = time.time()
    current_state = initial_state
    path = [current_state]
    current_h = manhattan_distance(current_state)
    while True:
        if time.time() - start_time > time_limit:
            print("Steepest Hill Climb Timeout/Stuck")
            return path
        if is_goal(current_state):
            return path
        neighbors = get_neighbors(current_state)
        best_neighbor = None
        best_h = current_h
        for neighbor in neighbors:
            h = manhattan_distance(neighbor)
            if h < best_h:
                best_h = h
                best_neighbor = neighbor
        if best_neighbor is None or best_h >= current_h:
            print("Steepest Hill Climb: Local optimum")
            return path
        current_state = best_neighbor
        current_h = best_h
        path.append(current_state)

def random_hill_climbing(initial_state, time_limit=30, max_iter_no_improve=500):
    start_time = time.time()
    current_state = initial_state
    path = [current_state]
    current_h = manhattan_distance(current_state)
    iter_no_improve = 0
    while True:
        if time.time() - start_time > time_limit:
            print("Stochastic Hill Climb Timeout")
            return path
        if is_goal(current_state):
            return path
        neighbors = get_neighbors(current_state)
        if not neighbors:
            print("Stochastic Hill Climb: No neighbors")
            break
        random_neighbor = random.choice(neighbors)
        neighbor_h = manhattan_distance(random_neighbor)
        if neighbor_h <= current_h:
            if neighbor_h < current_h:
                iter_no_improve = 0
            else:
                iter_no_improve += 1
            current_state = random_neighbor
            current_h = neighbor_h
            path.append(current_state)
        else:
            iter_no_improve += 1
        if iter_no_improve >= max_iter_no_improve:
            print(f"Stochastic Hill Climb: Stuck ({max_iter_no_improve} iters)")
            return path
    return path

def simulated_annealing(initial_state, initial_temp=1000, cooling_rate=0.99, min_temp=0.1, time_limit=30):
    start_time = time.time()
    current_state = initial_state
    current_h = manhattan_distance(current_state)
    path = [current_state]
    best_state = initial_state
    best_h = current_h
    temp = initial_temp
    while temp > min_temp:
        if time.time() - start_time > time_limit:
            print(f"SA Timeout ({time_limit}s)")
            return path
        if is_goal(current_state):
            print("SA: Goal found.")
            return path
        neighbors = get_neighbors(current_state)
        if not neighbors:
            break
        next_state = random.choice(neighbors)
        next_h = manhattan_distance(next_state)
        delta_h = next_h - current_h
        if delta_h < 0:
            current_state = next_state
            current_h = next_h
            path.append(current_state)
            if current_h < best_h:
                best_h = current_h
                best_state = current_state
        else:
            if temp > 0:
                acceptance_prob = math.exp(-delta_h / temp)
                if random.random() < acceptance_prob:
                    current_state = next_state
                    current_h = next_h
                    path.append(current_state)
        temp *= cooling_rate
    print(f"SA: Cooled down. Best h={best_h}")
    return path

def beam_search(initial_state, beam_width=5, time_limit=30):
    start_time = time.time()
    if is_goal(initial_state):
        return [initial_state]
    h_init = manhattan_distance(initial_state)
    beam = [(initial_state, [initial_state], h_init)]
    visited = set()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        return None
    visited.add(init_tuple)
    best_goal_path = None
    while beam:
        if time.time() - start_time > time_limit:
            print("Beam Search Timeout")
            return best_goal_path
        next_level_candidates = []
        processed_in_level = set()
        for current_state, path, _ in beam:
            for neighbor in get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple is None or neighbor_tuple in visited or neighbor_tuple in processed_in_level:
                    continue
                visited.add(neighbor_tuple)
                processed_in_level.add(neighbor_tuple)
                new_path = path + [neighbor]
                if is_goal(neighbor):
                    if best_goal_path is None or len(new_path) < len(best_goal_path):
                        best_goal_path = new_path
                h_neighbor = manhattan_distance(neighbor)
                next_level_candidates.append((neighbor, new_path, h_neighbor))
        if not next_level_candidates:
            break
        next_level_candidates.sort(key=lambda x: x[2])
        beam = next_level_candidates[:beam_width]
    if best_goal_path:
        print(f"Beam Search: Goal found len={len(best_goal_path)}")
    else:
        print("Beam Search: Beam empty or timeout before finding goal.")
    return best_goal_path

def and_or_search(initial_state, time_limit=30):
    start_time = time.time()
    solved_states = set()
    unsolved_states = set()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        print("AND-OR: Invalid initial state.")
        return None
    status, solution_path = _and_or_recursive(initial_state, [initial_state], solved_states, unsolved_states, start_time, time_limit, 0)
    if status == "SOLVED":
        print("AND-OR: Solution found.")
        return solution_path
    elif status == "Timeout":
        print(f"AND-OR: Timeout ({time_limit}s).")
        return None
    else:
        print("AND-OR: No solution found (exhausted/depth limit).")
        return None

def _and_or_recursive(state, path, solved_states, unsolved_states, start_time, time_limit, depth):
    state_tuple = state_to_tuple(state)
    if state_tuple is None:
        return "UNSOLVED", None
    if time.time() - start_time > time_limit:
        return "Timeout", None
    if depth > 50:
        return "UNSOLVED", None
    if is_goal(state):
        solved_states.add(state_tuple)
        return "SOLVED", path
    if state_tuple in solved_states:
        return "SOLVED", path
    if state_tuple in unsolved_states:
        return "UNSOLVED", None
    unsolved_states.add(state_tuple)
    for neighbor in get_neighbors(state):
        neighbor_tuple = state_to_tuple(neighbor)
        if neighbor_tuple is None or neighbor_tuple in unsolved_states:
            continue
        status, solution_path = _and_or_recursive(neighbor, path + [neighbor], solved_states, unsolved_states, start_time, time_limit, depth + 1)
        if status == "Timeout":
            return "Timeout", None
        if status == "SOLVED":
            solved_states.add(state_tuple)
            if state_tuple in unsolved_states:
                unsolved_states.remove(state_tuple)
            return "SOLVED", solution_path
    return "UNSOLVED", None

def generate_random_state():
    numbers = list(range(9))
    random.shuffle(numbers)
    state = []
    for i in range(0, 9, 3):
        state.append(numbers[i:i+3])
    return state

def is_solvable(state):
    flat_state = [tile for row in state for tile in row if tile != 0]
    inversions = 0
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] > flat_state[j]:
                inversions += 1
    return inversions % 2 == 0

def backtracking_search(goal_state, selected_algorithm, max_attempts=100, time_limit=30):
    global initial_state
    start_time = time.time()
    attempts = 0
    algo_func = None
    time_limit_per_attempt = time_limit / max_attempts if max_attempts > 0 else time_limit
    algo_map = {
        'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 'Greedy': greedy,
        'IDA*': ida_star, 'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing,
        'Stochastic Hill': random_hill_climbing, 'SA': simulated_annealing, 'Beam Search': beam_search,
        'AND-OR': and_or_search
    }
    algo_func = algo_map.get(selected_algorithm)
    if not algo_func:
        return None, f"Algorithm '{selected_algorithm}' is not supported."
    while attempts < max_attempts:
        if time.time() - start_time > time_limit:
            return None, "Backtracking: Global timeout."
        new_initial_state = generate_random_state()
        if not is_solvable(new_initial_state):
            attempts += 1
            continue
        initial_state = copy.deepcopy(new_initial_state)
        algo_params = [new_initial_state]
        func_params = algo_func.__code__.co_varnames[:algo_func.__code__.co_argcount]
        if 'time_limit' in func_params:
            algo_params.append(time_limit_per_attempt)
        try:
            solution_path = algo_func(*algo_params)
            if solution_path and len(solution_path) > 0 and is_goal(solution_path[-1]):
                print(f"Backtracking: Found solution with {selected_algorithm} after {attempts + 1} attempts.")
                return solution_path, None
            else:
                print(f"Backtracking: Attempt {attempts + 1} failed with {selected_algorithm}.")
        except Exception as e:
            print(f"Backtracking: Error in attempt {attempts + 1} with {selected_algorithm}: {str(e)}")
        attempts += 1
    return None, f"Backtracking: No solution found after {max_attempts} attempts."

def generate_belief_states(partial_state):
    """
    Tạo tất cả các trạng thái có thể từ trạng thái đầu không đầy đủ.
    Các ô có giá trị -1 sẽ được thay bằng các số còn thiếu.
    """
    flat_state = [tile for row in partial_state for tile in row]
    used_numbers = [tile for tile in flat_state if tile != -1]
    missing_numbers = [i for i in range(9) if i not in used_numbers]
    unknown_positions = [i for i, tile in enumerate(flat_state) if tile == -1]
    
    belief_states = []
    for perm in itertools.permutations(missing_numbers, len(unknown_positions)):
        new_state = flat_state.copy()
        for pos, num in zip(unknown_positions, perm):
            new_state[pos] = num
        state_2d = [new_state[i:i+3] for i in range(0, 9, 3)]
        if is_solvable(state_2d):
            belief_states.append(state_2d)
    
    return belief_states

def is_belief_state_goal(belief_states):
    """
    Kiểm tra xem tất cả các trạng thái trong belief states có khớp với goal_state không.
    Với các ô -1 trong goal_state, bỏ qua kiểm tra tại vị trí đó.
    """
    for state in belief_states:
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if goal_state[i][j] != -1 and state[i][j] != goal_state[i][j]:
                    return False
    return True

def apply_action_to_belief_states(belief_states, action):
    """
    Áp dụng một hành động cho tất cả các trạng thái trong belief states.
    """
    new_belief_states = []
    seen_tuples = set()
    for state in belief_states:
        new_state = apply_action_to_state(state, action)
        if new_state:
            state_tuple = state_to_tuple(new_state)
            if state_tuple and state_tuple not in seen_tuples:
                new_belief_states.append(new_state)
                seen_tuples.add(state_tuple)
    return new_belief_states

def sensorless_search(initial_state, time_limit=60):
    """
    Thuật toán Sensorless Search cho 8-puzzle với trạng thái đầu không đầy đủ.
    Tìm một chuỗi hành động cố định dẫn tất cả trạng thái trong belief states đến goal_state.
    """
    start_time = time.time()
    belief_states = generate_belief_states(initial_state)
    if not belief_states:
        print("Sensorless: No valid belief states generated.")
        return None
    
    queue = deque([([], belief_states)])
    visited = set()
    actions = ['Up', 'Down', 'Left', 'Right']
    
    while queue:
        if time.time() - start_time > time_limit:
            print("Sensorless Search Timeout")
            return None
        
        action_plan, current_belief_states = queue.popleft()
        belief_tuple = tuple(sorted([state_to_tuple(state) for state in current_belief_states if state_to_tuple(state)]))
        if belief_tuple in visited:
            continue
        visited.add(belief_tuple)
        
        if is_belief_state_goal(current_belief_states):
            print(f"Sensorless: Found plan with {len(action_plan)} actions.")
            return action_plan
        
        for action in actions:
            new_belief_states = apply_action_to_belief_states(current_belief_states, action)
            if new_belief_states:
                new_plan = action_plan + [action]
                queue.append((new_plan, new_belief_states))
    
    print("Sensorless: No plan found.")
    return None

def algorithm_selection_popup():
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
    title_font = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf = title_font.render("Select Algorithm", True, INFO_COLOR)
    title_rect = title_surf.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf, title_rect)
    algorithms = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search')
    ]
    button_width = 150
    button_height = 40
    button_margin = 10
    columns = 3
    start_x = 50
    start_y = title_rect.bottom + 30
    button_rects = {}
    algo_buttons = []
    for idx, (algo_id, algo_name) in enumerate(algorithms):
        col = idx % columns
        row = idx // columns
        button_x = start_x + col * (button_width + button_margin)
        button_y = start_y + row * (button_height + button_margin)
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        button_rects[algo_id] = button_rect
        algo_buttons.append((algo_id, algo_name, button_rect))
    cancel_button_rect = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface, RED, cancel_button_rect, border_radius=8)
    cancel_text_surf = BUTTON_FONT.render("Cancel", True, WHITE)
    cancel_text_rect = cancel_text_surf.get_rect(center=cancel_button_rect.center)
    popup_surface.blit(cancel_text_surf, cancel_text_rect)
    popup_rect = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    selected_algorithm = None
    running = True
    clock = pygame.time.Clock()
    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_pos_rel = (mouse_pos[0] - popup_rect.left, mouse_pos[1] - popup_rect.top)
        popup_surface.fill(INFO_BG)
        pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
        popup_surface.blit(title_surf, title_rect)
        for algo_id, algo_name, button_rect in algo_buttons:
            is_hovered = button_rect.collidepoint(mouse_pos_rel)
            button_color = MENU_HOVER_COLOR if is_hovered else MENU_BUTTON_COLOR
            pygame.draw.rect(popup_surface, button_color, button_rect, border_radius=8)
            text_surf = BUTTON_FONT.render(algo_name, True, WHITE)
            text_rect = text_surf.get_rect(center=button_rect.center)
            popup_surface.blit(text_surf, text_rect)
        pygame.draw.rect(popup_surface, RED, cancel_button_rect, border_radius=8)
        popup_surface.blit(cancel_text_surf, cancel_text_rect)
        screen.blit(popup_surface, popup_rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button_rect.collidepoint(mouse_pos_rel):
                    return None
                for algo_id, _, button_rect in algo_buttons:
                    if button_rect.collidepoint(mouse_pos_rel):
                        selected_algorithm = algo_id
                        running = False
                        break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
        clock.tick(60)
    return selected_algorithm

scroll_y = 0
menu_surface = None
total_menu_height = 0

def draw_state(state, x, y, title, is_current=False, is_goal_state=False):
    title_font = BUTTON_FONT
    title_text = title_font.render(title, True, BLACK)
    title_x = x + (GRID_DISPLAY_WIDTH // 2 - title_text.get_width() // 2)
    title_y = y - title_text.get_height() - 5
    screen.blit(title_text, (title_x, title_y))
    pygame.draw.rect(screen, BLACK, (x - 1, y - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell_x = x + j * CELL_SIZE
            cell_y = y + i * CELL_SIZE
            cell_rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
            tile = state[i][j]
            if tile != -1 and tile != 0:
                color = GREEN if is_goal_state or (is_current and is_goal(state)) else BLUE
                pygame.draw.rect(screen, color, cell_rect.inflate(-6, -6), border_radius=8)
                number = FONT.render(str(tile), True, WHITE)
                screen.blit(number, number.get_rect(center=cell_rect.center))
            else:
                pygame.draw.rect(screen, GRAY, cell_rect.inflate(-6, -6), border_radius=8)
            pygame.draw.rect(screen, BLACK, cell_rect, 1)

def show_popup(message, title="Info"):
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
    title_font = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf = title_font.render(title, True, INFO_COLOR)
    title_rect = title_surf.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf, title_rect)
    words = message.split(' ')
    lines = []
    current_line = ""
    text_width_limit = POPUP_WIDTH - 50
    for word in words:
        if "\n" in word:
            parts = word.split("\n")
            for i, part in enumerate(parts):
                if not part:
                    continue
                test_line_part = current_line + (" " if current_line else "") + part
                line_width_part = INFO_FONT.size(test_line_part)[0] if INFO_FONT else 0
                if line_width_part <= text_width_limit:
                    current_line = test_line_part
                else:
                    lines.append(current_line)
                    current_line = part
                if i < len(parts) - 1:
                    lines.append(current_line)
                    current_line = ""
        else:
            test_line = current_line + (" " if current_line else "") + word
            line_width = INFO_FONT.size(test_line)[0] if INFO_FONT else 0
            if line_width <= text_width_limit:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
    lines.append(current_line)
    line_height = INFO_FONT.get_linesize() if INFO_FONT else 20
    start_y = title_rect.bottom + 20
    for i, line in enumerate(lines):
        if INFO_FONT:
            text_surf = INFO_FONT.render(line, True, BLACK)
            text_rect = text_surf.get_rect(center=(POPUP_WIDTH // 2, start_y + i * line_height))
            popup_surface.blit(text_surf, text_rect)
    ok_button_rect = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface, INFO_COLOR, ok_button_rect, border_radius=8)
    ok_text_surf = BUTTON_FONT.render("OK", True, WHITE) if BUTTON_FONT else None
    if ok_text_surf:
        ok_text_rect = ok_text_surf.get_rect(center=ok_button_rect.center)
        popup_surface.blit(ok_text_surf, ok_text_rect)
    popup_rect = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(popup_surface, popup_rect)
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                if ok_button_rect.collidepoint(mouse_x - popup_rect.left, mouse_y - popup_rect.top):
                    waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    waiting = False
        pygame.time.delay(20)

def draw_menu(show_menu, mouse_pos, current_algorithm):
    global scroll_y, menu_surface, total_menu_height
    menu_elements = {}
    if not show_menu:
        menu_button_rect = pygame.Rect(10, 10, 50, 40)
        pygame.draw.rect(screen, MENU_COLOR, menu_button_rect, border_radius=5)
        bar_width, bar_height, space = 30, 4, 7
        start_x = menu_button_rect.centerx - bar_width // 2
        start_y = menu_button_rect.centery - (bar_height * 3 + space * 2) // 2
        for i in range(3):
            pygame.draw.rect(screen, WHITE, (start_x, start_y + i * (bar_height + space), bar_width, bar_height), border_radius=2)
        menu_elements['open_button'] = menu_button_rect
        return menu_elements
    algorithms = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Backtracking', 'Backtracking'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search'),
        ('Sensorless', 'Sensorless Plan')
    ]
    button_height, padding, button_margin = 55, 10, 8
    total_menu_height = (len(algorithms) * (button_height + button_margin)) - button_margin + (2 * padding)
    display_height = max(total_menu_height, HEIGHT)
    if menu_surface is None or menu_surface.get_height() != display_height:
        menu_surface = pygame.Surface((MENU_WIDTH, display_height))
    menu_surface.fill(MENU_COLOR)
    buttons_dict = {}
    y_position = padding
    mouse_x_rel, mouse_y_rel = mouse_pos[0], mouse_pos[1] + scroll_y
    for algo_id, algo_name in algorithms:
        button_rect_local = pygame.Rect(padding, y_position, MENU_WIDTH - 2 * padding, button_height)
        is_hover = button_rect_local.collidepoint(mouse_x_rel, mouse_y_rel)
        is_selected = (current_algorithm == algo_id)
        button_color = MENU_SELECTED_COLOR if is_selected else (MENU_HOVER_COLOR if is_hover else MENU_BUTTON_COLOR)
        pygame.draw.rect(menu_surface, button_color, button_rect_local, border_radius=5)
        text = BUTTON_FONT.render(algo_name, True, WHITE)
        menu_surface.blit(text, text.get_rect(center=button_rect_local.center))
        buttons_dict[algo_id] = button_rect_local
        y_position += button_height + button_margin
    visible_menu_area = pygame.Rect(0, scroll_y, MENU_WIDTH, HEIGHT)
    screen.blit(menu_surface, (0, 0), visible_menu_area)
    close_button_rect = pygame.Rect(MENU_WIDTH - 40, 10, 30, 30)
    pygame.draw.rect(screen, RED, close_button_rect, border_radius=5)
    cx, cy = close_button_rect.center
    pygame.draw.line(screen, WHITE, (cx - 7, cy - 7), (cx + 7, cy + 7), 3)
    pygame.draw.line(screen, WHITE, (cx - 7, cy + 7), (cx + 7, cy - 7), 3)
    menu_elements['close_button'] = close_button_rect
    menu_elements['buttons'] = buttons_dict
    menu_elements['menu_area'] = pygame.Rect(0, 0, MENU_WIDTH, HEIGHT)
    if total_menu_height > HEIGHT:
        scrollbar_height = max(20, HEIGHT * (HEIGHT / total_menu_height))
        scrollbar_max_y = HEIGHT - scrollbar_height
        scroll_ratio = scroll_y / (total_menu_height - HEIGHT) if total_menu_height > HEIGHT else 0
        scrollbar_y = scroll_ratio * scrollbar_max_y
        scrollbar_rect = pygame.Rect(MENU_WIDTH - 8, scrollbar_y, 6, scrollbar_height)
        pygame.draw.rect(screen, GRAY, scrollbar_rect, border_radius=3)
    return menu_elements

def draw_grid_and_ui(state, show_menu, current_algorithm, solve_times, last_solved_info, belief_state_size=None):
    screen.fill(WHITE)
    mouse_pos = pygame.mouse.get_pos()
    main_area_x = MENU_WIDTH if show_menu else 0
    main_area_width = WIDTH - main_area_x
    center_x_main = main_area_x + main_area_width // 2
    top_row_y = GRID_PADDING + 40
    grid_spacing_top = GRID_PADDING * 1.5
    total_width_top = 2 * GRID_DISPLAY_WIDTH + grid_spacing_top
    start_x_top = center_x_main - total_width_top // 2
    initial_x = start_x_top
    goal_x = start_x_top + GRID_DISPLAY_WIDTH + grid_spacing_top
    bottom_row_y = top_row_y + GRID_DISPLAY_WIDTH + GRID_PADDING + 60
    button_width, button_height = 130, 45
    button_x = main_area_x + GRID_PADDING
    button_mid_y = bottom_row_y + GRID_DISPLAY_WIDTH // 2
    solve_button_y = button_mid_y - button_height - 8
    reset_solution_button_y = button_mid_y + 8
    reset_all_button_y = button_mid_y + button_height + 16
    reset_initial_button_y = reset_all_button_y + button_height + 8
    solve_button_rect = pygame.Rect(button_x, solve_button_y, button_width, button_height)
    reset_solution_button_rect = pygame.Rect(button_x, reset_solution_button_y, button_width, button_height)
    reset_all_button_rect = pygame.Rect(button_x, reset_all_button_y, button_width, button_height)
    reset_initial_button_rect = pygame.Rect(button_x, reset_initial_button_y, button_width, button_height)
    current_state_x = button_x + button_width + GRID_PADDING * 1.5
    current_state_y = bottom_row_y
    info_area_x = current_state_x + GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5
    info_area_y = bottom_row_y
    info_area_width = max(150, main_area_x + main_area_width - info_area_x - GRID_PADDING)
    info_area_height = GRID_DISPLAY_WIDTH
    info_area_rect = pygame.Rect(info_area_x, info_area_y, info_area_width, info_area_height)
    draw_state(initial_state, initial_x, top_row_y, "Initial State")
    draw_state(goal_state, goal_x, top_row_y, "Goal State", is_goal_state=True)
    if current_algorithm not in ['Sensorless', 'Backtracking'] or belief_state_size is None:
        draw_state(state, current_state_x, current_state_y, f"Current ({current_algorithm})", is_current=True)
    else:
        belief_text = f"Belief States: {belief_state_size}"
        draw_state(state, current_state_x, current_state_y, belief_text, is_current=True)
    pygame.draw.rect(screen, RED, solve_button_rect, border_radius=5)
    solve_text = BUTTON_FONT.render("SOLVE", True, WHITE)
    screen.blit(solve_text, solve_text.get_rect(center=solve_button_rect.center))
    pygame.draw.rect(screen, BLUE, reset_solution_button_rect, border_radius=5)
    reset_solution_text = BUTTON_FONT.render("Reset Solution", True, WHITE)
    screen.blit(reset_solution_text, reset_solution_text.get_rect(center=reset_solution_button_rect.center))
    pygame.draw.rect(screen, BLUE, reset_all_button_rect, border_radius=5)
    reset_all_text = BUTTON_FONT.render("Reset All", True, WHITE)
    screen.blit(reset_all_text, reset_all_text.get_rect(center=reset_all_button_rect.center))
    pygame.draw.rect(screen, BLUE, reset_initial_button_rect, border_radius=5)
    reset_initial_text = BUTTON_FONT.render("Reset Initial", True, WHITE)
    screen.blit(reset_initial_text, reset_initial_text.get_rect(center=reset_initial_button_rect.center))
    pygame.draw.rect(screen, INFO_BG, info_area_rect, border_radius=8)
    pygame.draw.rect(screen, GRAY, info_area_rect, 2, border_radius=8)
    info_pad_x, info_pad_y = 15, 10
    line_height = INFO_FONT.get_linesize() + 4
    current_info_y = info_area_y + info_pad_y
    compare_title_surf = TITLE_FONT.render("Comparison", True, BLACK)
    compare_title_x = info_area_rect.centerx - compare_title_surf.get_width() // 2
    screen.blit(compare_title_surf, (compare_title_x, current_info_y))
    current_info_y += compare_title_surf.get_height() + 8
    if solve_times:
        sorted_times = sorted(solve_times.items(), key=lambda item: item[1])
        for algo, time_val in sorted_times:
            steps_val = last_solved_info.get(f"{algo}_steps")
            actions_val = last_solved_info.get(f"{algo}_actions")
            reached_goal = last_solved_info.get(f"{algo}_reached_goal")
            base_str = f"{algo}: {time_val:.3f}s"
            count_str = ""
            if steps_val is not None:
                count_str = f" ({steps_val} steps)"
            elif actions_val is not None:
                count_str = f" ({actions_val} actions)"
            else:
                count_str = " (--)"
            goal_str = " (Not Goal)" if reached_goal is False else ""
            comp_str = base_str + count_str + goal_str
            comp_surf = INFO_FONT.render(comp_str, True, BLACK)
            text_fits = info_area_x + info_pad_x + comp_surf.get_width() < info_area_x + info_area_width - info_pad_x / 2
            if current_info_y + line_height < info_area_y + info_area_height - info_pad_y:
                if text_fits:
                    screen.blit(comp_surf, (info_area_x + info_pad_x, current_info_y))
                    current_info_y += line_height
                else:
                    comp_str_short = base_str + goal_str
                    comp_surf_short = INFO_FONT.render(comp_str_short, True, BLACK)
                    if info_area_x + info_pad_x + comp_surf_short.get_width() < info_area_x + info_area_width - info_pad_x / 2:
                        screen.blit(comp_surf_short, (info_area_x + info_pad_x, current_info_y))
                        current_info_y += line_height
            else:
                screen.blit(INFO_FONT.render("...", True, BLACK), (info_area_x + info_pad_x, current_info_y))
                break
    else:
        no_comp_surf = INFO_FONT.render("(No results yet)", True, GRAY)
        screen.blit(no_comp_surf, (info_area_x + info_pad_x, current_info_y))
    menu_elements = draw_menu(show_menu, mouse_pos, current_algorithm)
    pygame.display.flip()
    return {
        'solve_button': solve_button_rect,
        'reset_solution_button': reset_solution_button_rect,
        'reset_all_button': reset_all_button_rect,
        'reset_initial_button': reset_initial_button_rect,
        'menu': menu_elements,
        'initial_grid_rect': pygame.Rect(initial_x, top_row_y, GRID_DISPLAY_WIDTH, GRID_DISPLAY_WIDTH),
        'goal_grid_rect': pygame.Rect(goal_x, top_row_y, GRID_DISPLAY_WIDTH, GRID_DISPLAY_WIDTH)
    }

def backtracking_menu():
    global scroll_y, initial_state, goal_state
    current_state = copy.deepcopy(initial_state)
    solution = None
    step_index = 0
    solving = False
    auto_solve = False
    last_step_time = 0
    show_menu = False
    current_algorithm = 'Backtracking'
    solve_times = {}
    last_solved_info = {}
    selected_cell = None
    clock = pygame.time.Clock()
    ui_elements = {}
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_handled = False
                if show_menu and not clicked_handled:
                    menu_data = ui_elements.get('menu', {})
                    menu_area = menu_data.get('menu_area')
                    if menu_area and menu_area.collidepoint(mouse_pos):
                        close_button = menu_data.get('close_button')
                        if close_button and close_button.collidepoint(mouse_pos):
                            show_menu = False
                            clicked_handled = True
                        if not clicked_handled:
                            buttons = menu_data.get('buttons', {})
                            for algo_id, button_rect_local in buttons.items():
                                button_rect_screen = button_rect_local.move(0, -scroll_y)
                                if button_rect_screen.collidepoint(mouse_pos):
                                    if algo_id != 'Backtracking':
                                        print(f"Switching to main menu with algorithm: {algo_id}")
                                        return algo_id
                                    clicked_handled = True
                                    break
                        if not clicked_handled:
                            clicked_handled = True
                if not show_menu and not clicked_handled:
                    menu_data = ui_elements.get('menu', {})
                    open_button = menu_data.get('open_button')
                    if open_button and open_button.collidepoint(mouse_pos):
                        show_menu = True
                        scroll_y = 0
                        clicked_handled = True
                if not clicked_handled:
                    solve_button = ui_elements.get('solve_button')
                    if solve_button and solve_button.collidepoint(mouse_pos):
                        if not auto_solve and not solving:
                            if is_valid_state(goal_state):
                                print("Opening algorithm selection popup for Backtracking...")
                                selected_algo = algorithm_selection_popup()
                                if selected_algo:
                                    print(f"Starting Backtracking with {selected_algo}...")
                                    solving = True
                                    solution = None
                                    step_index = 0
                                    auto_solve = False
                            else:
                                show_popup("Goal state is incomplete or invalid!\nEnter all numbers 0-8.", "Invalid State")
                        clicked_handled = True
                if not clicked_handled:
                    reset_solution_button = ui_elements.get('reset_solution_button')
                    if reset_solution_button and reset_solution_button.collidepoint(mouse_pos):
                        print("Resetting solution state.")
                        current_state = copy.deepcopy(initial_state)
                        solution = None
                        step_index = 0
                        solving = False
                        auto_solve = False
                        selected_cell = None
                        clicked_handled = True
                if not clicked_handled:
                    reset_all_button = ui_elements.get('reset_all_button')
                    if reset_all_button and reset_all_button.collidepoint(mouse_pos):
                        print("Resetting all states.")
                        initial_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
                        goal_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
                        current_state = copy.deepcopy(initial_state)
                        solution = None
                        step_index = 0
                        solving = False
                        auto_solve = False
                        selected_cell = None
                        clicked_handled = True
                if not clicked_handled:
                    reset_initial_button = ui_elements.get('reset_initial_button')
                    if reset_initial_button and reset_initial_button.collidepoint(mouse_pos):
                        print("Resetting initial state.")
                        initial_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
                        current_state = copy.deepcopy(initial_state)
                        solution = None
                        step_index = 0
                        solving = False
                        auto_solve = False
                        selected_cell = None
                        clicked_handled = True
                if not clicked_handled:
                    goal_grid_rect = ui_elements.get('goal_grid_rect')
                    if goal_grid_rect and goal_grid_rect.collidepoint(mouse_pos):
                        goal_x = goal_grid_rect.left
                        goal_y = goal_grid_rect.top
                        for i in range(GRID_SIZE):
                            for j in range(GRID_SIZE):
                                cell_x = goal_x + j * CELL_SIZE
                                cell_y = goal_y + i * CELL_SIZE
                                cell_rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
                                if cell_rect.collidepoint(mouse_pos):
                                    selected_cell = (i, j, 'goal')
                                    clicked_handled = True
                                    break
            if event.type == pygame.KEYDOWN and selected_cell:
                i, j, grid_type = selected_cell
                key = event.key
                if pygame.K_0 <= key <= pygame.K_8:
                    num = key - pygame.K_0
                    target_state = goal_state
                    flat_state = [tile for row in target_state for tile in row if tile != -1]
                    if num not in flat_state or target_state[i][j] == num:
                        target_state[i][j] = num
                        current_state = copy.deepcopy(initial_state)
                    else:
                        show_popup("Each number (0-8) must be unique!", "Invalid Input")
                selected_cell = None
            if event.type == pygame.MOUSEWHEEL and show_menu:
                menu_area = ui_elements.get('menu', {}).get('menu_area')
                if menu_area and menu_area.collidepoint(mouse_pos) and total_menu_height > HEIGHT:
                    scroll_amount = event.y * 35
                    max_scroll = max(0, total_menu_height - HEIGHT)
                    scroll_y = max(0, min(scroll_y - scroll_amount, max_scroll))
        if not running:
            break
        if solving:
            solving = False
            solve_start_time = time.time()
            error_occurred = False
            solution_path = None
            error_msg = None
            try:
                solution_path, error_msg = backtracking_search(goal_state, selected_algo, max_attempts=100, time_limit=30)
            except Exception as e:
                show_popup(f"Error during Backtracking:\n{traceback.format_exc()}", "Solver Error")
                traceback.print_exc()
                error_occurred = True
            if not error_occurred:
                solve_duration = time.time() - solve_start_time
                solve_times[current_algorithm] = solve_duration
                if solution_path and len(solution_path) > 0:
                    steps = len(solution_path) - 1
                    last_solved_info[f"{current_algorithm}_steps"] = steps
                    last_solved_info[f"{current_algorithm}_reached_goal"] = True
                    solution = solution_path
                    auto_solve = True
                    step_index = 0
                    last_step_time = time.time()
                    print(f"Backtracking: {steps} steps, {solve_duration:.4f}s")
                    show_popup(f"Backtracking with {selected_algo} found solution!\n{steps} steps\nTime: {solve_duration:.4f}s", "Solution Found")
                else:
                    solution = None
                    auto_solve = False
                    if f"{current_algorithm}_steps" in last_solved_info:
                        del last_solved_info[f"{current_algorithm}_steps"]
                    if f"{current_algorithm}_reached_goal" in last_solved_info:
                        del last_solved_info[f"{current_algorithm}_reached_goal"]
                    error_message = error_msg or "Unknown error"
                    print(f"Backtracking failed: {error_message}")
                    show_popup(f"Backtracking failed: {error_message}", "No Solution")
        if auto_solve and solution and isinstance(solution, list) and len(solution) > 0 and isinstance(solution[0], list):
            current_time = time.time()
            anim_delay = 0.3 if len(solution) < 30 else (0.2 if len(solution) < 60 else 0.1)
            if current_time - last_step_time >= anim_delay:
                if step_index < len(solution) - 1:
                    step_index += 1
                    current_state = copy.deepcopy(solution[step_index])
                    last_step_time = current_time
                else:
                    auto_solve = False
                    is_final_state_goal = is_goal(current_state)
                    print(f"Animation complete: {'Goal reached!' if is_final_state_goal else 'Final state reached (Not Goal).'}")
        ui_elements = draw_grid_and_ui(current_state, show_menu, current_algorithm, solve_times, last_solved_info)
        clock.tick(60)
    return None

def main():
    global scroll_y, initial_state, goal_state
    current_state = copy.deepcopy(initial_state)
    solution = None
    step_index = 0
    solving = False
    auto_solve = False
    last_step_time = 0
    show_menu = False
    current_algorithm = 'A*'
    solve_times = {}
    last_solved_info = {}
    selected_cell = None
    clock = pygame.time.Clock()
    ui_elements = {}
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_handled = False
                if show_menu and not clicked_handled:
                    menu_data = ui_elements.get('menu', {})
                    menu_area = menu_data.get('menu_area')
                    if menu_area and menu_area.collidepoint(mouse_pos):
                        close_button = menu_data.get('close_button')
                        if close_button and close_button.collidepoint(mouse_pos):
                            show_menu = False
                            clicked_handled = True
                        if not clicked_handled:
                            buttons = menu_data.get('buttons', {})
                            for algo_id, button_rect_local in buttons.items():
                                button_rect_screen = button_rect_local.move(0, -scroll_y)
                                if button_rect_screen.collidepoint(mouse_pos):
                                    if current_algorithm != algo_id:
                                        print(f"Algorithm changed to: {algo_id}")
                                        if algo_id == 'Backtracking':
                                            new_algo = backtracking_menu()
                                            if new_algo:
                                                current_algorithm = new_algo
                                                solution = None
                                                step_index = 0
                                                auto_solve = False
                                        else:
                                            current_algorithm = algo_id
                                            solution = None
                                            step_index = 0
                                            auto_solve = False
                                    show_menu = False
                                    clicked_handled = True
                                    break
                        if not clicked_handled:
                            clicked_handled = True
                if not show_menu and not clicked_handled:
                    menu_data = ui_elements.get('menu', {})
                    open_button = menu_data.get('open_button')
                    if open_button and open_button.collidepoint(mouse_pos):
                        show_menu = True
                        scroll_y = 0
                        clicked_handled = True
                if not clicked_handled:
                    solve_button = ui_elements.get('solve_button')
                    if solve_button and solve_button.collidepoint(mouse_pos):
                        if not auto_solve and not solving:
                            if current_algorithm != 'Sensorless' and (not is_valid_state(initial_state) or not is_valid_state(goal_state)):
                                show_popup("Initial or Goal state is incomplete or invalid!\nEnter all numbers 0-8.", "Invalid State")
                            else:
                                print(f"Starting solve with {current_algorithm}...")
                                solving = True
                                solution = None
                                step_index = 0
                                auto_solve = False
                        clicked_handled = True
                if not clicked_handled:
                    reset_solution_button = ui_elements.get('reset_solution_button')
                    if reset_solution_button and reset_solution_button.collidepoint(mouse_pos):
                        print("Resetting solution state.")
                        current_state = copy.deepcopy(initial_state)
                        solution = None
                        step_index = 0
                        solving = False
                        auto_solve = False
                        selected_cell = None
                        clicked_handled = True
                if not clicked_handled:
                    reset_all_button = ui_elements.get('reset_all_button')
                    if reset_all_button and reset_all_button.collidepoint(mouse_pos):
                        print("Resetting all states.")
                        initial_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
                        goal_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
                        current_state = copy.deepcopy(initial_state)
                        solution = None
                        step_index = 0
                        solving = False
                        auto_solve = False
                        selected_cell = None
                        clicked_handled = True
                if not clicked_handled:
                    reset_initial_button = ui_elements.get('reset_initial_button')
                    if reset_initial_button and reset_initial_button.collidepoint(mouse_pos):
                        print("Resetting initial state.")
                        initial_state = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
                        current_state = copy.deepcopy(initial_state)
                        solution = None
                        step_index = 0
                        solving = False
                        auto_solve = False
                        selected_cell = None
                        clicked_handled = True
                if not clicked_handled:
                    initial_grid_rect = ui_elements.get('initial_grid_rect')
                    if initial_grid_rect and initial_grid_rect.collidepoint(mouse_pos):
                        initial_x = initial_grid_rect.left
                        initial_y = initial_grid_rect.top
                        for i in range(GRID_SIZE):
                            for j in range(GRID_SIZE):
                                cell_x = initial_x + j * CELL_SIZE
                                cell_y = initial_y + i * CELL_SIZE
                                cell_rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
                                if cell_rect.collidepoint(mouse_pos):
                                    selected_cell = (i, j, 'initial')
                                    clicked_handled = True
                                    break
                if not clicked_handled:
                    goal_grid_rect = ui_elements.get('goal_grid_rect')
                    if goal_grid_rect and goal_grid_rect.collidepoint(mouse_pos):
                        goal_x = goal_grid_rect.left
                        goal_y = goal_grid_rect.top
                        for i in range(GRID_SIZE):
                            for j in range(GRID_SIZE):
                                cell_x = goal_x + j * CELL_SIZE
                                cell_y = goal_y + i * CELL_SIZE
                                cell_rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
                                if cell_rect.collidepoint(mouse_pos):
                                    selected_cell = (i, j, 'goal')
                                    clicked_handled = True
                                    break
            if event.type == pygame.KEYDOWN and selected_cell:
                i, j, grid_type = selected_cell
                key = event.key
                if pygame.K_0 <= key <= pygame.K_8:
                    num = key - pygame.K_0
                    target_state = initial_state if grid_type == 'initial' else goal_state
                    flat_state = [tile for row in target_state for tile in row if tile != -1]
                    if num not in flat_state or target_state[i][j] == num:
                        target_state[i][j] = num
                        if grid_type == 'initial':
                            current_state = copy.deepcopy(initial_state)
                    else:
                        show_popup("Each number (0-8) must be unique!", "Invalid Input")
                selected_cell = None
            if event.type == pygame.MOUSEWHEEL and show_menu:
                menu_area = ui_elements.get('menu', {}).get('menu_area')
                if menu_area and menu_area.collidepoint(mouse_pos) and total_menu_height > HEIGHT:
                    scroll_amount = event.y * 35
                    max_scroll = max(0, total_menu_height - HEIGHT)
                    scroll_y = max(0, min(scroll_y - scroll_amount, max_scroll))
        if not running:
            break
        if solving:
            solving = False
            solve_start_time = time.time()
            found_solution_path = None
            found_action_plan = None
            error_occurred = False
            is_sensorless_algo = (current_algorithm == 'Sensorless')
            algo_func = None
            algo_args = []
            time_limit = 30
            try:
                state_to_solve = copy.deepcopy(current_state)
                if current_algorithm == 'BFS':
                    algo_func = bfs
                    algo_args = [state_to_solve]
                elif current_algorithm == 'DFS':
                    algo_func = dfs
                    algo_args = [state_to_solve]
                elif current_algorithm == 'IDS':
                    algo_func = ids
                    algo_args = [state_to_solve]
                elif current_algorithm == 'UCS':
                    algo_func = ucs
                    algo_args = [state_to_solve]
                elif current_algorithm == 'A*':
                    algo_func = astar
                    algo_args = [state_to_solve]
                elif current_algorithm == 'Greedy':
                    algo_func = greedy
                    algo_args = [state_to_solve]
                elif current_algorithm == 'IDA*':
                    algo_func = ida_star
                    algo_args = [state_to_solve]
                    time_limit = 60
                elif current_algorithm == 'Hill Climbing':
                    algo_func = simple_hill_climbing
                    algo_args = [state_to_solve]
                elif current_algorithm == 'Steepest Hill':
                    algo_func = steepest_hill_climbing
                    algo_args = [state_to_solve]
                elif current_algorithm == 'Stochastic Hill':
                    algo_func = random_hill_climbing
                    algo_args = [state_to_solve]
                elif current_algorithm == 'SA':
                    algo_func = simulated_annealing
                    algo_args = [state_to_solve]
                elif current_algorithm == 'Beam Search':
                    algo_func = beam_search
                    algo_args = [state_to_solve]
                elif current_algorithm == 'AND-OR':
                    algo_func = and_or_search
                    algo_args = [state_to_solve]
                elif current_algorithm == 'Sensorless':
                    algo_func = sensorless_search
                    algo_args = [initial_state]
                    time_limit = 60
                else:
                    show_popup(f"Algorithm '{current_algorithm}' is not implemented.", "Error")
                    error_occurred = True
                if algo_func and not error_occurred:
                    func_params = algo_func.__code__.co_varnames[:algo_func.__code__.co_argcount]
                    if 'time_limit' in func_params:
                        algo_args.append(time_limit)
                    if is_sensorless_algo:
                        found_action_plan = algo_func(*algo_args)
                    else:
                        found_solution_path = algo_func(*algo_args)
            except Exception as e:
                show_popup(f"Error during {current_algorithm} solve:\n{traceback.format_exc()}", "Solver Error")
                traceback.print_exc()
                error_occurred = True
            if not error_occurred:
                solve_duration = time.time() - solve_start_time
                solve_times[current_algorithm] = solve_duration
                if is_sensorless_algo:
                    belief_states = generate_belief_states(initial_state)
                    belief_state_size = len(belief_states) if belief_states else 0
                    if found_action_plan is not None:
                        num_actions = len(found_action_plan)
                        last_solved_info[f"{current_algorithm}_actions"] = num_actions
                        last_solved_info[f"{current_algorithm}_reached_goal"] = True
                        print(f"Sensorless plan found: {num_actions} actions, {solve_duration:.4f}s.")
                        print("Simulating plan execution on a sample state for visualization...")
                        sim_start_state = belief_states[0] if belief_states else initial_state
                        simulated_state_path = execute_plan(sim_start_state, found_action_plan)
                        solution = simulated_state_path
                        auto_solve = True
                        step_index = 0
                        last_step_time = time.time()
                        show_popup(f"Sensorless plan found!\n{num_actions} actions over {belief_state_size} belief states.\nTime: {solve_duration:.4f}s\n(Visualizing on one state)", "Plan Found")
                    else:
                        solution = None
                        auto_solve = False
                        if f"{current_algorithm}_actions" in last_solved_info:
                            del last_solved_info[f"{current_algorithm}_actions"]
                        if f"{current_algorithm}_reached_goal" in last_solved_info:
                            del last_solved_info[f"{current_algorithm}_reached_goal"]
                        if solve_duration >= time_limit - 0.1:
                            print(f"{current_algorithm} timed out")
                            show_popup(f"{current_algorithm} timed out after ~{time_limit}s.", "Timeout")
                        else:
                            print(f"No plan found by {current_algorithm}")
                            show_popup(f"No valid plan found by {current_algorithm}.", "No Plan Found")
                else:
                    if found_solution_path and len(found_solution_path) > 0:
                        steps = len(found_solution_path) - 1
                        final_state = found_solution_path[-1]
                        is_actually_goal = is_goal(final_state)
                        last_solved_info[f"{current_algorithm}_steps"] = steps
                        last_solved_info[f"{current_algorithm}_reached_goal"] = is_actually_goal
                        solution = found_solution_path
                        step_index = 0
                        auto_solve = True
                        last_step_time = time.time()
                        if is_actually_goal:
                            print(f"{current_algorithm}: {steps} steps, {solve_duration:.4f}s")
                        else:
                            print(f"{current_algorithm} finished (Not Goal): {steps} steps, {solve_duration:.4f}s")
                            show_popup(f"{current_algorithm} finished ({steps} steps).\nFinal state NOT goal.", "Search Complete")
                    else:
                        solution = None
                        auto_solve = False
                        if f"{current_algorithm}_steps" in last_solved_info:
                            del last_solved_info[f"{current_algorithm}_steps"]
                        if f"{current_algorithm}_reached_goal" in last_solved_info:
                            del last_solved_info[f"{current_algorithm}_reached_goal"]
                        if solve_duration >= time_limit - 0.1:
                            print(f"{current_algorithm} timed out")
                            show_popup(f"{current_algorithm} timed out after ~{time_limit}s.", "Timeout")
                        else:
                            print(f"No solution found by {current_algorithm}")
                            show_popup(f"No solution found by {current_algorithm}.", "No Solution")
        if auto_solve and solution and isinstance(solution, list) and len(solution) > 0 and isinstance(solution[0], list):
            current_time = time.time()
            anim_delay = 0.3 if len(solution) < 30 else (0.2 if len(solution) < 60 else 0.1)
            if current_time - last_step_time >= anim_delay:
                if step_index < len(solution) - 1:
                    step_index += 1
                    current_state = copy.deepcopy(solution[step_index])
                    last_step_time = current_time
                else:
                    auto_solve = False
                    is_final_state_goal = is_goal(current_state)
                    print(f"Animation complete: {'Goal reached!' if is_final_state_goal else 'Final state reached (Not Goal).'}")
        belief_state_size = len(generate_belief_states(initial_state)) if current_algorithm == 'Sensorless' else None
        ui_elements = draw_grid_and_ui(current_state, show_menu, current_algorithm,
                                       solve_times, last_solved_info, belief_state_size)
        clock.tick(60)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n--- An unexpected error occurred ---")
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)