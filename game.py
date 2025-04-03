import pygame
import sys
from collections import deque
import copy
import time
from queue import PriorityQueue
import traceback
import math
import random

# --- Khởi tạo Pygame và các hằng số (Giữ nguyên) ---
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
INFO_FONT = pygame.font.SysFont('Arial', 18) # Giảm font info để vừa nhiều chữ hơn
TITLE_FONT = pygame.font.SysFont('Arial', 26, bold=True)
MENU_COLOR = (50, 50, 50)
MENU_BUTTON_COLOR = (70, 70, 70)
MENU_HOVER_COLOR = (90, 90, 90)
MENU_SELECTED_COLOR = pygame.Color('dodgerblue')
MENU_HIDDEN = True
INFO_COLOR = (50, 50, 150)
INFO_BG = (245, 245, 245)
POPUP_WIDTH = 450
POPUP_HEIGHT = 250

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LamVanDi-23110191 - 8 Puzzle Solver")

initial_state = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# --- Các hàm giải thuật (bao gồm SA) và hàm hỗ trợ (Giữ nguyên) ---
# ... (find_empty, is_goal, get_neighbors, state_to_tuple, bfs, dfs, ids, ucs, manhattan_distance,
#      astar, greedy, search_ida, ida_star, simple_hill_climbing, steepest_hill_climbing,
#      random_hill_climbing, simulated_annealing) ...
# Find empty, is_goal, get_neighbors, state_to_tuple... (không thay đổi)
def find_empty(state):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] == 0:
                return i, j
    return -1, -1

def is_goal(state):
    # Thêm kiểm tra type để tránh lỗi nếu state không đúng định dạng
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            return False
    return state == goal_state


def get_neighbors(state):
    neighbors = []
    empty_i, empty_j = find_empty(state)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, dj in directions:
        new_i, new_j = empty_i + di, empty_j + dj
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
            new_state = copy.deepcopy(state)
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    return neighbors

def state_to_tuple(state):
     # Thêm kiểm tra type để tránh lỗi
    if not isinstance(state, list): return None # Hoặc raise exception
    try:
        return tuple(tuple(row) for row in state)
    except TypeError:
        print(f"Error converting state to tuple. State: {state}")
        return None


def bfs(initial_state):
    start_time = time.time()
    queue = deque([(initial_state, [initial_state])])
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None # Handle conversion error
    visited = {init_tuple}
    while queue and time.time() - start_time < 30:
        current_state, path = queue.popleft()
        if is_goal(current_state): return path
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple); queue.append((neighbor, path + [neighbor]))
    return None

def dfs(initial_state, max_depth=30):
    start_time = time.time(); stack = [(initial_state, [initial_state], 0)]; visited = {}
    while stack and time.time() - start_time < 30:
        current_state, path, depth = stack.pop()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None: continue # Skip if conversion failed

        if current_tuple in visited and visited[current_tuple] <= depth: continue
        if depth > max_depth: continue
        visited[current_tuple] = depth
        if is_goal(current_state): return path
        neighbors = get_neighbors(current_state)
        for neighbor in reversed(neighbors):
             neighbor_tuple = state_to_tuple(neighbor)
             if neighbor_tuple is not None and (neighbor_tuple not in visited or visited[neighbor_tuple] > depth + 1):
                 stack.append((neighbor, path + [neighbor], depth + 1))
    return None

def ids(initial_state, max_depth_limit=30):
    start_time = time.time()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None
    for depth_limit in range(max_depth_limit + 1):
        stack = [(initial_state, [initial_state], 0)]; visited_in_iteration = {init_tuple: 0}
        while stack and time.time() - start_time < 30:
            current_state, path, depth = stack.pop()
            if is_goal(current_state): return path
            if depth < depth_limit:
                neighbors = get_neighbors(current_state)
                for neighbor in reversed(neighbors):
                    neighbor_tuple = state_to_tuple(neighbor)
                    if neighbor_tuple is not None and (neighbor_tuple not in visited_in_iteration or visited_in_iteration[neighbor_tuple] > depth + 1):
                        visited_in_iteration[neighbor_tuple] = depth + 1
                        stack.append((neighbor, path + [neighbor], depth + 1))
        if time.time() - start_time >= 30: break
    return None

def ucs(initial_state):
    start_time = time.time(); frontier = PriorityQueue()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None
    frontier.put((0, initial_state, [initial_state]))
    visited = {init_tuple: 0}
    while not frontier.empty() and time.time() - start_time < 30:
        cost, current_state, path = frontier.get()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None: continue

        if is_goal(current_state): return path
        # Check visited before accessing cost comparison
        if current_tuple in visited and cost > visited[current_tuple]: continue
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor);
            if neighbor_tuple is None: continue
            new_cost = cost + 1
            if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                visited[neighbor_tuple] = new_cost; frontier.put((new_cost, neighbor, path + [neighbor]))
    return None

def manhattan_distance(state):
    distance = 0; goal_pos = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE): val = goal_state[r][c];
        if val != 0: goal_pos[val] = (r, c)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE): tile = state[i][j]
        if tile != 0:
            if tile in goal_pos: goal_r, goal_c = goal_pos[tile]; distance += abs(goal_r - i) + abs(goal_c - j)
    return distance

def astar(initial_state):
    start_time = time.time(); frontier = PriorityQueue()
    g_init=0; h_init=manhattan_distance(initial_state); f_init = g_init + h_init
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None
    frontier.put((f_init, g_init, initial_state, [initial_state]))
    visited = {init_tuple: g_init}
    while not frontier.empty() and time.time() - start_time < 30:
        f_score, g_score, current_state, path = frontier.get()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None: continue

        if is_goal(current_state): return path
        # Check visited before accessing cost comparison
        if current_tuple in visited and g_score > visited[current_tuple]: continue
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor);
            if neighbor_tuple is None: continue
            tentative_g = g_score + 1
            if neighbor_tuple not in visited or tentative_g < visited[neighbor_tuple]:
                visited[neighbor_tuple] = tentative_g; h = manhattan_distance(neighbor); f = tentative_g + h
                frontier.put((f, tentative_g, neighbor, path + [neighbor]))
    return None

def greedy(initial_state):
    start_time = time.time(); frontier = PriorityQueue()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None
    frontier.put((manhattan_distance(initial_state), initial_state, [initial_state]))
    visited = {init_tuple}
    while not frontier.empty() and time.time() - start_time < 30:
        h_val, current_state, path = frontier.get()
        if is_goal(current_state): return path
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple); h = manhattan_distance(neighbor); frontier.put((h, neighbor, path + [neighbor]))
    return None

def search_ida(path, g, threshold, visited_in_iteration, start_time):
    current_state = path[-1]; h = manhattan_distance(current_state); f = g + h
    if f > threshold: return None, f
    if is_goal(current_state): return path, threshold
    if time.time() - start_time >= 60: return "Timeout", float('inf')
    min_new_threshold = float('inf')
    for neighbor in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor);
        if neighbor_tuple is None: continue # Skip bad neighbors
        new_g = g + 1
        if neighbor_tuple not in visited_in_iteration or new_g < visited_in_iteration[neighbor_tuple]:
            visited_in_iteration[neighbor_tuple] = new_g
            path.append(neighbor)
            result, recursive_threshold = search_ida(path, new_g, threshold, visited_in_iteration, start_time)
            path.pop()
            if result == "Timeout": return "Timeout", float('inf')
            if result is not None: return result, threshold
            min_new_threshold = min(min_new_threshold, recursive_threshold)
    return None, min_new_threshold

def ida_star(initial_state):
    start_time = time.time();
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None
    threshold = manhattan_distance(initial_state); path = [initial_state]
    while time.time() - start_time < 60:
        visited_in_iteration = {init_tuple: 0}
        result, new_threshold = search_ida(path, 0, threshold, visited_in_iteration, start_time)
        if result == "Timeout": return None
        if result is not None: return result
        if new_threshold == float('inf'): return None
        threshold = new_threshold
    return None

def simple_hill_climbing(initial_state):
    start_time = time.time(); current_state = initial_state; path = [current_state]
    current_h = manhattan_distance(current_state)
    while time.time() - start_time < 30:
        if is_goal(current_state): return path
        neighbors = get_neighbors(current_state); best_neighbor = None; found_better = False
        for neighbor in neighbors:
            h = manhattan_distance(neighbor)
            if h < current_h: best_neighbor = neighbor; current_h = h; found_better = True; break
        if not found_better: return path
        current_state = best_neighbor; path.append(current_state)
    return path

def steepest_hill_climbing(initial_state):
    start_time = time.time(); current_state = initial_state; path = [current_state]
    current_h = manhattan_distance(current_state)
    while time.time() - start_time < 30:
        if is_goal(current_state): return path
        neighbors = get_neighbors(current_state); best_h = current_h; candidates = []
        for neighbor in neighbors:
            h = manhattan_distance(neighbor)
            if h < best_h: candidates = [(h, neighbor)]; best_h = h
            elif h == best_h and candidates: candidates.append((h, neighbor))
        if not candidates: return path
        # Check if chosen candidate is None before accessing index
        if candidates:
             chosen_h, chosen_neighbor = candidates[0]
             current_state = chosen_neighbor; current_h = chosen_h; path.append(current_state)
        else: # Should theoretically not happen if candidates is non-empty, but safe check
             return path

    return path


def random_hill_climbing(initial_state):
    start_time = time.time(); current_state = initial_state; path = [current_state]
    current_h = manhattan_distance(current_state); max_iter_no_improve = 500; iter_no_improve = 0
    while time.time() - start_time < 30:
        if is_goal(current_state): return path
        neighbors = get_neighbors(current_state);
        if not neighbors: break
        random_neighbor = random.choice(neighbors); neighbor_h = manhattan_distance(random_neighbor)
        if neighbor_h <= current_h:
            if neighbor_h < current_h: iter_no_improve = 0
            else: iter_no_improve += 1
            current_state = random_neighbor; current_h = neighbor_h; path.append(current_state)
        else: iter_no_improve += 1
        if iter_no_improve >= max_iter_no_improve: return path
    return path

def simulated_annealing(initial_state, initial_temp=1000, cooling_rate=0.99, min_temp=0.1, time_limit=30):
    start_time = time.time()
    current_state = initial_state
    current_h = manhattan_distance(current_state)
    path = [current_state]
    temp = initial_temp
    while temp > min_temp and time.time() - start_time < time_limit:
        if is_goal(current_state): return path
        neighbors = get_neighbors(current_state)
        if not neighbors: break
        next_state = random.choice(neighbors)
        next_h = manhattan_distance(next_state)
        delta_h = next_h - current_h
        if delta_h < 0:
            current_state = next_state; current_h = next_h; path.append(current_state)
        else:
            # Check temp is positive before dividing
            if temp > 0:
                 acceptance_prob = math.exp(-delta_h / temp)
                 if random.random() < acceptance_prob:
                     current_state = next_state; current_h = next_h; path.append(current_state)
            # If temp is zero or negative, don't accept worse moves
        temp *= cooling_rate
    return path

# --- Hàm vẽ (draw_state, show_popup, draw_menu) ---
# ... (Giữ nguyên draw_state, show_popup, draw_menu) ...
def draw_state(state, x, y, title):
    title_font = BUTTON_FONT
    title_text = title_font.render(title, True, BLACK)
    title_x = x + (GRID_DISPLAY_WIDTH // 2 - title_text.get_width() // 2)
    title_y = y - title_text.get_height() - 5
    screen.blit(title_text, (title_x, title_y))
    pygame.draw.rect(screen, BLACK, (x - 1, y - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell_x = x + j * CELL_SIZE; cell_y = y + i * CELL_SIZE
            cell_rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
            if state[i][j] != 0:
                is_correct_pos = False
                # Add check for goal_state structure before accessing
                if isinstance(goal_state, list) and len(goal_state) > i and isinstance(goal_state[i], list) and len(goal_state[i]) > j:
                    is_correct_pos = (state[i][j] == goal_state[i][j])

                color = GREEN if is_correct_pos else BLUE
                pygame.draw.rect(screen, color, cell_rect.inflate(-6, -6), border_radius=8)
                number = FONT.render(str(state[i][j]), True, WHITE)
                screen.blit(number, number.get_rect(center=cell_rect.center))
            else: pygame.draw.rect(screen, GRAY, cell_rect.inflate(-6, -6), border_radius=8)
            pygame.draw.rect(screen, BLACK, cell_rect, 1)

def show_popup(message, title="Info"):
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT)); popup_surface.fill(INFO_BG)
    border_rect = popup_surface.get_rect(); pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10)
    title_font = pygame.font.SysFont('Arial', 28, bold=True); title_surf = title_font.render(title, True, INFO_COLOR)
    title_rect = title_surf.get_rect(center=(POPUP_WIDTH // 2, 30)); popup_surface.blit(title_surf, title_rect)
    words = message.split(' '); lines = []; current_line = ""; text_width_limit = POPUP_WIDTH - 50
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        # Use INFO_FONT which should be initialized
        line_width = INFO_FONT.size(test_line)[0] if INFO_FONT else 0

        if line_width <= text_width_limit: current_line = test_line
        else: lines.append(current_line); current_line = word
    lines.append(current_line)
    line_height = INFO_FONT.get_linesize() if INFO_FONT else 20 # Default line height
    start_y = title_rect.bottom + 20
    for i, line in enumerate(lines):
         if INFO_FONT: # Check if font exists
             text_surf = INFO_FONT.render(line, True, BLACK)
             text_rect = text_surf.get_rect(center=(POPUP_WIDTH // 2, start_y + i * line_height))
             popup_surface.blit(text_surf, text_rect)
    ok_button_rect = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface, INFO_COLOR, ok_button_rect, border_radius=8)
    # Use BUTTON_FONT for OK button text
    ok_text_surf = BUTTON_FONT.render("OK", True, WHITE) if BUTTON_FONT else None
    if ok_text_surf:
         ok_text_rect = ok_text_surf.get_rect(center=ok_button_rect.center); popup_surface.blit(ok_text_surf, ok_text_rect)
    popup_rect = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2)); screen.blit(popup_surface, popup_rect)
    pygame.display.flip(); waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                if ok_button_rect.collidepoint(mouse_x - popup_rect.left, mouse_y - popup_rect.top): waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE: waiting = False
        pygame.time.delay(20)

scroll_y = 0; menu_surface = None; total_menu_height = 0
def draw_menu(show_menu, mouse_pos, current_algorithm):
    global scroll_y, menu_surface, total_menu_height
    if not show_menu:
        menu_button_rect = pygame.Rect(10, 10, 50, 40); pygame.draw.rect(screen, MENU_COLOR, menu_button_rect, border_radius=5)
        bar_width, bar_height, space = 30, 4, 7; start_x = menu_button_rect.centerx - bar_width // 2
        start_y = menu_button_rect.centery - (bar_height * 3 + space * 2) // 2
        for i in range(3): pygame.draw.rect(screen, WHITE, (start_x, start_y + i * (bar_height + space), bar_width, bar_height), border_radius=2)
        return {'open_button': menu_button_rect}
    algorithms = [('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'), ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'), ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'), ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing')]
    button_height, padding, button_margin = 55, 10, 8
    total_menu_height = (len(algorithms) * (button_height + button_margin)) - button_margin + (2 * padding)
    display_height = max(total_menu_height, HEIGHT)
    if menu_surface is None or menu_surface.get_height() != display_height: menu_surface = pygame.Surface((MENU_WIDTH, display_height))
    menu_surface.fill(MENU_COLOR); buttons_dict = {}; y_position = padding
    mouse_x_rel, mouse_y_rel = mouse_pos[0], mouse_pos[1] + scroll_y
    for algo_id, algo_name in algorithms:
        button_rect_local = pygame.Rect(padding, y_position, MENU_WIDTH - 2 * padding, button_height)
        is_hover = button_rect_local.collidepoint(mouse_x_rel, mouse_y_rel); is_selected = (current_algorithm == algo_id)
        button_color = MENU_SELECTED_COLOR if is_selected else (MENU_HOVER_COLOR if is_hover else MENU_BUTTON_COLOR)
        pygame.draw.rect(menu_surface, button_color, button_rect_local, border_radius=5)
        text = BUTTON_FONT.render(algo_name, True, WHITE); menu_surface.blit(text, text.get_rect(center=button_rect_local.center))
        buttons_dict[algo_id] = button_rect_local; y_position += button_height + button_margin
    visible_menu_area = pygame.Rect(0, scroll_y, MENU_WIDTH, HEIGHT); screen.blit(menu_surface, (0, 0), visible_menu_area)
    close_button_rect = pygame.Rect(MENU_WIDTH - 40, 10, 30, 30); pygame.draw.rect(screen, RED, close_button_rect, border_radius=5)
    cx, cy = close_button_rect.center; pygame.draw.line(screen, WHITE, (cx - 7, cy - 7), (cx + 7, cy + 7), 3); pygame.draw.line(screen, WHITE, (cx - 7, cy + 7), (cx + 7, cy - 7), 3)
    return {'close_button': close_button_rect, 'buttons': buttons_dict, 'menu_area': pygame.Rect(0, 0, MENU_WIDTH, HEIGHT)}

# --- Hàm Vẽ Giao Diện Chính (Cập nhật hiển thị Comparison) ---
def draw_grid_and_ui(state, show_menu, current_algorithm, solve_times, last_solved_info):
    """Vẽ giao diện, hiển thị 'Goal not found' trong Comparison nếu cần."""
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
    reset_button_y = button_mid_y + 8
    solve_button_rect = pygame.Rect(button_x, solve_button_y, button_width, button_height)
    reset_button_rect = pygame.Rect(button_x, reset_button_y, button_width, button_height)
    current_state_x = button_x + button_width + GRID_PADDING * 1.5
    current_state_y = bottom_row_y
    info_area_x = current_state_x + GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5
    info_area_y = bottom_row_y
    info_area_width = max(150, main_area_x + main_area_width - info_area_x - GRID_PADDING)
    info_area_height = GRID_DISPLAY_WIDTH
    info_area_rect = pygame.Rect(info_area_x, info_area_y, info_area_width, info_area_height)

    # Vẽ Initial, Goal, Buttons, Current State (như cũ)
    draw_state(initial_state, initial_x, top_row_y, "Initial State")
    draw_state(goal_state, goal_x, top_row_y, "Goal State")
    pygame.draw.rect(screen, RED, solve_button_rect, border_radius=5)
    solve_text = BUTTON_FONT.render("SOLVE", True, WHITE)
    screen.blit(solve_text, solve_text.get_rect(center=solve_button_rect.center))
    pygame.draw.rect(screen, BLUE, reset_button_rect, border_radius=5)
    reset_text = BUTTON_FONT.render("RESET", True, WHITE)
    screen.blit(reset_text, reset_text.get_rect(center=reset_button_rect.center))
    draw_state(state, current_state_x, current_state_y, f"Current ({current_algorithm})")

    # Vẽ Khu vực Thông tin (Comparison)
    pygame.draw.rect(screen, INFO_BG, info_area_rect, border_radius=8)
    pygame.draw.rect(screen, GRAY, info_area_rect, 2, border_radius=8)
    info_pad_x = 15; info_pad_y = 10; line_height = INFO_FONT.get_linesize() + 4 # Tăng line_height chút
    current_info_y = info_area_y + info_pad_y
    compare_title_surf = TITLE_FONT.render("Comparison", True, BLACK)
    compare_title_x = info_area_rect.centerx - compare_title_surf.get_width() // 2
    screen.blit(compare_title_surf, (compare_title_x, current_info_y))
    current_info_y += compare_title_surf.get_height() + 8

    # Bảng So sánh (CẬP NHẬT HIỂN THỊ)
    if solve_times:
        sorted_times = sorted(solve_times.items(), key=lambda item: item[1])
        for algo, time_val in sorted_times:
            steps_val = last_solved_info.get(f"{algo}_steps", "N/A")
            # LẤY THÔNG TIN REACHED_GOAL
            reached_goal = last_solved_info.get(f"{algo}_reached_goal", None)

            # Xây dựng chuỗi hiển thị
            base_str = f"{algo}: {time_val:.3f}s"
            steps_str = f" ({steps_val} steps)" if steps_val != "N/A" else " (-- steps)"
            goal_str = " (Goal not found)" if reached_goal is False else "" # Chỉ thêm nếu False

            comp_str = base_str + steps_str + goal_str

            comp_surf = INFO_FONT.render(comp_str, True, BLACK)
            # Hiển thị nếu vừa
            if info_area_x + info_pad_x + comp_surf.get_width() < info_area_x + info_area_width - info_pad_x / 2:
                 screen.blit(comp_surf, (info_area_x + info_pad_x, current_info_y))
                 current_info_y += line_height
            else: # Thử rút gọn nếu quá dài (bỏ steps)
                 comp_str_short = base_str + goal_str
                 comp_surf_short = INFO_FONT.render(comp_str_short, True, BLACK)
                 if info_area_x + info_pad_x + comp_surf_short.get_width() < info_area_x + info_area_width - info_pad_x / 2:
                     screen.blit(comp_surf_short, (info_area_x + info_pad_x, current_info_y))
                     current_info_y += line_height
                 # else: bỏ qua

            if current_info_y > info_area_y + info_area_height - info_pad_y - line_height:
                screen.blit(INFO_FONT.render("...", True, BLACK), (info_area_x + info_pad_x, current_info_y))
                break
    else:
        no_comp_surf = INFO_FONT.render("(No results yet)", True, GRAY)
        screen.blit(no_comp_surf, (info_area_x + info_pad_x, current_info_y))

    # Vẽ Menu
    menu_elements = draw_menu(show_menu, mouse_pos, current_algorithm)
    pygame.display.flip()
    return {'solve_button': solve_button_rect, 'reset_button': reset_button_rect, 'menu': menu_elements}


# --- Hàm Main (Cập nhật logic lưu kết quả) ---
def main():
    global scroll_y
    current_state = copy.deepcopy(initial_state)
    solution = None
    step_index = 0
    solving = False
    auto_solve = False
    last_step_time = 0
    show_menu = False
    running = True
    current_algorithm = 'A*'
    solve_times = {}
    # Lưu cả steps và reached_goal cho từng thuật toán
    last_solved_info = {} # {'algo_steps': steps, 'algo_reached_goal': True/False}

    clock = pygame.time.Clock()
    ui_elements = {}

    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get(): # Xử lý event như cũ
            if event.type == pygame.QUIT: running = False; break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_handled = False
                if show_menu and not clicked_handled:
                    menu_data=ui_elements.get('menu', {}); menu_area=menu_data.get('menu_area')
                    if menu_area and menu_area.collidepoint(mouse_pos):
                        close_button=menu_data.get('close_button')
                        if close_button and close_button.collidepoint(mouse_pos): show_menu=False; clicked_handled=True
                        if not clicked_handled:
                            buttons=menu_data.get('buttons', {})
                            for algo_id, button_rect_local in buttons.items():
                                button_rect_screen = button_rect_local.move(0, -scroll_y)
                                if button_rect_screen.collidepoint(mouse_pos):
                                    if current_algorithm != algo_id: print(f"Algorithm changed to: {algo_id}"); current_algorithm=algo_id; solution=None; step_index=0; auto_solve=False
                                    show_menu=False; clicked_handled=True; break
                        if not clicked_handled: clicked_handled=True
                if not clicked_handled:
                    if not show_menu:
                        menu_data=ui_elements.get('menu', {}); open_button=menu_data.get('open_button')
                        if open_button and open_button.collidepoint(mouse_pos): show_menu=True; scroll_y=0; clicked_handled=True
                    if not clicked_handled:
                         solve_button = ui_elements.get('solve_button')
                         if solve_button and solve_button.collidepoint(mouse_pos):
                             if not auto_solve and not solving: print(f"Starting solve with {current_algorithm}..."); solving=True; solution=None; step_index=0; auto_solve=False
                             clicked_handled=True
                    if not clicked_handled:
                         reset_button = ui_elements.get('reset_button')
                         if reset_button and reset_button.collidepoint(mouse_pos):
                             print("Resetting puzzle state."); current_state=copy.deepcopy(initial_state); solution=None; step_index=0; solving=False; auto_solve=False
                             clicked_handled=True
            if event.type == pygame.MOUSEWHEEL and show_menu:
                menu_area=ui_elements.get('menu', {}).get('menu_area')
                if menu_area and menu_area.collidepoint(mouse_pos):
                    scroll_amount=event.y * 35; max_scroll=max(0, total_menu_height - HEIGHT); scroll_y=max(0, min(scroll_y - scroll_amount, max_scroll))
        if not running: break

        # --- Logic giải thuật (CẬP NHẬT LƯU KẾT QUẢ) ---
        if solving:
            solving = False; solve_start_time = time.time()
            found_solution_path = None; error_occurred = False
            try:
                state_to_solve = copy.deepcopy(current_state); 
                # Gọi thuật toán (như cũ, bao gồm SA)
                if current_algorithm == 'BFS': found_solution_path = bfs(state_to_solve)
                elif current_algorithm == 'DFS': found_solution_path = dfs(state_to_solve)
                elif current_algorithm == 'IDS': found_solution_path = ids(state_to_solve)
                elif current_algorithm == 'UCS': found_solution_path = ucs(state_to_solve)
                elif current_algorithm == 'A*': found_solution_path = astar(state_to_solve)
                elif current_algorithm == 'Greedy': found_solution_path = greedy(state_to_solve)
                elif current_algorithm == 'IDA*': found_solution_path = ida_star(state_to_solve)
                elif current_algorithm == 'Hill Climbing': found_solution_path = simple_hill_climbing(state_to_solve)
                elif current_algorithm == 'Steepest Hill': found_solution_path = steepest_hill_climbing(state_to_solve)
                elif current_algorithm == 'Stochastic Hill': found_solution_path = random_hill_climbing(state_to_solve)
                elif current_algorithm == 'SA': found_solution_path = simulated_annealing(state_to_solve)
                else: show_popup(f"Algorithm '{current_algorithm}' is not implemented.", "Error"); error_occurred = True
            except Exception as e: show_popup(f"Error during {current_algorithm} solve:\n{e}", "Solver Error"); traceback.print_exc(); error_occurred = True

            # Xử lý kết quả
            if not error_occurred:
                solve_duration = time.time() - solve_start_time
                solve_times[current_algorithm] = solve_duration # Luôn lưu thời gian

                if found_solution_path and len(found_solution_path) > 0: # Check path không rỗng
                    steps = len(found_solution_path) - 1
                    final_state = found_solution_path[-1]
                    is_actually_goal = is_goal(final_state) # KIỂM TRA GOAL

                    # LƯU CẢ STEPS VÀ REACHED_GOAL
                    last_solved_info[f"{current_algorithm}_steps"] = steps
                    last_solved_info[f"{current_algorithm}_reached_goal"] = is_actually_goal

                    solution = found_solution_path; step_index = 0; auto_solve = True; last_step_time = time.time()
                    if is_actually_goal: print(f"Solution found by {current_algorithm}: {steps} steps, {solve_duration:.4f}s")
                    else: show_popup(f"{current_algorithm} finished in {solve_duration:.4f}s ({steps} steps).\nThe final state is NOT the goal.", "Search Complete")

                else: # Không tìm thấy đường đi hoặc path rỗng
                    solution = None; auto_solve = False
                    # XÓA THÔNG TIN CŨ (steps, reached_goal) NẾU THẤT BẠI
                    if f"{current_algorithm}_steps" in last_solved_info: del last_solved_info[f"{current_algorithm}_steps"]
                    if f"{current_algorithm}_reached_goal" in last_solved_info: del last_solved_info[f"{current_algorithm}_reached_goal"]

                    if solve_duration >= 29.9: print(f"{current_algorithm} timed out ({solve_duration:.4f}s)."); show_popup(f"{current_algorithm} timed out after ~30 seconds.", "Timeout")
                    else: print(f"No solution found by {current_algorithm} ({solve_duration:.4f}s)."); show_popup(f"No solution path found by {current_algorithm}.", "No Solution")


        # --- Logic animation (Giữ nguyên) ---
        if auto_solve and solution:
            current_time = time.time()
            animation_speed = 0.3 if len(solution) < 30 else (0.2 if len(solution) < 60 else 0.1)
            if current_time - last_step_time >= animation_speed:
                if step_index < len(solution) - 1:
                    step_index += 1; current_state = copy.deepcopy(solution[step_index]); last_step_time = current_time
                else:
                    auto_solve = False
                    # Kiểm tra lại goal cuối cùng sau animation
                    is_final_state_goal = is_goal(current_state)
                    if is_final_state_goal: print("Animation complete: Goal reached!")
                    else: print("Animation complete: Final state reached (Not Goal).")


        # --- Vẽ lại giao diện ---
        ui_elements = draw_grid_and_ui(current_state, show_menu, current_algorithm,
                                       solve_times, last_solved_info)
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()