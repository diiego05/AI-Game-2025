import pygame
import sys
from collections import deque
import copy
import time
from queue import PriorityQueue
# Khởi tạo pygame
pygame.init()

# Các hằng số
MENU_WIDTH = 200
WIDTH = 1100  # Increased to accommodate menu
HEIGHT = 800  # Larger window
GRID_SIZE = 3
CELL_SIZE = 100  # Smaller cells for better layout
GRID_PADDING = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (30, 144, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FONT = pygame.font.SysFont('Arial', 60)
BUTTON_FONT = pygame.font.SysFont('Arial', 30)
MENU_COLOR = (50, 50, 50)
MENU_BUTTON_COLOR = (70, 70, 70)
MENU_HOVER_COLOR = (90, 90, 90)
MENU_HIDDEN = True  # Start with hidden menu
INFO_COLOR = (50, 50, 150)
INFO_BG = (220, 220, 220)
POPUP_WIDTH = 400
POPUP_HEIGHT = 200

# Tạo cửa sổ
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Fixed method name

# Trạng thái ban đầu và trạng thái đích
initial_state = [
    [2, 6, 5],
    [0, 8, 7],
    [4, 3, 1]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Tìm vị trí ô trống (0)
def find_empty(state):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] == 0:
                return i, j
    return -1, -1

# Kiểm tra trạng thái đích
def is_goal(state):
    return state == goal_state

# Hàm sinh các trạng thái kế tiếp
def get_neighbors(state):
    neighbors = []
    empty_i, empty_j = find_empty(state)
    
    # Các hướng di chuyển: lên, xuống, trái, phải
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for di, dj in directions:
        new_i, new_j = empty_i + di, empty_j + dj
        
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
            new_state = copy.deepcopy(state)
            # Hoán đổi vị trí ô trống với ô xung quanh
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    
    return neighbors

# Chuyển đổi trạng thái ma trận thành tuple để có thể hash
def state_to_tuple(state):
    return tuple(tuple(row) for row in state)

# Thuật toán BFS
def bfs(initial_state):
    start_time = time.time()
    queue = deque([(initial_state, [initial_state])])  # Store state and path
    visited = {state_to_tuple(initial_state)}
    
    while queue and time.time() - start_time < 30:
        current_state, path = queue.popleft()
        
        if is_goal(current_state):
            return path
        
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor, path + [neighbor]))
    
    return None

def dfs(initial_state, max_depth=30):
    start_time = time.time()
    stack = [(initial_state, [initial_state], 0)]  # (state, path, depth)
    visited = {state_to_tuple(initial_state)}
    
    while stack and time.time() - start_time < 30:
        current_state, path, depth = stack.pop()
        
        if is_goal(current_state):
            return path
            
        if depth < max_depth:
            neighbors = get_neighbors(current_state)
            for neighbor in reversed(neighbors):  # Reverse to maintain correct order
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    stack.append((neighbor, path + [neighbor], depth + 1))
    
    return None

def ids(initial_state, max_depth=30):
    start_time = time.time()
    
    for depth_limit in range(max_depth):
        stack = [(initial_state, [initial_state], 0)]  # (state, path, depth)
        visited = {state_to_tuple(initial_state)}
        
        while stack and time.time() - start_time < 30:
            current_state, path, depth = stack.pop()
            
            if is_goal(current_state):
                return path
                
            if depth < depth_limit:
                neighbors = get_neighbors(current_state)
                for neighbor in reversed(neighbors):
                    neighbor_tuple = state_to_tuple(neighbor)
                    if neighbor_tuple not in visited:
                        visited.add(neighbor_tuple)
                        stack.append((neighbor, path + [neighbor], depth + 1))
        
        # Clear visited states for next iteration
        if time.time() - start_time >= 30:
            break
            
    return None


def ucs(initial_state):
    """Uniform Cost Search implementation"""
    from queue import PriorityQueue
    start_time = time.time()
    
    # Priority queue of (cost, state, path)
    frontier = PriorityQueue()
    frontier.put((0, initial_state, [initial_state]))
    visited = {state_to_tuple(initial_state)}
    
    while not frontier.empty() and time.time() - start_time < 30:
        cost, current_state, path = frontier.get()
        
        if is_goal(current_state):
            return path
        
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                # Cost is incremented by 1 for each move
                frontier.put((cost + 1, neighbor, path + [neighbor]))
    
    return None
def draw_grid(state, solution=None, step_index=0, show_menu=False, current_algorithm='BFS', solve_time=0):
    screen.fill(WHITE)
    
    # Calculate x offset based on menu state
    x_offset = MENU_WIDTH if show_menu else 0
    main_width = WIDTH - x_offset
    
    # Draw initial state (left-top) - Added title
    draw_state(initial_state, 
              x_offset + GRID_PADDING + 20, 
              GRID_PADDING,
              "Initial State")
    
    # Draw goal state (right-top) - Added title
    draw_state(goal_state, 
              x_offset + main_width - GRID_PADDING - 300, 
              GRID_PADDING,
              "Goal State")
    
    # Get current state position
    current_state_x = x_offset + main_width//2 - (CELL_SIZE * GRID_SIZE)//2
    current_state_y = GRID_PADDING + CELL_SIZE * GRID_SIZE + 100
    
    # Draw current solving state (middle-bottom)
    draw_state(state, 
              current_state_x,
              current_state_y,
              "Current State")
    
    # Draw SOLVE button to the left of current state
    solve_button = pygame.draw.rect(screen, RED, 
                                  (current_state_x - 170,  # Left of current state
                                   current_state_y + CELL_SIZE//2,
                                   150, 50))
    solve_text = BUTTON_FONT.render("SOLVE", True, WHITE)
    screen.blit(solve_text, (solve_button.centerx - solve_text.get_width()//2,
                            solve_button.centery - solve_text.get_height()//2))
    
    # Draw solving information on the right if solution exists
    if solution:
        info_x = current_state_x + (CELL_SIZE * GRID_SIZE) + 20  # Right of current state
        
        # Time info
        time_text = BUTTON_FONT.render(f"Time: {solve_time:.3f}s", True, BLACK)
        screen.blit(time_text, (info_x, current_state_y))
        
        # Steps info
        steps_text = BUTTON_FONT.render(f"Steps: {len(solution)-1}", True, BLACK)
        screen.blit(steps_text, (info_x, current_state_y + 35))
    
    # Draw menu
    mouse_pos = pygame.mouse.get_pos()
    menu_elements = draw_menu(show_menu, mouse_pos, current_algorithm)
    
    pygame.display.flip()
    return menu_elements

def draw_state(state, x, y, title):
    # Draw title
    title_text = BUTTON_FONT.render(title, True, BLACK)
    screen.blit(title_text, (x, y - 30))
    
    # Draw puzzle grid
    pygame.draw.rect(screen, GRAY, 
                    (x, y, CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE))
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] != 0:
                color = BLUE
                if state[i][j] == goal_state[i][j]:
                    color = GREEN
                    
                pygame.draw.rect(screen, color, 
                               (x + j * CELL_SIZE + 5, 
                                y + i * CELL_SIZE + 5, 
                                CELL_SIZE - 10, CELL_SIZE - 10))
                
                number = FONT.render(str(state[i][j]), True, WHITE)
                screen.blit(number, 
                          (x + j * CELL_SIZE + CELL_SIZE//2 - number.get_width()//2,
                           y + i * CELL_SIZE + CELL_SIZE//2 - number.get_height()//2))

def draw_menu(show_menu, mouse_pos, current_algorithm):
    if not show_menu:
        menu_button = pygame.draw.rect(screen, MENU_COLOR, (10, 10, 40, 40))
        for i in range(3):
            pygame.draw.rect(screen, WHITE, (15, 15 + i*12, 30, 4))
        return menu_button
    
    # Draw menu background
    menu_rect = pygame.draw.rect(screen, MENU_COLOR, (0, 0, MENU_WIDTH, HEIGHT))
    
    # Draw close button
    close_button = pygame.draw.rect(screen, RED, (MENU_WIDTH - 40, 10, 30, 30))
    pygame.draw.line(screen, WHITE, (MENU_WIDTH - 35, 15), (MENU_WIDTH - 15, 35), 2)
    pygame.draw.line(screen, WHITE, (MENU_WIDTH - 15, 15), (MENU_WIDTH - 35, 35), 2)
    
    # BFS button
    bfs_rect = pygame.Rect(20, 100, MENU_WIDTH - 40, 50)
    bfs_color = MENU_HOVER_COLOR if current_algorithm == 'BFS' else MENU_BUTTON_COLOR
    if bfs_rect.collidepoint(mouse_pos):
        bfs_color = MENU_HOVER_COLOR
    bfs_button = pygame.draw.rect(screen, bfs_color, bfs_rect)
    bfs_text = BUTTON_FONT.render("BFS", True, WHITE)
    screen.blit(bfs_text, (bfs_button.centerx - bfs_text.get_width()//2,
                          bfs_button.centery - bfs_text.get_height()//2))
    
    # DFS button
    dfs_rect = pygame.Rect(20, 170, MENU_WIDTH - 40, 50)
    dfs_color = MENU_HOVER_COLOR if current_algorithm == 'DFS' else MENU_BUTTON_COLOR
    if dfs_rect.collidepoint(mouse_pos):
        dfs_color = MENU_HOVER_COLOR
    dfs_button = pygame.draw.rect(screen, dfs_color, dfs_rect)
    dfs_text = BUTTON_FONT.render("DFS", True, WHITE)
    screen.blit(dfs_text, (dfs_button.centerx - dfs_text.get_width()//2,
                          dfs_button.centery - dfs_text.get_height()//2))
    # IDS button
    ids_rect = pygame.Rect(20, 240, MENU_WIDTH - 40, 50)
    ids_color = MENU_HOVER_COLOR if current_algorithm == 'IDS' else MENU_BUTTON_COLOR
    if ids_rect.collidepoint(mouse_pos):
        ids_color = MENU_HOVER_COLOR
    ids_button = pygame.draw.rect(screen, ids_color, ids_rect)
    ids_text = BUTTON_FONT.render("IDS", True, WHITE)
    screen.blit(ids_text, (ids_button.centerx - ids_text.get_width()//2,
                          ids_button.centery - ids_text.get_height()//2))
    
    # UCS button (add after IDS button)
    ucs_rect = pygame.Rect(20, 310, MENU_WIDTH - 40, 50)
    ucs_color = MENU_HOVER_COLOR if current_algorithm == 'UCS' else MENU_BUTTON_COLOR
    if ucs_rect.collidepoint(mouse_pos):
        ucs_color = MENU_HOVER_COLOR
    ucs_button = pygame.draw.rect(screen, ucs_color, ucs_rect)
    ucs_text = BUTTON_FONT.render("UCS", True, WHITE)
    screen.blit(ucs_text, (ucs_button.centerx - ucs_text.get_width()//2,
                          ucs_button.centery - ucs_text.get_height()//2))
    
    return menu_rect, close_button, bfs_button, dfs_button, ids_button, ucs_button

def show_popup(message):
    """Show a popup message window"""
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG)
    pygame.draw.rect(popup_surface, INFO_COLOR, popup_surface.get_rect(), 3)
    
    # Split message into lines if too long
    words = message.split()
    lines = []
    line = []
    for word in words:
        line.append(word)
        if len(' '.join(line)) > POPUP_WIDTH - 40:
            lines.append(' '.join(line[:-1]))
            line = [word]
    if line:
        lines.append(' '.join(line))
    
    # Render each line
    y = 30
    for line in lines:
        text = BUTTON_FONT.render(line, True, BLACK)
        text_rect = text.get_rect(center=(POPUP_WIDTH//2, y))
        popup_surface.blit(text, text_rect)
        y += 40
    
    # OK button
    ok_button = pygame.draw.rect(popup_surface, INFO_COLOR, 
                               (POPUP_WIDTH//2 - 40, POPUP_HEIGHT - 50, 80, 35))
    ok_text = BUTTON_FONT.render("OK", True, WHITE)
    ok_rect = ok_text.get_rect(center=ok_button.center)
    popup_surface.blit(ok_text, ok_rect)
    
    # Show popup and wait for response
    popup_rect = popup_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(popup_surface, popup_rect)
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                button_x = popup_rect.x + POPUP_WIDTH//2 - 40
                button_y = popup_rect.y + POPUP_HEIGHT - 50
                if (button_x <= x <= button_x + 80 and 
                    button_y <= y <= button_y + 35):
                    waiting = False

def main():
    current_state = copy.deepcopy(initial_state)
    solution = None
    step_index = 0
    solving = False
    auto_solve = False
    last_step_time = 0
    show_menu = False
    menu_elements = None
    running = True
    current_algorithm = 'BFS'  # Default algorithm
    solve_time = 0  # Add this variable
    
    while running:
        current_time = time.time()
        mouse_pos = pygame.mouse.get_pos()
        menu_elements = draw_grid(current_state, solution, step_index, show_menu, 
                                current_algorithm, solve_time)  # Add solve_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                
                # Menu button click handling
                if not show_menu:
                    if x <= 50 and y <= 50:  # Menu button area
                        show_menu = True
                    else:
                        # SOLVE button click - Updated coordinates
                        current_state_x = (WIDTH - (MENU_WIDTH if show_menu else 0))//2 - (CELL_SIZE * GRID_SIZE)//2
                        current_state_y = GRID_PADDING + CELL_SIZE * GRID_SIZE + 100
                        solve_button_x = current_state_x - 170
                        solve_button_y = current_state_y + CELL_SIZE//2
                        
                        if (solve_button_x <= x <= solve_button_x + 150 and 
                            solve_button_y <= y <= solve_button_y + 50):
                            solving = True
                            auto_solve = True
                            solution = None  # Reset solution when starting new solve
                else:
                    if isinstance(menu_elements, tuple):
                        _, close_button, bfs_button, dfs_button, ids_button, ucs_button = menu_elements
                        if close_button.collidepoint(x, y):
                            show_menu = False
                        elif bfs_button.collidepoint(x, y):
                            current_algorithm = 'BFS'
                            show_menu = False
                        elif dfs_button.collidepoint(x, y):
                            current_algorithm = 'DFS'
                            show_menu = False
                        elif ids_button.collidepoint(x, y):
                            current_algorithm = 'IDS'
                            show_menu = False
                        elif ucs_button.collidepoint(x, y):
                            current_algorithm = 'UCS'
                            show_menu = False

        # Solve puzzle with selected algorithm
        if solving:
            solving = False
            try:
                solve_start_time = time.time()  # Track start time
                if current_algorithm == 'BFS':
                    solution = bfs(current_state)
                elif current_algorithm == 'DFS':
                    solution = dfs(current_state)
                elif current_algorithm == 'IDS':
                    solution = ids(current_state)
                else:  # UCS
                    solution = ucs(current_state)
                solve_time = time.time() - solve_start_time  # Calculate solving time
                
                if solution:
                    step_index = 0
                    current_state = copy.deepcopy(solution[step_index])
                else:
                    show_popup(f"No solution found with {current_algorithm}!")
                    auto_solve = False
            except Exception as e:
                show_popup(f"Error during solving: {str(e)}")
                solution = None
                auto_solve = False

        # Auto-solve animation - Added error checking
        if auto_solve and solution and current_time - last_step_time >= 0.5:
            try:
                if step_index < len(solution) - 1:
                    step_index += 1
                    current_state = copy.deepcopy(solution[step_index])
                    last_step_time = current_time
                else:
                    auto_solve = False
            except Exception as e:
                print(f"Error during animation: {str(e)}")
                auto_solve = False

        pygame.time.delay(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()