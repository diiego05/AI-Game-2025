import pygame
import sys
from collections import deque
import copy
import time
from queue import PriorityQueue
import traceback
import math
import random

# Constants and Pygame Initialization (Keep as is)
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
pygame.display.set_caption("LamVanDi-23110191")

initial_state = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# --- Helper Functions (find_empty, is_goal, get_neighbors, state_to_tuple) ---
# Keep these functions as they are.
def find_empty(state):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] == 0:
                return i, j
    return -1, -1

def is_goal(state):
    # Added type checking for robustness
    if not isinstance(state, list) or len(state) != GRID_SIZE:
        return False
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE:
            return False
    # Direct comparison assumes goal_state is correctly defined
    return state == goal_state


def get_neighbors(state):
    neighbors = []
    empty_i, empty_j = find_empty(state)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
    for di, dj in directions:
        new_i, new_j = empty_i + di, empty_j + dj
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE:
            # Use deepcopy to ensure states are independent
            new_state = copy.deepcopy(state)
            # Swap the empty tile with the neighbor
            new_state[empty_i][empty_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    return neighbors

def state_to_tuple(state):
    """Converts a list-of-lists state into a tuple-of-tuples for hashing."""
    # Added error handling
    if not isinstance(state, list):
        # print(f"Warning: state_to_tuple received non-list input: {state}")
        return None # Or raise a TypeError
    try:
        return tuple(tuple(row) for row in state)
    except TypeError as e:
        print(f"Error converting state to tuple. State: {state}, Error: {e}")
        return None

# --- Search Algorithms (BFS, DFS, IDS, UCS, A*, Greedy, IDA*, Hill Climbing, SA, Beam Search) ---
# Keep the existing search algorithms as they are.
# ... (BFS, DFS, IDS, UCS, A*, Greedy, IDA*, Hill Climbing variants, SA, Beam Search code remains here) ...

def bfs(initial_state):
    start_time = time.time()
    queue = deque([(initial_state, [initial_state])])
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None # Handle conversion error
    visited = {init_tuple}
    while queue:
        # Time limit check
        if time.time() - start_time > 30:
            print("BFS Timeout")
            return None
        current_state, path = queue.popleft()
        if is_goal(current_state):
            return path
        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor, path + [neighbor])) # Append new state to path
    return None

def dfs(initial_state, max_depth=30): # Added default max_depth
    start_time = time.time()
    stack = [(initial_state, [initial_state], 0)] # state, path, depth
    visited = {} # Store visited state tuple -> min depth found

    while stack:
         # Time limit check
        if time.time() - start_time > 30:
            print("DFS Timeout")
            return None

        current_state, path, depth = stack.pop()
        current_tuple = state_to_tuple(current_state)

        if current_tuple is None: continue # Skip if conversion failed

        # Pruning based on visited depth or max_depth
        if current_tuple in visited and visited[current_tuple] <= depth:
            continue
        if depth > max_depth:
             continue # Depth limit exceeded for this path

        visited[current_tuple] = depth # Mark visited at this depth

        if is_goal(current_state):
            return path

        # Explore neighbors
        neighbors = get_neighbors(current_state)
        # Add neighbors to stack in reverse order to explore "leftmost" first
        for neighbor in reversed(neighbors):
             neighbor_tuple = state_to_tuple(neighbor)
             # Add to stack only if not visited or found a shorter path
             if neighbor_tuple is not None and (neighbor_tuple not in visited or visited[neighbor_tuple] > depth + 1):
                 stack.append((neighbor, path + [neighbor], depth + 1))

    return None # Goal not found within depth limit or time limit

def ids(initial_state, max_depth_limit=30): # Added default max_depth_limit
    start_time = time.time()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None

    for depth_limit in range(max_depth_limit + 1):
        # print(f"IDS: Trying depth {depth_limit}") # Optional: for debugging
        stack = [(initial_state, [initial_state], 0)] # state, path, depth
        # Visited set for the current iteration only to handle cycles within the depth limit
        visited_in_iteration = {init_tuple: 0}

        while stack:
             # Global time limit check
            if time.time() - start_time > 30:
                print("IDS Timeout")
                return None

            current_state, path, depth = stack.pop()
            if is_goal(current_state):
                return path

            # Explore neighbors only if within the current depth limit
            if depth < depth_limit:
                neighbors = get_neighbors(current_state)
                for neighbor in reversed(neighbors): # Maintain DFS exploration order
                    neighbor_tuple = state_to_tuple(neighbor)
                    # Add if valid and not visited in this iteration, or found shorter path in this iteration
                    if neighbor_tuple is not None and (neighbor_tuple not in visited_in_iteration or visited_in_iteration[neighbor_tuple] > depth + 1):
                         visited_in_iteration[neighbor_tuple] = depth + 1
                         stack.append((neighbor, path + [neighbor], depth + 1))


        if time.time() - start_time >= 30:
             print("IDS Timeout before starting next depth")
             break # Exit outer loop if time limit exceeded

    return None # Goal not found within the overall depth and time limits

def ucs(initial_state):
    start_time = time.time()
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None

    frontier.put((0, initial_state, [initial_state])) # (cost, state, path)
    visited = {init_tuple: 0} # state_tuple -> minimum cost found so far

    while not frontier.empty():
        # Time limit check
        if time.time() - start_time > 30:
            print("UCS Timeout")
            return None

        cost, current_state, path = frontier.get()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None: continue

        if is_goal(current_state):
            return path

        if current_tuple in visited and cost > visited[current_tuple]:
             continue

        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is None: continue # Skip invalid neighbor

            new_cost = cost + 1 # Assuming uniform cost of 1 per move

            # If neighbor not visited OR found a cheaper path to it
            if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                visited[neighbor_tuple] = new_cost # Update cost
                frontier.put((new_cost, neighbor, path + [neighbor])) # Add to queue

    return None # Goal not reachable or time limit exceeded


def manhattan_distance(state):
    """Calculates the Manhattan distance heuristic for a given state."""
    distance = 0
    goal_pos = {}
    # Precompute goal positions for faster lookup
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            val = goal_state[r][c]
            if val != 0: # Don't calculate distance for the empty tile
                goal_pos[val] = (r, c)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tile = state[i][j]
            if tile != 0:
                if tile in goal_pos: # Ensure tile exists in goal (robustness)
                    goal_r, goal_c = goal_pos[tile]
                    distance += abs(goal_r - i) + abs(goal_c - j)
                # else: print(f"Warning: Tile {tile} not found in goal_pos") # Should not happen
    return distance

def astar(initial_state):
    start_time = time.time()
    frontier = PriorityQueue()
    g_init = 0
    h_init = manhattan_distance(initial_state)
    f_init = g_init + h_init
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None

    # Store (f_score, g_score, state, path)
    frontier.put((f_init, g_init, initial_state, [initial_state]))
    visited = {init_tuple: g_init}

    while not frontier.empty():
        # Time limit check
        if time.time() - start_time > 30:
            print("A* Timeout")
            return None

        f_score, g_score, current_state, path = frontier.get()
        current_tuple = state_to_tuple(current_state)
        if current_tuple is None: continue

        if is_goal(current_state):
            return path

        # Optimization: If we've already found a path to this state with a
        # lower or equal g_score, we don't need to explore this one.
        if current_tuple in visited and g_score > visited[current_tuple]:
             continue


        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple is None: continue

            tentative_g = g_score + 1 # Cost of move is 1

            # If this path to the neighbor is better than any previous path found
            if neighbor_tuple not in visited or tentative_g < visited[neighbor_tuple]:
                visited[neighbor_tuple] = tentative_g # Update the cost
                h = manhattan_distance(neighbor)
                f = tentative_g + h
                frontier.put((f, tentative_g, neighbor, path + [neighbor]))

    return None # Goal not reachable or time limit exceeded


def greedy(initial_state):
    start_time = time.time()
    frontier = PriorityQueue()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None

    # Store (heuristic_value, state, path)
    frontier.put((manhattan_distance(initial_state), initial_state, [initial_state]))
    visited = {init_tuple} # Only need to track visited states, not costs

    while not frontier.empty():
        # Time limit check
        if time.time() - start_time > 30:
            print("Greedy Timeout")
            return None

        h_val, current_state, path = frontier.get()

        if is_goal(current_state):
            return path

        for neighbor in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor)
            # Add to frontier if valid and not visited
            if neighbor_tuple is not None and neighbor_tuple not in visited:
                visited.add(neighbor_tuple) # Mark as visited
                h = manhattan_distance(neighbor)
                frontier.put((h, neighbor, path + [neighbor]))

    return None # Goal not reachable (or stuck in local minimum) or time limit exceeded

# --- IDA* Helper Function ---
def search_ida(path, g, threshold, visited_in_iteration, start_time):
    """Recursive helper for IDA*."""
    current_state = path[-1]
    h = manhattan_distance(current_state)
    f = g + h

    # Prune if f exceeds the current threshold
    if f > threshold:
        return None, f # Return indicates no solution found below threshold, and the f-value causing prune

    if is_goal(current_state):
        return path, threshold # Solution found! Return path and current threshold

    # Time limit check inside recursion
    if time.time() - start_time >= 60: # Increased timeout for potentially deeper search
        return "Timeout", float('inf')

    min_new_threshold = float('inf') # Track the minimum f-value that exceeded the threshold

    for neighbor in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor)
        if neighbor_tuple is None: continue # Skip bad neighbors

        new_g = g + 1 # Cost increases by 1

        if neighbor_tuple not in visited_in_iteration or new_g < visited_in_iteration[neighbor_tuple]:
            visited_in_iteration[neighbor_tuple] = new_g # Mark visited at this g-cost for this iteration
            path.append(neighbor) # Add neighbor to current path

            result, recursive_threshold = search_ida(path, new_g, threshold, visited_in_iteration, start_time)

            path.pop() # Backtrack: remove neighbor from path

            if result == "Timeout":
                 return "Timeout", float('inf')
            if result is not None: # Solution found down this branch
                return result, threshold

            # If this branch was pruned (result is None), update the minimum threshold needed for next iteration
            min_new_threshold = min(min_new_threshold, recursive_threshold)

        # If neighbor was already visited with lower/equal g-cost in this iteration, skip

    # If no neighbor led to a solution within the threshold
    return None, min_new_threshold

def ida_star(initial_state):
    start_time = time.time()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None

    threshold = manhattan_distance(initial_state)
    path = [initial_state]

    while True: # Loop indefinitely, increasing threshold
        # print(f"IDA*: Trying threshold {threshold}") # Optional debug
        # Time limit check for the entire algorithm
        if time.time() - start_time >= 60: # Increased time limit for IDA*
            print("IDA* Global Timeout")
            return None

        # Visited set for the current iteration to prune states explored *within this threshold search*
        visited_in_iteration = {init_tuple: 0} # state -> min g-cost found in this iteration

        result, new_threshold = search_ida(path, 0, threshold, visited_in_iteration, start_time)

        if result == "Timeout":
             print("IDA* Timeout during search")
             return None
        if result is not None: # Solution found
            return result
        if new_threshold == float('inf'): # No solution possible (all branches explored)
            print("IDA*: Search exhausted, no solution found.")
            return None
        if new_threshold <= threshold: # Should not happen with admissible heuristic, but safety check
            print(f"Warning: IDA* new threshold ({new_threshold}) not greater than old ({threshold}). Incrementing.")
            new_threshold = threshold + 1

        threshold = new_threshold # Increase threshold to the minimum value that caused pruning


def simple_hill_climbing(initial_state):
    """Finds the first neighbor better than current."""
    start_time = time.time()
    current_state = initial_state
    path = [current_state]
    current_h = manhattan_distance(current_state)

    while True:
        # Time limit
        if time.time() - start_time > 30:
            print("Simple Hill Climbing Timeout/Stuck")
            return path # Return path found so far

        if is_goal(current_state):
            return path

        neighbors = get_neighbors(current_state)
        best_neighbor = None
        found_better = False

        # Find the *first* neighbor that is strictly better
        for neighbor in neighbors:
            h = manhattan_distance(neighbor)
            if h < current_h:
                best_neighbor = neighbor
                current_h = h # Update heuristic value for the next iteration
                found_better = True
                break # Move immediately once a better neighbor is found

        if not found_better:
            # Local maximum or plateau reached
            print("Simple Hill Climbing: Reached local optimum or plateau.")
            return path # Return the path to the local optimum

        # Move to the better neighbor
        current_state = best_neighbor
        path.append(current_state)
        # Loop continues with the new current_state

def steepest_hill_climbing(initial_state):
    """Finds the best neighbor among all neighbors."""
    start_time = time.time()
    current_state = initial_state
    path = [current_state]
    current_h = manhattan_distance(current_state)

    while True:
         # Time limit
        if time.time() - start_time > 30:
            print("Steepest Hill Climbing Timeout/Stuck")
            return path # Return path found so far

        if is_goal(current_state):
            return path

        neighbors = get_neighbors(current_state)
        best_neighbor = None
        best_h = current_h # Initialize best heuristic found so far in this step

        # Evaluate all neighbors to find the one with the lowest heuristic value
        for neighbor in neighbors:
            h = manhattan_distance(neighbor)
            if h < best_h: # Found a new best neighbor
                best_h = h
                best_neighbor = neighbor
            # Note: Does not handle ties explicitly (first one found with min h is chosen)

        # If no neighbor is strictly better than the current state
        if best_neighbor is None or best_h >= current_h: # Need best_neighbor check in case no neighbors
            print("Steepest Hill Climbing: Reached local optimum or plateau.")
            return path # Return the path to the local optimum/plateau

        # Move to the best neighbor found
        current_state = best_neighbor
        current_h = best_h # Update the heuristic value
        path.append(current_state)
        # Loop continues with the new current_state

def random_hill_climbing(initial_state): # Also known as Stochastic Hill Climbing
    """Randomly selects a neighbor and moves if it's better or equal."""
    start_time = time.time()
    current_state = initial_state
    path = [current_state]
    current_h = manhattan_distance(current_state)
    max_iter_no_improve = 500 # Stop if no improvement for a while (helps escape plateaus slowly)
    iter_no_improve = 0

    while True:
         # Time limit
        if time.time() - start_time > 30:
            print("Stochastic Hill Climbing Timeout")
            return path

        if is_goal(current_state):
            return path

        neighbors = get_neighbors(current_state)
        if not neighbors: # Should not happen in 8-puzzle unless already solved?
             print("Stochastic Hill Climbing: No neighbors found.")
             break # Stuck

        # Choose a random neighbor
        random_neighbor = random.choice(neighbors)
        neighbor_h = manhattan_distance(random_neighbor)

        # Move if the random neighbor is better or equal (allows sideways moves on plateaus)
        if neighbor_h <= current_h:
            if neighbor_h < current_h:
                 iter_no_improve = 0 # Reset counter if strictly better
            else:
                 iter_no_improve += 1 # Increment if equal (sideways move)

            current_state = random_neighbor
            current_h = neighbor_h
            path.append(current_state)
        else:
            # Didn't move, count as no improvement
            iter_no_improve += 1

        # Check if stuck for too long (might be on a plateau or local optimum)
        if iter_no_improve >= max_iter_no_improve:
            print(f"Stochastic Hill Climbing: No improvement for {max_iter_no_improve} iterations.")
            return path

    return path # Should ideally be reached only if goal found or no neighbors


def simulated_annealing(initial_state, initial_temp=1000, cooling_rate=0.99, min_temp=0.1, time_limit=30):
    start_time = time.time()
    current_state = initial_state
    current_h = manhattan_distance(current_state)
    path = [current_state] # Track the history of accepted states
    best_state = current_state # Track the best state found so far
    best_h = current_h

    temp = initial_temp

    while temp > min_temp:
        # Time limit check
        if time.time() - start_time > time_limit:
            print(f"Simulated Annealing Timeout ({time_limit}s)")
    
            return path # Return the path of accepted moves

        if is_goal(current_state): # Check if current state is goal
             # We might have reached goal temporarily then moved away.
             # If current is goal, it's the best possible.
             best_state = current_state
             best_h = 0
             # Consider stopping early if goal found? Or let it cool further?
             # Let's return immediately if goal is found for efficiency.
             print("Simulated Annealing: Goal found.")
             return path


        neighbors = get_neighbors(current_state)
        if not neighbors: break # Should not happen

        # Choose a random neighbor
        next_state = random.choice(neighbors)
        next_h = manhattan_distance(next_state)

        # Calculate change in heuristic (energy)
        delta_h = next_h - current_h

        # If the new state is better, always accept it
        if delta_h < 0:
            current_state = next_state
            current_h = next_h
            path.append(current_state) # Add accepted state to path
            # Update best state if this is better than the current best
            if current_h < best_h:
                best_h = current_h
                best_state = current_state # Keep track of the best state visited
        # If the new state is worse, accept it with a probability based on temperature
        else:
            # Check temp is positive before dividing and calculating exp
            if temp > 0:
                 acceptance_prob = math.exp(-delta_h / temp)
                 if random.random() < acceptance_prob:
                     current_state = next_state
                     current_h = next_h
                     path.append(current_state) # Add accepted state to path
            # If temp is zero or negative (or prob check fails), do not accept worse move

        # Cool down the temperature
        temp *= cooling_rate

    # End of loop (temperature cooled down)
    print(f"Simulated Annealing: Temperature cooled down. Best h found: {best_h}")

    return path

def beam_search(initial_state, beam_width=3):
    start_time = time.time()

    # Check if initial state is already the goal
    if is_goal(initial_state):
        return [initial_state]

    # Initialize the beam with the initial state: (state, path, heuristic)
    # We use heuristic (Manhattan distance) to evaluate states
    h_init = manhattan_distance(initial_state)
    beam = [(initial_state, [initial_state], h_init)]

    visited = set()
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None: return None
    visited.add(init_tuple) # Add initial state tuple to visited

    # Keep track of the best path found ending in the goal state
    best_goal_path = None

    while beam:
        # Time limit check
        if time.time() - start_time > 30:
            print("Beam Search Timeout")
            # Return the best goal path found so far, or None
            return best_goal_path if best_goal_path else None # Return None if timeout before finding goal


        next_level_candidates = [] # Candidates for the next beam

        # Expand all states currently in the beam
        for current_state, path, _ in beam:
            # Generate neighbors
            for neighbor in get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple is None: continue # Skip invalid states

                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple) # Mark as visited
                    new_path = path + [neighbor]

                    # Check if the neighbor is the goal
                    if is_goal(neighbor):
                        # Found a potential solution path
                        # Keep track of the shortest goal path found so far
                        if best_goal_path is None or len(new_path) < len(best_goal_path):
                             best_goal_path = new_path
                       
             
                    h_neighbor = manhattan_distance(neighbor)
                    next_level_candidates.append((neighbor, new_path, h_neighbor))

        # If no new candidates generated, search is stuck or finished
        if not next_level_candidates:
            break # Exit the main loop

        # Sort candidates based on heuristic value (lower is better)
        next_level_candidates.sort(key=lambda x: x[2])

        # Select the top 'beam_width' candidates for the next beam
        beam = next_level_candidates[:beam_width]

    if best_goal_path:
        print(f"Beam Search: Goal found with length {len(best_goal_path)}")
    else:
        print("Beam Search: Beam empty, goal not found.")

    return best_goal_path




SOLVED = "SOLVED"
UNSOLVED = "UNSOLVED"
MAX_AND_OR_DEPTH = 50 # Limit recursion depth

def _and_or_recursive(state, path, solved_states, unsolved_states, start_time, time_limit, depth):
    """Recursive helper for AND-OR search simulation."""
    state_tuple = state_to_tuple(state)
    if state_tuple is None: return UNSOLVED, None

    # Time and depth limits
    if time.time() - start_time > time_limit: return "Timeout", None
    if depth > MAX_AND_OR_DEPTH: return UNSOLVED, None # Depth limit reached

    # Base case: Goal state
    if is_goal(state):
        solved_states.add(state_tuple)
        return SOLVED, path # Return the path that reached the goal

    # Check memoization tables
    if state_tuple in solved_states: return SOLVED, path # Already known to be solvable (path here might differ, but state is solvable)
    if state_tuple in unsolved_states: return UNSOLVED, None # Already known to be unsolvable

    unsolved_states.add(state_tuple) 

    # --- OR Node Logic ---
    for neighbor in get_neighbors(state):
        neighbor_tuple = state_to_tuple(neighbor)
        if neighbor_tuple is None: continue

        # Check if neighbor is already known to be unsolvable
        if neighbor_tuple in unsolved_states: continue

        # Recursive call for the neighbor
        status, solution_path = _and_or_recursive(neighbor, path + [neighbor], solved_states, unsolved_states, start_time, time_limit, depth + 1)

        if status == "Timeout": return "Timeout", None # Propagate timeout
        if status == SOLVED:
            # Success! This state is solvable because a neighbor is.
            solved_states.add(state_tuple)
            # Crucially, remove from unsolved if it was marked temporarily
            if state_tuple in unsolved_states: unsolved_states.remove(state_tuple)
            return SOLVED, solution_path # Return the successful path found

   
    return UNSOLVED, None

def and_or_search(initial_state, time_limit=30):
    """
    Conceptual adaptation of AND-OR search for the 8-puzzle pathfinding problem.
    Acts similarly to recursive DFS with memoization.
    """
    start_time = time.time()
    solved_states = set()    # Memoization for states known to reach the goal
    unsolved_states = set()  # Memoization for states known NOT to reach the goal (or currently exploring)

    # Ensure initial state is valid before starting
    init_tuple = state_to_tuple(initial_state)
    if init_tuple is None:
        print("AND-OR Search: Invalid initial state.")
        return None

    status, solution_path = _and_or_recursive(initial_state, [initial_state], solved_states, unsolved_states, start_time, time_limit, 0)

    if status == SOLVED:
        print("AND-OR Search: Solution found.")
        return solution_path
    elif status == "Timeout":
        print(f"AND-OR Search: Timeout after {time_limit} seconds.")
        return None # Indicate timeout, potentially return partial path if needed? No, return None.
    else: # status == UNSOLVED
        print("AND-OR Search: No solution found (search space exhausted or depth limit reached).")
        return None


scroll_y = 0
menu_surface = None
total_menu_height = 0

def draw_state(state, x, y, title):
    """Draws a single state of the puzzle grid."""
    title_font = BUTTON_FONT # Use button font for consistency
    title_text = title_font.render(title, True, BLACK)
    # Center title above the grid
    title_x = x + (GRID_DISPLAY_WIDTH // 2 - title_text.get_width() // 2)
    title_y = y - title_text.get_height() - 5 # Position above the grid
    screen.blit(title_text, (title_x, title_y))

    # Draw border around the grid
    pygame.draw.rect(screen, BLACK, (x - 1, y - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell_x = x + j * CELL_SIZE
            cell_y = y + i * CELL_SIZE
            cell_rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)

            if state[i][j] != 0:
                # Determine if tile is in its correct goal position
                is_correct_pos = False

                try:
                    if isinstance(goal_state, list) and len(goal_state) > i and isinstance(goal_state[i], list) and len(goal_state[i]) > j:
                         is_correct_pos = (state[i][j] == goal_state[i][j])
                except IndexError:
                     pass # Handle potential index errors gracefully

                color = GREEN if is_correct_pos else BLUE
                # Draw the tile background
                pygame.draw.rect(screen, color, cell_rect.inflate(-6, -6), border_radius=8) # Use inflate for padding
                # Draw the number
                number = FONT.render(str(state[i][j]), True, WHITE)
                # Center the number in the cell
                screen.blit(number, number.get_rect(center=cell_rect.center))
            else:
                # Draw the empty cell background
                 pygame.draw.rect(screen, GRAY, cell_rect.inflate(-6, -6), border_radius=8)

            # Draw cell border
            pygame.draw.rect(screen, BLACK, cell_rect, 1)

def show_popup(message, title="Info"):
    """Displays a modal popup message box."""
    popup_surface = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT))
    popup_surface.fill(INFO_BG) # Light background
    border_rect = popup_surface.get_rect()
    pygame.draw.rect(popup_surface, INFO_COLOR, border_rect, 4, border_radius=10) # Border

    # Title
    title_font = pygame.font.SysFont('Arial', 28, bold=True)
    title_surf = title_font.render(title, True, INFO_COLOR)
    title_rect = title_surf.get_rect(center=(POPUP_WIDTH // 2, 30))
    popup_surface.blit(title_surf, title_rect)

    # Message Text (with word wrap)
    words = message.split(' ')
    lines = []
    current_line = ""
    text_width_limit = POPUP_WIDTH - 50 # Padding on sides

    for word in words:
        # Handle potential newline characters in the message
        if "\n" in word:
            parts = word.split("\n")
            for i, part in enumerate(parts):
                if not part: continue # Skip empty parts resulting from consecutive newlines
                test_line_part = current_line + (" " if current_line else "") + part
                line_width_part = INFO_FONT.size(test_line_part)[0] if INFO_FONT else 0
                if line_width_part <= text_width_limit:
                    current_line = test_line_part
                else:
                    lines.append(current_line)
                    current_line = part
                # Add line break after each part except the last if newline was present
                if i < len(parts) - 1:
                    lines.append(current_line)
                    current_line = ""
        else:
            # Normal word processing
            test_line = current_line + (" " if current_line else "") + word
            # Use INFO_FONT which should be initialized
            line_width = INFO_FONT.size(test_line)[0] if INFO_FONT else 0

            if line_width <= text_width_limit:
                current_line = test_line
            else:
                lines.append(current_line) # Finish previous line
                current_line = word       # Start new line

    lines.append(current_line) # Add the last line

    line_height = INFO_FONT.get_linesize() if INFO_FONT else 20 # Default line height
    start_y = title_rect.bottom + 20

    for i, line in enumerate(lines):
         if INFO_FONT: # Check if font exists
             text_surf = INFO_FONT.render(line, True, BLACK)
             # Center each line of text
             text_rect = text_surf.get_rect(center=(POPUP_WIDTH // 2, start_y + i * line_height))
             popup_surface.blit(text_surf, text_rect)
         # Limit number of lines displayed? Optional.

    # OK Button
    ok_button_rect = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface, INFO_COLOR, ok_button_rect, border_radius=8)
    # Use BUTTON_FONT for OK button text
    ok_text_surf = BUTTON_FONT.render("OK", True, WHITE) if BUTTON_FONT else None
    if ok_text_surf:
         ok_text_rect = ok_text_surf.get_rect(center=ok_button_rect.center)
         popup_surface.blit(ok_text_surf, ok_text_rect)

    # Blit the popup onto the main screen
    popup_rect = popup_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(popup_surface, popup_rect)
    pygame.display.flip()

    # Wait for user interaction
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                # Adjust mouse coordinates relative to the popup surface
                if ok_button_rect.collidepoint(mouse_x - popup_rect.left, mouse_y - popup_rect.top):
                    waiting = False
            if event.type == pygame.KEYDOWN:
                # Allow closing with Enter or Escape key
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    waiting = False
        pygame.time.delay(20) # Reduce CPU usage while waiting

def draw_menu(show_menu, mouse_pos, current_algorithm):
    """Draws the algorithm selection menu, handles scrolling."""
    global scroll_y, menu_surface, total_menu_height
    menu_elements = {}

    if not show_menu:
        # Draw Menu Hamburger Button when menu is hidden
        menu_button_rect = pygame.Rect(10, 10, 50, 40)
        pygame.draw.rect(screen, MENU_COLOR, menu_button_rect, border_radius=5)
        # Draw hamburger icon lines
        bar_width, bar_height, space = 30, 4, 7
        start_x = menu_button_rect.centerx - bar_width // 2
        start_y = menu_button_rect.centery - (bar_height * 3 + space * 2) // 2
        for i in range(3):
            pygame.draw.rect(screen, WHITE, (start_x, start_y + i * (bar_height + space), bar_width, bar_height), border_radius=2)
        menu_elements['open_button'] = menu_button_rect
        return menu_elements

    # --- Menu is Shown ---
    algorithms = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('A*', 'A*'), ('Greedy', 'Greedy'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('Beam Search', 'Beam Search'), ('AND-OR', 'AND-OR Search') # Added AND-OR
    ]
    button_height = 55
    padding = 10
    button_margin = 8 # Space between buttons

    # Calculate total height needed for all buttons
    total_menu_height = (len(algorithms) * (button_height + button_margin)) - button_margin + (2 * padding)

    # Create or resize menu surface if necessary
    # Make surface scrollable if content exceeds screen height
    display_height = max(total_menu_height, HEIGHT) # Use total height for surface if scroll needed
    if menu_surface is None or menu_surface.get_height() != display_height:
        menu_surface = pygame.Surface((MENU_WIDTH, display_height))

    menu_surface.fill(MENU_COLOR) # Background

    buttons_dict = {}
    y_position = padding # Starting Y for the first button


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

    # --- Draw Close Button (on top of the scrolled menu) ---
    close_button_rect = pygame.Rect(MENU_WIDTH - 40, 10, 30, 30) # Position relative to screen
    pygame.draw.rect(screen, RED, close_button_rect, border_radius=5)
    # Draw 'X' icon
    cx, cy = close_button_rect.center
    pygame.draw.line(screen, WHITE, (cx - 7, cy - 7), (cx + 7, cy + 7), 3)
    pygame.draw.line(screen, WHITE, (cx - 7, cy + 7), (cx + 7, cy - 7), 3)

    menu_elements['close_button'] = close_button_rect
    menu_elements['buttons'] = buttons_dict
    # Define the screen area occupied by the menu for mouse click/wheel detection
    menu_elements['menu_area'] = pygame.Rect(0, 0, MENU_WIDTH, HEIGHT)

    # Add Scrollbar (Optional visual enhancement)
    if total_menu_height > HEIGHT:
         scrollbar_height = HEIGHT * (HEIGHT / total_menu_height)
         scrollbar_y = (scroll_y / (total_menu_height - HEIGHT)) * (HEIGHT - scrollbar_height)
         scrollbar_rect = pygame.Rect(MENU_WIDTH - 8, scrollbar_y, 6, scrollbar_height)
         pygame.draw.rect(screen, GRAY, scrollbar_rect, border_radius=3)


    return menu_elements

def draw_grid_and_ui(state, show_menu, current_algorithm, solve_times, last_solved_info):
    """Draws the entire UI including grids, buttons, info panel, and menu."""
    screen.fill(WHITE) # Clear screen
    mouse_pos = pygame.mouse.get_pos()

    # Determine main content area based on menu visibility
    main_area_x = MENU_WIDTH if show_menu else 0
    main_area_width = WIDTH - main_area_x

    # --- Layout Calculations ---
    center_x_main = main_area_x + main_area_width // 2

    # Top Row: Initial and Goal States
    top_row_y = GRID_PADDING + 40 # Space from top + space for title
    grid_spacing_top = GRID_PADDING * 1.5 # Space between Initial and Goal grids
    total_width_top = 2 * GRID_DISPLAY_WIDTH + grid_spacing_top
    start_x_top = center_x_main - total_width_top // 2 # Center the row
    initial_x = start_x_top
    goal_x = start_x_top + GRID_DISPLAY_WIDTH + grid_spacing_top

    # Bottom Row: Buttons, Current State, Info Panel
    bottom_row_y = top_row_y + GRID_DISPLAY_WIDTH + GRID_PADDING + 60 # Y position for bottom elements
    button_width, button_height = 130, 45
    button_x = main_area_x + GRID_PADDING # Buttons on the left of this row
    button_mid_y = bottom_row_y + GRID_DISPLAY_WIDTH // 2 # Mid-height of the current state grid
    solve_button_y = button_mid_y - button_height - 8 # Position Solve button
    reset_button_y = button_mid_y + 8            # Position Reset button
    solve_button_rect = pygame.Rect(button_x, solve_button_y, button_width, button_height)
    reset_button_rect = pygame.Rect(button_x, reset_button_y, button_width, button_height)

    # Current State Grid
    current_state_x = button_x + button_width + GRID_PADDING * 1.5 # To the right of buttons
    current_state_y = bottom_row_y

    # Info/Comparison Panel
    info_area_x = current_state_x + GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5 # To the right of current state
    info_area_y = bottom_row_y
    # Calculate width dynamically to fill remaining space
    info_area_width = max(150, main_area_x + main_area_width - info_area_x - GRID_PADDING) # Ensure min width
    info_area_height = GRID_DISPLAY_WIDTH # Same height as current grid
    info_area_rect = pygame.Rect(info_area_x, info_area_y, info_area_width, info_area_height)


  
    draw_state(initial_state, initial_x, top_row_y, "Initial State")
    draw_state(goal_state, goal_x, top_row_y, "Goal State")
    draw_state(state, current_state_x, current_state_y, f"Current ({current_algorithm})")

    
    pygame.draw.rect(screen, RED, solve_button_rect, border_radius=5)
    solve_text = BUTTON_FONT.render("SOLVE", True, WHITE)
    screen.blit(solve_text, solve_text.get_rect(center=solve_button_rect.center))

    pygame.draw.rect(screen, BLUE, reset_button_rect, border_radius=5)
    reset_text = BUTTON_FONT.render("RESET", True, WHITE)
    screen.blit(reset_text, reset_text.get_rect(center=reset_button_rect.center))

    pygame.draw.rect(screen, INFO_BG, info_area_rect, border_radius=8)
    pygame.draw.rect(screen, GRAY, info_area_rect, 2, border_radius=8) 

    info_pad_x = 15
    info_pad_y = 10
    line_height = INFO_FONT.get_linesize() + 4 
    current_info_y = info_area_y + info_pad_y 


    compare_title_surf = TITLE_FONT.render("Comparison", True, BLACK)
    compare_title_x = info_area_rect.centerx - compare_title_surf.get_width() // 2
    screen.blit(compare_title_surf, (compare_title_x, current_info_y))
    current_info_y += compare_title_surf.get_height() + 8


    if solve_times:
        sorted_times = sorted(solve_times.items(), key=lambda item: item[1])

        for algo, time_val in sorted_times:
            steps_val = last_solved_info.get(f"{algo}_steps", "N/A")
            reached_goal = last_solved_info.get(f"{algo}_reached_goal", None) 

            base_str = f"{algo}: {time_val:.3f}s"
            steps_str = f" ({steps_val} steps)" if steps_val != "N/A" else " (-- steps)"
            goal_str = " (Goal not found)" if reached_goal is False else ""

            comp_str = base_str + steps_str + goal_str

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
        'reset_button': reset_button_rect,
        'menu': menu_elements 
    }


def main():
    global scroll_y, initial_state 
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
    last_solved_info = {} 

    clock = pygame.time.Clock()
    ui_elements = {} 

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
                             print(f"Starting solve with {current_algorithm}...")
                             solving = True 
                             solution = None
                             step_index = 0
                             auto_solve = False 
                        clicked_handled = True


                if not clicked_handled:
                    reset_button = ui_elements.get('reset_button')
                    if reset_button and reset_button.collidepoint(mouse_pos):
                        print("Resetting puzzle state.")
                        current_state = copy.deepcopy(initial_state)
                        solution = None
                        step_index = 0
                        solving = False
                        auto_solve = False
                        clicked_handled = True


            if event.type == pygame.MOUSEWHEEL and show_menu:
                 menu_area = ui_elements.get('menu', {}).get('menu_area')
                 if menu_area and menu_area.collidepoint(mouse_pos) and total_menu_height > HEIGHT:
                     scroll_amount = event.y * 35
                     max_scroll = max(0, total_menu_height - HEIGHT)
                     scroll_y = max(0, min(scroll_y - scroll_amount, max_scroll))


        if not running: break 


        if solving:
            solving = False
            solve_start_time = time.time()
            found_solution_path = None
            error_occurred = False

            try:
                state_to_solve = copy.deepcopy(current_state)

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
                elif current_algorithm == 'Beam Search': found_solution_path = beam_search(state_to_solve, beam_width=5) # Example beam width
                elif current_algorithm == 'AND-OR': found_solution_path = and_or_search(state_to_solve) # Call new algorithm
                else:
                    show_popup(f"Algorithm '{current_algorithm}' is not implemented.", "Error")
                    error_occurred = True

            except Exception as e:
                show_popup(f"Error during {current_algorithm} solve:\n{traceback.format_exc()}", "Solver Error")
                traceback.print_exc()
                error_occurred = True


            if not error_occurred:
                solve_duration = time.time() - solve_start_time
                solve_times[current_algorithm] = solve_duration


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
                         print(f"Solution found by {current_algorithm}: {steps} steps, {solve_duration:.4f}s")
                    else:
                     
                         print(f"{current_algorithm} finished: {steps} steps, {solve_duration:.4f}s. Final state NOT goal.")
                         show_popup(f"{current_algorithm} finished in {solve_duration:.4f}s ({steps} steps).\nThe final state reached is NOT the goal state.", "Search Complete (Not Goal)")


                else: 
                    solution = None
                    auto_solve = False
                    if f"{current_algorithm}_steps" in last_solved_info: del last_solved_info[f"{current_algorithm}_steps"]
                    if f"{current_algorithm}_reached_goal" in last_solved_info: del last_solved_info[f"{current_algorithm}_reached_goal"]
                    timeout_threshold = 29.9 
                    if solve_duration >= timeout_threshold:
                         print(f"{current_algorithm} timed out ({solve_duration:.4f}s).")
                         show_popup(f"{current_algorithm} timed out after ~{int(timeout_threshold)+1} seconds.", "Timeout")
                    else:
                         print(f"No solution found by {current_algorithm} ({solve_duration:.4f}s).")
                         show_popup(f"No solution path found by {current_algorithm}.", "No Solution")



        if auto_solve and solution:
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
                    if is_final_state_goal:
                        print("Animation complete: Goal reached!")
                    else:
                    
                        print("Animation complete: Final state reached (Not Goal).")


        
        ui_elements = draw_grid_and_ui(current_state, show_menu, current_algorithm,
                                       solve_times, last_solved_info)

        
        clock.tick(60) # Limit FPS to 60

    
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