import pygame
import sys
from collections import deque, defaultdict 
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

# --- Helper Functions (State Manipulation, Validation, etc.) ---
def draw_state(state_to_draw, x_pos, y_pos, title_str, is_current_anim_state=False, is_fixed_goal_display=False, is_editable=False, selected_cell_coords=None):
    title_font_ds = pygame.font.Font(None, 28) 
    title_text_ds = title_font_ds.render(title_str, True, BLACK)
    title_x_ds = x_pos + (GRID_DISPLAY_WIDTH // 2 - title_text_ds.get_width() // 2)
    title_y_ds = y_pos - title_text_ds.get_height() - 5 
    screen.blit(title_text_ds, (title_x_ds, title_y_ds))
    pygame.draw.rect(screen, BLACK, (x_pos - 1, y_pos - 1, GRID_DISPLAY_WIDTH + 2, GRID_DISPLAY_WIDTH + 2), 2)
    is_valid_structure = isinstance(state_to_draw, list) and len(state_to_draw) == GRID_SIZE and \
                         all(isinstance(row, list) and len(row) == GRID_SIZE for row in state_to_draw)
    for r_ds in range(GRID_SIZE):
        for c_ds in range(GRID_SIZE):
            cell_x_ds = x_pos + c_ds * CELL_SIZE; cell_y_ds = y_pos + r_ds * CELL_SIZE
            cell_rect_ds = pygame.Rect(cell_x_ds, cell_y_ds, CELL_SIZE, CELL_SIZE)
            tile_val = state_to_draw[r_ds][c_ds] if is_valid_structure else None
            if tile_val is None:
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
                q_surf = pygame.font.SysFont('Arial', 40).render("?", True, BLACK)
                screen.blit(q_surf, q_surf.get_rect(center=cell_rect_ds.center))
            elif tile_val == 0:
                pygame.draw.rect(screen, GRAY, cell_rect_ds.inflate(-6, -6), border_radius=8)
            else:
                cell_fill_color = GREEN if is_fixed_goal_display or \
                                   (is_current_anim_state and is_valid_structure and state_to_draw == goal_state_fixed_global) \
                                   else BLUE
                pygame.draw.rect(screen, cell_fill_color, cell_rect_ds.inflate(-6, -6), border_radius=8)
                try:
                    screen.blit(FONT.render(str(tile_val), True, WHITE), FONT.render(str(tile_val), True, WHITE).get_rect(center=cell_rect_ds.center))
                except (ValueError, TypeError): pygame.draw.rect(screen, RED, cell_rect_ds.inflate(-10,-10)) 
            pygame.draw.rect(screen, BLACK, cell_rect_ds, 1)
            if is_editable and selected_cell_coords == (r_ds, c_ds):
                 pygame.draw.rect(screen, RED, cell_rect_ds, 3) 

def find_empty(state):
    if not isinstance(state, list) or len(state) != GRID_SIZE: return -1, -1
    for i in range(GRID_SIZE):
        if not isinstance(state[i], list) or len(state[i]) != GRID_SIZE: return -1, -1
        for j in range(GRID_SIZE):
            if isinstance(state[i][j], int) and state[i][j] == 0: return i, j
    return -1, -1 

def is_goal(state):
    if not isinstance(state, list) or len(state) != GRID_SIZE or \
       not all(isinstance(row, list) and len(row) == GRID_SIZE for row in state) or \
       not is_valid_state_for_solve(state): return False
    return state == goal_state_fixed_global

def get_neighbors(state):
    neighbors = []; empty_i, empty_j = find_empty(state)
    if empty_i == -1: return []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        tile_i, tile_j = empty_i + di, empty_j + dj 
        if 0 <= tile_i < GRID_SIZE and 0 <= tile_j < GRID_SIZE:
            new_state = copy.deepcopy(state)
            new_state[empty_i][empty_j], new_state[tile_i][tile_j] = new_state[tile_i][tile_j], new_state[empty_i][empty_j]
            neighbors.append(new_state)
    return neighbors

def get_valid_actions(state):
    actions = []; empty_i, empty_j = find_empty(state)
    if empty_i == -1: return []
    if empty_i > 0: actions.append('Up')
    if empty_i < GRID_SIZE - 1: actions.append('Down')
    if empty_j > 0: actions.append('Left')
    if empty_j < GRID_SIZE - 1: actions.append('Right')
    return actions

def apply_action_to_state(state_list, action):
    if state_list is None or not is_valid_state_for_solve(state_list): return None 
    new_state = copy.deepcopy(state_list); empty_i, empty_j = find_empty(new_state)
    if empty_i == -1: return None 
    tile_i, tile_j = -1,-1
    if action == 'Up' and empty_i > 0: tile_i, tile_j = empty_i - 1, empty_j
    elif action == 'Down' and empty_i < GRID_SIZE - 1: tile_i, tile_j = empty_i + 1, empty_j
    elif action == 'Left' and empty_j > 0: tile_i, tile_j = empty_i, empty_j - 1
    elif action == 'Right' and empty_j < GRID_SIZE - 1: tile_i, tile_j = empty_i, empty_j + 1
    else: return None 
    if 0 <= tile_i < GRID_SIZE and 0 <= tile_j < GRID_SIZE:
        new_state[empty_i][empty_j], new_state[tile_i][tile_j] = new_state[tile_i][tile_j], new_state[empty_i][empty_j]
        return new_state
    return None 

def state_to_tuple(state):
    if not isinstance(state, list) or len(state) != GRID_SIZE: return None
    try:
        tuple_rows = []
        for r_idx in range(GRID_SIZE):
            row = state[r_idx]
            if not isinstance(row, list) or len(row) != GRID_SIZE: return None
            row_content = []
            for c_val in row:
                if not isinstance(c_val, int) and c_val is not None: return None
                row_content.append(c_val)
            tuple_rows.append(tuple(row_content))
        return tuple(tuple_rows)
    except (TypeError, IndexError): return None

def tuple_to_list(state_tuple):
    if not isinstance(state_tuple, tuple) or len(state_tuple) != GRID_SIZE: return None
    try:
        new_list = []
        for row_tuple in state_tuple:
            if not isinstance(row_tuple, tuple) or len(row_tuple) != GRID_SIZE: return None
            list_row = []
            for item in row_tuple:
                 if not isinstance(item, int) and item is not None: return None
                 list_row.append(item)
            new_list.append(list_row)
        return new_list
    except (TypeError, IndexError): return None

def manhattan_distance(state):
    distance = 0; goal_pos = {val: (r,c) for r,row in enumerate(goal_state_fixed_global) for c,val in enumerate(row) if val != 0}
    if not isinstance(state, list) or len(state) != GRID_SIZE : return float('inf')
    for r_curr, row_curr in enumerate(state):
        if not isinstance(row_curr, list) or len(row_curr) != GRID_SIZE: return float('inf')
        for c_curr, tile in enumerate(row_curr):
            if tile is None or not isinstance(tile, int): return float('inf') 
            if tile != 0:
                gr, gc = goal_pos.get(tile, (None, None))
                if gr is None: return float('inf')
                distance += abs(r_curr - gr) + abs(c_curr - gc)
    return distance

def is_valid_state_for_solve(state_to_check): # For path-finders
    flat_state = []
    try:
        if not isinstance(state_to_check, list) or len(state_to_check) != GRID_SIZE: return False
        for r in state_to_check:
            if not isinstance(r, list) or len(r) != GRID_SIZE: return False
            for c_val in r:
                if not isinstance(c_val, int): return False 
                flat_state.append(c_val)
    except (TypeError, IndexError): return False 
    return len(flat_state) == GRID_SIZE*GRID_SIZE and set(flat_state) == set(range(GRID_SIZE*GRID_SIZE))

def is_valid_template_for_csp(state_template): # For Sensorless/Unknown
    numbers_present = []
    try:
        if not isinstance(state_template, list) or len(state_template) != GRID_SIZE: return False
        for r in state_template:
            if not isinstance(r, list) or len(r) != GRID_SIZE: return False
            for tile in r:
                if tile is not None:
                    if not isinstance(tile, int) or not (0 <= tile < GRID_SIZE*GRID_SIZE): return False 
                    numbers_present.append(tile)
    except (TypeError, IndexError): return False
    return len(numbers_present) == len(set(numbers_present))

def get_inversions(flat_state_no_zero):
    inv = 0
    for i in range(len(flat_state_no_zero)):
        for j in range(i + 1, len(flat_state_no_zero)):
            if flat_state_no_zero[i] > flat_state_no_zero[j]: inv += 1
    return inv

def is_solvable(state_to_check):
    if not is_valid_state_for_solve(state_to_check): return False
    flat = [t for r in state_to_check for t in r if t != 0]
    inv_s = get_inversions(flat)
    flat_g = [t for r in goal_state_fixed_global for t in r if t != 0]
    inv_g = get_inversions(flat_g)
    if GRID_SIZE % 2 == 1: return (inv_s % 2) == (inv_g % 2)
    else:
        blank_r_s = next((r for r,row in enumerate(state_to_check) if 0 in row), -1)
        blank_r_g = next((r for r,row in enumerate(goal_state_fixed_global) if 0 in row), -1)
        if blank_r_s == -1 or blank_r_g == -1 : return False 
        return ((inv_s + (GRID_SIZE - blank_r_s)) % 2) == ((inv_g + (GRID_SIZE - blank_r_g)) % 2)

# --- Path-Finding Search Algorithms ---
def bfs(s, tl=30):
    st=time.time(); init_t = state_to_tuple(s)
    if not init_t : return 
    q=deque([(s, [s])]); v={init_t}
    while q:
        if time.time()-st > tl: return
        cs, p = q.popleft()
        if is_goal(cs): return p
        for n in get_neighbors(cs):
            nt = state_to_tuple(n)
            if nt and nt not in v: v.add(nt); q.append((n, p + [n]))
def dfs(s, md=30, tl=30):
    st=time.time(); S=[(s, [s], 0)]; v={}
    init_t = state_to_tuple(s)
    if not init_t : return
    while S:
        if time.time()-st > tl: return
        cs, p, d = S.pop(); cst = state_to_tuple(cs)
        if not cst or (cst in v and v[cst] <= d): continue
        v[cst] = d
        if is_goal(cs): return p
        if d >= md: continue
        for n in reversed(get_neighbors(cs)): S.append((n, p + [n], d + 1))
def ids(s, mdl=30, tl=60):
    stg=time.time()
    init_t = state_to_tuple(s)
    if not init_t : return
    for dl in range(mdl + 1):
        if time.time()-stg > tl: return
        S=[(s, [s], 0)]; vit={init_t:0}
        while S:
            if time.time()-stg > tl: return
            cs, p, d = S.pop()
            if is_goal(cs): return p
            if d < dl:
                for n in reversed(get_neighbors(cs)):
                    nt = state_to_tuple(n)
                    if nt and (nt not in vit or vit[nt] > d+1): vit[nt]=d+1; S.append((n,p+[n],d+1))
def ucs(s, tl=30):
    st=time.time(); pq=PriorityQueue(); init_t = state_to_tuple(s)
    if not init_t : return
    pq.put((0,s,[s])); v={init_t:0}
    while not pq.empty():
        if time.time()-st > tl: return
        c, cs, p = pq.get(); cst=state_to_tuple(cs)
        if not cst or c > v.get(cst, float('inf')): continue
        if is_goal(cs): return p
        for n in get_neighbors(cs):
            nt=state_to_tuple(n)
            if not nt: continue
            nc = c+1
            if nc < v.get(nt, float('inf')): v[nt]=nc; pq.put((nc,n,p+[n]))
def astar(s, tl=30):
    st=time.time(); h=manhattan_distance(s)
    if h==float('inf'): return
    pq=PriorityQueue(); init_t = state_to_tuple(s)
    if not init_t : return
    pq.put((h,0,s,[s])); vg={init_t:0}
    while not pq.empty():
        if time.time()-st > tl: return
        f,g,cs,p = pq.get(); cst=state_to_tuple(cs)
        if not cst or g > vg.get(cst, float('inf')): continue
        if is_goal(cs): return p
        for n in get_neighbors(cs):
            nt=state_to_tuple(n)
            if not nt: continue
            tg=g+1
            if tg < vg.get(nt, float('inf')):
                hn=manhattan_distance(n)
                if hn==float('inf'): continue
                vg[nt]=tg; pq.put((tg+hn,tg,n,p+[n]))
def greedy(s, tl=30):
    st=time.time(); h=manhattan_distance(s)
    if h==float('inf'): return
    pq=PriorityQueue(); init_t = state_to_tuple(s)
    if not init_t : return
    pq.put((h,s,[s])); v={init_t}
    while not pq.empty():
        if time.time()-st > tl: return
        hv,cs,p = pq.get()
        if is_goal(cs): return p
        for n in get_neighbors(cs):
            nt=state_to_tuple(n)
            if nt and nt not in v:
                v.add(nt); hn=manhattan_distance(n)
                if hn!=float('inf'): pq.put((hn,n,p+[n]))
def _sr(pi,gs,th,sti,tli,vip): 
    if time.time()-sti >= tli: return "Timeout", float('inf')
    cs=pi[-1]; h=manhattan_distance(cs)
    if h==float('inf'): return None, float('inf')
    fs=gs+h
    if fs > th: return None, fs
    if is_goal(cs): return pi[:], th
    mnt=float('inf')
    for n in get_neighbors(cs):
        nt=state_to_tuple(n)
        if not nt or nt in vip: continue
        pi.append(n); vip.add(nt)
        r,rt = _sr(pi,gs+1,th,sti,tli,vip)
        pi.pop(); vip.remove(nt)
        if r=="Timeout": return "Timeout",float('inf')
        if r: return r,th
        mnt=min(mnt,rt)
    return None,mnt
def ida_star(s, tl=60): 
    stg=time.time(); h=manhattan_distance(s)
    if h==float('inf'): return
    init_t = state_to_tuple(s)
    if not init_t : return
    th=h
    while True:
        if time.time()-stg >= tl: return
        r,nct = _sr([s],0,th,stg,tl,{init_t})
        if r=="Timeout": return
        if r: return r
        if nct==float('inf') or nct <= th: return
        th=nct
def simple_hill_climbing(s,tl=30):
    st=time.time(); cs=s; p=[cs]; ch=manhattan_distance(cs)
    if ch==float('inf'): return p
    while True:
        if time.time()-st>tl or is_goal(cs): return p
        ngh=get_neighbors(cs); random.shuffle(ngh)
        mvd=False
        for n in ngh:
            hn=manhattan_distance(n)
            if hn!=float('inf') and hn < ch: cs=n;ch=hn;p.append(cs);mvd=True;break
        if not mvd: return p
def steepest_hill_climbing(s,tl=30):
    st=time.time(); cs=s; p=[cs]; ch=manhattan_distance(cs)
    if ch==float('inf'): return p
    while True:
        if time.time()-st>tl or is_goal(cs): return p
        ngh=get_neighbors(cs); bn=None; bnh=ch
        for n in ngh:
            hn=manhattan_distance(n)
            if hn!=float('inf') and hn < bnh: bnh=hn; bn=n
        if not bn or bnh >= ch: return p
        cs=bn; ch=bnh; p.append(cs)
def random_hill_climbing(s,tl=30,mni=500):
    st=time.time(); cs=s; p=[cs]; ch=manhattan_distance(cs)
    if ch==float('inf'): return p
    ni=0
    while True:
        if time.time()-st>tl or is_goal(cs): return p
        ngh=get_neighbors(cs)
        if not ngh: break
        rn=random.choice(ngh); nh=manhattan_distance(rn)
        if nh==float('inf'): continue
        if nh <= ch:
            ni=0 if nh < ch else ni+1
            cs=rn; ch=nh; p.append(cs)
        else: ni+=1
        if ni >= mni: return p
    return p
def simulated_annealing(s,it=20.0,cr=0.95,mt=0.1,tl=15):
    stt=time.time(); cs=s; ch=manhattan_distance(s)
    if ch==float('inf'): return [s]
    p=[cs]; bsf=copy.deepcopy(cs); bhsf=ch; bpsf=[copy.deepcopy(cs)]
    t=it; itr=0; nic=0; mnr=500; miter=5000
    while t > mt and itr < miter:
        if time.time()-stt > tl: break
        if is_goal(cs):
            if ch < bhsf: bsf=copy.deepcopy(cs); bhsf=ch; bpsf=p[:]
            break
        ngh=get_neighbors(cs)
        if not ngh: break
        cnn=random.choice(ngh); cnh=manhattan_distance(cnn)
        if cnh==float('inf'): itr+=1; nic+=1; continue
        dh=cnh-ch
        if dh < 0 or (t>0 and random.random() < math.exp(-dh/t)):
            cs=cnn; ch=cnh; p.append(copy.deepcopy(cs))
            if ch < bhsf: bsf=copy.deepcopy(cs); bhsf=ch; bpsf=p[:]; nic=0
            else: nic+=1
        else: nic+=1
        if nic>=mnr: cs=copy.deepcopy(bsf); ch=bhsf; p=bpsf[:]; nic=0
        t*=cr; itr+=1
    if not is_goal(bsf) and is_valid_state_for_solve(bsf):
        rem_t=max(1,tl-(time.time()-stt))
        acp=astar(bsf,time_limit=rem_t)
        if acp and len(acp)>0:
             if bpsf and bpsf[-1] == acp[0]: return bpsf[:-1]+acp
             else: return bpsf+acp 
    return bpsf
def beam_search(s,bw=5,tl=30):
    stt=time.time()
    if not is_valid_state_for_solve(s): return
    if is_goal(s): return [s]
    ih=manhattan_distance(s)
    if ih==float('inf'): return
    init_t = state_to_tuple(s)
    if not init_t : return
    bm=[(s,[s],ih)]; vtg={init_t}; bfp=None; mit=100
    for _ in range(mit):
        if time.time()-stt > tl: return bfp
        nlc=[]
        for cs,cp,_ in bm:
            for n in get_neighbors(cs):
                nt=state_to_tuple(n)
                if not nt or nt in vtg: continue
                vtg.add(nt); np=cp+[n]
                if is_goal(n) and (not bfp or len(np) < len(bfp)): bfp=np
                hn=manhattan_distance(n)
                if hn!=float('inf'): nlc.append((n,np,hn))
        if not nlc: break
        nlc.sort(key=lambda x:x[2]); bm=nlc[:bw]
        if not bm: break
    return bfp
def _aor(st_ao, p_ao, v_ao, s_t_ao, t_l_ao, d_ao, md_ao=50): 
    s_tup_ao = state_to_tuple(st_ao)
    if not s_tup_ao: return "Fail",None
    if time.time()-s_t_ao > t_l_ao: return "Timeout",None
    if d_ao > md_ao: return "Fail",None
    if is_goal(st_ao): return "Solved", p_ao[:]
    if s_tup_ao in v_ao: return "Fail",None
    v_ao.add(s_tup_ao)
    for n_ao in get_neighbors(st_ao):
        stat_ao, sol_p_ao = _aor(n_ao,p_ao+[n_ao],v_ao,s_t_ao,t_l_ao,d_ao+1,md_ao)
        if stat_ao=="Timeout": v_ao.remove(s_tup_ao); return "Timeout",None
        if stat_ao=="Solved": v_ao.remove(s_tup_ao); return "Solved",sol_p_ao
    v_ao.remove(s_tup_ao); return "Fail",None
def and_or_search(s,tl=30,md=50): 
    stt=time.time(); itup=state_to_tuple(s)
    if not itup: return
    stat,sol_p = _aor(s,[s],{itup},stt,tl,0,md)
    return sol_p if stat=="Solved" else None

# ----- Sensorless Search / Unknown Env Functions -----
def generate_belief_states(partial_state_template):
    flat_state_template = []; unknown_positions_indices_flat = []
    known_numbers_set = set(); is_template_with_unknowns = False
    if not isinstance(partial_state_template, list) or len(partial_state_template) != GRID_SIZE: return []
    for r_idx, r_val in enumerate(partial_state_template):
        if not isinstance(r_val, list) or len(r_val) != GRID_SIZE: return []
        for c_idx, tile in enumerate(r_val):
            flat_state_template.append(tile)
            if tile is None: is_template_with_unknowns = True; unknown_positions_indices_flat.append(r_idx * GRID_SIZE + c_idx)
            elif isinstance(tile, int):
                if not (0 <= tile < GRID_SIZE*GRID_SIZE) or tile in known_numbers_set: return []
                known_numbers_set.add(tile)
            else: return [] 
    if len(flat_state_template) != GRID_SIZE * GRID_SIZE: return []
    if not is_template_with_unknowns:
        return [copy.deepcopy(partial_state_template)] if is_valid_state_for_solve(partial_state_template) and is_solvable(partial_state_template) else []
    all_possible_tiles = list(range(GRID_SIZE * GRID_SIZE))
    missing_numbers_to_fill = [num for num in all_possible_tiles if num not in known_numbers_set]
    if len(missing_numbers_to_fill) != len(unknown_positions_indices_flat): return []
    belief_states_generated = []
    for perm_fill_nums in itertools.permutations(missing_numbers_to_fill):
        new_flat_state_filled = list(flat_state_template)
        for i, unknown_idx in enumerate(unknown_positions_indices_flat):
            new_flat_state_filled[unknown_idx] = perm_fill_nums[i]
        state_2d_filled = [new_flat_state_filled[r*GRID_SIZE:(r+1)*GRID_SIZE] for r in range(GRID_SIZE)]
        if is_valid_state_for_solve(state_2d_filled) and is_solvable(state_2d_filled):
            belief_states_generated.append(state_2d_filled)
    return belief_states_generated
def is_belief_state_goal(list_of_belief_states):
    return bool(list_of_belief_states) and all(is_goal(state_b) for state_b in list_of_belief_states)

def apply_action_to_belief_states(list_of_belief_states, action_str):
    new_belief_states_set = set()
    for state_b in list_of_belief_states:
        next_s = apply_action_to_state(state_b, action_str)
        if next_s:
            next_s_tuple = state_to_tuple(next_s)
            if next_s_tuple:
                new_belief_states_set.add(next_s_tuple)
    return [tuple_to_list(s_tuple) for s_tuple in new_belief_states_set if s_tuple is not None]

def sensorless_search(start_belief_state_template, time_limit=60):
    st=time.time(); ibl=generate_belief_states(start_belief_state_template)
    if not ibl: return None,0
    try: ibt={state_to_tuple(s) for s in ibl}; assert None not in ibt
    except: return None,len(ibl) if ibl else 0
    q=deque([([],ibt)]); vbs={frozenset(ibt)}; pa=['Up','Down','Left','Right']; mpl=30; lpbs=len(ibl)
    while q:
        if time.time()-st>time_limit: return None,len(q[0][1]) if q else lpbs
        ap,cbt=q.popleft(); lpbs=len(cbt)
        if len(ap)>mpl: continue
        cbl=[tuple_to_list(s) for s in cbt if s]
        if not cbl or len(cbl)!=len(cbt): continue
        if is_belief_state_goal(cbl): return ap,len(cbl)
        for ac in pa:
            nbsl=apply_action_to_belief_states(cbl,ac)
            if not nbsl: continue
            try: nbst={state_to_tuple(s) for s in nbsl}; assert None not in nbst and nbst
            except: continue
            nbf=frozenset(nbst)
            if nbf not in vbs: vbs.add(nbf); q.append((ap+[ac],nbst))
    return None,lpbs
def execute_plan(start_state, action_plan):
    if not start_state or not isinstance(start_state,list) or not is_valid_state_for_solve(start_state): return [start_state] if start_state else None
    cs=copy.deepcopy(start_state); seq=[cs]
    if not action_plan: return seq
    for ac in action_plan:
        ns=apply_action_to_state(cs,ac)
        if not ns: return seq
        cs=ns; seq.append(copy.deepcopy(cs))
    return seq

# --- CSP Helper Functions ---
def get_empty_cell(board):
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell is None:
                return r, c
    return None
def get_used_numbers(board):
    return {cell for row in board for cell in row if cell is not None}

# --- CSP Solvers (Backtracking, AC3, Kiểm Thử) ---
def csp_backtracking_solver(start_template_board=None, goal_state=None, time_limit=10):
    st = time.time()
    board = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
    steps = [copy.deepcopy(board)]
    cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]

    def solve(idx):
        if time.time() - st > time_limit:
            return False, "Timeout"
        if idx == len(cells):
            return board == goal_state, None

        r, c = cells[idx]
        used_nums = get_used_numbers(board)
        available_nums = [n for n in range(GRID_SIZE * GRID_SIZE) if n not in used_nums]

        for num in available_nums:
            board[r][c] = num
            steps.append(copy.deepcopy(board))
            result, status = solve(idx + 1)
            if status == "Timeout":
                return False, "Timeout"
            if result:
                return True, None
            board[r][c] = None
            steps.append(copy.deepcopy(board))

        return False, None

    success, status = solve(0)
    if status == "Timeout":
        return steps, "Timeout"
    return steps, success

def kiem_thu_solver(start_template_board=None, goal_state=None, time_limit=10):
    st = time.time()
    board = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
    steps = []

    cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]

    for idx, (r, c) in enumerate(cells):
        if time.time() - st > time_limit:
            return steps, "Timeout"
        val = goal_state[r][c]
        if val in get_used_numbers(board):
            return steps, False  # duplicate
        board[r][c] = val
        steps.append(copy.deepcopy(board))

    return steps, board == goal_state

def ac3_then_backtracking_solver(start_template_board=None, goal_state=None, time_limit=10):
    import time
    st = time.time()
    board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    steps = []

    # 1. Khởi tạo miền cho từng ô
    domains = {(r, c): set(range(9)) for r in range(GRID_SIZE) for c in range(GRID_SIZE)}

    # 2. Xây dựng hàm lấy các ô liên hệ ràng buộc
    def get_neighbors(pos):
        return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r, c) != pos]

    # 3. Hàng đợi gồm các cặp có ràng buộc trực tiếp
    queue = [(xi, xj) for xi in domains for xj in get_neighbors(xi)]
    processed_pairs = set(queue)

    # 4. AC3
    while queue:
        if time.time() - st > time_limit:
            return steps, "Timeout"
        xi, xj = queue.pop(0)
        if len(domains[xj]) == 1:
            val = next(iter(domains[xj]))
            if val in domains[xi]:
                domains[xi].remove(val)
                if not domains[xi]:
                    return steps, False
                for xk in get_neighbors(xi):
                    if (xk, xi) not in processed_pairs:
                        queue.append((xk, xi))
                        processed_pairs.add((xk, xi))

    # 5. Gán các ô có miền đơn
    for (r, c), dom in domains.items():
        if len(dom) == 1:
            board[r][c] = next(iter(dom))
            steps.append(copy.deepcopy(board))

    # 6. Backtracking nếu còn ô trống
    def get_used_numbers(bd):
        return {cell for row in bd for cell in row if cell is not None}

    def backtrack():
        if time.time() - st > time_limit:
            return False
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if board[r][c] is None:
                    for val in sorted(domains[(r, c)] - get_used_numbers(board)):
                        board[r][c] = val
                        steps.append(copy.deepcopy(board))
                        if backtrack():
                            return True
                        board[r][c] = None
                        steps.append(copy.deepcopy(board))
                    return False
        return board == goal_state

    success = backtrack()
    return steps, success


# --- Q-Learning and Genetic Algorithm ---
GA_POPULATION_SIZE = 100; GA_NUM_GENERATIONS = 100; GA_MAX_INITIAL_MOVES = 35
GA_MAX_MOVES_PER_INDIVIDUAL = 70; GA_MUTATION_RATE = 0.1; GA_POINT_MUTATION_RATE = 0.05
GA_CROSSOVER_RATE = 0.8; GA_TOURNAMENT_SIZE = 5; GA_ELITISM_COUNT = 5

def q_learning_train_and_solve(start_node_state, time_limit=60):
    st=time.time(); g_tup=state_to_tuple(goal_state_fixed_global)
    if not is_valid_state_for_solve(start_node_state) or not is_solvable(start_node_state) or not g_tup: return
    s_tup=state_to_tuple(start_node_state); 
    if not s_tup: return
    lr,df,es,ed,em=0.1,0.95,1.0,0.9995,0.05; ne,mspe=1500,300
    qt=defaultdict(lambda:defaultdict(float)); eps=es
    for ep in range(ne):
        if time.time()-st > time_limit: break
        csl=copy.deepcopy(start_node_state)
        for _ in range(mspe):
            cst=state_to_tuple(csl); 
            if not cst or cst==g_tup: break
            v_a=get_valid_actions(csl); 
            if not v_a: break
            ca = random.choice(v_a) if random.random()<eps else max(qt[cst],key=qt[cst].get) if qt.get(cst) else random.choice(v_a) 
            if not ca: break
            nsl=apply_action_to_state(csl,ca); 
            if not nsl: continue
            nst=state_to_tuple(nsl); 
            if not nst: continue
            rwd,term = (-1,False) if nst!=g_tup else (100,True)
            oq=qt[cst][ca]; max_qn=max(qt[nst].values()) if qt.get(nst) and not term else 0.0 
            qt[cst][ca]=oq+lr*(rwd+df*max_qn-oq)
            csl=nsl
            if term: break
        eps=max(em,eps*ed)
    p=[start_node_state]; csl=copy.deepcopy(start_node_state)
    for _ in range(max(500,mspe*2)):
        if time.time()-st > time_limit: break
        cst=state_to_tuple(csl)
        if not cst or cst==g_tup: break
        v_a=get_valid_actions(csl); 
        if not v_a: break
        cqv=qt.get(cst); ba=max(v_a,key=lambda act: cqv.get(act,-float('inf'))) if cqv else random.choice(v_a) 
        nsl=apply_action_to_state(csl,ba)
        if not nsl: break
        p.append(nsl); csl=nsl
    return p

class GAIndividual:
    def __init__(self, m): self.moves=list(m); self.fitness=0.0
    def __lt__(self,o): return self.fitness < o.fitness
def _ga_fit(ind,s,gt):
    cs=copy.deepcopy(s); plen=0; grs=-1; agm=None
    if not is_valid_state_for_solve(cs): return 0.0,None
    if not ind.moves: md=manhattan_distance(cs); return (0.0,None) if md==float('inf') else (max(0.01,(30.0-md)/30.0 if 30.0>0 else 0.5),None)
    for i,ma in enumerate(ind.moves):
        ns=apply_action_to_state(cs,ma)
        if not ns: md=manhattan_distance(cs); return (0.0,None) if md==float('inf') else (1.0/(1.0+md+(len(ind.moves)-i)*5.0),None)
        cs=ns; plen+=1; cst=state_to_tuple(cs)
        if not cst: return 0.0,None
        if cst==gt: grs=plen; agm=ind.moves[:plen]; break
    if not is_valid_state_for_solve(cs): return 0.0,None
    if grs!=-1 and agm: return 1000.0+(GA_MAX_MOVES_PER_INDIVIDUAL-grs),agm
    md=manhattan_distance(cs); 
    if md==float('inf'): return 0.0,None
    fit=(30.0-md)/30.0 if 30.0>0 else 0.5; fit-=(len(ind.moves)/(GA_MAX_MOVES_PER_INDIVIDUAL or 1))*0.1
    return max(0.01,fit),None
def _ga_pop(s,ps,mim):
    pop=[]
    for _ in range(ps):
        ct=copy.deepcopy(s); ms=[]
        nmi=random.randint(max(1,max(1,mim)//2),max(1,mim))
        for _ in range(nmi):
            if not is_valid_state_for_solve(ct): break
            va=get_valid_actions(ct); 
            if not va: break
            ca=random.choice(va); nt=apply_action_to_state(ct,ca)
            if not nt: break
            ms.append(ca); ct=nt
            if len(ms)>=GA_MAX_MOVES_PER_INDIVIDUAL: break
        pop.append(GAIndividual(ms))
    return pop
def _ga_sel(pop,ts):
    sp=[]; ats=min(ts,len(pop))
    if ats==0: return [] if not pop else [pop[0]] 
    for _ in range(len(pop)): sp.append(max(random.sample(pop,ats),key=lambda i:i.fitness))
    return sp
def _ga_co(p1,p2):
    if random.random()<GA_CROSSOVER_RATE:
        l1,l2=len(p1.moves),len(p2.moves)
        if min(l1,l2)<1: cm1,cm2=p1.moves[:]+p2.moves[l2//2:],p2.moves[:]+p1.moves[l1//2:]
        else: pt1,pt2=random.randint(0,l1-1 if l1>0 else 0),random.randint(0,l2-1 if l2>0 else 0); cm1,cm2=p1.moves[:pt1]+p2.moves[pt2:],p2.moves[:pt2]+p1.moves[pt1:]
        return GAIndividual(cm1[:GA_MAX_MOVES_PER_INDIVIDUAL]),GAIndividual(cm2[:GA_MAX_MOVES_PER_INDIVIDUAL])
    return GAIndividual(p1.moves[:]),GAIndividual(p2.moves[:])
def _ga_mut(ind,pmr,ssv):
    if random.random()<GA_MUTATION_RATE:
        nm=list(ind.moves); mf=False
        for i in range(len(nm)):
            if random.random()<pmr:
                pnm=[m for m in ['Up','Down','Left','Right'] if m!=nm[i]]
                if pnm: nm[i]=random.choice(pnm); mf=True
        if random.random()<0.2:
            if random.random()<0.5 and len(nm)<GA_MAX_MOVES_PER_INDIVIDUAL: ap=random.randint(0,len(nm)) if len(nm)>0 else 0; nm.insert(ap,random.choice(['Up','Down','Left','Right'])); mf=True
            elif len(nm)>0: del nm[random.randint(0,len(nm)-1)]; mf=True
        if mf: ind.moves=nm[:GA_MAX_MOVES_PER_INDIVIDUAL]
        if not ind.moves and mf: va=get_valid_actions(ssv); ind.moves=[random.choice(va)] if va else []
    return ind
def genetic_algorithm_solve(s,tl=60):
    st=time.time()
    if not is_valid_state_for_solve(s) or not is_solvable(s): return
    gt=state_to_tuple(goal_state_fixed_global); 
    if not gt: return
    pop=_ga_pop(s,GA_POPULATION_SIZE,GA_MAX_INITIAL_MOVES); 
    if not pop: return [s] 
    bsi=None; bfo=-float('inf'); bie=pop[0] if pop else GAIndividual([])
    for _ in range(GA_NUM_GENERATIONS):
        if time.time()-st > tl: break
        for ind in pop:
            fv,mig=_ga_fit(ind,s,gt); ind.fitness=fv
            if fv>bfo: bfo=fv; bie=copy.deepcopy(ind)
            if mig and (not bsi or fv > bsi.fitness): bsi=GAIndividual(mig); bsi.fitness=fv
        if bsi and bsi.fitness >= 1000.0+(GA_MAX_MOVES_PER_INDIVIDUAL*0.5): break
        np=[]
        if not pop: break
        pop.sort(key=lambda i:i.fitness,reverse=True); np.extend(pop[:GA_ELITISM_COUNT])
        ppool=_ga_sel(pop,GA_TOURNAMENT_SIZE); 
        if not ppool: ppool=pop[:max(1,len(pop)//2)] if pop else []
        while len(np)<GA_POPULATION_SIZE:
            if not ppool: break
            p1_idx = random.randrange(len(ppool)) 
            p2_idx = random.randrange(len(ppool))
            p1 = ppool[p1_idx]
            p2 = ppool[p2_idx]
            c1,c2=_ga_co(p1,p2); np.append(_ga_mut(c1,GA_POINT_MUTATION_RATE,s))
            if len(np)<GA_POPULATION_SIZE: np.append(_ga_mut(c2,GA_POINT_MUTATION_RATE,s))
        population = np if np else pop 
        if not population: break
    fsm=bsi.moves if bsi else (bie.moves if bie and bie.moves else None) 
    return execute_plan(s,fsm) if fsm else [s]

# --- UIBrick and Drawing Functions ---
scroll_y = 0; menu_surface = None; total_menu_height = 0
def draw_menu(show_menu_flag, mouse_pos_dm, current_selected_algorithm_dm):
    global scroll_y, menu_surface, total_menu_height
    menu_elements_dict = {}
    if not show_menu_flag:
        menu_button_rect_dm = pygame.Rect(10, 10, 50, 40)
        pygame.draw.rect(screen, MENU_COLOR, menu_button_rect_dm, border_radius=5)
        bar_width_dm, bar_height_dm, space_dm = 30, 4, 7
        start_x_dm = menu_button_rect_dm.centerx - bar_width_dm // 2
        start_y_dm = menu_button_rect_dm.centery - (bar_height_dm * 3 + space_dm * 2) // 2 + bar_height_dm // 2
        for i in range(3): pygame.draw.rect(screen, WHITE, (start_x_dm, start_y_dm + i * (bar_height_dm + space_dm), bar_width_dm, bar_height_dm), border_radius=2)
        menu_elements_dict['open_button'] = menu_button_rect_dm
        return menu_elements_dict
    algorithms_list_dm = [
        ('BFS', 'BFS'), ('DFS', 'DFS'), ('IDS', 'IDS'), ('UCS', 'UCS'),
        ('Greedy', 'Greedy'), ('A*', 'A*'), ('IDA*', 'IDA*'),
        ('Hill Climbing', 'Simple Hill (SHC)'), ('Steepest Hill', 'Steepest Hill'),
        ('Stochastic Hill', 'Stochastic Hill'), ('SA', 'Simulated Annealing'),
        ('GA', 'Genetic Algorithm'), ('Beam Search', 'Beam Search'),
        ('AND-OR', 'AND-OR Search'), ('Sensorless', 'Sensorless Plan'),
        ('Unknown', 'Unknown Env'), ('KiemThu', 'Kiểm Thử'), 
        ('Backtracking', 'Backtracking'), ('AC3', 'AC3'), 
        ('Q-Learning', 'Q-Learning'),
    ]
    button_h_dm, padding_dm, button_margin_dm = 55, 10, 8
    total_menu_height = (len(algorithms_list_dm) * (button_h_dm + button_margin_dm)) - button_margin_dm + (2 * padding_dm)
    display_height_menu_surf = max(total_menu_height, HEIGHT)
    if menu_surface is None or menu_surface.get_height() != display_height_menu_surf:
        menu_surface = pygame.Surface((MENU_WIDTH, display_height_menu_surf))
    menu_surface.fill(MENU_COLOR); buttons_dict_dm = {}; y_pos_current_button = padding_dm
    mouse_x_rel_dm, mouse_y_rel_dm = mouse_pos_dm[0], mouse_pos_dm[1] + scroll_y
    for algo_id_dm, algo_name_dm in algorithms_list_dm:
        button_rect_local_dm = pygame.Rect(padding_dm, y_pos_current_button, MENU_WIDTH - 2 * padding_dm, button_h_dm)
        is_hover_dm = button_rect_local_dm.collidepoint(mouse_x_rel_dm, mouse_y_rel_dm)
        is_selected_dm = (current_selected_algorithm_dm == algo_id_dm)
        button_color_dm = MENU_SELECTED_COLOR if is_selected_dm else (MENU_HOVER_COLOR if is_hover_dm else MENU_BUTTON_COLOR)
        pygame.draw.rect(menu_surface, button_color_dm, button_rect_local_dm, border_radius=5)
        text_surf_dm = BUTTON_FONT.render(algo_name_dm, True, WHITE)
        menu_surface.blit(text_surf_dm, text_surf_dm.get_rect(center=button_rect_local_dm.center))
        buttons_dict_dm[algo_id_dm] = button_rect_local_dm
        y_pos_current_button += button_h_dm + button_margin_dm
    screen.blit(menu_surface, (0,0), pygame.Rect(0, scroll_y, MENU_WIDTH, HEIGHT))
    close_button_rect_dm = pygame.Rect(MENU_WIDTH - 40, 10, 30, 30)
    pygame.draw.rect(screen, RED, close_button_rect_dm, border_radius=5)
    cx_dm, cy_dm = close_button_rect_dm.center
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm - 7), (cx_dm + 7, cy_dm + 7), 3)
    pygame.draw.line(screen, WHITE, (cx_dm - 7, cy_dm + 7), (cx_dm + 7, cy_dm - 7), 3)
    menu_elements_dict['close_button'] = close_button_rect_dm
    menu_elements_dict['buttons'] = {algo_id: rect_local.move(0, -scroll_y) for algo_id, rect_local in buttons_dict_dm.items()}
    menu_elements_dict['menu_area'] = pygame.Rect(0, 0, MENU_WIDTH, HEIGHT)
    if total_menu_height > HEIGHT:
        scrollbar_track_height = HEIGHT - 2*padding_dm
        scrollbar_height_val = max(20, scrollbar_track_height * (HEIGHT / total_menu_height))
        max_scroll_y = total_menu_height - HEIGHT
        scroll_ratio = scroll_y / max_scroll_y if max_scroll_y > 0 else 0 
        scrollbar_y_pos_thumb = padding_dm + scroll_ratio * (scrollbar_track_height - scrollbar_height_val)
        pygame.draw.rect(screen, GRAY, pygame.Rect(MENU_WIDTH - 10, scrollbar_y_pos_thumb, 6, scrollbar_height_val), border_radius=3)
    return menu_elements_dict

def show_popup(message_str, title_str="Info"):
    popup_surface_sp = pygame.Surface((POPUP_WIDTH, POPUP_HEIGHT)); popup_surface_sp.fill(INFO_BG)
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, popup_surface_sp.get_rect(), 4, border_radius=10)
    title_surf_sp = pygame.font.SysFont('Arial', 28, bold=True).render(title_str, True, INFO_COLOR)
    popup_surface_sp.blit(title_surf_sp, title_surf_sp.get_rect(center=(POPUP_WIDTH // 2, 30)))
    words_sp = message_str.replace('\n', ' \n ').split(' '); lines_sp = []; current_line_sp = ""
    text_width_limit_sp = POPUP_WIDTH - 60; line_height_sp = INFO_FONT.get_linesize()
    max_lines_display = (POPUP_HEIGHT - title_surf_sp.get_rect().bottom - 70) // line_height_sp if line_height_sp > 0 else 0
    for word_sp in words_sp:
        if word_sp == '\n': lines_sp.append(current_line_sp); current_line_sp = ""; continue
        test_line_sp = current_line_sp + (" " if current_line_sp else "") + word_sp
        if INFO_FONT.size(test_line_sp)[0] <= text_width_limit_sp: current_line_sp = test_line_sp
        else: lines_sp.append(current_line_sp); current_line_sp = word_sp
    lines_sp.append(current_line_sp)
    text_start_y_sp = title_surf_sp.get_rect().bottom + 20
    for i, line_text_sp in enumerate(lines_sp):
        if i >= max_lines_display and max_lines_display > 0 : 
            popup_surface_sp.blit(INFO_FONT.render("...", True, BLACK), ((POPUP_WIDTH - INFO_FONT.size("...")[0]) // 2, text_start_y_sp + i * line_height_sp)); break
        text_surf_sp = INFO_FONT.render(line_text_sp, True, BLACK)
        popup_surface_sp.blit(text_surf_sp, text_surf_sp.get_rect(centerx=POPUP_WIDTH // 2, top=text_start_y_sp + i * line_height_sp))
    ok_button_rect_sp = pygame.Rect(POPUP_WIDTH // 2 - 60, POPUP_HEIGHT - 65, 120, 40)
    pygame.draw.rect(popup_surface_sp, INFO_COLOR, ok_button_rect_sp, border_radius=8)
    popup_surface_sp.blit(BUTTON_FONT.render("OK", True, WHITE), BUTTON_FONT.render("OK", True, WHITE).get_rect(center=ok_button_rect_sp.center))
    popup_rect_on_screen_sp = popup_surface_sp.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(popup_surface_sp, popup_rect_on_screen_sp); pygame.display.flip()
    waiting_for_ok = True
    while waiting_for_ok:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if ok_button_rect_sp.collidepoint(event.pos[0] - popup_rect_on_screen_sp.left, event.pos[1] - popup_rect_on_screen_sp.top): waiting_for_ok = False
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE): waiting_for_ok = False
        pygame.time.delay(20)

def draw_grid_and_ui(current_anim_state_dgui, show_menu_dgui, current_algo_name_dgui, active_solver_name_display, solve_times_dgui, last_solved_run_info_dgui, current_belief_state_size_dgui=None, selected_cell_for_input_coords=None):
    screen.fill(WHITE)
    main_area_x_offset = MENU_WIDTH if show_menu_dgui else 0
    main_area_width_dgui = WIDTH - main_area_x_offset
    center_x_main_area = main_area_x_offset + main_area_width_dgui // 2
    top_row_y_grids = GRID_PADDING + 40
    total_width_top_grids = 2 * GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5
    start_x_top_grids = center_x_main_area - total_width_top_grids // 2
    draw_state(initial_state, start_x_top_grids, top_row_y_grids, "Initial State (Editable)", is_editable=True, selected_cell_coords=selected_cell_for_input_coords)
    draw_state(goal_state_fixed_global, start_x_top_grids + GRID_DISPLAY_WIDTH + GRID_PADDING * 1.5, top_row_y_grids, "Goal State", is_fixed_goal_display=True)
    bottom_row_y_start = top_row_y_grids + GRID_DISPLAY_WIDTH + GRID_PADDING + 60
    button_width_ctrl, button_height_ctrl = 140, 45
    buttons_start_x = main_area_x_offset + GRID_PADDING
    buttons_mid_y_anchor = bottom_row_y_start + GRID_DISPLAY_WIDTH // 2
    solve_btn_rect = pygame.Rect(buttons_start_x, buttons_mid_y_anchor - button_height_ctrl*2 - 16, button_width_ctrl, button_height_ctrl)
    reset_sol_btn_rect = pygame.Rect(buttons_start_x, solve_btn_rect.bottom + 8, button_width_ctrl, button_height_ctrl)
    reset_all_btn_rect = pygame.Rect(buttons_start_x, reset_sol_btn_rect.bottom + 8, button_width_ctrl, button_height_ctrl)
    reset_disp_btn_rect = pygame.Rect(buttons_start_x, reset_all_btn_rect.bottom + 8, button_width_ctrl, button_height_ctrl)
    pygame.draw.rect(screen, RED, solve_btn_rect, border_radius=5); screen.blit(BUTTON_FONT.render("SOLVE", True, WHITE), solve_btn_rect.inflate(-20, -10))
    pygame.draw.rect(screen, BLUE, reset_sol_btn_rect, border_radius=5); screen.blit(BUTTON_FONT.render("Reset Solution", True, WHITE), BUTTON_FONT.render("Reset Solution", True, WHITE).get_rect(center=reset_sol_btn_rect.center))
    pygame.draw.rect(screen, BLUE, reset_all_btn_rect, border_radius=5); screen.blit(BUTTON_FONT.render("Reset All", True, WHITE), BUTTON_FONT.render("Reset All", True, WHITE).get_rect(center=reset_all_btn_rect.center))
    pygame.draw.rect(screen, BLUE, reset_disp_btn_rect, border_radius=5); screen.blit(BUTTON_FONT.render("Reset Display", True, WHITE), BUTTON_FONT.render("Reset Display", True, WHITE).get_rect(center=reset_disp_btn_rect.center))
    current_state_grid_x = buttons_start_x + button_width_ctrl + GRID_PADDING * 1.5
    current_state_title_text = f"Current ({active_solver_name_display})"
    if active_solver_name_display in ['Sensorless', 'Unknown'] and current_belief_state_size_dgui is not None:
        current_state_title_text = f"Belief States: {current_belief_state_size_dgui}"
    draw_state(current_anim_state_dgui, current_state_grid_x, bottom_row_y_start, current_state_title_text, is_current_anim_state=True)
    info_area_rect = pygame.Rect(current_state_grid_x + GRID_DISPLAY_WIDTH + GRID_PADDING*1.5, bottom_row_y_start, max(150, (main_area_x_offset + main_area_width_dgui) - (current_state_grid_x + GRID_DISPLAY_WIDTH + GRID_PADDING*1.5) - GRID_PADDING), GRID_DISPLAY_WIDTH)
    pygame.draw.rect(screen, INFO_BG, info_area_rect, border_radius=8); pygame.draw.rect(screen, GRAY, info_area_rect, 2, border_radius=8)
    info_pad_x_ia, info_pad_y_ia = 15, 10; line_h_ia = INFO_FONT.get_linesize() + 4
    current_info_y_draw = info_area_rect.top + info_pad_y_ia
    title_surf = TITLE_FONT.render("Solver Stats", True, BLACK)
    screen.blit(title_surf, (info_area_rect.centerx - title_surf.get_width()//2, current_info_y_draw)); current_info_y_draw += title_surf.get_height() + 8
    if solve_times_dgui:
        for algo_name_st, time_val_st in sorted(solve_times_dgui.items(), key=lambda item: item[0]):
            if current_info_y_draw + line_h_ia > info_area_rect.bottom - info_pad_y_ia: screen.blit(INFO_FONT.render("...", True, BLACK), (info_area_rect.left + info_pad_x_ia, current_info_y_draw)); break
            steps_str = ""
            if last_solved_run_info_dgui.get(f"{algo_name_st}_actions") is not None: steps_str = f" ({last_solved_run_info_dgui[f'{algo_name_st}_actions']} actions)"
            elif last_solved_run_info_dgui.get(f"{algo_name_st}_steps") is not None: steps_str = f" ({last_solved_run_info_dgui[f'{algo_name_st}_steps']} steps)"
            goal_ind = ""
            reached_goal_val = last_solved_run_info_dgui.get(f"{algo_name_st}_reached_goal")
            if reached_goal_val is False: goal_ind = " (Not Goal)"
            elif reached_goal_val == "Timeout": goal_ind = " (Timeout)"
            elif algo_name_st in ['Sensorless', 'Unknown'] and reached_goal_val is None: goal_ind = " (Plan Fail)"
            elif reached_goal_val is None and algo_name_st not in ['AC3', 'Backtracking', 'KiemThu']: goal_ind = " (?)"
            text = f"{algo_name_st}: {time_val_st:.3f}s{steps_str}{goal_ind}"
            if INFO_FONT.size(text)[0] > info_area_rect.width - 2*info_pad_x_ia: text = f"{algo_name_st.split('(')[0][:10]}..: {time_val_st:.3f}s{goal_ind}"
            screen.blit(INFO_FONT.render(text, True, BLACK), (info_area_rect.left + info_pad_x_ia, current_info_y_draw)); current_info_y_draw += line_h_ia
    else: screen.blit(INFO_FONT.render("(No results yet)", True, GRAY), (info_area_rect.left + info_pad_x_ia, current_info_y_draw))
    return {
        'solve_button': solve_btn_rect, 'reset_solution_button': reset_sol_btn_rect,
        'reset_all_button': reset_all_btn_rect, 'reset_initial_button': reset_disp_btn_rect,
        'menu': draw_menu(show_menu_dgui, pygame.mouse.get_pos(), current_algo_name_dgui),
        'initial_grid_area': pygame.Rect(start_x_top_grids, top_row_y_grids, GRID_DISPLAY_WIDTH, GRID_DISPLAY_WIDTH)
    }
    
    
# --- Main Game Loop ---
def main():
    global scroll_y, initial_state
    current_state_for_animation = copy.deepcopy(initial_state)
    solution_path_anim = None; action_plan_anim = None; current_step_in_anim = 0
    is_solving_flag = False; is_auto_animating_flag = False; last_anim_step_time = 0
    show_algo_menu = False; current_selected_algorithm = 'A*' 
    solver_algorithm_to_run = None; solver_name_for_stats = None      
    all_solve_times = {}; last_run_solver_info = {}
    game_clock = pygame.time.Clock(); ui_elements_rects = {}
    running_main_loop = True
    current_sensorless_belief_size_for_display = None
    selected_cell_for_input_coords = None

    while running_main_loop:
        mouse_pos_main = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running_main_loop = False; break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_something_handled = False
                initial_grid_area_rect = ui_elements_rects.get('initial_grid_area')
                if initial_grid_area_rect and initial_grid_area_rect.collidepoint(mouse_pos_main):
                    grid_offset_x = initial_grid_area_rect.left; grid_offset_y = initial_grid_area_rect.top
                    clicked_col = (mouse_pos_main[0] - grid_offset_x) // CELL_SIZE
                    clicked_row = (mouse_pos_main[1] - grid_offset_y) // CELL_SIZE
                    selected_cell_for_input_coords = (clicked_row, clicked_col) if 0 <= clicked_row < GRID_SIZE and 0 <= clicked_col < GRID_SIZE else None
                    clicked_something_handled = True
                elif selected_cell_for_input_coords and (not show_algo_menu or not ui_elements_rects.get('menu', {}).get('menu_area', pygame.Rect(0,0,0,0)).collidepoint(mouse_pos_main)):
                     selected_cell_for_input_coords = None 

                menu_data = ui_elements_rects.get('menu', {})
                if show_algo_menu and not clicked_something_handled and menu_data.get('menu_area', pygame.Rect(0,0,0,0)).collidepoint(mouse_pos_main):
                    if menu_data.get('close_button', pygame.Rect(0,0,0,0)).collidepoint(mouse_pos_main):
                        show_algo_menu = False; clicked_something_handled = True
                    if not clicked_something_handled:
                        for algo_id, btn_rect_screen in menu_data.get('buttons', {}).items():
                            if btn_rect_screen.collidepoint(mouse_pos_main):
                                if current_selected_algorithm != algo_id:
                                    current_selected_algorithm = algo_id
                                    solution_path_anim = None; action_plan_anim = None; current_step_in_anim = 0
                                    is_auto_animating_flag = False; is_solving_flag = False
                                    current_sensorless_belief_size_for_display = None
                                show_algo_menu = False; clicked_something_handled = True; break
                    if not clicked_something_handled: clicked_something_handled = True 
                elif not show_algo_menu and not clicked_something_handled and menu_data.get('open_button', pygame.Rect(0,0,0,0)).collidepoint(mouse_pos_main):
                    show_algo_menu = True; scroll_y = 0; clicked_something_handled = True

                if not clicked_something_handled: 
                    btns = {k: ui_elements_rects.get(k) for k in ['solve_button', 'reset_solution_button', 'reset_all_button', 'reset_initial_button']}
                    if btns['solve_button'] and btns['solve_button'].collidepoint(mouse_pos_main) and not is_auto_animating_flag and not is_solving_flag:
                        should_start_solving = False
                        solver_algorithm_to_run = current_selected_algorithm 
                        solver_name_for_stats = current_selected_algorithm                         
                        is_csp_filler_algo = solver_algorithm_to_run in ['AC3', 'Backtracking', 'KiemThu']
                        is_sensorless_type = solver_algorithm_to_run in ['Sensorless', 'Unknown']
                        
                        if is_csp_filler_algo: 
                            should_start_solving = True
                            current_state_for_animation = [[None]*GRID_SIZE for _ in range(GRID_SIZE)] 
                        elif is_sensorless_type:
                            if is_valid_template_for_csp(initial_state): should_start_solving = True
                            else: show_popup("Initial state is not a valid template for Unknown/Sensorless.", "Invalid Template")
                        else: 
                            if not is_valid_state_for_solve(initial_state): show_popup(f"Initial state must be complete and valid for {solver_algorithm_to_run}.", "Invalid State")
                            elif not is_solvable(initial_state): show_popup(f"Initial state is not solvable for path-finding.", "Unsolvable State")
                            else: should_start_solving = True
                        
                        if should_start_solving:
                            is_solving_flag = True; solution_path_anim = None; action_plan_anim = None
                            current_step_in_anim = 0; is_auto_animating_flag = False
                            current_sensorless_belief_size_for_display = None
                    elif btns['reset_solution_button'] and btns['reset_solution_button'].collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state); solution_path_anim = None; action_plan_anim = None
                        current_step_in_anim = 0; is_solving_flag = False; is_auto_animating_flag = False
                        current_sensorless_belief_size_for_display = None
                    elif btns['reset_all_button'] and btns['reset_all_button'].collidepoint(mouse_pos_main):
                        initial_state = copy.deepcopy(initial_state_fixed_global); current_state_for_animation = copy.deepcopy(initial_state)
                        solution_path_anim = None; action_plan_anim = None; current_step_in_anim = 0
                        is_solving_flag = False; is_auto_animating_flag = False; all_solve_times.clear(); last_run_solver_info.clear()
                        current_sensorless_belief_size_for_display = None; selected_cell_for_input_coords = None
                    elif btns['reset_initial_button'] and btns['reset_initial_button'].collidepoint(mouse_pos_main):
                        current_state_for_animation = copy.deepcopy(initial_state); solution_path_anim = None; action_plan_anim = None
                        current_step_in_anim = 0; is_solving_flag = False; is_auto_animating_flag = False
                        current_sensorless_belief_size_for_display = None
            elif event.type == pygame.MOUSEWHEEL and show_algo_menu and menu_data.get('menu_area', pygame.Rect(0,0,0,0)).collidepoint(mouse_pos_main) and total_menu_height > HEIGHT:
                scroll_y = max(0, min(scroll_y - event.y * 35, total_menu_height - HEIGHT if total_menu_height > HEIGHT else 0)) 
            elif event.type == pygame.KEYDOWN and selected_cell_for_input_coords and not is_solving_flag and not is_auto_animating_flag:
                r, c = selected_cell_for_input_coords
                if pygame.K_0 <= event.key <= pygame.K_8:
                    num = event.key - pygame.K_0
                    can_place = True 
                    if num != 0 :
                        for row_idx, row_val in enumerate(initial_state):
                            for col_idx, cell_val in enumerate(row_val):
                                if cell_val == num and (row_idx != r or col_idx != c):
                                    can_place = False; break
                            if not can_place: break
                    if can_place: initial_state[r][c] = num
                elif event.key in [pygame.K_DELETE, pygame.K_BACKSPACE, pygame.K_SPACE]: initial_state[r][c] = None
                elif event.key in [pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_TAB]: selected_cell_for_input_coords = None
                elif event.key == pygame.K_UP: selected_cell_for_input_coords = ((r - 1 + GRID_SIZE) % GRID_SIZE, c)
                elif event.key == pygame.K_DOWN: selected_cell_for_input_coords = ((r + 1) % GRID_SIZE, c)
                elif event.key == pygame.K_LEFT: selected_cell_for_input_coords = (r, (c - 1 + GRID_SIZE) % GRID_SIZE)
                elif event.key == pygame.K_RIGHT: selected_cell_for_input_coords = (r, (c + 1) % GRID_SIZE)
                current_state_for_animation = copy.deepcopy(initial_state) 
                solution_path_anim = None; action_plan_anim = None; current_step_in_anim = 0
        if not running_main_loop: break

        if is_solving_flag: 
            is_solving_flag = False; solve_start_t = time.time()
            found_path_algo = None; found_action_plan_algo = None; error_during_solve = False; error_message_solve = ""
            solve_result_flag = False 
            
            algo_func_map = { 
                'BFS': bfs, 'DFS': dfs, 'IDS': ids, 'UCS': ucs, 'A*': astar, 'Greedy': greedy, 'IDA*': ida_star,
                'Hill Climbing': simple_hill_climbing, 'Steepest Hill': steepest_hill_climbing, 'Stochastic Hill': random_hill_climbing,
                'SA': simulated_annealing, 'GA': genetic_algorithm_solve, 'Beam Search': beam_search, 'AND-OR': and_or_search,
                'Sensorless': sensorless_search, 'Unknown': sensorless_search, 'Q-Learning': q_learning_train_and_solve,
                # The first argument (None) for CSP solvers is a placeholder for start_template_board as they start blank.
                'Backtracking': lambda _,tgt,lim: csp_backtracking_solver([[None]*GRID_SIZE for _ in range(GRID_SIZE)], tgt, lim), 
                'KiemThu': lambda _,tgt,lim: kiem_thu_solver([[None]*GRID_SIZE for _ in range(GRID_SIZE)], tgt, lim), 
                'AC3': lambda _, tgt, lim: ac3_then_backtracking_solver(None, tgt, lim),

            }
            try:
                state_to_solve_from = copy.deepcopy(initial_state) 
                if solver_algorithm_to_run not in ['Backtracking', 'KiemThu', 'AC3']:
                    current_state_for_animation = copy.deepcopy(state_to_solve_from)

                selected_algo_function = algo_func_map.get(solver_algorithm_to_run)
                if selected_algo_function:
                    if solver_algorithm_to_run in ['Backtracking', 'KiemThu', 'AC3']:
                        found_path_algo, solve_result_flag = selected_algo_function(None, goal_state_fixed_global, 30) 
                    elif solver_algorithm_to_run in ['Sensorless', 'Unknown']:
                        found_action_plan_algo, belief_size = sensorless_search(state_to_solve_from, 60)
                        current_sensorless_belief_size_for_display = belief_size
                        if found_action_plan_algo:
                            action_plan_anim = found_action_plan_algo
                            sample_states = generate_belief_states(state_to_solve_from)
                            start_vis = sample_states[0] if sample_states else state_to_solve_from
                            if is_valid_state_for_solve(start_vis): 
                                found_path_algo = execute_plan(start_vis, action_plan_anim)
                            else: 
                                found_path_algo = [start_vis] + [None] * len(action_plan_anim) 
                            current_state_for_animation = copy.deepcopy(start_vis)
                            solve_result_flag = True
                        else: solve_result_flag = False
                    else: 
                        algo_args_list = [state_to_solve_from]
                        func_varnames = selected_algo_function.__code__.co_varnames[:selected_algo_function.__code__.co_argcount]
                        default_time_limit_path = 30
                        if solver_algorithm_to_run in ['IDA*', 'IDS', 'Q-Learning', 'GA']: default_time_limit_path = 60
                        if 'time_limit' in func_varnames: algo_args_list.append(default_time_limit_path)
                        elif solver_algorithm_to_run == 'DFS' and 'max_depth' in func_varnames: algo_args_list.append(30)
                        elif solver_algorithm_to_run == 'IDS' and 'max_depth_limit' in func_varnames: algo_args_list.append(30)
                        elif solver_algorithm_to_run == 'Stochastic Hill' and 'max_iter_no_improve' in func_varnames: algo_args_list.append(500)
                        elif solver_algorithm_to_run == 'Beam Search' and 'beam_width' in func_varnames: algo_args_list.append(5)

                        found_path_algo = selected_algo_function(*algo_args_list)
                        action_plan_anim = None
                        if found_path_algo and len(found_path_algo) > 0 and found_path_algo[-1] and is_valid_state_for_solve(found_path_algo[-1]):
                            solve_result_flag = is_goal(found_path_algo[-1])
                        else: solve_result_flag = False 
                else: error_during_solve = True; error_message_solve = f"Algo func for '{solver_algorithm_to_run}' not found."
            except Exception as e: error_during_solve = True; error_message_solve = f"Runtime Error: {traceback.format_exc()}"
            
            solve_duration_t = time.time() - solve_start_t
            if solver_name_for_stats: all_solve_times[solver_name_for_stats] = solve_duration_t
            
            if error_during_solve:
                show_popup(error_message_solve, "Solver Error")
                for key_suffix in ["_reached_goal", "_steps", "_actions"]: last_run_solver_info.pop(f"{solver_name_for_stats}{key_suffix}", None)
            else:
                solution_path_anim = found_path_algo
                if solution_path_anim or action_plan_anim :
                    is_auto_animating_flag = bool(solution_path_anim and len(solution_path_anim) > 1)
                    last_anim_step_time = time.time()
                    steps_or_actions_count = 0; popup_detail = ""
                    if action_plan_anim:
                        steps_or_actions_count = len(action_plan_anim)
                        last_run_solver_info[f"{solver_name_for_stats}_actions"] = steps_or_actions_count
                        popup_detail = f"{steps_or_actions_count} actions."
                    elif solution_path_anim:
                        steps_or_actions_count = max(0, len(solution_path_anim) -1)
                        last_run_solver_info[f"{solver_name_for_stats}_steps"] = steps_or_actions_count
                        popup_detail = f"{steps_or_actions_count} {'config steps' if solver_algorithm_to_run in ['AC3','Backtracking','KiemThu'] else 'path steps'}."
                    last_run_solver_info[f"{solver_name_for_stats}_reached_goal"] = solve_result_flag
                    popup_title = "Solve Complete" if solve_result_flag is True else ("Solver Timeout" if solve_result_flag=="Timeout" else "Search Finished")
                    popup_msg = f"{solver_name_for_stats} {'succeeded!' if solve_result_flag is True else ('timed out.' if solve_result_flag=='Timeout' else 'finished (Goal Not Reached).')}"
                    show_popup(f"{popup_msg}\n{popup_detail}\nTime: {solve_duration_t:.3f}s", popup_title)
                else: 
                    last_run_solver_info[f"{solver_name_for_stats}_reached_goal"] = False
                    time_limit_check = 30
                    if solver_algorithm_to_run in ['IDA*', 'IDS', 'Sensorless', 'Unknown', 'Q-Learning', 'GA', 'AC3', 'Backtracking', 'KiemThu']: time_limit_check = 60
                    if solver_algorithm_to_run in ['AC3', 'Backtracking', 'KiemThu']: time_limit_check=30 
                    msg = f"No solution by {solver_name_for_stats}."
                    if solve_duration_t >= time_limit_check * 0.98: msg = f"{solver_name_for_stats} likely timed out."
                    show_popup(msg, "No Solution / Timeout")
            solver_algorithm_to_run = None 

        if is_auto_animating_flag and solution_path_anim and current_step_in_anim < len(solution_path_anim) - 1:
            anim_delay = 0.3
            if len(solution_path_anim) > 30: anim_delay = 0.15
            if len(solution_path_anim) > 60: anim_delay = 0.08
            if current_selected_algorithm in ['AC3', 'Backtracking', 'KiemThu'] and len(solution_path_anim) > 20: anim_delay = 0.05
            if time.time() - last_anim_step_time >= anim_delay:
                current_step_in_anim += 1
                next_state_anim = solution_path_anim[current_step_in_anim]
                if next_state_anim and isinstance(next_state_anim, list) and len(next_state_anim) == GRID_SIZE and all(isinstance(r,list) and len(r)==GRID_SIZE for r in next_state_anim):
                    current_state_for_animation = copy.deepcopy(next_state_anim)
                else: is_auto_animating_flag = False 
                last_anim_step_time = time.time()
        elif is_auto_animating_flag and solution_path_anim and current_step_in_anim >= len(solution_path_anim) -1 :
             is_auto_animating_flag = False 

        belief_display_size = current_sensorless_belief_size_for_display
        active_algo_ui_name = solver_name_for_stats if solver_name_for_stats else current_selected_algorithm
        if active_algo_ui_name in ['Sensorless', 'Unknown'] and belief_display_size is None and not is_auto_animating_flag:
             try:
                 belief_display_size = len(generate_belief_states(initial_state)) if isinstance(initial_state, list) else 0
             except: belief_display_size = 0 
        
        ui_elements_rects = draw_grid_and_ui(current_state_for_animation, show_algo_menu, current_selected_algorithm, active_algo_ui_name, all_solve_times, last_run_solver_info, belief_display_size, selected_cell_for_input_coords)
        pygame.display.flip()
        game_clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    try: main()
    except Exception as e: 
        print(f"\n--- UNHANDLED ERROR: {e} ---")
        traceback.print_exc()
    finally: 
        pygame.quit()
        sys.exit(1) 

