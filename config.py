"""
Global configuration settings for PyVorenndro 2D engine.
Optimized for random, dynamic map generation with pure visual raycasting and a global target toggle.
Tuned for maximum CPU execution speed with Numba JIT compilation.
"""

from typing import Tuple

# ------ Display & Grid Layout ------
SCREEN_WIDTH: int = 1280   # pixels
SCREEN_HEIGHT: int = 720   # pixels
FPS: int = 60   # hertz
GRID_ROWS: int = 4   # count
GRID_COLS: int = 4   # count

# ------ World & Map Parameters ------
MAP_TYPE: str = "BRANCHING WALLS"   # style
MAP_WIDTH: int = 12   # tiles
MAP_HEIGHT: int = 9   # tiles
TILE_SIZE: int = 24   # pixels
WALL_DENSITY: float = 0.35   # ratio
MAX_MAP_GEN_ATTEMPTS: int = 150   # attempts
MIN_PATH_DIFFICULTY_RATIO: float = 0.40   # ratio

# ------ Sensory & Vision Arc (NO COMPASS, NUMBA-OPTIMIZED) ------
VISION_RAYS: int = 19   # 9 rays across 120° field of view
VISION_ARC_ANGLE: float = 120.0   # degrees
VISION_MAX_DIST: float = 6.0   # tiles
INCLUDE_COMPASS: bool = False   # Pure visual raycasting navigation

# ------ Kinematics & Health Tuning (REDUCED CULLING AGGRESSIVENESS) ------
KINEMATICS_PROFILE: str = "CAR"   # style
MOVE_SPEED: float = 0.15   # tiles
TURN_SPEED: float = 1800.0   # dpsec
PLAYER_RADIUS_RATIO: float = 0.45   # ratio
PLAYER_CAMERA_ZOOM: float = 0.5   # scale
HEALTH_COLL_DMG_PER_FRAME: float = 0.008   # Smoother penalty to give agents time to learn steering
HEALTH_IDLE_DMG_PER_FRAME: float = 0.002   # Balanced idle termination rate
HEALTH_IDLE_MOVE_THRESHOLD: float = 0.005   # ratio
HEALTH_RECOVERY_RATIO: float = 0.05   # ratio

# ------ Anti-Spin & Stagnation Guards ------
SPIN_PENALTY_ENABLED: bool = True   # toggle
SPIN_ANGLE_THRESHOLD_DEG: float = 360.0   # degrees
SPIN_RESET_ANGLE_DEG: float = 5.0   # degrees
SPIN_DMG_PER_FRAME: float = 0.008   # Balanced culling on spinning agents
STAGNATION_ENABLED: bool = True   # toggle
STAGNATION_TIMEOUT_RATIO: float = 0.30   # Times out stuck agents after ~240 ticks
STAGNATION_DMG_PER_FRAME: float = 0.010   # Balanced culling in dead ends

# ------ Neural Architecture ------
NEURAL_HIDDEN_LAYERS: int = 1   # layers
NEURAL_NEURONS: int = 24   # Compact hidden layer for ultra-fast vector dot products
NEURAL_INPUT_SIZE: int = VISION_RAYS + 1  # 9 ray distances + 1 global target toggle indicator

# ------ Genetic Algorithm & Training (BALANCED MUTATION & LARGER POOL) ------
LEARNING_GENERATIONS: int = 350   # generations
POPULATION_SIZE: int = 150   # Increased gene pool for better diversity and local minima escape
MAX_SIMULATION_STEPS: int = 800   # Step cap
MAP_REGIME_GENERATIONS: int = 25   # Keep the same map for 25 generations
REGIME_MIN_GENERATIONS: int = 15   # Force at least 15 generations on a map
REGIME_SOLVE_TARGET: int = 5       # Allow early rotation only if 5 agents solve it
REGIME_TRANSITION_MUTATION_BOOST: float = 1.0   # scale
MAP_DIFFICULTY_MIN: float = 0.5
MAP_DIFFICULTY_MAX: float = 0.5  # Lowered from 0.85 to avoid brutal layouts
MUTATION_RATE: float = 0.08   # Slightly reduced to protect high-performing gene structures
MUTATION_SCALE: float = 0.05   # Reduced to prevent catastrophic regression in top generations
ELITISM_RATIO: float = 0.30   # Preserves top generalist visual controllers
DEFAULT_PLAYBACK_SPEED: int = 1   # multiplier
SCRUBBER_ARROW_JUMP_FRAMES: int = 25   # frames
SCRUBBER_PAGE_JUMP_GENS: int = 1   # generations
DIST_TO_TIME_BONUS_RATIO: float = 1.2   # ratio
LOST_HP_SCORE_IMPACT_RATIO: float = 0.6   # ratio

# ------ Parallel Simulation ------
SIMULATION_WORKERS: int = 4   # Parallel worker processes

# ------ Curriculum (Disabled per user instructions) ------
CURRICULUM_ENABLED: bool = False
CURRICULUM_DIFFICULTY_RATIO: float = 0.5
CURRICULUM_START_BFS: int = 4
CURRICULUM_BFS_STEP: int = 1
CURRICULUM_BFS_WINDOW: int = 4
CURRICULUM_MAX_BFS: int = 22
CURRICULUM_MAP_ATTEMPTS: int = 15
CURRICULUM_FAILURES_BEFORE_EASE: int = 5

# ------ GUI Element Layout Rectangles ------
LAYOUT_GRID_RECT: Tuple[int, int, int, int] = (20, 20, 800, 600)
LAYOUT_PANEL_RECT: Tuple[int, int, int, int] = (840, 20, 420, 140)
LAYOUT_GRAPH_RECT: Tuple[int, int, int, int] = (840, 180, 420, 440)
LAYOUT_SCRUBBER_RECT: Tuple[int, int, int, int] = (20, 650, 1240, 60)

# ------ HUD Typography & Sizing ------
HUD_PANEL_TITLE_FONT_SIZE: int = 16
HUD_PANEL_BODY_FONT_SIZE: int = 15
HUD_SCRUBBER_BTN_HEIGHT: int = 36
HUD_SCRUBBER_BTN_FONT_SIZE: int = 13
HUD_SCRUBBER_BAR_HEIGHT: int = 16
HUD_SCRUBBER_MARKER_RADIUS: int = 8
HUD_GRAPH_NODE_RADIUS: int = 8
HUD_GRAPH_FONT_SIZE: int = 12
HUD_GRAPH_LABEL_PADDING: int = 6

# ------ Visual Theme & Colors ------
COLOR_BG: Tuple[int, int, int] = (15, 15, 20)
COLOR_WALL: Tuple[int, int, int] = (45, 45, 55)
COLOR_WALL_BORDER: Tuple[int, int, int] = (80, 80, 95)
COLOR_FLOOR: Tuple[int, int, int] = (25, 25, 32)
COLOR_FLOOR_BORDER: Tuple[int, int, int] = (35, 35, 45)
COLOR_START: Tuple[int, int, int] = (40, 160, 220)
COLOR_EXIT: Tuple[int, int, int] = (50, 200, 100)
COLOR_PLAYER: Tuple[int, int, int] = (240, 180, 50)
COLOR_PLAYER_HIGHLIGHT: Tuple[int, int, int] = (255, 220, 80)
COLOR_PLAYER_TEXT: Tuple[int, int, int] = (10, 10, 15)
COLOR_VISION_ARC: Tuple[int, int, int, int] = (255, 220, 80, 30)
PLAYER_FACE_TEXT_SCALE: float = 0.80
PLAYER_HEADING_LINE_LENGTH: float = 1.25
PLAYER_HEADING_LINE_WIDTH: int = 1
COLOR_PLAYER_HEADING_LINE: Tuple[int, int, int, int] = (255, 255, 255, 120)

# ------ Health Bar & Status Frame Colors ------
COLOR_HEALTH_FULL: Tuple[int, int, int] = (50, 200, 100)
COLOR_HEALTH_MID: Tuple[int, int, int] = (255, 140, 0)
COLOR_HEALTH_LOW: Tuple[int, int, int] = (220, 50, 50)
COLOR_FRAME_SOLVED: Tuple[int, int, int] = (20, 140, 50)
COLOR_FRAME_DEAD: Tuple[int, int, int] = (160, 20, 20)

# ------ HUD & Activation Graph Colors ------
COLOR_NODE_INACTIVE: Tuple[int, int, int] = (80, 15, 15)
COLOR_NODE_ACTIVE: Tuple[int, int, int] = (255, 140, 0)
COLOR_TIMELINE_BAR: Tuple[int, int, int] = (60, 60, 75)
COLOR_MARKER: Tuple[int, int, int] = (255, 140, 0)
COLOR_BUTTON: Tuple[int, int, int] = (45, 45, 55)
COLOR_BUTTON_ACTIVE: Tuple[int, int, int] = (80, 80, 100)

# ------ Candidate Facial Expressions ------
FACE_WALK: str = "o_o"
FACE_WALL: str = ">*<"
FACE_DEAD: str = "T_T"
FACE_EXIT: str = "^*^"