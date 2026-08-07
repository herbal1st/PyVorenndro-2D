"""Global configuration settings for PyVorenndro 2D engine."""

from typing import Tuple

# ------ Active Profile Selection ------
ACTIVE_PROFILE_ID: str = "herbal1st"  # profile

# ------ Display & Window Layout ------
SCREEN_WIDTH: int = 1280  # pixels
SCREEN_HEIGHT: int = 720  # pixels
FPS: int = 60  # hertz
VIEWPORT_GRID_ROWS: int = 4  # count
VIEWPORT_GRID_COLS: int = 4  # count
WINNER_CELEBRATION_FRAMES: int = 60  # frames

# ------ World & Map Parameters ------
MAP_TYPE: str = "BRANCHING WALLS"  # style
MAP_WIDTH: int = 12  # tiles
MAP_HEIGHT: int = 9  # tiles
TILE_SIZE: int = 24  # pixels
WALL_DENSITY: float = 0.45  # ratio
MAX_MAP_GEN_ATTEMPTS: int = 100  # attempts
MIN_PATH_DIFFICULTY_RATIO: float = 0.85  # ratio

# ------ Genetic Algorithm & Training ------
LEARNING_GENERATIONS: int = 100  # generations
POPULATION_SIZE: int = 16  # candidates
MAPS_PER_CANDIDATE: int = 8  # mazes
SIMULATION_RUNS: int = POPULATION_SIZE * MAPS_PER_CANDIDATE  # simulations
MAX_SIMULATION_STEPS: int = 1000  # ticks
MAP_REGIME_GENERATIONS: int = 1  # generations
REGIME_MIN_GENERATIONS: int = 8  # generations
REGIME_SOLVE_TARGET: int = 1  # solvers
REGIME_TRANSITION_MUTATION_BOOST: float = 1.0  # ratio
MAP_DIFFICULTY_MIN: float = 0.55  # ratio
MAP_DIFFICULTY_MAX: float = 0.85  # ratio
MUTATION_RATE: float = 0.25  # ratio
MUTATION_SCALE: float = 0.08  # scale
MUTATION_SCALE_MIN: float = 0.002  # scale
MUTATION_SCALE_MAX: float = 0.25  # scale
MUTATION_ADAPT_WINDOW: int = 8  # generations
STAGNATION_BUMP_GENERATIONS: int = 8  # generations
STAGNATION_BUMP_FACTOR: float = 4.0  # multiplier
ELITISM_RATIO: float = 0.20  # ratio
DEFAULT_PLAYBACK_SPEED: int = 1  # multiplier
DIST_TO_TIME_BONUS_RATIO: float = 1.0  # ratio
LOST_HP_SCORE_IMPACT_RATIO: float = 0.5  # ratio
RECORDER_MAX_GENERATIONS: int = 1000  # generations

# ------ Parallel Simulation ------
SIMULATION_WORKERS: int = 0  # workers
TRAINING_USE_LOW_PRIORITY: bool = True  # toggle

# ------ Hardware & Auto-Tuning ------
TRAINING_BACKEND: str = "auto"  # backend
AUTO_TUNE: bool = False  # toggle
TARGET_GEN_TIME: float = 0.75  # seconds
AUTO_TUNE_MIN_RUNS: int = 64  # simulations
AUTO_TUNE_MAX_RUNS: int = 32768  # simulations
AUTO_TUNE_PROBE_RUNS: int = 256  # simulations
AUTO_TUNE_PROBE_RUNS_CPU: int = 32  # simulations

# ------ Curriculum (Adaptive Difficulty) ------
CURRICULUM_ENABLED: bool = False  # toggle
CURRICULUM_DIFFICULTY_RATIO: float = 0.5  # ratio
CURRICULUM_START_BFS: int = 8  # tiles
CURRICULUM_BFS_STEP: int = 2  # tiles
CURRICULUM_BFS_WINDOW: int = 4  # tiles
CURRICULUM_MAX_BFS: int = 20  # tiles
CURRICULUM_MAP_ATTEMPTS: int = 10  # attempts
CURRICULUM_FAILURES_BEFORE_EASE: int = 2  # regimes

# ------ GUI Element Layout Rectangles ------
LAYOUT_GRID_RECT: Tuple[int, int, int, int] = (20, 20, 800, 600)  # rect
LAYOUT_PANEL_RECT: Tuple[int, int, int, int] = (840, 20, 420, 140)  # rect
LAYOUT_GRAPH_RECT: Tuple[int, int, int, int] = (840, 180, 420, 440)  # rect
LAYOUT_SCRUBBER_RECT: Tuple[int, int, int, int] = (
    20, 650, 1240, 60
)  # rect

# ------ HUD Typography & Element Sizing ------
HUD_PANEL_TITLE_FONT_SIZE: int = 16  # pt
HUD_PANEL_BODY_FONT_SIZE: int = 15  # pt
HUD_SCRUBBER_BTN_HEIGHT: int = 36  # pixels
HUD_SCRUBBER_BTN_FONT_SIZE: int = 13  # pt
HUD_SCRUBBER_BAR_HEIGHT: int = 16  # pixels
HUD_SCRUBBER_MARKER_RADIUS: int = 8  # pixels
HUD_GRAPH_NODE_RADIUS: int = 8  # pixels
HUD_GRAPH_FONT_SIZE: int = 12  # pt
HUD_GRAPH_LABEL_PADDING: int = 6  # pixels

# ------ Visual Theme & Colors ------
COLOR_BG: Tuple[int, int, int] = (15, 15, 20)  # rgb
COLOR_WALL: Tuple[int, int, int] = (45, 45, 55)  # rgb
COLOR_WALL_BORDER: Tuple[int, int, int] = (80, 80, 95)  # rgb
COLOR_FLOOR: Tuple[int, int, int] = (25, 25, 32)  # rgb
COLOR_FLOOR_BORDER: Tuple[int, int, int] = (35, 35, 45)  # rgb
COLOR_START: Tuple[int, int, int] = (40, 160, 220)  # rgb
COLOR_EXIT: Tuple[int, int, int] = (50, 200, 100)  # rgb
COLOR_PLAYER: Tuple[int, int, int] = (240, 180, 50)  # rgb
COLOR_PLAYER_HIGHLIGHT: Tuple[int, int, int] = (255, 220, 80)  # rgb
COLOR_PLAYER_TEXT: Tuple[int, int, int] = (10, 10, 15)  # rgb
COLOR_VISION_ARC: Tuple[int, int, int, int] = (255, 220, 80, 30)  # rgba
PLAYER_FACE_TEXT_SCALE: float = 0.80  # scale
PLAYER_HEADING_LINE_LENGTH: float = 1.25  # scale
PLAYER_HEADING_LINE_WIDTH: int = 1  # pixels
COLOR_PLAYER_HEADING_LINE: Tuple[int, int, int, int] = (
    255, 255, 255, 120
)  # rgba

# ------ Health Bar & Status Frame Colors ------
COLOR_HEALTH_FULL: Tuple[int, int, int] = (50, 200, 100)  # rgb
COLOR_HEALTH_MID: Tuple[int, int, int] = (255, 140, 0)  # rgb
COLOR_HEALTH_LOW: Tuple[int, int, int] = (220, 50, 50)  # rgb
COLOR_FRAME_SOLVED: Tuple[int, int, int] = (20, 140, 50)  # rgb
COLOR_FRAME_DEAD: Tuple[int, int, int] = (160, 20, 20)  # rgb

# ------ HUD & Activation Graph Colors ------
COLOR_NODE_INACTIVE: Tuple[int, int, int] = (80, 15, 15)  # rgb
COLOR_NODE_ACTIVE: Tuple[int, int, int] = (255, 140, 0)  # rgb
COLOR_TIMELINE_BAR: Tuple[int, int, int] = (60, 60, 75)  # rgb
COLOR_MARKER: Tuple[int, int, int] = (255, 140, 0)  # rgb
COLOR_BUTTON: Tuple[int, int, int] = (45, 45, 55)  # rgb
COLOR_BUTTON_ACTIVE: Tuple[int, int, int] = (80, 80, 100)  # rgb

# ------ Candidate Facial Expressions ------
FACE_WALK: str = "o_o"  # expression
FACE_WALL: str = ">*<"  # expression
FACE_DEAD: str = "T_T"  # expression
FACE_EXIT: str = "^*^"  # expression
