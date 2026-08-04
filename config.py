"""
Global configuration settings for PyVorenndro 2D engine.
"""

from typing import Tuple

# ------ Display & Window Layout ------

SCREEN_WIDTH: int = 1280  # pixels
SCREEN_HEIGHT: int = 720  # pixels
FPS: int = 60  # hertz

# ------ World & Map Parameters ------

MAP_TYPE: str = "BRANCHING WALLS"  # style
MAP_WIDTH: int = 12  # tiles
MAP_HEIGHT: int = 9  # tiles
TILE_SIZE: int = 24  # pixels
WALL_DENSITY: float = 0.45  # ratio
MAX_MAP_GEN_ATTEMPTS: int = 100  # attempts
MIN_PATH_DIFFICULTY_RATIO: float = 0.85  # ratio

# ------ Sensory & Vision Arc ------

VISION_RAYS: int = 9  # rays
VISION_ARC_ANGLE: float = 120.0  # degrees
VISION_MAX_DIST: float = 6.0  # tiles

# ------ Kinematics ------

KINEMATICS_PROFILE: str = "TANK"  # style
MOVE_SPEED: float = 0.1  # tiles
TURN_SPEED: float = 1080.0  # dpsec
PLAYER_RADIUS_RATIO: float = 0.5  # ratio
PLAYER_CAMERA_ZOOM: float = 0.5  # scale
HEALTH_COLL_DMG_PER_FRAME: float = 0.005  # damage
HEALTH_IDLE_DMG_PER_FRAME: float = 0.005  # damage
HEALTH_RECOVERY_RATIO: float = 0.5  # ratio

# ------ Neural Architecture ------

NEURAL_HIDDEN_LAYERS: int = 2  # layers
NEURAL_NEURONS: int = 32  # neurons

# Stereo compass sensor: two channels encoding the exit's bearing relative
# to heading. Enabling it lets the agent steer toward the exit from its own
# position; disabling it forces agents to learn maze structure from local
# wall rays alone (e.g. wall-following) instead of steering toward a beacon.
INCLUDE_COMPASS: bool = False

# Whether the agent receives the current tile's BFS distance to the exit
# (normalized by the map's maximum path span). A goal-gradient sensor that
# makes maze navigation substantially more learnable for the small MLP.
INCLUDE_BFS_SENSOR: bool = False

# ------ Genetic Algorithm & Training ------

# Single-agent evolution: one champion genome is re-evaluated every generation
# across SIMULATION_RUNS parallel, independently-seeded simulation runs. The
# aggregated fitness across those runs is corroborated in the single agent.

# The champion and its POPULATION_SIZE-1 mutated offspring share the SAME map
# set each generation (fair, paired selection), so the total simulation budget
# per generation is SIMULATION_RUNS (POPULATION_SIZE agents x shared maps).
LEARNING_GENERATIONS: int = 100  # generations
SIMULATION_RUNS: int = 100  # total parallel simulations per generation
POPULATION_SIZE: int = 2  # candidate agents evaluated per generation (1 champion + offspring)
MAX_SIMULATION_STEPS: int = 1000  # ticks
MAP_REGIME_GENERATIONS: int = 1  # max generations per map before regenerating
REGIME_MIN_GENERATIONS: int = 8  # minimum gens before mastery-based switch
REGIME_SOLVE_TARGET: int = 1  # solvers required to switch map early
REGIME_TRANSITION_MUTATION_BOOST: float = 1.0  # mutation scale on new maps
MAP_DIFFICULTY_MIN: float = 0.55  # sampled training difficulty range (min)
MAP_DIFFICULTY_MAX: float = 0.85  # sampled training difficulty range (max)
MUTATION_RATE: float = 0.25  # ratio
MUTATION_SCALE: float = 0.12  # initial scale
MUTATION_SCALE_MIN: float = 0.002  # adaptive floor
MUTATION_SCALE_MAX: float = 0.5  # adaptive ceiling
MUTATION_ADAPT_WINDOW: int = 8  # generations for the 1/5 success rule
STAGNATION_BUMP_GENERATIONS: int = 8  # gens without accept before scale bump
STAGNATION_BUMP_FACTOR: float = 4.0  # scale multiplier on stagnation
ELITISM_RATIO: float = 0.20  # ratio
DEFAULT_PLAYBACK_SPEED: int = 1  # multiplier
DIST_TO_TIME_BONUS_RATIO: float = 1.0  # ratio
LOST_HP_SCORE_IMPACT_RATIO: float = 0.5  # ratio
RECORDER_MAX_GENERATIONS: int = 1000  # metric history cap

# ------ Parallel Simulation ------

# Number of worker processes used to simulate candidates in parallel.

# 0 = automatically choose a sensible number based on available CPUs.

# The implementation uses portable multiprocessing, so this works on

# Windows, Linux, and macOS.

SIMULATION_WORKERS: int = 0

# ------ GPU (PyTorch) Training Backend ------

# "auto" picks the GPU backend when a CUDA device is available, otherwise it
# falls back to the portable CPU backend. "gpu" forces GPU (errors if CUDA is
# unavailable), "cpu" forces the CPU backend.

TRAINING_BACKEND: str = "auto"

# When the GPU backend is active these replace the CPU defaults so a single
# champion generation runs thousands of parallel simulations on the device.

SIMULATION_RUNS_GPU: int = 16384  # total parallel simulations per GPU generation
POPULATION_SIZE_GPU: int = 16  # candidate agents evaluated per GPU generation

# Hardware-aware auto-configuration. On startup the trainer measures the real
# simulation throughput of this machine's backend (a short calibration run on
# the actual device) and picks the per-generation run budget, population size,
# and worker count so one generation lands near TARGET_GEN_TIME seconds while
# staying within hardware limits (VRAM for GPU, core count for CPU). Set
# AUTO_TUNE = False to force the static values above instead.

AUTO_TUNE: bool = False
TARGET_GEN_TIME: float = 0.75  # target seconds per training generation
AUTO_TUNE_MIN_RUNS: int = 64  # never drop below this run budget
AUTO_TUNE_MAX_RUNS: int = 32768  # never exceed this run budget
AUTO_TUNE_PROBE_RUNS: int = 256  # calibration runs used on the GPU backend
AUTO_TUNE_PROBE_RUNS_CPU: int = 32  # calibration runs used on the CPU backend

# ------ Curriculum (Adaptive Difficulty) ------

CURRICULUM_ENABLED: bool = False
CURRICULUM_DIFFICULTY_RATIO: float = 0.5  # map wall density used by curriculum
CURRICULUM_START_BFS: int = 8  # initial target path length (tiles)
CURRICULUM_BFS_STEP: int = 2  # path length increase per solved regime
CURRICULUM_BFS_WINDOW: int = 4  # acceptable path length range
CURRICULUM_MAX_BFS: int = 20  # max targeted path length
CURRICULUM_MAP_ATTEMPTS: int = 10  # map generation retries per regime
CURRICULUM_FAILURES_BEFORE_EASE: int = 2  # unsolved regimes before easing

# ------ GUI Element Layout Rectangles ------

LAYOUT_GRID_RECT: Tuple[int, int, int, int] = (20, 20, 800, 600)  # rect
LAYOUT_PANEL_RECT: Tuple[int, int, int, int] = (840, 20, 420, 140)  # rect
LAYOUT_GRAPH_RECT: Tuple[int, int, int, int] = (840, 180, 420, 440)  # rect
LAYOUT_SCRUBBER_RECT: Tuple[int, int, int, int] = (20, 650, 1240, 60)  # rect

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
