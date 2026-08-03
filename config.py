"""
Global configuration settings for PyVorenndro 2D engine.
"""

from typing import Tuple

# ------ Display & Grid Layout ------
SCREEN_WIDTH: int = 1280  # pixels
SCREEN_HEIGHT: int = 720  # pixels
FPS: int = 60  # hertz
GRID_ROWS: int = 4  # count
GRID_COLS: int = 4  # count

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
NEURAL_NEURONS: int = 10  # neurons

# ------ Genetic Algorithm & Training ------
LEARNING_GENERATIONS: int = 200  # generations
POPULATION_SIZE: int = 16  # candidates
MAX_SIMULATION_STEPS: int = 1000  # ticks
MUTATION_RATE: float = 0.15  # ratio
MUTATION_SCALE: float = 0.05  # scale
ELITISM_RATIO: float = 0.20  # ratio
DEFAULT_PLAYBACK_SPEED: int = 1  # multiplier
DIST_TO_TIME_BONUS_RATIO: float = 1.0  # ratio
LOST_HP_SCORE_IMPACT_RATIO: float = 0.5  # ratio

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
FACE_WALL: str = ">_<"  # expression
FACE_DEAD: str = "T_T"  # expression
FACE_EXIT: str = "^_^"  # expression
