```text
 ____           __  __                                             
/\  _`\        /\ \/\ \                                      __    
\ \ \L\ \__  __\ \ \ \ \    ___   _ __    __    ___      __ /\_\   
 \ \ ,__/\ \/\ \\ \ \ \ \  / __`\/\`'__\/'__`\/' _ `\  /'_ `\/\ \  
  \ \ \/\ \ \_\ \\ \ \_/ \/\ \L\ \ \ \//\  __//\ \/\ \/\ \L\ \ \ \ 
   \ \_\ \/`____ \\ `\___/\ \____/\ \_\\ \____\ \_\ \_\ \____ \ \_\
    \/_/  `/___/> \`\/__/  \/___/  \/_/ \/____/\/_/\/_/\/___L\ \/_/
             /\___/                                      /\____/   
             \/__/                                       \_/__/    
===============================================================================
               PYVORENNDRO 2D - SYSTEM SPECIFICATIONS & GUIDE
===============================================================================

[1.0 SYSTEM OVERVIEW]
-------------------------------------------------------------------------------
Core Philosophy: Sovereign Compute, Matrix-Isolated, Zero-Dependency.
Architecture   : 2D Neuroevolution Visualizer utilizing vectorized NumPy 
                 matrix math, PyBiwis 64-bit bitmask compression, continuous 
                 Circle-to-AABB smooth wall physics, dual steering dynamics 
                 (Car and Tank profiles), and an 11-channel Multi-Layer 
                 Perceptron (MLP) neural engine.
Primary Goal   : Train autonomous 2D candidates to navigate procedural 2D 
                 labyrinths from randomized start tiles to exit tiles using 
                 a 7-ray vision fan and a stereo binocular target compass.
Presentation   : Interactive GUI featuring dual camera tracking modes (Map-
                 Centered and Player-Centered), 16x turbo speed controls, 
                 repeat modes, pre-rendered background caching, standardized 
                 0-1000 score normalization, and a 2-column telemetry HUD.

[2.0 MEMORY, MAPS & PROCEDURAL GENERATION (PYBIWIS & NUMPY)]
-------------------------------------------------------------------------------
Grid Storage   : Rectangular tile grids (MAP_WIDTH x MAP_HEIGHT) mapped 
                 directly into flat C-contiguous NumPy memory arrays.
PyBiwis Chunks : Packed 64-bit integer words storing wall collision layouts. 
                 Tile maps compress into lightweight integer words, enabling 
                 register-speed bitmask queries and tiny save snapshots.
100% Floor Fill: All procedural generators use flood-fill analysis to 
                 guarantee 100% floor connectivity with zero isolated 
                 dead-zone pockets or unreachable corridors.
BFS Distance   : Every level builds an O(1) step-distance matrix. Each tile 
                 stores its exact topological path step-distance to the exit.
Dual Map Styles: Selectable procedural level generation algorithms:
                 - "BRANCHING WALLS": Organic tree-like maze crawler that 
                   grows continuous wall stems, sharp 90-degree corners, 
                   and T-junction branches to form realistic corridors and 
                   rooms without 2x2 solid wall blocks or dead pockets.
                 - "RANDOM": Classic random scattered wall noise layout.

[3.0 SPATIAL PERCEPTION & STEREO COMPASS (11 INPUT CHANNELS)]
-------------------------------------------------------------------------------
Vision Fan     : 7 probe rays distributed evenly across a 120-degree Field 
                 of View (VISION_ARC_ANGLE). Rays measure wall proximity:
                 - Channels 0..6 (-60° to +60°): 0.0 for open space up to 1.0 
                   for a point-blank wall collision.
Stereo Compass : 2 local cockpit directional sensors tracking goal angle:
                 - Channel 7 (TG-L): Left eye target intensity [0.0, 1.0].
                 - Channel 8 (TG-R): Right eye target intensity [0.0, 1.0].
Proprioception : 2 physical status channels:
                 - Channel 9 (SPD): Current forward movement speed.
                 - Channel 10 (HP): Active candidate health ratio [0.0, 1.0].

[4.0 NEURAL NETWORK ARCHITECTURE (MLP ENGINE)]
-------------------------------------------------------------------------------
Inputs (Fixed) : 11 continuous sensory channels (7 Wall Rays + 2 Stereo 
                 Compass Channels + Speed + Health Ratio).
Hidden Layers  : Configurable multi-layer dense stack (NEURAL_HIDDEN_LAYERS, 
                 NEURAL_NEURONS) using ReLU activations and Xavier weights.
Outputs (Fixed): 2 continuous motor control channels:
                 - Move Effort (Sigmoid [0.0, 1.0]): Forward acceleration.
                 - Turn Effort (Tanh [-1.0, 1.0]): Steering rotation delta.

[5.0 KINEMATICS, HEALTH & DUAL STEERING PROFILES]
-------------------------------------------------------------------------------
Steering Dynamics: Selectable via KINEMATICS_PROFILE in config:
                 - "CAR": Steering rotation is scaled by forward movement 
                   velocity (turn * move). Standing still prevents turning, 
                   forcing smooth driving arcs.
                 - "TANK": In-place differential steering. Turning while 
                   standing still (move < 0.05) is allowed but incurs an 
                   idle health penalty per frame to penalize stationary 
                   spinning.
Unblocked Spawn: Initial candidate spawn headings are probed to guarantee 
                 candidates face an open, walkable exit path upon spawning.
Smooth Collision: Continuous Circle-to-AABB penetration resolution uses 
                 minimum translation vectors (MTV) to push candidate bodies 
                 out of wall tiles. Candidates slide smoothly along flat wall 
                 faces and roll naturally around 90-degree outer corners.
Health & Damage: Candidates start at 100% health. Drains occur from:
                 - Wall Collision Penalty (HEALTH_COLL_DMG_PER_FRAME).
                 - Idle & Stationary Turn Penalty (HEALTH_IDLE_DMG_PER_FRAME): 
                   Drains health when standing still or turning in place.
BFS Recovery   : Achieving a NEW record best BFS step-distance restores 
                 health (Heal = dist_reduced * penalty_base * RECOVERY_RATIO).
Expressive UI  : Facial ASCII states map directly to physics events:
                 - FACE_WALK (o_o) : Normal walking / forward traversal.
                 - FACE_WALL (>_<) : Active wall collision / impact.
                 - FACE_DEAD (T_T) : Expired / depleted health state.
                 - FACE_EXIT (^_^) : Successfully reached the exit tile.

[6.0 VISUALIZER INTERFACE & PERFORMANCE OPTIMIZATIONS]
-------------------------------------------------------------------------------
Pre-Rendered Maps: Pre-renders the static level layout (walls, floors, start, 
                 and exit tiles) ONCE per generation into a single background 
                 image. Reusing this background reduces over 10,000 tile draw 
                 calls per frame down to just 16 fast image copies (blits), 
                 locking playback at a smooth 60 FPS in all camera views.
Font Caching   : Caches OS font instances in memory by point size, 
                 eliminating expensive OS font instantiations during render.
Alpha Scratchpad: Reuses a single pre-allocated transparent Surface for 
                 vision arc fan polygon rendering, eliminating dynamic RAM 
                 allocations and garbage collection pauses.
Dual Camera    : Interactive ENTER key toggle switches between:
                 - Map-Centered Mode: Fits entire level into viewport.
                 - Player-Centered Mode: Locks active candidate to center 
                   with fixed TILE_SIZE camera tracking & pixel-perfect SDL 
                   clipping masks (zero border bleed).

[7.0 NEUROEVOLUTION, FITNESS MATH & TIMELINE HUD]
-------------------------------------------------------------------------------
GA Engine      : Headless multi-generation simulation coordinator running 
                 candidate runs over population pools (POPULATION_SIZE), 
                 clamped to viewport grid capacity (GRID_ROWS * GRID_COLS).
Fitness Math   : Unit-aligned scoring formula normalized to [0, 1000]:
                 - Unit Alignment: Distance progress is converted to frame 
                   tick equivalents (dist / MOVE_SPEED). 1 tile equals 8 
                   frame ticks at speed 0.125, giving distance and speed 
                   bonus a natural 1:1 physical unit ratio.
                 - Ratio Control (DIST_TO_TIME_BONUS_RATIO): Controls the 
                   relative weight ratio of distance progress to time saved 
                   (e.g., ratio = 1.0 means 1:1 physical parity).
                 - Health Factor (LOST_HP_SCORE_IMPACT_RATIO): Applies a 
                   weighted multiplier ((1 - w) + w * HP) to punish wall 
                   scraping and reward clean navigation.
                 - 0-1000 Scale: Normalizes raw scores against a theoretical 
                   perfect run max (incorporating 0.586-tile corner turn 
                   savings) and scales to an integer range [0, 1000].

[7.1 CLI PROGRESS TABLE]
-------------------------------------------------------------------------------
CLI Layout     : Right-aligned 7-column real-time console progress table 
                 tracking generation metrics during headless execution:
                 - GEN   : Active evolutionary generation number (1-based).
                 - TOP   : Scaled top-candidate score [0, 1000].
                 - AVG   : Average scaled population score [0.0, 1000.0].
                 - WAY   : Initial BFS topological step-distance to exit.
                 - FIRST : Winning candidate index string (e.g. # 0).
                 - FRAME : Step tick count of fastest exit solver (- if none).
                 - EXITS : Ratio of candidate solvers to population pool size.

[7.2 GUI VISUALIZER & TIMELINE HUD]
-------------------------------------------------------------------------------
Breeding Ops   : Evolves next-generation networks via Elitism, Tournament 
                 Selection, Uniform Crossover, and Gaussian Mutation.
Status Frames  : Solved runs highlighted with Dark Green inset borders, 
                 dead runs highlighted with Dark Red inset borders, and the 
                 active selected candidate framed with an outer Yellow border.
Interactive UI : Transport controls with PLAY/PAUSE, REP ALL / REP 1 loop 
                 toggle, bidirectional speed controls (1x to 16x Turbo via 
                 Left/Right click), and 5-button mouse indexing.
Telemetry HUD  : Scaled 2-column layout with Light Blue header, Yellow 
                 selected callout, White step/gen stats, Green winner/top 
                 score metrics, and Grey average score.
Activation Graph: Real-time neural activation visualization featuring 
                 larger node circles and right-side input channel labels.

[8.0 DEVELOPMENT ROADMAP]
-------------------------------------------------------------------------------
Phase 1 [DONE] : MapData, PyBiwis Bitmasks, BFS Pathfinder, Map Generator.
Phase 2 [DONE] : Config, Kinematics, PlayerState, Expressive Face Engine.
Phase 3 [DONE] : Vision Arc DDA, Spatial Transformer, Neural MLP Engine.
Phase 4 [DONE] : Headless GA Trainer, Fitness Evaluator, Frame Recorder.
Phase 5 [DONE] : Multi-Map Grid Viewport, Neural Activation Graph, Timeline.
Phase 6 [DONE] : Smooth Wall-Sliding Physics, Dual Camera Tracking, 16x Turbo 
                 Transport, Repeat Modes & 2-Column Telemetry Dashboard.
Phase 7 [DONE] : Organic Branching Wall Generator, Pre-Rendered Tilemap Caching, 
                 Dynamic Font Caching & Alpha Scratchpad Optimization.
Phase 8 [DONE] : Dual Kinematics Profiles (Car & Tank), Unit-Aligned Fitness 
                 Math, Health Penalty Weighting & Standardized 0-1000 Score Engine.

===============================================================================
[!] PYVORENNDRO 2D | SOVEREIGN NEUROEVOLUTION ENGINE
===============================================================================