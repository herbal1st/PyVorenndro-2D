```text
. ____           __  __                                          __                 
/\  _`\        /\ \/\ \                                        /\ \                
\ \ \L\ \__  __\ \ \ \ \    ___   _ __    __    ___     ___    \_\ \  _ __   ___   
 \ \ ,__/\ \/\ \\ \ \ \ \  / __`\/\`'__\/'__`\/' _ `\ /' _ `\  /'_` \/\`'__\/ __`\ 
  \ \ \/\ \ \_\ \\ \ \_/ \/\ \L\ \ \ \//\  __//\ \/\ \/\ \/\ \/\ \L\ \ \ \//\ \L\ \
   \ \_\ \/`____ \\ `\___/\ \____/\ \_\\ \____\ \_\ \_\ \_\ \_\ \___,_\ \_\\ \____/
    \/_/  `/___/> \`\/__/  \/___/  \/_/ \/____/\/_/\/_/\/_/\/_/\/__,_ /\/_/ \/___/ 
             /\___/                                                                
             \/__/ 
===============================================================================
               PYVORENNDRO 2D - SYSTEM SPECIFICATIONS & GUIDE
===============================================================================

[1.0 SYSTEM OVERVIEW]
-------------------------------------------------------------------------------
Core Philosophy: Sovereign Compute, Matrix-Isolated, Zero-Dependency.
Architecture   : High-performance 2D Neuroevolution Visualizer powered by a
                 vectorized NumPy batch simulation engine (feature batch ->
                 neural batch -> physics batch), PyBiwis 64-bit bitmask grid
                 compression, grid-aligned Amanatides-Woo DDA raycasting,
                 continuous Circle-to-AABB smooth wall physics, dual driving
                 dynamics (Car and Tank profiles), adaptive curriculum
                 training, and a dynamic Multi-Layer Perceptron (MLP).
Primary Goal   : Train autonomous 2D candidate agents to navigate procedural
                 labyrinths from randomized start locations to exit tiles
                 using a 9-ray visual sensor fan and optional target guidance.
Presentation   : Interactive GUI featuring dual camera tracking modes (Map-
                 Centered and Player-Centered), 16x turbo speed controls,
                 dual-bar scrubber transport, pre-rendered background surface
                 caching, standardized 0-1000 score normalization, real-time
                 neural activation graphs, and an 8-column CLI console table.

[2.0 MEMORY, MAPS & PROCEDURAL GENERATION (PYBIWIS & NUMPY)]
-------------------------------------------------------------------------------
Grid Storage   : Rectangular tile grids (MAP_WIDTH x MAP_HEIGHT) stored in
                 C-contiguous NumPy arrays for instant vectorized lookups.
PyBiwis Chunks : Packed 64-bit integer words storing wall collision layouts.
                 Compresses tile maps into tiny save snapshots and fast memory.
100% Floor Fill: All level generators use flood-fill analysis to guarantee
                 100% floor connectivity with zero isolated dead-end pockets.
BFS Distance   : Every level builds an O(1) step-distance matrix. Each tile
                 stores its exact topological step-distance to the exit.
Dual Map Styles: Selectable procedural level generation algorithms:
                 - "BRANCHING WALLS": Organic tree-like maze crawler growing
                   continuous wall stems, 90-degree corners, and T-junctions
                   without 2x2 solid wall blocks or isolated dead zones.
                 - "RANDOM": Classic random scattered wall noise layout.
Map Regimes    : Organizes training generations into controlled "regimes" on
                 a single map layout. This gives agents time to master a
                 specific maze before rotating environments. Transitioning
                 to a new map temporarily boosts mutation rates to encourage
                 exploration on new layouts.
Curriculum     : Adaptive difficulty system that acts like a training ladder.
Learning         It starts agents on short, easy paths (e.g. 8 tiles) and
                 automatically steps up maze complexity as agents succeed. If
                 the population struggles, it automatically eases difficulty.

[3.0 SPATIAL PERCEPTION (9 TO 13 INPUT CHANNELS)]
-------------------------------------------------------------------------------
Vision Fan     : 9 probe rays distributed evenly across a 120-degree Field
                 of View (VISION_ARC_ANGLE), calculated via grid-aligned
                 Amanatides-Woo DDA. Measures proximity to walls:
                 - Channels 0..8 (-60° to +60°): 0.0 for open space up to 1.0
                   for a point-blank wall collision.
Proprioception : 2 physical status channels:
                 - Speed Channel (SPD): Current forward translation speed.
                 - Health Channel (HP): Active candidate health ratio [0, 1].
Optional       : Toggleable via INCLUDE_COMPASS in config. Adds 2 local
Stereo Compass   target-guidance channels (TG-L, TG-R) encoding exit bearing
                 relative to heading (expanding inputs from 11 to 13).
                 Disabled by default so agents must learn maze structure from
                 wall rays alone (e.g. wall-following).

[4.0 NEURAL NETWORK ARCHITECTURE (MLP ENGINE)]
-------------------------------------------------------------------------------
Inputs         : 11 continuous sensory channels by default (9 Wall Rays +
                 Speed + Health Ratio), or 13 when INCLUDE_COMPASS is True.
Batched Pass   : Tensorized matrix multiplication (`np.einsum`) evaluating the
                 entire population's neural decisions simultaneously.
Hidden Layers  : Configurable multi-layer dense stack (NEURAL_HIDDEN_LAYERS,
                 NEURAL_NEURONS) using ReLU activations and Xavier weights.
Outputs (Fixed): 2 continuous motor control channels:
                 - Move Effort (Sigmoid [0.0, 1.0]): Forward acceleration.
                 - Turn Effort (Tanh [-1.0, 1.0]): Steering rotation delta.

[5.0 KINEMATICS, HEALTH & DUAL STEERING PROFILES]
-------------------------------------------------------------------------------
Steering Profiles: Selectable via KINEMATICS_PROFILE in config:
                 - "CAR": Turning rotation is scaled by forward velocity.
                   Standing still prevents turning, forcing driving arcs.
                 - "TANK": In-place differential steering. Turning while
                   standing still is allowed but incurs an idle health
                   penalty to discourage stationary spinning.
Smooth Collision : Continuous Circle-to-AABB penetration resolution uses
                   minimum translation vectors (MTV) to push agent bodies
                   out of wall tiles, sliding smoothly along wall faces.
Health & Damage  : Candidates start at 100% health. Health drains from:
                   - Wall Collision Penalty (HEALTH_COLL_DMG_PER_FRAME).
                   - Idle & Spin Penalty (HEALTH_IDLE_DMG_PER_FRAME).
                   - Cumulative Spin Penalty (SPIN_DMG_PER_FRAME): Triggers
                     once absolute turn accumulation exceeds
                     SPIN_ANGLE_THRESHOLD_DEG without BFS progress.
                   - Stagnation Penalty (STAGNATION_DMG_PER_FRAME): Triggers
                     once a candidate fails to set a new record BFS distance
                     within int(MAX_SIMULATION_STEPS *
                     STAGNATION_TIMEOUT_RATIO) consecutive ticks.
BFS Recovery     : Setting a NEW best record for closeness to the exit
                   restores health based on distance closed. Also resets
                   both the cumulative spin counter and the stagnation
                   clock, so legitimate corridor navigation is never
                   penalised for the turns it requires.
Expressive UI    : Facial ASCII states map directly to physics events:
                   - FACE_WALK : Normal walking traversal.
                   - FACE_WALL : Active wall impact collision.
                   - FACE_DEAD : Expired / zero health state.
                   - FACE_EXIT : Successfully reached exit tile.

[6.0 VISUALIZER INTERFACE & PERFORMANCE OPTIMIZATIONS]
-------------------------------------------------------------------------------
Batched Engine   : Computes entire generations in unified NumPy array operations
                   (features -> neural decisions -> physics steps -> health),
                   eliminating per-candidate Python loop overhead.
Pre-Rendered Maps: Pre-renders static level geography into a single background
                   image per generation, reducing thousands of tile draw
                   calls down to 16 fast image copies (blits) at 60 FPS.
Font Caching     : Memoizes OS font instances by point size to eliminate
                   runtime font creation overhead.
Alpha Scratchpad : Reuses a single pre-allocated transparent Surface for
                   vision arc fan polygons, preventing memory allocations.
Dual Camera      : Interactive ENTER key toggle switches between:
                   - Map-Centered Mode: Fits entire level into viewport.
                   - Player-Centered Mode: Locks selected agent to center
                     with pixel-perfect SDL clipping masks.

[7.0 NEUROEVOLUTION, FITNESS MATH & TIMELINE HUD]
-------------------------------------------------------------------------------
GA Engine        : Multi-generation evolutionary manager executing Elitism,
                   Tournament Selection, Uniform Crossover, and Gaussian
                   Weight Mutations across stacked 3D genome matrices.
Fitness Math     : Unit-aligned scoring formula normalized to [0, 1000]:
                   - Unit Alignment: Distance progress converts to frame tick
                     equivalents (dist / MOVE_SPEED). 1 tile equals 10 ticks
                     at speed 0.10, giving distance and speed bonus parity.
                   - Ratio Control: DIST_TO_TIME_BONUS_RATIO balances distance
                     progress against time saved.
                   - Health Weight: LOST_HP_SCORE_IMPACT_RATIO penalizes wall
                     scraping to reward clean navigation.

[7.1 CLI PROGRESS TABLE]
-------------------------------------------------------------------------------
CLI Layout       : Right-aligned 8-column real-time console table tracking
                   per-generation simulation progress during training:
                   - GEN   : Active evolutionary generation number (1-based).
                   - TOP   : Scaled top-candidate score [0, 1000].
                   - AVG   : Average scaled population score [0.0, 1000.0].
                   - WAY   : Initial BFS topological step-distance to exit.
                   - TARG  : Target curriculum BFS distance.
                   - FRAME : Step tick count of fastest exit solver (- if none).
                   - EXITS : Solvers ratio out of population pool size.
                   - TIME  : Execution wall-clock duration in seconds.

[7.2 GUI VISUALIZER & TIMELINE HUD]
-------------------------------------------------------------------------------
Status Frames    : Solved runs feature Dark Green inset borders, dead runs
                   feature Dark Red inset borders, and the selected agent
                   is highlighted with a Yellow outer border.
Interactive UI   : Transport controls with PLAY/PAUSE, REP ALL / REP 1 loop
                   toggle, speed controls (1x to 16x Turbo), and timeline
                   scrubbing for both frame steps and generations.
Telemetry HUD    : Scaled 2-column overlay displaying selected candidate index,
                   frame step count, generation status, top scores, average
                   scores, and winning candidate indices.
Activation Graph : Real-time neural activation graph rendering node layer
                   intensities with dark-red-to-orange color shifts and
                   shorthand input channel labels (-60°..+60°, SPD, HP).

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
Phase 8 [DONE] : Dual Kinematics Profiles, Unit-Aligned Fitness Math, Health
                 Penalty Weighting & Standardized 0-1000 Score Engine.
Phase 9 [DONE] : Vectorized Batch Simulation Engine, Grid-Aligned DDA Raycasting,
                 Map Regime Dynamics, Adaptive Curriculum Learning, Optional
                 Target Compass Toggle, and Execution Timing Telemetry.

[9.0 RUNNING THE SIMULATOR]
-------------------------------------------------------------------------------
Requirements : Python 3.10+, NumPy, Pygame (2.6+ tested).
Launch       : python main.py    (from the project root)
                 - Executes LEARNING_GENERATIONS training runs headlessly,
                   prints the per-generation CLI table, and boots the
                   interactive visualizer GUI upon completion.
GUI Controls : 
                 - ENTER         : Toggle camera (Map-Centered / Player-Centered).
                 - Left-Click    : Select viewport candidate or scrub timeline.
                 - Double-Click  : Toggle single candidate zoom view.
                 - Right-Click   : Decrease speed on speed control button.

===============================================================================
CONTRIBUTORS
===============================================================================
This project is developed and maintained by:
* herbal1st (Lead Maintainer)
* Elijah (scooterking38) — Simulation engine vectorisation & curriculum logic.

===============================================================================
[!] PYVORENNDRO 2D | SOVEREIGN NEUROEVOLUTION ENGINE
===============================================================================
Distributed under the PyVorengi Source-Available End User License Agreement.
Copyright (c) 2026 herbal1st and Elijah. All Rights Reserved.
Strictly for personal evaluation, education, private editing, and non-commercial
research.
