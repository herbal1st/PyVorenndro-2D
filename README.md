```text
 ____           __  __                                          __                 
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
Architecture   : High-performance 2D Neuroevolution Engine built with pure
                 Python, NumPy, Pygame, and optional PyTorch CUDA support.
                 Features a Single-Agent Champion evolutionary strategy,
                 data-driven YAML profiles, persistent champion weights,
                 PyBiwis 64-bit bitmask grid compression, vectorized DDA
                 raycasting, continuous Circle-to-AABB smooth wall physics,
                 dual kinematics steering profiles (Car and Tank), adaptive
                 curriculum training, hardware auto-tuning, and low-priority
                 process scheduling.
Primary Goal   : Train autonomous candidate agents to navigate procedural
                 labyrinths from randomized start locations to exit tiles
                 using multi-ray visual probe fans, target compass guidance,
                 topological distance gradients, and real-time damage
                 indicator feedback channels.
Presentation   : Interactive GUI featuring dual camera tracking modes (Map-
                 Centered and Player-Centered), 16x turbo speed controls,
                 dual-bar scrubber transport, pre-rendered background surface
                 caching, standardized 0-1000 score normalization, real-time
                 neural activation graphs, and an 8-column CLI console table.


[2.0 SINGLE-AGENT CHAMPION PARADIGM & MAZE GENERALIZATION]
-------------------------------------------------------------------------------
Single Champion: Rather than training a large pool of distinct agents on a
                 single maze, the system maintains one master Champion Agent.
                 In each generation, the trainer evaluates the champion and
                 a set of mutated offspring across multiple independently-
                 seeded procedural mazes simultaneously (paired evaluation).
Generalization : An agent cannot memorize or "cheat" on a single maze layout.
                 Its fitness score is the average performance across all
                 evaluated mazes, encouraging true spatial navigation logic.
Grid Storage   : Rectangular tile grids (MAP_WIDTH x MAP_HEIGHT) stored in
                 C-contiguous NumPy arrays for instant vectorized lookups.
PyBiwis Chunks : Packed 64-bit integer words storing wall collision layouts,
                 compressing tile maps into tiny save snapshots.
100% Floor Fill: All level generators use flood-fill analysis to guarantee
                 100% floor connectivity with zero isolated dead-end pockets.
BFS Distance   : Every level builds an O(1) step-distance matrix where each
                 tile stores its exact topological distance to the exit.
Branching Walls: Organic tree-like maze crawler growing continuous wall
                 stems, 90-degree corners, and T-junctions without 2x2 solid
                 blocks or isolated dead zones.


[3.0 DATA-DRIVEN PROFILE REGISTRY & WEIGHT PERSISTENCE]
-------------------------------------------------------------------------------
Brain Library  : Agent configurations (sensory channels, neural layer sizes,
                 kinematics, health metabolics, anti-spin guards, and file
                 paths) are defined in 'brain_library.yaml', keeping system
                 code separate from agent data.
Profiles       : Includes pre-configured profiles for 'scooterking38' (9 rays,
                 120° arc, 32 neurons/layer) and 'herbal1st' (21 rays, 200° arc,
                 3 hidden layers x 20 neurons).
Fail-Fast Check: The engine validates profile schemas at boot. Missing or
                 misspelled YAML keys trigger immediate, clear errors rather
                 than running on hidden default fallback settings.
Persistence    : Champion weight and bias arrays auto-save to compressed
                 '.npz' files in 'champions/' on generation completion or new
                 fitness records. Rerunning training automatically reloads
                 saved weights to continue fine-tuning.


[4.0 SPATIAL PERCEPTION & SENSORY CHANNELS]
-------------------------------------------------------------------------------
Vision Fan     : Configurable probe rays (e.g. 9 to 21 rays) distributed
                 evenly across a configurable Field of View (e.g. 120° to 200°),
                 calculated via grid-aligned Amanatides-Woo DDA. Measures
                 proximity to walls (0.0 for open space to 1.0 for point-blank
                 wall collision).
Proprioception : 2 physical status channels:
                 - Speed Channel (SPD): Current forward translation speed.
                 - Health Channel (HP): Active candidate health ratio [0, 1].
Damage Inputs  : 3 dedicated penalty indicator channels:
                 - Collision Flag (HIT): 1.0 on wall impact (0.0 otherwise).
                 - Idle Flag (IDL): 1.0 when stationary or turning in place.
                 - Spin Flag (SPN): 1.0 when over the cumulative turn limit.
Optional Compass: Toggleable via 'include_compass' in profile YAML. Adds 2
                 stereo target-guidance channels (TG-L, TG-R) encoding exit
                 bearing relative to agent heading.
Optional BFS    : Toggleable via 'include_bfs_sensor' in profile YAML. Adds a
                 normalized topological goal-gradient sensor (distance to
                 exit / max maze span).
Range & LOS    : Supports 'goal_sensor_max_range' for smooth distance falloff
                 and 'enable_los_gating' for line-of-sight raycast checks.


[5.0 NEURAL NETWORK ARCHITECTURE & FORWARD PASS]
-------------------------------------------------------------------------------
Inputs         : Dynamic sensory vector compiled per profile (e.g. 14, 16,
                 or 26 continuous channels).
Batched Pass   : Vectorized matrix multiplication ('np.einsum' on CPU or
                 PyTorch CUDA tensors on GPU) evaluating all candidates and
                 parallel mazes simultaneously in unified tensor batches.
Hidden Layers  : Configurable multi-layer dense stack ('hidden_layers',
                 'neurons_per_layer') using ReLU activations and Xavier
                 weight initialization.
Outputs (Fixed): 2 continuous motor control channels:
                 - Move Effort (Sigmoid [0.0, 1.0]): Forward acceleration.
                 - Turn Effort (Tanh [-1.0, 1.0]): Steering rotation delta.


[6.0 KINEMATICS, HEALTH & DUAL STEERING PROFILES]
-------------------------------------------------------------------------------
Steering       : Selectable via 'profile_style' in brain_library.yaml:
                 - "CAR": Turning rotation is scaled by forward velocity.
                   Standing still prevents turning, forcing driving arcs.
                 - "TANK": In-place differential steering. Turning while
                   standing still is allowed but incurs an idle health
                   penalty to discourage stationary spinning.
Smooth Collision: Continuous Circle-to-AABB penetration resolution uses
                  minimum translation vectors (MTV) to push agent bodies
                  out of wall tiles, sliding smoothly along wall faces.
Health & Damage : Candidates start at 100% health. Health drains from:
                  - Wall Collision Penalty ('collision_damage').
                  - Idle Penalty ('idle_damage').
                  - Cumulative Spin Penalty ('spin_damage_per_frame'):
                    Triggers once absolute turn accumulation exceeds
                    'spin_angle_threshold_deg' without driving straight for
                    'spin_reset_hold_frames' consecutive frames.
                  - Stagnation Penalty ('stagnation_damage_per_frame'):
                    Triggers once an agent fails to set a new record BFS
                    distance within consecutive steps.
BFS Recovery    : Setting a NEW best record for closeness to the exit
                  restores health based on distance closed and resets the
                  cumulative spin counter, straight hold timer, and
                  stagnation clock.
Expressive UI   : Facial ASCII states map directly to physics events:
                  - FACE_WALK : Normal walking traversal ("o_o").
                  - FACE_WALL : Active wall impact collision (">*<").
                  - FACE_DEAD : Expired / zero health state ("T_T").
                  - FACE_EXIT : Successfully reached exit tile ("^*^").


[7.0 DUAL COMPUTE BACKENDS (CPU NUMPY & GPU PYTORCH)]
-------------------------------------------------------------------------------
CPU Backend    : Multi-candidate vectorized NumPy batch engine executing
                 all candidates and mazes in unified tensor operations,
                 delivering fast generation times (~0.2s - 0.5s per gen).
GPU Backend    : PyTorch CUDA engine ('GpuHeadlessTrainer') evaluating thousands
                 of parallel maze simulations simultaneously as batched CUDA
                 tensors in GPU VRAM, paired with an asynchronous multi-core
                 CPU map prefetching pool.
Auto Fallback  : If PyTorch or CUDA hardware is unavailable, the trainer
                 automatically reverts to the optimized CPU NumPy engine.


[8.0 HARDWARE AUTO-TUNING & PROCESS SCHEDULING]
-------------------------------------------------------------------------------
Auto-Tuning    : At startup, the system benchmarks hardware throughput during
                 a calibration probe run, automatically adjusting candidate
                 population and maze count to hit a target generation
                 duration (e.g. 0.75s).
Core Capping   : Auto mode caps CPU workers at physical cores minus 1
                 ('max(1, (logical_cores // 2) - 1)'). On an 8-core / 16-thread
                 chip, workers cap at 3 or 4 processes, preventing 100% CPU
                 thread saturation and thermal throttling.
Low Priority   : Worker processes drop OS scheduling priority
                 ('BELOW_NORMAL_PRIORITY_CLASS' on Windows or 'os.nice(10)' on
                 Unix/macOS), ensuring desktop UI, browser, and display loops
                 always receive priority execution cycles.


[9.0 VISUALIZER INTERFACE & INTERACTIVE GUI]
-------------------------------------------------------------------------------
Process Split  : The headless trainer runs inside a background worker process,
                 publishing champion weights to shared memory. The display
                 runner re-simulates the champion live at human playback speed
                 on a single viewport at 60 FPS.
Pre-Rendered   : Pre-renders static level geography into a single background
                 image per run, reducing tile draw calls to fast surface copies.
Alpha Scratchpad: Reuses a single pre-allocated transparent Surface for vision
                 arc fan polygons and heading lines, preventing memory churn.
Dual Camera    : ENTER key toggle switches between:
                 - Map-Centered Mode: Fits entire level into viewport.
                 - Player-Centered Mode: Locks agent to center with SDL
                   clipping masks.
Interactive UI : Transport controls with PLAY/PAUSE, REP ALL / REP 1 loop
                 toggle, speed controls (1x to 16x Turbo), and timeline
                 scrubbing for frame steps and generations.
Telemetry HUD  : Overlay panel displaying active run step count, generation,
                 top scores, average scores, and winner callouts.
Activation Graph: Real-time neural activation graph rendering node layer
                 intensities with dark-red-to-orange color shifts and input
                 channel shorthand labels (-60°..+60°, TG-L, TG-R, SPD, HP,
                 HIT, IDL, SPN).


[10.0 CLI PROGRESS TABLE]
-------------------------------------------------------------------------------
CLI Layout     : Right-aligned 8-column real-time console table tracking
                 per-generation simulation progress during training:
                 - GEN   : Active evolutionary generation number (1-based).
                 - BEST  : Current generation's top candidate score [0, 1000].
                 - FIT   : Average scaled population score [0.0, 1000.0].
                 - WAY   : Average initial BFS step-distance to exit.
                 - FRAME : Step tick count of fastest exit solver (- if none).
                 - EXITS : Solvers ratio out of parallel simulation run count.
                 - SCALE : Active Gaussian mutation noise scale.
                 - TIME  : Execution wall-clock duration in seconds.


[11.0 RUNNING THE SIMULATOR & CONTROLS]
-------------------------------------------------------------------------------
Prerequisites  : Python 3.10+, NumPy, Pygame (Pygame-CE 2.5+ tested).
                 Optional: PyTorch with CUDA for GPU acceleration.
Launch         : python main.py
                 - Executes single-agent training in the background, prints the
                   per-generation CLI table, auto-saves champion weights to
                   'champions/', and renders the live re-simulation GUI.
GUI Controls   : 
                 - ENTER         : Toggle camera (Map-Centered / Player-Centered).
                 - ESC           : Close visualizer GUI and terminate trainer.
                 - SPACE         : Toggle Play/Pause transport.
                 - LEFT / RIGHT  : Jump 25 frame steps backward/forward.
                 - PAGEUP / DOWN : Jump 1 generation backward/forward.
                 - Left-Click    : Toggle camera zoom or scrub timeline.
                 - Right-Click   : Decrease playback speed on speed button.

===============================================================================
CONTRIBUTORS
===============================================================================
This project is developed and maintained by:
* herbal1st (Lead Maintainer)
* Elijah (scooterking38) — Simulation engine vectorization, GPU PyTorch
  backend, and single-agent champion paradigm.

===============================================================================
[!] PYVORENNDRO 2D | SOVEREIGN NEUROEVOLUTION ENGINE
===============================================================================
Distributed under the PyVorengi Source-Available End User License Agreement.
Copyright (c) 2026 herbal1st and Elijah. All Rights Reserved.
Strictly for personal evaluation, education, private editing, and non-commercial
research.