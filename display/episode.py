"""
Episode simulation runner for human-speed champion re-simulations.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

import config
from core.kinematics import CandidateKinematics
from core.map_data import MapData
from entities.player_express import PlayerExpress
from entities.player_state import PlayerState
from perception.spatial_transformer import SpatialTransformer
from training.agent import Agent


class Episode:
    """
    One human-speed re-simulation of a champion genome on a single map.
    """

    def __init__(
        self,
        genome: Tuple[List[np.ndarray], List[np.ndarray]],
        map_data: MapData,
        dist_grid: np.ndarray,
        initial_bfs: int,
        spawn_heading: float,
        run_number: int,
        profile_config: Dict[str, Any],
        max_steps: int = config.MAX_SIMULATION_STEPS,
        generation: int = 0
    ) -> None:
        """
        Initializes physical parameters and pre-simulates run to completion.
        """
        self.weights, self.biases = genome
        self.map_data: MapData = map_data
        self.dist_grid: np.ndarray = dist_grid
        self.initial_bfs: int = initial_bfs
        self.spawn_heading: float = spawn_heading
        self.run_number: int = run_number
        self.generation: int = generation
        self.profile_config: Dict[str, Any] = profile_config
        self.max_steps: int = max_steps

        self.state: PlayerState = PlayerState(
            float(map_data.start_pos[0]) + 0.5,
            float(map_data.start_pos[1]) + 0.5
        )
        self.state.heading = spawn_heading
        self.state.best_step_dist = initial_bfs

        self.frames: List[Dict[str, Any]] = []
        self.finished: bool = False
        self.solved: bool = False
        self.finish_step: int = max_steps
        self._celebration_count: int = 0

        self._kinematics: CandidateKinematics = CandidateKinematics(
            profile_config=profile_config
        )
        self._transformer: SpatialTransformer = SpatialTransformer(
            profile_config=profile_config
        )

        metabolics: Dict[str, Any] = profile_config.get("metabolics", {})
        kinematics_cfg: Dict[str, Any] = profile_config.get("kinematics", {})

        self.coll_damage: float = float(
            metabolics.get("collision_damage", 0.003)
        )
        self.idle_damage: float = float(
            metabolics.get("idle_damage", 0.001)
        )
        self.recovery_ratio: float = float(
            metabolics.get("recovery_ratio", 0.05)
        )
        self.move_speed: float = float(
            kinematics_cfg.get("move_speed", 0.1)
        )

        self.spin_enabled: bool = bool(
            metabolics.get("spin_penalty_enabled", True)
        )
        self.spin_thresh_rad: float = math.radians(
            float(metabolics.get("spin_angle_threshold_deg", 360.0))
        )
        self.spin_reset_rad: float = math.radians(
            float(metabolics.get("spin_reset_angle_deg", 5.0))
        )
        self.spin_hold_frames: int = int(
            metabolics.get("spin_reset_hold_frames", 15)
        )
        self.spin_dmg: float = float(
            metabolics.get("spin_damage_per_frame", 0.003)
        )

        self.stag_enabled: bool = bool(
            metabolics.get("stagnation_enabled", True)
        )
        self.stag_limit: int = int(
            self.max_steps * float(
                metabolics.get("stagnation_timeout_ratio", 0.75)
            )
        )
        self.stag_dmg: float = float(
            metabolics.get("stagnation_damage_per_frame", 0.001)
        )

        self.cum_rotation: float = 0.0
        self.straight_ticks: int = 0
        self.stagnation_ticks: int = 0

        self.last_hit: bool = False
        self.is_idle: bool = False
        self.is_spinning: bool = False

        self.simulate_up_to(self.max_steps)

    def run_data(self) -> Dict[str, Any]:
        """
        Exposes episode telemetry dictionary including generation index.
        """
        return {
            "map_data": self.map_data,
            "frames": self.frames,
            "run_number": self.run_number,
            "generation": self.generation,
            "initial_bfs": self.initial_bfs,
            "max_steps": self.max_steps,
            "finished": self.finished,
            "solved": self.solved,
            "finish_step": self.finish_step
        }

    def simulate_up_to(self, target_step: int) -> None:
        """
        Advances the episode until target_step frames exist or it ends.
        """
        while len(self.frames) < target_step and not self.finished:
            self._step()

    def _step(self) -> None:
        """
        Runs one simulation tick for the agent and records its frame.
        """
        if self.solved:
            self._step_celebration()
            return

        state: PlayerState = self.state
        step: int = len(self.frames) + 1

        tx: int = int(state.x)
        ty: int = int(state.y)

        current_dist: float = 9999.0
        if (
            0 <= tx < self.map_data.width
            and 0 <= ty < self.map_data.height
        ):
            current_dist = float(self.dist_grid[ty, tx])

        features: np.ndarray = self._transformer.compile_feature_vector(
            state.x,
            state.y,
            state.heading,
            self.move_speed,
            state.health,
            self.map_data,
            current_dist=current_dist,
            last_hit=self.last_hit,
            is_idle=self.is_idle,
            is_spinning=self.is_spinning,
            profile_config=self.profile_config
        )

        outputs, acts = Agent.forward_batch(
            self.weights,
            self.biases,
            features.reshape(1, -1),
            return_acts=True
        )

        move_eff: float = float(outputs[0, 0])
        turn_eff: float = float(outputs[0, 1])

        old_heading: float = state.heading
        state.heading, is_stationary_turn = (
            self._kinematics.apply_rotation(
                state.heading,
                turn_eff,
                move_eff
            )
        )

        heading_delta: float = abs(state.heading - old_heading)
        if heading_delta > math.pi:
            heading_delta = (2.0 * math.pi) - heading_delta

        if heading_delta >= self.spin_reset_rad:
            self.straight_ticks = 0
            self.cum_rotation += heading_delta
        else:
            self.straight_ticks += 1

        if self.straight_ticks >= self.spin_hold_frames:
            self.cum_rotation = 0.0

        self.stagnation_ticks += 1

        nx, ny, hit = self._kinematics.calculate_forward_step(
            state.x,
            state.y,
            state.heading,
            move_eff,
            self.map_data
        )

        is_idle: bool = (
            move_eff < 0.05
            or (abs(nx - state.x) < 1e-4 and abs(ny - state.y) < 1e-4)
            or is_stationary_turn
        )

        state.x = nx
        state.y = ny
        state.has_collided = hit
        state.frames_survived += 1

        if hit:
            state.health = max(0.0, state.health - self.coll_damage)

        if is_idle:
            state.health = max(0.0, state.health - self.idle_damage)

        is_spinning: bool = False
        if self.spin_enabled and self.cum_rotation >= self.spin_thresh_rad:
            is_spinning = True
            state.health = max(0.0, state.health - self.spin_dmg)

        if self.stag_enabled and self.stagnation_ticks >= self.stag_limit:
            state.health = max(0.0, state.health - self.stag_dmg)

        if state.health <= 0.0:
            state.is_alive = False

        tx = int(state.x)
        ty = int(state.y)

        curr_dist: int = 9999
        if (
            0 <= tx < self.map_data.width
            and 0 <= ty < self.map_data.height
        ):
            curr_dist = int(self.dist_grid[ty, tx])

        if curr_dist < state.best_step_dist:
            heal_amount: float = (
                (state.best_step_dist - curr_dist)
                * self.coll_damage
                * self.recovery_ratio
            )
            state.health = min(1.0, state.health + heal_amount)
            state.best_step_dist = curr_dist
            self.stagnation_ticks = 0
            self.cum_rotation = 0.0
            self.straight_ticks = 0

        if (
            tx == self.map_data.exit_pos[0]
            and ty == self.map_data.exit_pos[1]
        ):
            state.has_reached_exit = True

        self.last_hit = hit
        self.is_idle = is_idle
        self.is_spinning = is_spinning

        face: str = PlayerExpress.resolve_face(
            state.has_reached_exit,
            state.has_collided,
            state.is_alive
        )

        layer_acts: List[List[float]] = []
        if acts:
            layer_acts = [acts[0][0].tolist()] + [
                layer[0].tolist() for layer in acts[1:]
            ]

        self.frames.append({
            "step": step,
            "x": state.x,
            "y": state.y,
            "heading": state.heading,
            "face": face,
            "hit_wall": hit,
            "health": state.health,
            "is_alive": state.is_alive,
            "reached_exit": state.has_reached_exit,
            "dist": curr_dist,
            "activations": layer_acts,
        })

        if state.has_reached_exit:
            self.solved = True
            self.finish_step = step
        elif not state.is_alive:
            self.finished = True
            self.solved = False
            self.finish_step = step
        elif step >= self.max_steps:
            self.finished = True
            self.finish_step = step

    def _step_celebration(self) -> None:
        """
        Advances celebration frames once candidate has reached exit tile.
        """
        step: int = len(self.frames) + 1
        self._celebration_count += 1

        celebration_cap: int = getattr(
            config, "WINNER_CELEBRATION_FRAMES", 60
        )
        last_frame: Dict[str, Any] = self.frames[-1] if self.frames else {}

        self.frames.append({
            "step": step,
            "x": self.state.x,
            "y": self.state.y,
            "heading": self.state.heading,
            "face": config.FACE_EXIT,
            "hit_wall": False,
            "health": 1.0,
            "is_alive": True,
            "reached_exit": True,
            "dist": 0,
            "activations": last_frame.get("activations", [])
        })

        if (
            self._celebration_count >= celebration_cap
            or step >= self.max_steps
        ):
            self.finished = True
            self.finish_step = step
