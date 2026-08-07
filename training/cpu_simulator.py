"""
Vectorized NumPy batch simulation engine for CPU candidate evaluation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.kinematics import CandidateKinematics
from entities.player_state import PlayerState
from perception.spatial_transformer import SpatialTransformer
from training.agent import Agent
from training.fitness import FitnessEvaluator


class CpuSimulator:
    """
    Executes candidate agent forward passes and continuous physics in NumPy.
    """

    def __init__(
        self,
        profile_config: Dict[str, Any],
        max_steps: int,
        kinematics: Optional[CandidateKinematics] = None,
        transformer: Optional[SpatialTransformer] = None
    ) -> None:
        """
        Initializes profile parameters, kinematics physics, and transformer.
        """
        self.profile_config: Dict[str, Any] = profile_config
        self.max_steps: int = max_steps
        self.kinematics: CandidateKinematics = (
            kinematics or CandidateKinematics(profile_config=profile_config)
        )
        self.transformer: SpatialTransformer = (
            transformer or SpatialTransformer(profile_config=profile_config)
        )

    def simulate_candidates(
        self,
        agents: List[Agent],
        runs: List[Tuple]
    ) -> List[Tuple[float, List[float], List[PlayerState]]]:
        """
        Simulates all candidate genomes across shared mazes in NumPy batch ops.
        """
        cand_count: int = len(agents)
        run_count: int = len(runs)
        total_instances: int = cand_count * run_count

        metabolics: Dict[str, Any] = self.profile_config["metabolics"]
        kinematics_cfg: Dict[str, Any] = self.profile_config["kinematics"]
        sensory: Dict[str, Any] = self.profile_config["sensory"]

        coll_dmg: float = float(metabolics["collision_damage"])
        idle_dmg: float = float(metabolics["idle_damage"])
        rec_ratio: float = float(metabolics["recovery_ratio"])
        move_speed: float = float(kinematics_cfg["move_speed"])
        include_bfs: bool = bool(sensory["include_bfs_sensor"])

        spin_enabled: bool = bool(
            metabolics.get("spin_penalty_enabled", True)
        )
        spin_thresh_rad: float = math.radians(
            float(metabolics.get("spin_angle_threshold_deg", 360.0))
        )
        spin_reset_rad: float = math.radians(
            float(metabolics.get("spin_reset_angle_deg", 5.0))
        )
        spin_hold_frames: int = int(
            metabolics.get("spin_reset_hold_frames", 15)
        )
        spin_dmg: float = float(
            metabolics.get("spin_damage_per_frame", 0.003)
        )

        stag_enabled: bool = bool(
            metabolics.get("stagnation_enabled", True)
        )
        stag_limit: int = int(
            self.max_steps * float(
                metabolics.get("stagnation_timeout_ratio", 0.75)
            )
        )
        stag_dmg: float = float(
            metabolics.get("stagnation_damage_per_frame", 0.001)
        )

        wall_grids: np.ndarray = np.stack(
            [runs[i % run_count][0].build_wall_grid()
             for i in range(total_instances)]
        )
        dist_grids: np.ndarray = np.stack(
            [np.asarray(runs[i % run_count][1], dtype=np.int64)
             for i in range(total_instances)]
        )
        exit_positions: np.ndarray = np.array(
            [runs[i % run_count][0].exit_pos for i in range(total_instances)],
            dtype=np.int64
        )
        initial_bfs_dists: np.ndarray = np.array(
            [runs[i % run_count][2] for i in range(total_instances)],
            dtype=np.int64
        )
        spawn_headings: np.ndarray = np.array(
            [runs[i % run_count][5] for i in range(total_instances)],
            dtype=np.float64
        )

        h, w = wall_grids.shape[1], wall_grids.shape[2]

        num_layers: int = len(agents[0].weights)
        w_stacks: List[np.ndarray] = []
        b_stacks: List[np.ndarray] = []

        for l_idx in range(num_layers):
            w_layer: np.ndarray = np.stack(
                [agents[i // run_count].weights[l_idx]
                 for i in range(total_instances)]
            )
            b_layer: np.ndarray = np.stack(
                [agents[i // run_count].biases[l_idx]
                 for i in range(total_instances)]
            )
            w_stacks.append(w_layer)
            b_stacks.append(b_layer)

        xs: np.ndarray = np.array(
            [float(runs[i % run_count][0].start_pos[0]) + 0.5
             for i in range(total_instances)],
            dtype=np.float64
        )
        ys: np.ndarray = np.array(
            [float(runs[i % run_count][0].start_pos[1]) + 0.5
             for i in range(total_instances)],
            dtype=np.float64
        )
        headings: np.ndarray = spawn_headings.copy()
        healths: np.ndarray = np.ones(total_instances, dtype=np.float64)
        best_dists: np.ndarray = initial_bfs_dists.astype(np.float64)

        cum_rotation: np.ndarray = np.zeros(
            total_instances, dtype=np.float64
        )
        straight_ticks: np.ndarray = np.zeros(
            total_instances, dtype=np.int64
        )
        stagnation_ticks: np.ndarray = np.zeros(
            total_instances, dtype=np.int64
        )

        last_hits: np.ndarray = np.zeros(total_instances, dtype=np.bool_)
        is_idles: np.ndarray = np.zeros(total_instances, dtype=np.bool_)
        is_spinnings: np.ndarray = np.zeros(total_instances, dtype=np.bool_)

        alive: np.ndarray = np.ones(total_instances, dtype=np.bool_)
        reached: np.ndarray = np.zeros(total_instances, dtype=np.bool_)
        frames_survived: np.ndarray = np.zeros(
            total_instances, dtype=np.int64
        )

        hidden_layers: int = num_layers - 1

        for step in range(1, self.max_steps + 1):
            active_mask: np.ndarray = alive & ~reached
            active_idx: np.ndarray = np.flatnonzero(active_mask)
            if active_idx.size == 0:
                break

            axs: np.ndarray = xs[active_idx]
            ays: np.ndarray = ys[active_idx]
            ahs: np.ndarray = headings[active_idx]
            ahp: np.ndarray = healths[active_idx]

            if include_bfs:
                sx: np.ndarray = np.floor(axs).astype(np.int64)
                sy: np.ndarray = np.floor(ays).astype(np.int64)
                in_s: np.ndarray = (
                    (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
                )
                sxc: np.ndarray = np.clip(sx, 0, w - 1)
                syc: np.ndarray = np.clip(sy, 0, h - 1)
                sensor_dist: Optional[np.ndarray] = np.where(
                    in_s, dist_grids[active_idx, syc, sxc], 9999
                ).astype(np.float64)
            else:
                sensor_dist = None

            features = self.transformer.compile_feature_batch(
                axs,
                ays,
                ahs,
                np.full(active_idx.size, move_speed, dtype=np.float64),
                ahp,
                wall_grids=wall_grids[active_idx],
                exit_positions=exit_positions[active_idx],
                current_dists=sensor_dist,
                last_hits=last_hits[active_idx],
                is_idles=is_idles[active_idx],
                is_spinnings=is_spinnings[active_idx],
                profile_config=self.profile_config
            )

            x_mat: np.ndarray = features.astype(np.float64, copy=False)
            for l_idx, (w_st, b_st) in enumerate(zip(w_stacks, b_stacks)):
                x_mat = np.einsum(
                    "ni,nij->nj",
                    x_mat,
                    w_st[active_idx],
                    optimize=False
                ) + b_st[active_idx].squeeze(1)

                if l_idx < hidden_layers:
                    x_mat = np.maximum(0.0, x_mat)

            move_eff: np.ndarray = 1.0 / (
                1.0 + np.exp(-np.clip(x_mat[:, 0], -500.0, 500.0))
            )
            turn_eff: np.ndarray = np.tanh(x_mat[:, 1])

            px, py, new_heading, hit, is_stationary = (
                self.kinematics.step_batch(
                    axs,
                    ays,
                    ahs,
                    move_eff,
                    turn_eff,
                    wall_grids=wall_grids[active_idx]
                )
            )

            heading_delta: np.ndarray = np.abs(new_heading - ahs)
            heading_delta = np.where(
                heading_delta > math.pi,
                (2.0 * math.pi) - heading_delta,
                heading_delta
            )

            is_turning: np.ndarray = heading_delta >= spin_reset_rad
            straight_ticks[active_idx] = np.where(
                is_turning, 0, straight_ticks[active_idx] + 1
            )
            cum_rotation[active_idx] = np.where(
                is_turning,
                cum_rotation[active_idx] + heading_delta,
                cum_rotation[active_idx]
            )

            should_reset_spin: np.ndarray = (
                straight_ticks[active_idx] >= spin_hold_frames
            )
            cum_rotation[active_idx] = np.where(
                should_reset_spin, 0.0, cum_rotation[active_idx]
            )

            stagnation_ticks[active_idx] += 1

            hlt: np.ndarray = ahp - (hit * coll_dmg)
            not_moved: np.ndarray = (
                (np.abs(px - axs) < 1e-4) & (np.abs(py - ays) < 1e-4)
            )
            idle: np.ndarray = (
                (move_eff < 0.05) | not_moved | is_stationary
            )
            hlt = hlt - (idle * idle_dmg)

            is_spinning: np.ndarray = np.zeros(
                active_idx.size, dtype=np.bool_
            )
            if spin_enabled:
                is_spinning = cum_rotation[active_idx] >= spin_thresh_rad
                hlt = hlt - (is_spinning.astype(np.float64) * spin_dmg)

            if stag_enabled:
                is_stagnated: np.ndarray = (
                    stagnation_ticks[active_idx] >= stag_limit
                )
                hlt = hlt - (is_stagnated.astype(np.float64) * stag_dmg)

            hlt = np.maximum(hlt, 0.0)

            tx: np.ndarray = np.floor(px).astype(np.int64)
            ty: np.ndarray = np.floor(py).astype(np.int64)

            inb: np.ndarray = (
                (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
            )
            txc: np.ndarray = np.clip(tx, 0, w - 1)
            tyc: np.ndarray = np.clip(ty, 0, h - 1)

            curr_dist: np.ndarray = np.where(
                inb, dist_grids[active_idx, tyc, txc], 9999
            )

            better: np.ndarray = curr_dist < best_dists[active_idx]
            heal: np.ndarray = (
                (best_dists[active_idx] - curr_dist) * coll_dmg * rec_ratio
            )
            hlt = np.where(better, np.minimum(1.0, hlt + heal), hlt)
            best_dists[active_idx] = np.where(
                better, curr_dist, best_dists[active_idx]
            )

            improved_idx: np.ndarray = active_idx[better]
            if improved_idx.size > 0:
                stagnation_ticks[improved_idx] = 0
                cum_rotation[improved_idx] = 0.0
                straight_ticks[improved_idx] = 0

            ex: np.ndarray = exit_positions[active_idx, 0]
            ey: np.ndarray = exit_positions[active_idx, 1]
            reached_run: np.ndarray = inb & (tx == ex) & (ty == ey)

            xs[active_idx] = px
            ys[active_idx] = py
            headings[active_idx] = new_heading
            healths[active_idx] = hlt
            alive[active_idx] = hlt > 0.0
            reached[active_idx] |= reached_run
            last_hits[active_idx] = hit
            is_idles[active_idx] = idle
            is_spinnings[active_idx] = is_spinning
            frames_survived[active_idx] += 1

        results: List[Tuple[float, List[float], List[PlayerState]]] = []

        for c_idx in range(cand_count):
            c_states: List[PlayerState] = []
            c_scaled_scores: List[float] = []

            for r_idx in range(run_count):
                inst_idx: int = c_idx * run_count + r_idx
                st: PlayerState = PlayerState(
                    float(runs[r_idx][0].start_pos[0]) + 0.5,
                    float(runs[r_idx][0].start_pos[1]) + 0.5
                )
                st.x = float(xs[inst_idx])
                st.y = float(ys[inst_idx])
                st.heading = float(headings[inst_idx])
                st.health = float(healths[inst_idx])
                st.has_collided = bool(last_hits[inst_idx])
                st.is_alive = bool(alive[inst_idx])
                st.has_reached_exit = bool(reached[inst_idx])
                st.best_step_dist = int(best_dists[inst_idx])
                st.frames_survived = int(frames_survived[inst_idx])
                c_states.append(st)

                raw: float = FitnessEvaluator.calculate_raw_score(
                    st, runs[r_idx][2], self.max_steps
                )
                scaled: float = FitnessEvaluator.calculate_scaled_score(
                    raw, runs[r_idx][4]
                )
                c_scaled_scores.append(scaled)

            c_fitness: float = (
                float(np.mean(c_scaled_scores)) if c_scaled_scores else 0.0
            )
            results.append((c_fitness, c_scaled_scores, c_states))

        return results
