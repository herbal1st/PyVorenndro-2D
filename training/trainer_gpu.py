"""
GPU-accelerated (PyTorch) single-agent trainer.

Simulates all candidates and mazes as batched CUDA tensors in GPU VRAM,
synchronizing sensory, perception, penalty, and scaling parity with CPU.
"""

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from entities.player_state import PlayerState
from training.agent import Agent
from training.base_trainer import BaseTrainer
from training.fitness import FitnessEvaluator
from training.map_gen_worker import _generate_run_map_worker

try:
    import torch
    import torch.nn as nn

    _TORCH_OK: bool = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_OK = False


def torch_available() -> bool:
    """Returns True only when PyTorch is importable and CUDA is usable."""
    if not _TORCH_OK or torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_FUSED_VISION_COMPILED: Any = None


def _sample_vision_fused(
    sim: "TorchSimulator",
    xs: torch.Tensor,
    ys: torch.Tensor,
    headings: torch.Tensor,
    wall_bool: torch.Tensor,
) -> torch.Tensor:
    """Breakless DDA variant of TorchSimulator.sample_vision."""
    n_cands: int = int(xs.shape[0])
    if n_cands == 0:
        return torch.zeros(
            (0, sim.num_rays), dtype=sim.dtype, device=sim.device
        )

    h: int = int(wall_bool.shape[1])
    w: int = int(wall_bool.shape[2])
    n_rays: int = sim.num_rays
    n_total: int = n_cands * n_rays

    angles: torch.Tensor = headings[:, None] + sim.offsets[None, :]
    dir_x: torch.Tensor = torch.cos(angles).reshape(-1)
    dir_y: torch.Tensor = torch.sin(angles).reshape(-1)

    ox: torch.Tensor = xs.repeat_interleave(n_rays)
    oy: torch.Tensor = ys.repeat_interleave(n_rays)
    run_idx: torch.Tensor = torch.arange(
        n_cands, device=sim.device
    ).repeat_interleave(n_rays)

    tile_x: torch.Tensor = torch.floor(ox).to(torch.int64)
    tile_y: torch.Tensor = torch.floor(oy).to(torch.int64)

    prox: torch.Tensor = torch.zeros(
        n_total, dtype=sim.dtype, device=sim.device
    )
    done: torch.Tensor = torch.zeros(
        n_total, dtype=torch.bool, device=sim.device
    )

    inb: torch.Tensor = (
        (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
    )
    txc: torch.Tensor = tile_x.clamp(0, w - 1)
    tyc: torch.Tensor = tile_y.clamp(0, h - 1)
    is_wall: torch.Tensor = wall_bool[run_idx, tyc, txc]
    inside_wall: torch.Tensor = (inb & is_wall).bool()
    prox = torch.where(inside_wall, torch.ones_like(prox), prox)
    done = done | inside_wall

    step_x: torch.Tensor = torch.where(
        dir_x > 0.0,
        torch.tensor(1, dtype=torch.int64, device=sim.device),
        torch.where(
            dir_x < 0.0,
            torch.tensor(-1, dtype=torch.int64, device=sim.device),
            torch.tensor(0, dtype=torch.int64, device=sim.device),
        ),
    )
    step_y: torch.Tensor = torch.where(
        dir_y > 0.0,
        torch.tensor(1, dtype=torch.int64, device=sim.device),
        torch.where(
            dir_y < 0.0,
            torch.tensor(-1, dtype=torch.int64, device=sim.device),
            torch.tensor(0, dtype=torch.int64, device=sim.device),
        ),
    )

    inf: torch.Tensor = torch.tensor(
        float("inf"), dtype=sim.dtype, device=sim.device
    )
    one: torch.Tensor = torch.tensor(
        1.0, dtype=sim.dtype, device=sim.device
    )
    zero: torch.Tensor = torch.tensor(
        0.0, dtype=sim.dtype, device=sim.device
    )

    t_delta_x: torch.Tensor = torch.where(
        dir_x.abs() > 1e-9, (one / dir_x).abs(), inf
    )
    t_delta_y: torch.Tensor = torch.where(
        dir_y.abs() > 1e-9, (one / dir_y).abs(), inf
    )
    t_max_x: torch.Tensor = torch.where(
        step_x > 0,
        (tile_x.to(sim.dtype) + one - ox) / dir_x,
        torch.where(
            step_x < 0,
            (tile_x.to(sim.dtype) - ox) / dir_x,
            inf,
        ),
    )
    t_max_y: torch.Tensor = torch.where(
        step_y > 0,
        (tile_y.to(sim.dtype) + one - oy) / dir_y,
        torch.where(
            step_y < 0,
            (tile_y.to(sim.dtype) - oy) / dir_y,
            inf,
        ),
    )

    max_t: float = sim.max_dist
    max_iter: int = int(2.0 * max_t) + 4

    for _ in range(max_iter):
        active: torch.Tensor = ~done
        move_x: torch.Tensor = t_max_x < t_max_y
        hit_t: torch.Tensor = torch.where(move_x, t_max_x, t_max_y)
        tile_x = torch.where(move_x, tile_x + step_x, tile_x)
        tile_y = torch.where(move_x, tile_y, tile_y + step_y)
        t_max_x = torch.where(move_x, t_max_x + t_delta_x, t_max_x)
        t_max_y = torch.where(move_x, t_max_y, t_max_y + t_delta_y)

        no_hit: torch.Tensor = hit_t > max_t
        prox = torch.where(active & no_hit, zero, prox)
        done = done | (active & no_hit)

        inb = (
            (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
        )
        txc = tile_x.clamp(0, w - 1)
        tyc = tile_y.clamp(0, h - 1)
        is_wall = wall_bool[run_idx, tyc, txc]
        wall_hit: torch.Tensor = (inb & is_wall).bool()
        blocked: torch.Tensor = (
            active & ~no_hit & (wall_hit | ~inb)
        )
        prox = torch.where(
            blocked,
            torch.clamp(one - (hit_t / max_t), 0.0, 1.0),
            prox,
        )
        done = done | blocked

    prox = torch.where(done, prox, zero)
    return prox.reshape(n_cands, n_rays)


def _compiled_sample_vision() -> Any:
    """Lazily compiles the fused breakless DDA once."""
    global _FUSED_VISION_COMPILED

    if _FUSED_VISION_COMPILED is None and _TORCH_OK:
        try:
            _FUSED_VISION_COMPILED = torch.compile(_sample_vision_fused)
        except Exception:
            _FUSED_VISION_COMPILED = False

    return _FUSED_VISION_COMPILED or None


class TorchAgentModule(nn.Module):
    """MLP mirroring Agent as PyTorch parameters for GPU execution."""

    def __init__(self, sizes: List[int]) -> None:
        super().__init__()
        self.sizes: List[int] = list(sizes)
        self.linears: nn.ModuleList = nn.ModuleList()
        for fin, fout in zip(self.sizes[:-1], self.sizes[1:]):
            self.linears.append(nn.Linear(fin, fout, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_layers: int = len(self.linears) - 1
        h: torch.Tensor = x
        for idx, linear in enumerate(self.linears):
            z: torch.Tensor = linear(h)
            if idx < hidden_layers:
                h = torch.relu(z)
            else:
                move_eff: torch.Tensor = torch.sigmoid(z[:, 0:1])
                turn_eff: torch.Tensor = torch.tanh(z[:, 1:2])
                h = torch.cat([move_eff, turn_eff], dim=1)
        return h

    def load_from_agent(self, agent: Agent) -> "TorchAgentModule":
        """Copies numpy genome arrays into PyTorch layer parameters."""
        with torch.no_grad():
            for linear, w, b in zip(
                self.linears, agent.weights, agent.biases
            ):
                linear.weight.copy_(
                    torch.as_tensor(
                        np.asarray(w).T,
                        dtype=torch.float32,
                        device=linear.weight.device
                    )
                )
                linear.bias.copy_(
                    torch.as_tensor(
                        np.asarray(b).squeeze(),
                        dtype=torch.float32,
                        device=linear.bias.device
                    )
                )
        return self


class TorchSimulator:
    """Vectorized step-by-step maze simulation on CUDA/PyTorch tensors."""

    def __init__(
        self,
        device: torch.device,
        profile_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.device: torch.device = device
        self.dtype: torch.dtype = torch.float32
        self.two_pi: float = 2.0 * math.pi

        if profile_config is None:
            profile_config = {}

        sensory: Dict[str, Any] = profile_config.get("sensory", {})
        kinematics: Dict[str, Any] = profile_config.get("kinematics", {})
        metabolics: Dict[str, Any] = profile_config.get("metabolics", {})

        self.num_rays: int = int(sensory.get("vision_rays", 9))
        self.max_dist: float = 5.0
        self.arc_angle: float = float(sensory.get("vision_arc_angle", 120.0))
        self.include_compass: bool = bool(sensory.get("include_compass", False))
        self.include_bfs: bool = bool(sensory.get("include_bfs_sensor", False))

        half_arc: float = math.radians(self.arc_angle / 2.0)
        if self.num_rays > 1:
            step: float = (2.0 * half_arc) / float(self.num_rays - 1)
            offsets: List[float] = [
                -half_arc + (i * step) for i in range(self.num_rays)
            ]
        else:
            offsets = [0.0]

        self.offsets: torch.Tensor = torch.as_tensor(
            offsets, dtype=self.dtype, device=self.device
        )

        turn_speed: float = float(
            kinematics.get("turn_speed_dpsec", 1080.0)
        )
        self.rad_per_frame: float = (
            math.radians(turn_speed) / float(config.FPS)
        )
        self.move_speed: float = float(
            kinematics.get("move_speed", 0.1)
        )
        self.radius: float = 0.5 * float(
            kinematics.get("player_radius_ratio", 0.5)
        )

        self.coll_damage: float = float(
            metabolics.get("collision_damage", 0.003)
        )
        self.idle_damage: float = float(
            metabolics.get("idle_damage", 0.001)
        )
        self.recovery_ratio: float = float(
            metabolics.get("recovery_ratio", 0.05)
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
        self.stag_timeout_ratio: float = float(
            metabolics.get("stagnation_timeout_ratio", 0.75)
        )
        self.stag_dmg: float = float(
            metabolics.get("stagnation_damage_per_frame", 0.001)
        )

    def sample_vision(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        headings: torch.Tensor,
        wall_bool: torch.Tensor
    ) -> torch.Tensor:
        """Batched Amanatides-Woo DDA across all (N,) origins at once."""
        if getattr(self.device, "type", "cpu") == "cuda":
            fused = _compiled_sample_vision()
            if fused is not None:
                try:
                    return fused(self, xs, ys, headings, wall_bool)
                except Exception:
                    pass
        return self._sample_vision_eager(xs, ys, headings, wall_bool)

    def _sample_vision_eager(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        headings: torch.Tensor,
        wall_bool: torch.Tensor
    ) -> torch.Tensor:
        """Eager (non-compiled) DDA fallback."""
        n_cands: int = int(xs.shape[0])
        if n_cands == 0:
            return torch.zeros(
                (0, self.num_rays), dtype=self.dtype, device=self.device
            )

        h: int = int(wall_bool.shape[1])
        w: int = int(wall_bool.shape[2])
        n_rays: int = self.num_rays
        n_total: int = n_cands * n_rays

        angles: torch.Tensor = (
            headings[:, None] + self.offsets[None, :]
        )
        dir_x: torch.Tensor = torch.cos(angles).reshape(-1)
        dir_y: torch.Tensor = torch.sin(angles).reshape(-1)

        ox: torch.Tensor = xs.repeat_interleave(n_rays)
        oy: torch.Tensor = ys.repeat_interleave(n_rays)
        run_idx: torch.Tensor = torch.arange(
            n_cands, device=self.device
        ).repeat_interleave(n_rays)

        tile_x: torch.Tensor = torch.floor(ox).to(torch.int64)
        tile_y: torch.Tensor = torch.floor(oy).to(torch.int64)

        prox: torch.Tensor = torch.zeros(
            n_total, dtype=self.dtype, device=self.device
        )
        done: torch.Tensor = torch.zeros(
            n_total, dtype=torch.bool, device=self.device
        )

        inb: torch.Tensor = (
            (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
        )
        txc: torch.Tensor = tile_x.clamp(0, w - 1)
        tyc: torch.Tensor = tile_y.clamp(0, h - 1)
        is_wall: torch.Tensor = wall_bool[run_idx, tyc, txc]
        inside_wall: torch.Tensor = inb & is_wall
        prox = torch.where(inside_wall, torch.ones_like(prox), prox)
        done = done | inside_wall

        step_x: torch.Tensor = torch.where(
            dir_x > 0.0,
            torch.tensor(1, dtype=torch.int64, device=self.device),
            torch.where(
                dir_x < 0.0,
                torch.tensor(-1, dtype=torch.int64, device=self.device),
                torch.tensor(0, dtype=torch.int64, device=self.device),
            ),
        )
        step_y: torch.Tensor = torch.where(
            dir_y > 0.0,
            torch.tensor(1, dtype=torch.int64, device=self.device),
            torch.where(
                dir_y < 0.0,
                torch.tensor(-1, dtype=torch.int64, device=self.device),
                torch.tensor(0, dtype=torch.int64, device=self.device),
            ),
        )

        inf: torch.Tensor = torch.tensor(
            float("inf"), dtype=self.dtype, device=self.device
        )
        one: torch.Tensor = torch.tensor(
            1.0, dtype=self.dtype, device=self.device
        )
        zero: torch.Tensor = torch.tensor(
            0.0, dtype=self.dtype, device=self.device
        )

        t_delta_x: torch.Tensor = torch.where(
            dir_x.abs() > 1e-9, (one / dir_x).abs(), inf
        )
        t_delta_y: torch.Tensor = torch.where(
            dir_y.abs() > 1e-9, (one / dir_y).abs(), inf
        )
        t_max_x: torch.Tensor = torch.where(
            step_x > 0,
            (tile_x.to(self.dtype) + one - ox) / dir_x,
            torch.where(
                step_x < 0,
                (tile_x.to(self.dtype) - ox) / dir_x,
                inf,
            ),
        )
        t_max_y: torch.Tensor = torch.where(
            step_y > 0,
            (tile_y.to(self.dtype) + one - oy) / dir_y,
            torch.where(
                step_y < 0,
                (tile_y.to(self.dtype) - oy) / dir_y,
                inf,
            ),
        )

        max_t: float = self.max_dist
        max_iter: int = int(2.0 * max_t) + 4

        for _ in range(max_iter):
            if bool(done.all()):
                break

            active: torch.Tensor = ~done
            move_x: torch.Tensor = t_max_x < t_max_y
            hit_t: torch.Tensor = torch.where(move_x, t_max_x, t_max_y)
            tile_x = torch.where(move_x, tile_x + step_x, tile_x)
            tile_y = torch.where(move_x, tile_y, tile_y + step_y)
            t_max_x = torch.where(move_x, t_max_x + t_delta_x, t_max_x)
            t_max_y = torch.where(move_x, t_max_y, t_max_y + t_delta_y)

            no_hit: torch.Tensor = hit_t > max_t
            prox = torch.where(active & no_hit, zero, prox)
            done = done | (active & no_hit)

            inb = (
                (tile_x >= 0) & (tile_x < w) & (tile_y >= 0) & (tile_y < h)
            )
            txc = tile_x.clamp(0, w - 1)
            tyc = tile_y.clamp(0, h - 1)
            is_wall = wall_bool[run_idx, tyc, txc]
            wall_hit: torch.Tensor = inb & is_wall
            blocked: torch.Tensor = (
                active & ~no_hit & (wall_hit | ~inb)
            )
            prox = torch.where(
                blocked,
                torch.clamp(one - (hit_t / max_t), 0.0, 1.0),
                prox,
            )
            done = done | blocked

        prox = torch.where(done, prox, zero)
        return prox.reshape(n_cands, n_rays)

    def compass(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        headings: torch.Tensor,
        exit_pos: torch.Tensor
    ) -> torch.Tensor:
        """Stereo binocular target compass."""
        ex: torch.Tensor = exit_pos[:, 0] + 0.5
        ey: torch.Tensor = exit_pos[:, 1] + 0.5

        dx: torch.Tensor = ex - xs
        dy: torch.Tensor = ey - ys

        target_angle: torch.Tensor = torch.atan2(dy, dx)
        angle_delta: torch.Tensor = (target_angle - headings) % self.two_pi
        angle_delta = torch.where(
            angle_delta > math.pi, angle_delta - self.two_pi, angle_delta
        )

        tg_left: torch.Tensor = torch.where(
            (angle_delta >= -math.pi) & (angle_delta <= 0.0),
            1.0 - (torch.abs(angle_delta) / math.pi),
            torch.zeros_like(angle_delta),
        )
        tg_right: torch.Tensor = torch.where(
            (angle_delta >= 0.0) & (angle_delta <= math.pi),
            1.0 - (angle_delta / math.pi),
            torch.zeros_like(angle_delta),
        )

        return torch.stack(
            [
                tg_left.clamp(0.0, 1.0),
                tg_right.clamp(0.0, 1.0),
            ],
            dim=1,
        )

    def kinematics_step(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        headings: torch.Tensor,
        move_effort: torch.Tensor,
        turn_effort: torch.Tensor,
        wall_grids: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Vectorized tank rotation, step, & Circle-to-AABB pushback."""
        n: int = int(xs.shape[0])
        h: int = int(wall_grids.shape[1])
        w: int = int(wall_grids.shape[2])
        runs: torch.Tensor = torch.arange(n, device=self.device)
        r: float = self.radius

        move_effort = move_effort.reshape(-1)
        turn_effort = turn_effort.reshape(-1)

        clamped_turn: torch.Tensor = turn_effort.clamp(-1.0, 1.0)
        clamped_move: torch.Tensor = move_effort.clamp(0.0, 1.0)

        new_headings: torch.Tensor = (
            headings + (clamped_turn * self.rad_per_frame)
        ) % self.two_pi
        is_stationary_turn: torch.Tensor = (
            (torch.abs(clamped_turn) > 0.05) & (clamped_move < 0.05)
        )

        no_move: torch.Tensor = clamped_move < 1e-4
        step_dist: torch.Tensor = clamped_move * self.move_speed
        px: torch.Tensor = xs + (torch.cos(new_headings) * step_dist)
        py: torch.Tensor = ys + (torch.sin(new_headings) * step_dist)
        px = torch.where(no_move, xs, px)
        py = torch.where(no_move, ys, py)

        hit: torch.Tensor = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )

        dx_offs: torch.Tensor = torch.tensor(
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            dtype=torch.int64,
            device=self.device
        )
        dy_offs: torch.Tensor = torch.tensor(
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            dtype=torch.int64,
            device=self.device
        )

        for _ in range(2):
            base_tx: torch.Tensor = torch.floor(px).to(torch.int64)
            base_ty: torch.Tensor = torch.floor(py).to(torch.int64)

            tx: torch.Tensor = base_tx[:, None] + dx_offs[None, :]
            ty: torch.Tensor = base_ty[:, None] + dy_offs[None, :]

            inb: torch.Tensor = (
                (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
            )
            txc: torch.Tensor = tx.clamp(0, w - 1)
            tyc: torch.Tensor = ty.clamp(0, h - 1)
            is_wall: torch.Tensor = (
                inb & wall_grids[runs[:, None], tyc, txc].to(torch.bool)
            )

            tx_f: torch.Tensor = tx.to(self.dtype)
            ty_f: torch.Tensor = ty.to(self.dtype)

            cx: torch.Tensor = px[:, None].clamp(tx_f, tx_f + 1.0)
            cy: torch.Tensor = py[:, None].clamp(ty_f, ty_f + 1.0)

            ddx: torch.Tensor = px[:, None] - cx
            ddy: torch.Tensor = py[:, None] - cy
            dist_sq: torch.Tensor = (ddx * ddx) + (ddy * ddy)

            pen: torch.Tensor = is_wall & (dist_sq < (r * r))
            hit = hit | pen.any(dim=1)

            dist: torch.Tensor = torch.sqrt(dist_sq)
            overlap: torch.Tensor = r - dist
            zero: torch.Tensor = pen & (dist_sq < 1e-12)

            nrm_x: torch.Tensor = torch.where(
                dist > 1e-6, ddx / dist, torch.zeros_like(ddx)
            )
            nrm_y: torch.Tensor = torch.where(
                dist > 1e-6, ddy / dist, torch.zeros_like(ddy)
            )

            push_x: torch.Tensor = torch.where(
                pen, nrm_x * overlap, torch.zeros_like(nrm_x)
            )
            push_y: torch.Tensor = torch.where(
                pen, nrm_y * overlap, torch.zeros_like(nrm_y)
            )

            px = px + push_x.sum(dim=1)
            py = py + push_y.sum(dim=1)

            nudged: torch.Tensor = zero.any(dim=1)
            px = torch.where(nudged, px + 0.01, px)
            py = torch.where(nudged, py + 0.01, py)

        px = torch.where(no_move, xs, px)
        py = torch.where(no_move, ys, py)
        hit = torch.where(no_move, torch.zeros_like(hit), hit)

        return px, py, new_headings, hit, is_stationary_turn

    def simulate_candidates(
        self,
        modules: List[TorchAgentModule],
        wall_grids: torch.Tensor,
        dist_grid: torch.Tensor,
        exit_pos: torch.Tensor,
        start_x: torch.Tensor,
        start_y: torch.Tensor,
        spawn_headings: torch.Tensor,
        initial_bfs_dists: torch.Tensor,
        max_steps: int
    ) -> List[List[PlayerState]]:
        """Runs candidate genomes in ONE batched CUDA tensor matrix job."""
        c_count: int = len(modules)
        n: int = int(wall_grids.shape[0])
        r_total: int = c_count * n

        wall_grids_c: torch.Tensor = wall_grids.repeat(c_count, 1, 1)
        dist_grid_c: torch.Tensor = dist_grid.repeat(c_count, 1, 1)
        exit_pos_c: torch.Tensor = exit_pos.repeat(c_count, 1)
        start_x_c: torch.Tensor = start_x.repeat(c_count)
        start_y_c: torch.Tensor = start_y.repeat(c_count)
        spawn_headings_c: torch.Tensor = spawn_headings.repeat(c_count)
        initial_bfs_dists_c: torch.Tensor = initial_bfs_dists.repeat(c_count)

        stack_w, stack_b = _stack_modules(modules)
        hidden_layers: int = len(stack_w) - 1

        def forward(
            features: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            h: torch.Tensor = features
            for layer_idx, (w_l, b_l) in enumerate(
                zip(stack_w, stack_b)
            ):
                hc: torch.Tensor = h.view(c_count, n, -1)
                z: torch.Tensor = torch.bmm(hc, w_l)
                z = z + b_l.unsqueeze(1)
                z = z.reshape(r_total, -1)
                if layer_idx < hidden_layers:
                    h = torch.relu(z)
                else:
                    move_eff: torch.Tensor = torch.sigmoid(z[:, 0:1])
                    turn_eff: torch.Tensor = torch.tanh(z[:, 1:2])
                    h = torch.cat([move_eff, turn_eff], dim=1)
            return h[:, 0:1], h[:, 1:2]

        rows = self._simulate_rows(
            forward,
            wall_grids_c,
            dist_grid_c,
            exit_pos_c,
            start_x_c,
            start_y_c,
            spawn_headings_c,
            initial_bfs_dists_c,
            max_steps
        )
        all_states: List[PlayerState] = _build_states(rows, 0, r_total)

        return [
            all_states[c_idx * n:(c_idx + 1) * n]
            for c_idx in range(c_count)
        ]

    def _simulate_rows(
        self,
        forward: Any,
        wall_grids: torch.Tensor,
        dist_grid: torch.Tensor,
        exit_pos: torch.Tensor,
        start_x: torch.Tensor,
        start_y: torch.Tensor,
        spawn_headings: torch.Tensor,
        initial_bfs_dists: torch.Tensor,
        max_steps: int
    ) -> Tuple[List[Any], ...]:
        """Shared step-by-step GPU simulation over (R,) rows."""
        n: int = int(wall_grids.shape[0])
        h: int = int(wall_grids.shape[1])
        w: int = int(wall_grids.shape[2])
        max_span: float = float(h + w)
        move_speed: float = self.move_speed
        stag_limit: int = int(max_steps * self.stag_timeout_ratio)

        wall_bool: torch.Tensor = wall_grids.to(torch.bool)

        xs: torch.Tensor = start_x.to(self.dtype).clone()
        ys: torch.Tensor = start_y.to(self.dtype).clone()
        headings: torch.Tensor = spawn_headings.to(self.dtype).clone()
        healths: torch.Tensor = torch.ones(
            n, dtype=self.dtype, device=self.device
        )
        best_dists: torch.Tensor = initial_bfs_dists.to(self.dtype).clone()
        cum_rotation: torch.Tensor = torch.zeros(
            n, dtype=self.dtype, device=self.device
        )
        straight_ticks: torch.Tensor = torch.zeros(
            n, dtype=torch.int64, device=self.device
        )
        stagnation_ticks: torch.Tensor = torch.zeros(
            n, dtype=torch.int64, device=self.device
        )

        frames: torch.Tensor = torch.zeros(
            n, dtype=torch.int64, device=self.device
        )
        alive: torch.Tensor = torch.ones(
            n, dtype=torch.bool, device=self.device
        )
        reached: torch.Tensor = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        last_hit: torch.Tensor = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        is_idle: torch.Tensor = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        is_spinning: torch.Tensor = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )

        inf_i: torch.Tensor = torch.tensor(
            9999, dtype=torch.int64, device=self.device
        )
        one_f: torch.Tensor = torch.tensor(
            1.0, dtype=self.dtype, device=self.device
        )
        zero_f: torch.Tensor = torch.tensor(
            0.0, dtype=self.dtype, device=self.device
        )
        zero_i: torch.Tensor = torch.tensor(
            0, dtype=torch.int64, device=self.device
        )

        rows: torch.Tensor = torch.arange(n, device=self.device)

        for step in range(1, max_steps + 1):
            eff: torch.Tensor = alive & ~reached

            if self.include_bfs:
                sx: torch.Tensor = torch.floor(xs).to(torch.int64)
                sy: torch.Tensor = torch.floor(ys).to(torch.int64)
                in_s: torch.Tensor = (
                    (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
                )
                sxc: torch.Tensor = sx.clamp(0, w - 1)
                syc: torch.Tensor = sy.clamp(0, h - 1)
                sensor_dist: torch.Tensor = torch.where(
                    in_s,
                    dist_grid[rows, syc, sxc],
                    inf_i,
                ).to(self.dtype)
            else:
                sensor_dist = None

            wall_ch: torch.Tensor = self.sample_vision(
                xs, ys, headings, wall_bool
            )
            speed_ch: torch.Tensor = torch.full(
                (n, 1), move_speed, dtype=self.dtype, device=self.device
            )
            health_ch: torch.Tensor = healths[:, None]

            features: torch.Tensor = torch.cat(
                [wall_ch, speed_ch, health_ch], dim=1
            )

            if self.include_compass:
                comp_ch: torch.Tensor = self.compass(
                    xs, ys, headings, exit_pos
                )
                features = torch.cat(
                    [wall_ch, comp_ch, speed_ch, health_ch], dim=1
                )

            if self.include_bfs and sensor_dist is not None:
                dist_ch: torch.Tensor = (
                    sensor_dist / max_span
                ).clamp(0.0, 1.0)[:, None]
                features = torch.cat([features, dist_ch], dim=1)

            # Add dedicated penalty channels (HIT, IDL, SPN)
            hit_ch: torch.Tensor = last_hit.to(self.dtype)[:, None]
            idle_ch: torch.Tensor = is_idle.to(self.dtype)[:, None]
            spin_ch: torch.Tensor = is_spinning.to(self.dtype)[:, None]
            features = torch.cat(
                [features, hit_ch, idle_ch, spin_ch], dim=1
            )

            move_eff, turn_eff = forward(features)
            move_eff = move_eff * eff.unsqueeze(1).to(self.dtype)
            turn_eff = turn_eff * eff.unsqueeze(1).to(self.dtype)

            (
                px,
                py,
                new_headings,
                hit,
                is_stationary,
            ) = self.kinematics_step(
                xs,
                ys,
                headings,
                move_eff,
                turn_eff,
                wall_bool,
            )

            heading_delta: torch.Tensor = torch.abs(new_headings - headings)
            heading_delta = torch.where(
                heading_delta > math.pi,
                self.two_pi - heading_delta,
                heading_delta
            )

            is_turning: torch.Tensor = heading_delta >= self.spin_reset_rad
            straight_ticks = torch.where(
                is_turning,
                zero_i,
                straight_ticks + eff.long()
            )
            cum_rotation = torch.where(
                is_turning,
                cum_rotation + heading_delta,
                cum_rotation
            )

            should_reset_spin: torch.Tensor = (
                straight_ticks >= self.spin_hold_frames
            )
            cum_rotation = torch.where(
                should_reset_spin,
                zero_f,
                cum_rotation
            )

            stagnation_ticks = stagnation_ticks + eff.long()

            hlt: torch.Tensor = healths - (
                hit.to(self.dtype) * self.coll_damage
            )

            not_moved: torch.Tensor = (
                (torch.abs(px - xs) < 1e-4)
                & (torch.abs(py - ys) < 1e-4)
            )
            is_idle = (
                (move_eff.squeeze(1) < 0.05)
                | not_moved
                | is_stationary
            )
            hlt = hlt - (
                is_idle.to(self.dtype)
                * eff.to(self.dtype)
                * self.idle_damage
            )

            if self.spin_enabled:
                is_spinning = (
                    cum_rotation >= self.spin_thresh_rad
                )
                hlt = hlt - (
                    is_spinning.to(self.dtype)
                    * eff.to(self.dtype)
                    * self.spin_dmg
                )

            if self.stag_enabled:
                is_stagnated: torch.Tensor = (
                    stagnation_ticks >= stag_limit
                )
                hlt = hlt - (
                    is_stagnated.to(self.dtype)
                    * eff.to(self.dtype)
                    * self.stag_dmg
                )

            hlt = hlt.clamp(min=0.0)

            tx: torch.Tensor = torch.floor(px).to(torch.int64)
            ty: torch.Tensor = torch.floor(py).to(torch.int64)

            inb: torch.Tensor = (
                (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
            )
            txc: torch.Tensor = tx.clamp(0, w - 1)
            tyc: torch.Tensor = ty.clamp(0, h - 1)

            curr_dist: torch.Tensor = torch.where(
                inb,
                dist_grid[rows, tyc, txc],
                inf_i,
            ).to(self.dtype)

            better: torch.Tensor = curr_dist < best_dists
            heal: torch.Tensor = (
                (best_dists - curr_dist)
                * self.coll_damage
                * self.recovery_ratio
            )
            hlt = torch.where(
                better,
                torch.minimum(one_f, hlt + heal),
                hlt,
            )
            best_dists = torch.where(
                better, curr_dist, best_dists
            )

            stagnation_ticks = torch.where(
                better, zero_i, stagnation_ticks
            )
            cum_rotation = torch.where(
                better, zero_f, cum_rotation
            )
            straight_ticks = torch.where(
                better, zero_i, straight_ticks
            )

            ex: torch.Tensor = exit_pos[:, 0].to(torch.int64)
            ey: torch.Tensor = exit_pos[:, 1].to(torch.int64)
            reached_run: torch.Tensor = (
                inb & (tx == ex) & (ty == ey)
            )

            xs = px
            ys = py
            headings = new_headings
            healths = hlt
            alive = hlt > 0.0
            reached = reached | reached_run
            last_hit = hit
            frames = frames + eff.long()

            if (
                step % 128 == 0
                and not bool((alive & ~reached).any())
            ):
                break

        return (
            xs.tolist(),
            ys.tolist(),
            headings.tolist(),
            healths.tolist(),
            alive.tolist(),
            reached.tolist(),
            last_hit.tolist(),
            best_dists.tolist(),
            frames.tolist(),
            start_x.tolist(),
            start_y.tolist(),
        )


def _stack_modules(
    modules: List[TorchAgentModule]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Stacks per-candidate layer weights into batched GPU tensors."""
    layer_count: int = len(modules[0].linears)

    stack_w: List[torch.Tensor] = []
    stack_b: List[torch.Tensor] = []

    for layer_idx in range(layer_count):
        weights = [
            modules[c].linears[layer_idx].weight for c in range(len(modules))
        ]
        biases = [
            modules[c].linears[layer_idx].bias for c in range(len(modules))
        ]
        stack_w.append(torch.stack(weights).transpose(1, 2))
        stack_b.append(torch.stack(biases))

    return stack_w, stack_b


def _build_states(
    rows: Tuple[List[Any], ...],
    offset: int,
    count: int
) -> List[PlayerState]:
    """Rebuilds PlayerState objects from raw per-row CPU lists."""
    (
        xs_cpu,
        ys_cpu,
        headings_cpu,
        healths_cpu,
        alive_cpu,
        reached_cpu,
        last_hit_cpu,
        best_dists_cpu,
        frames_cpu,
        start_x_cpu,
        start_y_cpu,
    ) = rows

    states: List[PlayerState] = []
    for i in range(offset, offset + count):
        state: PlayerState = PlayerState(
            float(start_x_cpu[i]) - 0.5,
            float(start_y_cpu[i]) - 0.5
        )
        state.x = xs_cpu[i]
        state.y = ys_cpu[i]
        state.heading = headings_cpu[i]
        state.health = healths_cpu[i]
        state.has_collided = bool(last_hit_cpu[i])
        state.is_alive = alive_cpu[i]
        state.has_reached_exit = reached_cpu[i]
        state.best_step_dist = int(best_dists_cpu[i])
        state.frames_survived = int(frames_cpu[i])
        states.append(state)

    return states


class GpuHeadlessTrainer(BaseTrainer):
    """GPU single-agent trainer running batched CUDA tensor jobs."""

    def __init__(
        self,
        num_runs: int = 0,
        max_steps: int = config.MAX_SIMULATION_STEPS,
        state: Optional[Dict[str, Any]] = None,
        stop_event: Any = None
    ) -> None:
        self.population_size = max(
            2, int(getattr(config, "POPULATION_SIZE", 16))
        )
        self.shared_runs = max(
            1, int(getattr(config, "MAPS_PER_CANDIDATE", 8))
        )

        calculated_runs: int = self.population_size * self.shared_runs
        if num_runs <= 0:
            num_runs = calculated_runs
            explicit_runs: bool = False
        else:
            explicit_runs = True

        super().__init__(
            num_runs=num_runs,
            max_steps=max_steps,
            state=state,
            stop_event=stop_event,
            auto_tune=False
        )

        self._explicit_num_runs = explicit_runs
        self.simulation_workers = 1
        self._backend_label = "GPU"
        self._map_pool: Optional[ProcessPoolExecutor] = None
        self._map_workers: int = max(1, (os.cpu_count() or 1) - 1)
        self._map_pool_context = "spawn"
        self._prefetch_futures: Optional[List[Any]] = None

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._simulator: TorchSimulator = TorchSimulator(
            self.device, profile_config=self.profile_config
        )

        from training import autotune
        autotune.apply_auto_tuning(self)

    def _generate_run_maps(self, count: int) -> List[Tuple]:
        """Builds shared map set across available CPU worker cores."""
        if count <= 0:
            return []

        if self._map_workers <= 1 or count < self._map_workers * 2:
            return [self._generate_run_map() for _ in range(count)]

        if self._map_pool is None:
            import multiprocessing

            mp_context = multiprocessing.get_context(
                self._map_pool_context
            )
            self._map_pool = ProcessPoolExecutor(
                max_workers=self._map_workers,
                mp_context=mp_context,
                initializer=_initialize_worker_priority
            )

        if (
            self._prefetch_futures is not None
            and len(self._prefetch_futures) == count
        ):
            runs: List[Tuple] = [
                future.result() for future in self._prefetch_futures
            ]
            self._prefetch_futures = None
        else:
            if self._prefetch_futures is not None:
                for future in self._prefetch_futures:
                    future.cancel()
                self._prefetch_futures = None
            runs = [
                future.result()
                for future in self._submit_map_batch(count)
            ]

        self._prefetch_futures = self._submit_map_batch(count)
        return runs

    def _submit_map_batch(self, count: int) -> List[Any]:
        """Submits count map-generation tasks with independent seeds."""
        seeds: List[int] = [
            random.getrandbits(64) for _ in range(count)
        ]
        return [
            self._map_pool.submit(
                _generate_run_map_worker,
                seed,
                self.max_steps,
            )
            for seed in seeds
        ]

    def _shutdown_process_pool(self) -> None:
        """Shuts down parallel map pool and prefetch futures."""
        super()._shutdown_process_pool()

        if self._prefetch_futures is not None:
            for future in self._prefetch_futures:
                future.cancel()
            self._prefetch_futures = None

        if self._map_pool is not None:
            self._map_pool.shutdown(wait=True)
            self._map_pool = None

    def _simulate_candidates(
        self,
        agents: List[Agent],
        runs: List[Tuple]
    ) -> List[Tuple[float, List[float], List[PlayerState]]]:
        """Runs every candidate across the shared map set on CUDA/PyTorch."""
        num_runs: int = len(runs)
        height: int = self._map_height
        width: int = self._map_width

        wall_grids: torch.Tensor = torch.zeros(
            (num_runs, height, width),
            dtype=torch.uint8,
            device=self.device
        )
        dist_grid: torch.Tensor = torch.zeros(
            (num_runs, height, width),
            dtype=torch.int64,
            device=self.device
        )

        for i in range(num_runs):
            map_data = runs[i][0]
            wall_grids[i].copy_(
                torch.as_tensor(
                    map_data.build_wall_grid(),
                    dtype=torch.uint8,
                    device=self.device
                )
            )
            dist_grid[i].copy_(
                torch.as_tensor(
                    np.asarray(runs[i][1], dtype=np.int64),
                    dtype=torch.int64,
                    device=self.device
                )
            )

        exit_pos: torch.Tensor = torch.as_tensor(
            [
                [runs[i][0].exit_pos[0], runs[i][0].exit_pos[1]]
                for i in range(num_runs)
            ],
            dtype=torch.float32,
            device=self.device
        )

        start_x: torch.Tensor = torch.as_tensor(
            [float(runs[i][0].start_pos[0]) + 0.5 for i in range(num_runs)],
            dtype=torch.float64,
            device=self.device
        )
        start_y: torch.Tensor = torch.as_tensor(
            [float(runs[i][0].start_pos[1]) + 0.5 for i in range(num_runs)],
            dtype=torch.float64,
            device=self.device
        )
        spawn_headings: torch.Tensor = torch.as_tensor(
            [runs[i][5] for i in range(num_runs)],
            dtype=torch.float64,
            device=self.device
        )
        initial_bfs_dists: torch.Tensor = torch.as_tensor(
            [runs[i][2] for i in range(num_runs)],
            dtype=torch.int64,
            device=self.device
        )

        all_results: List[Tuple[float, List[float], List[PlayerState]]] = []

        modules: List[TorchAgentModule] = []
        for agent in agents:
            module: TorchAgentModule = (
                TorchAgentModule(agent.sizes)
                .to(self.device)
                .load_from_agent(agent)
            )
            module.eval()
            modules.append(module)

        with torch.no_grad():
            per_candidate_states: List[List[PlayerState]] = (
                self._simulator.simulate_candidates(
                    modules,
                    wall_grids,
                    dist_grid,
                    exit_pos,
                    start_x,
                    start_y,
                    spawn_headings,
                    initial_bfs_dists,
                    self.max_steps
                )
            )

        for candidate_states in per_candidate_states:
            raw_scores: List[float] = [
                FitnessEvaluator.calculate_raw_score(
                    candidate_states[i],
                    runs[i][2],
                    self.max_steps
                )
                for i in range(num_runs)
            ]

            scaled_scores: List[float] = [
                FitnessEvaluator.calculate_scaled_score(
                    raw_scores[i],
                    runs[i][4]
                )
                for i in range(num_runs)
            ]

            fitness: float = float(
                np.mean(scaled_scores)
            ) if scaled_scores else 0.0

            all_results.append(
                (fitness, scaled_scores, candidate_states)
            )

        return all_results
