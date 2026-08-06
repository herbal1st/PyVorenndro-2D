"""
Hardware-aware auto-configuration of training scale settings.

The trainer's search budget (runs per generation), candidate population, and
parallel worker count are all machine-dependent. A value that fits an
integrated GPU or a 4-core laptop is wasteful on a 16 GB discrete GPU with 16
cores. Rather than hard-code guesses, this module measures the actual
throughput of the running machine once at startup: it generates a small probe
map set, simulates one candidate through the real trainer backend, and divides
simulations by wall-clock time. It then chooses a run budget that lands one
generation near ``config.TARGET_GEN_TIME`` seconds, a population size from the
device tier, and (on CPU) a worker count that leaves a core free for the
display runner. All measured values are clamped by
``AUTO_TUNE_MIN_RUNS``/``AUTO_TUNE_MAX_RUNS`` and by the device's VRAM when a
GPU is present.

The calibration runs exactly once per process (guarded by a module flag) and
restores the global RNG state around itself so training starts from the same
seeded sequence whether or not auto-tuning ran.
"""

import os
import random
import time
from typing import Any, Dict

import numpy as np

import config

_TUNED: bool = False


def is_tuned() -> bool:
    """True once this process has auto-tuned (or been told not to)."""
    return _TUNED


def detect_hardware() -> Dict[str, Any]:
    """
    Returns a profile dict describing the machine's compute resources.
    """
    profile: Dict[str, Any] = {
        "backend": "cpu",
        "device_name": "CPU",
        "vram_bytes": 0,
        "cpu_cores": os.cpu_count() or 1,
        "ram_bytes": 0,
    }

    try:
        from training.trainer_gpu import torch_available

        if torch_available():
            import torch

            profile["backend"] = "gpu"
            props: Any = torch.cuda.get_device_properties(0)
            profile["device_name"] = str(props.name)
            profile["vram_bytes"] = int(props.total_memory)
    except Exception:
        pass

    try:
        profile["ram_bytes"] = int(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        )
    except Exception:
        pass

    return profile


def _save_rng() -> tuple:
    return (random.getstate(), np.random.get_state())


def _restore_rng(saved: tuple) -> None:
    random.setstate(saved[0])
    np.random.set_state(saved[1])


def _calibrate_gpu(trainer: Any) -> Tuple[float, float, float]:
    """
    Returns ``(fixed, per_run, per_map)`` seconds for the GPU simulator and the
    trainer's (parallel) map generation. One probe map set is generated once
    through the trainer's own pool (timing real map throughput) and re-used
    across two simulation slices so startup stays small while the per-run
    estimate comes from a representative large-batch interval.
    """
    base: int = int(getattr(config, "AUTO_TUNE_PROBE_RUNS", 256))
    b_hi: int = max(64, min(base * 8, 4096))
    b_lo: int = max(32, min(base * 4, b_hi // 2))
    if b_lo >= b_hi:
        b_lo = max(32, b_hi // 2)

    t0: float = time.perf_counter()
    probe_runs = trainer._generate_run_maps(b_hi)
    gen_elapsed: float = time.perf_counter() - t0
    per_map: float = gen_elapsed / float(b_hi) if b_hi else 0.0

    candidate = trainer.agent.mutate(mutation_scale=0.05)

    t_lo: float = _time_simulate(trainer, candidate, probe_runs[:b_lo])
    t_hi: float = _time_simulate(trainer, candidate, probe_runs)

    denom: float = float(b_hi - b_lo)

    if denom <= 0.0 or t_hi <= 0.0:
        return 0.0, 0.0, per_map

    per_run: float = (t_hi - t_lo) / denom
    fixed: float = t_lo - per_run * b_lo

    if per_run < 0.0:
        per_run = 0.0
    if fixed < 0.0:
        fixed = 0.0

    return fixed, per_run, per_map


def _time_simulate(trainer: Any, candidate: Any, runs: List[Any]) -> float:
    """
    Times one candidate through the GPU simulator over the given map set.
    """
    t0: float = time.perf_counter()
    trainer._simulate_candidates([candidate], runs)
    return time.perf_counter() - t0


def _calibrate_cpu(trainer: Any) -> float:
    """
    Times one candidate through the inline numpy simulator over a probe map
    set on a single worker and returns simulations per second.
    """
    probe_runs: int = int(
        getattr(config, "AUTO_TUNE_PROBE_RUNS_CPU", 32)
    )
    probe_runs = max(8, min(probe_runs, 128))

    runs = [trainer._generate_run_map() for _ in range(probe_runs)]
    candidate = trainer.agent.mutate(mutation_scale=0.05)

    saved_workers: int = trainer.simulation_workers
    trainer.simulation_workers = 1

    t0: float = time.perf_counter()
    trainer._simulate_candidates([candidate], runs)
    elapsed: float = time.perf_counter() - t0

    trainer.simulation_workers = saved_workers

    if elapsed <= 0.0:
        return 0.0
    return probe_runs / elapsed


def _tune_gpu(trainer: Any, profile: Dict[str, Any]) -> None:
    target: float = float(getattr(config, "TARGET_GEN_TIME", 0.75))
    min_runs: int = int(getattr(config, "AUTO_TUNE_MIN_RUNS", 64))
    max_runs: int = int(getattr(config, "AUTO_TUNE_MAX_RUNS", 8192))
    vram: int = int(profile.get("vram_bytes", 0))

    fixed_cost, per_run, per_map = _calibrate_gpu(trainer)

    vram_gb: float = vram / (1024 ** 3)
    if vram_gb >= 16.0:
        pop: int = 16
    elif vram_gb >= 6.0:
        pop = 8
    else:
        pop = 4

    if fixed_cost > 0.0 or per_run > 0.0:
        if fixed_cost >= target:
            runs = max_runs
        else:
            map_workers: int = int(
                getattr(trainer, "_map_workers", 1) or 1
            )
            cost_per_sim: float = (
                per_run + per_map / float(map_workers * pop)
            )
            if cost_per_sim > 0.0:
                runs = int((target - fixed_cost) / cost_per_sim)
            else:
                runs = int(getattr(config, "SIMULATION_RUNS_GPU", 1024))
    else:
        runs = int(getattr(config, "SIMULATION_RUNS_GPU", 1024))

    if vram > 0:
        h: int = trainer._map_height
        w: int = trainer._map_width
        bytes_per_run: int = max(1, h * w * 9)
        runs = min(runs, int(vram // bytes_per_run))

    runs = int(min(max_runs, max(min_runs, runs)))

    explicit_runs: bool = bool(
        getattr(trainer, "_explicit_num_runs", False)
    )
    if not explicit_runs:
        trainer.num_runs = runs
    trainer.population_size = pop
    trainer.shared_runs = max(1, trainer.num_runs // pop)

    config.SIMULATION_RUNS_GPU = int(trainer.num_runs)
    config.POPULATION_SIZE_GPU = pop
    config.SIMULATION_RUNS = int(trainer.num_runs)
    config.POPULATION_SIZE = pop

    est_sims: float = runs / target if target > 0.0 else 0.0

    print(
        f"[Auto-tune] GPU backend: {profile.get('device_name', 'CUDA')} | "
        f"~{est_sims:,.0f} sims/s | runs/gen: {trainer.num_runs} | "
        f"pop: {pop} | maps/gen: {trainer.shared_runs} | "
        f"target: {target:.2f}s/gen"
    )


def _tune_cpu(trainer: Any, profile: Dict[str, Any]) -> None:
    target: float = float(getattr(config, "TARGET_GEN_TIME", 0.75))
    min_runs: int = int(getattr(config, "AUTO_TUNE_MIN_RUNS", 64))
    max_runs: int = int(getattr(config, "AUTO_TUNE_MAX_RUNS", 8192))

    cores: int = int(profile.get("cpu_cores", os.cpu_count() or 1))

    workers: int = max(1, cores - 1)
    workers = min(workers, 16)

    throughput: float = _calibrate_cpu(trainer)

    if throughput > 0.0:
        runs: int = int(throughput * workers * target)
    else:
        runs = int(getattr(config, "SIMULATION_RUNS", 100))

    runs = int(min(max_runs, max(min_runs, runs)))

    pop: int = 2

    explicit_runs: bool = bool(
        getattr(trainer, "_explicit_num_runs", False)
    )

    trainer.simulation_workers = workers
    if not explicit_runs:
        trainer.num_runs = runs
    trainer.population_size = pop
    trainer.shared_runs = max(1, trainer.num_runs // pop)

    config.SIMULATION_WORKERS = workers
    config.SIMULATION_RUNS = int(trainer.num_runs)
    config.POPULATION_SIZE = pop

    print(
        f"[Auto-tune] CPU backend: {cores} cores | "
        f"~{throughput * workers:,.0f} sims/s ({workers} workers) | "
        f"runs/gen: {trainer.num_runs} | pop: {pop} | "
        f"maps/gen: {trainer.shared_runs} | target: {target:.2f}s/gen"
    )


def apply_auto_tuning(trainer: Any) -> None:
    """
    Calibrates this machine once and writes the chosen scale settings into the
    trainer and config. A no-op after the first call or when AUTO_TUNE is off.
    """
    global _TUNED

    if _TUNED:
        return

    if not getattr(config, "AUTO_TUNE", True):
        _TUNED = True
        return

    _TUNED = True

    saved_rng: tuple = _save_rng()
    profile: Dict[str, Any] = detect_hardware()

    try:
        if getattr(trainer, "_backend_label", "CPU") == "GPU":
            _tune_gpu(trainer, profile)
        else:
            _tune_cpu(trainer, profile)
        config.AUTOTUNE_PROFILE = profile
    except Exception as exc:  # pragma: no cover - hardware edge cases
        print(f"[Auto-tune] calibration failed ({exc!r}); using config defaults")
    finally:
        _restore_rng(saved_rng)
