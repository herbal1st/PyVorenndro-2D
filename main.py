"""
Application entry point.

The headless trainer runs in its own subprocess so rendering never slows down
training. This main process runs the live display runner, which repeatedly
asks the trainer for the latest champion model state, re-simulates it on a
random maze at human speed, and keeps replaying the final champion after
training completes until the window closes or Ctrl+C is pressed.
"""
import os

# Mute the Pygame support prompt before any other modules load
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import multiprocessing as mp
from typing import Any, Dict

from training.trainer import run_training_process
from display.runner import DisplayRunner


def main() -> None:
    """
    Spawns the trainer subprocess and runs the live display runner.
    """
    ctx: mp.context.BaseContext = mp.get_context("spawn")

    manager = ctx.Manager()
    state: Dict[str, Any] = manager.dict()
    stop_event = ctx.Event()

    trainer_proc = ctx.Process(
        target=run_training_process,
        args=(state, stop_event),
        name="pyvorenndro-trainer"
    )
    trainer_proc.start()

    runner: DisplayRunner = DisplayRunner(state, stop_event)

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[Display runner interrupted by Ctrl+C]")
    finally:
        print("[Shutting down trainer...]")
        stop_event.set()
        trainer_proc.join(timeout=30)

        if trainer_proc.is_alive():
            trainer_proc.terminate()
            trainer_proc.join(timeout=5)

        manager.shutdown()
        print("[Done.]")


if __name__ == "__main__":
    main()
