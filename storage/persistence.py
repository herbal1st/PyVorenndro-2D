"""
Centralized neural weight persistence and brain library registry.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import yaml
import numpy as np


class WeightStorageManager:
    """
    Manages loading and saving raw neural network weight matrices on disk.
    """

    def __init__(self, brain_dir: Optional[Path] = None) -> None:
        """
        Initializes storage directory and in-memory numpy weight cache.
        """
        if brain_dir is None:
            self.brain_dir: Path = (
                Path(__file__).resolve().parents[1] / "champions"
            )
        else:
            self.brain_dir = brain_dir

        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def get_weights(
        self, filename: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieves raw weight arrays from cache or reads compressed .npz.
        """
        if filename in self._cache:
            return self._cache[filename]

        file_path: Path = self.brain_dir / filename
        if not file_path.exists():
            root_path: Path = (
                Path(__file__).resolve().parents[1] / filename
            )
            if root_path.exists():
                file_path = root_path
            else:
                return None

        try:
            with np.load(file_path) as data:
                loaded: Dict[str, np.ndarray] = {
                    k: np.array(data[k]) for k in data.files
                }
                self._cache[filename] = loaded
                return loaded
        except Exception:
            return None

    def save_weights(
        self,
        filename: str,
        weights: List[np.ndarray],
        biases: List[np.ndarray]
    ) -> Path:
        """
        Persists layer weight and bias arrays into a compressed .npz file.
        """
        target_path: Path = self.brain_dir / filename
        if not target_path.suffix:
            target_path = target_path.with_suffix(".npz")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        archive: Dict[str, np.ndarray] = {}
        for idx, (w, b) in enumerate(zip(weights, biases)):
            archive[f"w{idx}"] = np.asarray(w, dtype=np.float64)
            archive[f"b{idx}"] = np.asarray(b, dtype=np.float64)

        np.savez_compressed(target_path, **archive)
        self._cache[filename] = archive
        return target_path


class BrainLibraryRegistry:
    """
    Parses and resolves data-driven agent profiles from brain_library.yaml.
    """

    def __init__(self, yaml_path: Optional[Path] = None) -> None:
        """
        Initializes registry path and loads YAML definitions into memory.
        """
        if yaml_path is None:
            self.yaml_path: Path = (
                Path(__file__).resolve().parents[1] / "brain_library.yaml"
            )
        else:
            self.yaml_path = yaml_path

        self.library: Dict[str, Any] = self._load_library()

    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        """
        Resolves inheritance and validates required schema keys (Fail-Fast).
        """
        if profile_id not in self.library:
            raise KeyError(
                f"[Registry Error] Profile '{profile_id}' not found in "
                f"brain_library.yaml. Available: {list(self.library.keys())}"
            )

        resolved: Dict[str, Any] = self._resolve_inheritance(profile_id)
        self._validate_schema(profile_id, resolved)
        return resolved

    def _load_library(self) -> Dict[str, Any]:
        """
        Reads and parses brain_library.yaml configuration file.
        """
        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"[Registry Error] Could not locate brain_library.yaml at "
                f"{self.yaml_path}"
            )

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return dict(data.get("library", {}))

    def _resolve_inheritance(self, profile_id: str) -> Dict[str, Any]:
        """
        Recursively merges parent profile properties into child profiles.
        """
        config: Dict[str, Any] = self.library.get(profile_id, {})
        if "extends" in config:
            parent_id: str = config["extends"]
            parent_config: Dict[str, Any] = self._resolve_inheritance(
                parent_id
            )
            return self._deep_merge(parent_config, config)

        return config.copy()

    def _deep_merge(
        self, parent: Dict[str, Any], child: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merges nested dictionary configurations recursively.
        """
        merged: Dict[str, Any] = parent.copy()
        for key, val in child.items():
            if (
                isinstance(val, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key] = self._deep_merge(merged[key], val)
            else:
                merged[key] = val
        return merged

    def _validate_schema(
        self, profile_id: str, config: Dict[str, Any]
    ) -> None:
        """
        Enforces required configuration keys; raises KeyError if missing.
        """
        required_sections: List[str] = [
            "file", "topology", "sensory", "kinematics", "metabolics"
        ]
        for section in required_sections:
            if section not in config:
                raise KeyError(
                    f"[Fail-Fast Error] Profile '{profile_id}' missing "
                    f"mandatory section '{section}' in brain_library.yaml."
                )

        topology_keys: List[str] = [
            "hidden_layers", "neurons_per_layer", "memory_frames"
        ]
        for key in topology_keys:
            if key not in config["topology"]:
                raise KeyError(
                    f"[Fail-Fast Error] Profile '{profile_id}.topology' "
                    f"missing key '{key}' in brain_library.yaml."
                )


class BrainWeightHandler:
    """
    Bridges Agent weight matrices with WeightStorageManager persistence.
    """

    def __init__(
        self, storage_manager: Optional[WeightStorageManager] = None
    ) -> None:
        """
        Initializes weight storage manager binding.
        """
        if storage_manager is None:
            self.storage: WeightStorageManager = WeightStorageManager()
        else:
            self.storage = storage_manager

    def load_champion(
        self, agent: Any, profile_config: Dict[str, Any]
    ) -> bool:
        """
        Loads saved weight matrices into an agent instance if file exists.
        """
        filename: str = profile_config.get("file", "")
        if not filename:
            return False

        data: Optional[Dict[str, np.ndarray]] = self.storage.get_weights(
            filename
        )
        if data is None:
            return False

        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []

        for idx in range(len(agent.weights)):
            w_key: str = f"w{idx}"
            b_key: str = f"b{idx}"

            if w_key in data and b_key in data:
                saved_w: np.ndarray = data[w_key]
                saved_b: np.ndarray = data[b_key]

                target_w_shape: Tuple[int, ...] = agent.weights[idx].shape
                target_b_shape: Tuple[int, ...] = agent.biases[idx].shape

                if saved_w.shape == target_w_shape:
                    weights.append(saved_w.astype(np.float64))
                else:
                    adapted_w = np.zeros(target_w_shape, dtype=np.float64)
                    r_lim: int = min(saved_w.shape[0], target_w_shape[0])
                    c_lim: int = min(saved_w.shape[1], target_w_shape[1])
                    adapted_w[:r_lim, :c_lim] = saved_w[:r_lim, :c_lim]
                    weights.append(adapted_w)

                if saved_b.shape == target_b_shape:
                    biases.append(saved_b.astype(np.float64))
                else:
                    adapted_b = np.zeros(target_b_shape, dtype=np.float64)
                    c_lim = min(saved_b.shape[1], target_b_shape[1])
                    adapted_b[0, :c_lim] = saved_b[0, :c_lim]
                    biases.append(adapted_b)
            else:
                return False

        agent.weights = weights
        agent.biases = biases
        return True

    def save_champion(
        self, agent: Any, profile_config: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Saves agent weight and bias arrays to disk under profile path.
        """
        filename: str = profile_config.get("file", "")
        if not filename:
            return None

        return self.storage.save_weights(
            filename, agent.weights, agent.biases
        )
