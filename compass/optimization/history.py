"""Optimization history tracking for COMPASS inverse design.

Records per-iteration data (parameters, objective value, metadata) and
provides retrieval, serialization, and summary utilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationHistory:
    """Track and query optimization progress.

    Records one entry per optimizer iteration containing the parameter
    vector, objective value, and optional metadata.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(
        self,
        iteration: int,
        params: np.ndarray,
        objective: float,
        metadata: dict | None = None,
    ) -> None:
        """Append a single iteration record.

        Args:
            iteration: Iteration index (0-based).
            params: Parameter vector for this iteration.
            objective: Scalar objective value.
            metadata: Optional extra info (e.g. per-pixel QE, timing).
        """
        entry = {
            "iteration": iteration,
            "params": params.tolist() if isinstance(params, np.ndarray) else list(params),
            "objective": float(objective),
            "metadata": metadata or {},
        }
        self._records.append(entry)

    @property
    def n_records(self) -> int:
        """Number of recorded iterations."""
        return len(self._records)

    def get(self, index: int) -> dict:
        """Retrieve a record by index.

        Args:
            index: 0-based record index.

        Returns:
            Record dict with keys: iteration, params, objective, metadata.
        """
        return self._records[index]

    def best(self) -> dict:
        """Return the record with the lowest objective value.

        Returns:
            Best record dict, or empty dict if no records exist.
        """
        if not self._records:
            return {}
        return min(self._records, key=lambda r: r["objective"])

    def objectives(self) -> list[float]:
        """Return list of objective values in iteration order."""
        return [r["objective"] for r in self._records]

    def params_history(self) -> list[list[float]]:
        """Return list of parameter vectors in iteration order."""
        return [r["params"] for r in self._records]

    def to_dict(self) -> dict:
        """Serialize history to a plain dict for JSON export.

        Returns:
            Dict with ``records`` key containing list of entries.
        """
        return {
            "n_records": self.n_records,
            "records": self._records,
        }

    def save(self, path: str | Path) -> None:
        """Save history to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Optimization history saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> OptimizationHistory:
        """Load history from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Populated OptimizationHistory instance.
        """
        with open(path) as f:
            data = json.load(f)
        history = cls()
        for rec in data.get("records", []):
            history.record(
                iteration=rec["iteration"],
                params=np.array(rec["params"]),
                objective=rec["objective"],
                metadata=rec.get("metadata", {}),
            )
        return history
