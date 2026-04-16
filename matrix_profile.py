"""
ECG matrix profile (stumpy) with window-level scores vs labels.
Edit constants at the top only.

Each entry in DATA_PATHS (or each file matched by DATA_GLOB) should be one .npz with:
  - ecg: 1d time series (or use PREPROCESS_FLATTEN to ravel (C,T) or (T,) per file)
  - binary_label (or TARGET): same length as the score vector from MP (see metrics.evaluate_matrixprofile)

Requires: pip install stumpy

Run: python matrix_profile.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from metrics import evaluate_matrixprofile
from preprocess import apply_bandpass

# --- Edit these ---
# List of .npz paths (one long ECG per file is typical). Leave empty if using DATA_GLOB.
DATA_PATHS: list[str] = ["data/processed_2s_40Hz/test.npz"]
# If non-empty, overrides / fills DATA_PATHS via glob, e.g. "data/testruns/testruns_2s/*.npz"
# DATA_GLOB: str | None = "data/processed_2s_8Hz/*.npz"

TARGET = "binary_label"
RESULTS_PATH = "results/matrix_profile_ecg_only.json"

# Matrix profile hyperparameters
MP_M = 80
MP_PERCENTILE = 95.0
MP_MIN_SEP = 10
MP_MIN_CLUSTER = 3
MP_MAX_GAP = 5

# Optional bandpass on ECG before stump (Hz); set USE_BANDPASS False to skip
USE_BANDPASS = False
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
BANDPASS_FS = 8.0

# If ecg in each file is (C, T) or (N, C, T), ravel / take first channel before stump
PREPROCESS_FLATTEN = True
# ------------------


class MatrixProfile:
    def __init__(
        self,
        m: int = 256,
        percentile: float = 98,
        min_sep: int = 10,
        min_cluster: int = 3,
        max_gap: int = 5,
    ):
        self.m = m
        self.percentile = percentile
        self.min_sep = min_sep
        self.min_cluster = min_cluster
        self.max_gap = max_gap

    def _clusters(self, idxs: np.ndarray) -> np.ndarray:
        if len(idxs) == 0:
            return np.array([])
        clusters, cur = [], [idxs[0]]
        for i in idxs[1:]:
            if i - cur[-1] <= self.max_gap:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)
        return np.array([int(np.mean(c)) for c in clusters if len(c) >= self.min_cluster])

    def mp_to_window_scores(self, mp_scores: np.ndarray, window_length: int, m: int) -> np.ndarray:
        scores = []
        W = window_length
        subseqs_per_window = W - m + 1
        start = 0
        while start + subseqs_per_window <= len(mp_scores):
            end = start + subseqs_per_window
            scores.append(mp_scores[start:end].max())
            start += W
        return np.array(scores)

    def predict(self, ecg: np.ndarray) -> dict:
        import stumpy

        mp = stumpy.stump(ecg, self.m)[:, 0]
        thr = np.percentile(mp, self.percentile)
        cand = np.where(mp >= thr)[0]
        if len(cand) == 0:
            return {
                "mp": mp,
                "scores": self.mp_to_window_scores(mp, self.m, self.m),
                "discords": np.array([]),
                "events": np.array([]),
            }
        disc = [cand[0]]
        for c in cand[1:]:
            if c - disc[-1] >= self.min_sep:
                disc.append(c)
        disc = np.array(disc)
        events = self._clusters(disc)
        return {
            "mp": mp,
            "scores": self.mp_to_window_scores(mp, self.m, self.m),
            "discords": disc,
            "events": events,
        }



def preprocess_ecg(ecg: np.ndarray) -> np.ndarray:
    x = np.asarray(ecg, dtype=np.float64)
    if PREPROCESS_FLATTEN:
        x = x.ravel() if x.ndim > 1 else x
    if USE_BANDPASS:
        x = apply_bandpass(x, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_FS)
    return np.asarray(x, dtype=np.float64).ravel()


def main() -> None:
    if not DATA_PATHS:
        raise SystemExit(
            "Set DATA_PATHS (list of .npz) and/or DATA_GLOB at the top of matrix_profile.py, then re-run."
        )

    mp = MatrixProfile(
        m=MP_M,
        percentile=MP_PERCENTILE,
        min_sep=MP_MIN_SEP,
        min_cluster=MP_MIN_CLUSTER,
        max_gap=MP_MAX_GAP,
    )

    results = evaluate_matrixprofile(
        mp,
        preprocess_ecg,
        data_paths=DATA_PATHS,
        target=TARGET,
    )

    selected = {k: results[k] for k in ("auc_score", "accuracy", "f1", "precision", "recall") if k in results}
    print(f"Files: {len(DATA_PATHS)}")
    print("Metrics:", selected)

    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "metrics": selected,
                "n_files": len(DATA_PATHS),
                "mp_m": MP_M,
                "mp_percentile": MP_PERCENTILE,
                "target": TARGET,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
