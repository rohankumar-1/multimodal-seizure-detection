""" Using stumpy to implement Matrix Profile for anomaly detection of ECG data """

import numpy as np
import stumpy

class MatrixProfile:
    def __init__(self, m=256, percentile=98, min_sep=10, min_cluster=3, max_gap=5):
        self.m = m
        self.percentile = percentile
        self.min_sep = min_sep
        self.min_cluster = min_cluster
        self.max_gap = max_gap

    def _clusters(self, idxs):
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

    def mp_to_window_scores(self, mp_scores, window_length):
        """
        Convert matrix profile scores (length T-W+1) into non-overlapping window scores.
        Each window gets the average of all MP scores that fall into it.

        Args:
            mp_scores: np.array, length T-W+1
            window_length: int, size of the window W

        Returns:
            window_scores: np.array, length ceil((T-W+1)/W)
        """
        num_scores = len(mp_scores)
        # Compute number of full windows
        num_windows = int(np.ceil(num_scores / window_length))

        window_scores = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * window_length
            end = min(start + window_length, num_scores)
            window_scores[i] = mp_scores[start:end].mean()

        return window_scores

    def predict(self, ecg: np.ndarray):
        # 1) Matrix Profile
        mp = stumpy.stump(ecg, self.m)[:, 0]

        # 2) Discord candidates (percentile threshold)
        thr = np.percentile(mp, self.percentile)
        cand = np.where(mp >= thr)[0]

        # 3) Overlap filtering
        if len(cand) == 0:
            return {"mp": mp, "discords": np.array([]), "events": np.array([])}
        disc = [cand[0]]
        for c in cand[1:]:
            if c - disc[-1] >= self.min_sep:
                disc.append(c)
        disc = np.array(disc)

        # 4) Temporal clustering
        events = self._clusters(disc)

        return {"mp": mp, "scores": self.mp_to_window_scores(mp, self.m), "discords": disc, "events": events}