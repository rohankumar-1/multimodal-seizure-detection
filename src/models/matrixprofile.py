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
        """ find clusters of indices separated by at least min_sep """
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
        """ takes matrix profile scores and returns window scores (averages of scores in each window), interpret as prob of anomaly"""
        num_scores = len(mp_scores)
        num_windows = int(np.ceil(num_scores / window_length))
        window_scores = np.zeros(num_windows)
        for i in range(num_windows):
            start = i * window_length
            end = min(start + window_length, num_scores)
            window_scores[i] = mp_scores[start:end].mean()
        return window_scores

    def predict(self, chunks: np.ndarray):
        """
        chunks: shape (num_chunks, chunk_size)
        Returns global MP and correct global window scores.
        """
        all_mp = []
        all_discords = []
        all_events = []

        offset = 0
        for chunk in chunks:
            mp = stumpy.stump(chunk, self.m)[:, 0]

            # keep only the first chunk_size - m + 1 entries
            valid_len = len(chunk) - self.m + 1
            mp_valid = mp[:valid_len]


            # threshold
            thr = np.percentile(mp_valid, self.percentile)
            cand = np.where(mp_valid >= thr)[0]

            if len(cand) > 0:
                disc = [cand[0]]
                for c in cand[1:]:
                    if c - disc[-1] >= self.min_sep:
                        disc.append(c)
                disc = np.array(disc)
                events = self._clusters(disc)

                # shift to global index space
                disc += offset
                events += offset
            else:
                disc = np.array([])
                events = np.array([])

            all_discords.append(disc)
            all_events.append(events)

            all_mp.append(mp_valid)
            offset += valid_len

        # ----- KEY FIX -----
        # concat MP globally first
        mp_full = np.concatenate(all_mp)

        # compute window scores ONCE based on the full MP
        window_scores = self.mp_to_window_scores(mp_full, self.m)

        return {
            "mp": mp_full,
            "scores": window_scores,
            "discords": np.concatenate(all_discords) if all_discords else np.array([]),
            "events": np.concatenate(all_events) if all_events else np.array([])
        }

    @staticmethod
    def chunk_timeseries(ts, chunk_size, m):
        overlap = m - 1
        chunks = []
        start = 0
        T = len(ts)

        while start < T:
            end = min(start + chunk_size + overlap, T)
            if end - start < chunk_size:
                return chunks
            chunks.append(ts[start:end])
            start += chunk_size

        return chunks
