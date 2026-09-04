"""Splits players into two teams (A/B) by dominant jersey color."""

from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

UNKNOWN = "?"
_KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


def extract_torso(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Crops the jersey region: upper-middle of the box, court/legs excluded."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    h, w = y2 - y1, x2 - x1
    # narrow horizontally to exclude bare arms — jersey only
    ty1, ty2 = y1 + int(0.2 * h), y1 + int(0.5 * h)
    tx1, tx2 = x1 + int(0.3 * w), x2 - int(0.3 * w)
    ty1, tx1 = max(ty1, 0), max(tx1, 0)
    return frame[ty1:ty2, tx1:tx2]


def dominant_color(crop: np.ndarray, k: int = 3) -> np.ndarray | None:
    """Returns the dominant HSV color of a crop via k-means, or None."""
    if crop.size == 0:
        return None
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return None
    _, labels, centers = cv2.kmeans(
        pixels, k, None, _KMEANS_CRITERIA, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers[np.argmax(np.bincount(labels.flatten()))]


def _occluded(xyxy: np.ndarray) -> np.ndarray:
    """True for boxes substantially overlapped by another box."""
    n = len(xyxy)
    mask = np.zeros(n, dtype=bool)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ix = min(xyxy[i, 2], xyxy[j, 2]) - max(xyxy[i, 0], xyxy[j, 0])
            iy = min(xyxy[i, 3], xyxy[j, 3]) - max(xyxy[i, 1], xyxy[j, 1])
            if ix > 0 and iy > 0 and ix * iy > 0.25 * areas[i]:
                mask[i] = True
                break
    return mask


class TeamClassifier:
    """Assigns A/B by matching jersey colors to persistent team centroids.

    Centroids are bootstrapped once via k-means on a clean frame, then each
    player is matched to the nearest centroid — stable even when one team
    dominates the frame. Occluded players contribute no color samples (their
    crop likely shows another player's jersey); they keep their voted label.
    """

    def __init__(self, smoothing: float = 0.995, vote_window: int = 15):
        self.smoothing = smoothing
        self.team_centroids: np.ndarray | None = None
        self.votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_window))

    def classify(self, frame: np.ndarray, detections: sv.Detections) -> list[str]:
        """Returns a team label ('A'/'B'/'?') per detection."""
        n = len(detections)
        labels = [UNKNOWN] * n
        if n == 0:
            return labels

        occluded = _occluded(detections.xyxy)
        colors = [
            None if occluded[i] else dominant_color(extract_torso(frame, box))
            for i, box in enumerate(detections.xyxy)
        ]

        if self.team_centroids is None and not self._bootstrap(colors):
            return labels

        for i in range(n):
            tid = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else None
            )
            team = None
            if colors[i] is not None:
                dists = np.linalg.norm(self.team_centroids - colors[i], axis=1)
                team = int(np.argmin(dists))
                # drift centroid only on confident assignments
                if dists[team] < 0.8 * dists[1 - team]:
                    self.team_centroids[team] = (
                        self.smoothing * self.team_centroids[team]
                        + (1 - self.smoothing) * colors[i]
                    )
                if tid is not None:
                    self.votes[tid].append(team)
            if tid is not None and self.votes[tid]:
                team = round(np.mean(self.votes[tid]))
            if team is not None:
                labels[i] = "A" if team == 0 else "B"
        return labels

    def _bootstrap(self, colors: list) -> bool:
        """Initializes team centroids via k-means on an unoccluded frame."""
        samples = np.float32([c for c in colors if c is not None])
        if len(samples) < 4:
            return False
        _, _, centers = cv2.kmeans(
            samples, 2, None, _KMEANS_CRITERIA, 3, cv2.KMEANS_PP_CENTERS
        )
        self.team_centroids = centers
        return True
