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
    ty1, ty2 = y1 + int(0.2 * h), y1 + int(0.55 * h)
    tx1, tx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    ty1, tx1 = max(ty1, 0), max(tx1, 0)
    return frame[ty1:ty2, tx1:tx2]


def dominant_color(crop: np.ndarray, k: int = 3) -> np.ndarray | None:
    """Returns the dominant HSV color of a crop via k-means, or None."""
    if crop.size == 0:
        return None
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    # prefer saturated pixels so court lines / skin / white noise don't dominate
    saturated = pixels[pixels[:, 1] > 40]
    if len(saturated) >= 50:
        pixels = saturated
    if len(pixels) < k:
        return None
    _, labels, centers = cv2.kmeans(
        pixels, k, None, _KMEANS_CRITERIA, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers[np.argmax(np.bincount(labels.flatten()))]


class TeamClassifier:
    def __init__(self, smoothing: float = 0.9, vote_window: int = 15):
        self.smoothing = smoothing
        self.team_centroids: np.ndarray | None = None
        self.votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_window))

    def classify(self, frame: np.ndarray, detections: sv.Detections) -> list[str]:
        """Returns a team label ('A'/'B'/'?') per detection."""
        labels = [UNKNOWN] * len(detections)
        colors = [
            dominant_color(extract_torso(frame, box)) for box in detections.xyxy
        ]
        valid = [(i, c) for i, c in enumerate(colors) if c is not None]
        if len(valid) < 2:
            return labels

        samples = np.float32([c for _, c in valid])
        _, assignments, centers = cv2.kmeans(
            samples, 2, None, _KMEANS_CRITERIA, 3, cv2.KMEANS_PP_CENTERS
        )
        assignments = assignments.flatten()
        assignments, centers = self._align_with_history(assignments, centers)

        for (i, _), team in zip(valid, assignments):
            tid = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else None
            )
            if tid is not None:
                self.votes[tid].append(int(team))
                team = round(np.mean(self.votes[tid]))
            labels[i] = "A" if team == 0 else "B"
        return labels

    def _align_with_history(self, assignments, centers):
        """Keeps cluster 0 = Team A across frames so labels don't swap."""
        if self.team_centroids is None:
            self.team_centroids = centers.copy()
            return assignments, centers
        direct = np.linalg.norm(centers - self.team_centroids)
        swapped = np.linalg.norm(centers[::-1] - self.team_centroids)
        if swapped < direct:
            assignments = 1 - assignments
            centers = centers[::-1]
        self.team_centroids = (
            self.smoothing * self.team_centroids + (1 - self.smoothing) * centers
        )
        return assignments, centers
