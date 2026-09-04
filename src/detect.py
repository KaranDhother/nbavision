"""Object detection for nba-vision.

Wraps a YOLOv8 model (ultralytics) and normalizes its output into
`supervision.Detections` so the rest of the pipeline never touches
ultralytics types directly.

Weights handling:
- Preferred: a fine-tuned model at ``models/best.pt`` with classes like
  ``player`` / ``ball`` / ``referee`` (e.g. trained on a Roboflow
  basketball dataset).
- Fallback: if the weights file is missing, we load stock ``yolov8n.pt``
  (auto-downloaded by ultralytics on first use). COCO has no basketball
  classes, so its ``person`` class stands in for "player" — good enough
  to exercise the tracking and team-classification stages.

Classes are matched by *name*, not hardcoded IDs, so swapping in custom
weights requires no code changes.
"""

from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

# Fallback model when no custom weights exist. Ultralytics downloads it
# automatically to the current directory on first use (~6 MB).
FALLBACK_WEIGHTS = "yolov8n.pt"

# Class names (lowercased) that count as a trackable person. "person" covers
# the COCO fallback; the rest cover typical basketball fine-tunes.
PERSON_LIKE_NAMES = {"person", "player", "referee", "ref", "coach"}

# Class names that count as the ball.
BALL_NAMES = {"ball", "basketball", "sports ball"}


class Detector:
    """Runs YOLOv8 inference on single frames.

    Attributes:
        person_class_ids: class IDs treated as people (players/referees).
        ball_class_ids: class IDs treated as the ball.
        class_names: id -> name mapping from the loaded model.
    """

    def __init__(self, weights_path: str = "models/best.pt", conf: float = 0.3):
        if Path(weights_path).exists():
            print(f"[detect] loading custom weights: {weights_path}")
            self.model = YOLO(weights_path)
        else:
            print(
                f"[detect] '{weights_path}' not found — falling back to COCO "
                f"'{FALLBACK_WEIGHTS}' (persons only, no ball/referee classes)"
            )
            self.model = YOLO(FALLBACK_WEIGHTS)

        self.conf = conf
        # Ultralytics exposes names as {id: name}; normalize to lowercase
        # so matching is robust to dataset labeling conventions.
        self.class_names: dict[int, str] = dict(self.model.names)
        self.person_class_ids = {
            cid for cid, name in self.class_names.items()
            if name.lower() in PERSON_LIKE_NAMES
        }
        self.ball_class_ids = {
            cid for cid, name in self.class_names.items()
            if name.lower() in BALL_NAMES
        }
        # Everything we care about; other COCO classes (chairs, etc.) get dropped.
        self._keep_ids = self.person_class_ids | self.ball_class_ids

        if not self.person_class_ids:
            raise ValueError(
                "Loaded model has no person-like class "
                f"(looked for {sorted(PERSON_LIKE_NAMES)} in {self.class_names})"
            )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Detect objects in a single BGR frame.

        Returns a `sv.Detections` filtered to person/ball classes only.
        """
        # verbose=False keeps per-frame ultralytics logging out of the console.
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        return detections[np.isin(detections.class_id, list(self._keep_ids))]

    # -- helpers for downstream stages -----------------------------------

    def split(self, detections: sv.Detections) -> tuple[sv.Detections, sv.Detections]:
        """Split detections into (persons, balls).

        Persons go through tracking + team classification; the ball (if the
        model detects one) is drawn as-is in v1.
        """
        person_mask = np.isin(detections.class_id, list(self.person_class_ids))
        return detections[person_mask], detections[~person_mask]
