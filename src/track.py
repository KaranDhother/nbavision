"""ByteTrack wrapper assigning persistent IDs across frames."""

import supervision as sv


class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Returns detections with `tracker_id` populated."""
        return self.tracker.update_with_detections(detections)

    def reset(self):
        self.tracker.reset()
