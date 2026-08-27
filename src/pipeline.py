"""End-to-end pipeline: detect -> track -> team-classify -> annotated video."""

import argparse

import cv2
import numpy as np
import supervision as sv

from .detect import Detector
from .team_classify import UNKNOWN, TeamClassifier
from .track import Tracker

TEAM_COLORS = {
    "A": sv.Color(220, 50, 50),
    "B": sv.Color(50, 90, 220),
    UNKNOWN: sv.Color(160, 160, 160),
}
BALL_COLOR = sv.Color(255, 140, 0)


class _Annotators:
    def __init__(self):
        self.boxes = {
            team: sv.BoxAnnotator(color=color) for team, color in TEAM_COLORS.items()
        }
        self.labels = {
            team: sv.LabelAnnotator(color=color, text_color=sv.Color.WHITE)
            for team, color in TEAM_COLORS.items()
        }
        self.ball_box = sv.BoxAnnotator(color=BALL_COLOR)

    def draw(self, frame, persons, team_labels, balls):
        for team in TEAM_COLORS:
            mask = np.array([t == team for t in team_labels], dtype=bool)
            group = persons[mask] if len(persons) else persons
            if len(group) == 0:
                continue
            texts = [
                f"#{tid} {team}" if tid is not None else team
                for tid in (group.tracker_id if group.tracker_id is not None else [None] * len(group))
            ]
            frame = self.boxes[team].annotate(frame, group)
            frame = self.labels[team].annotate(frame, group, labels=texts)
        if len(balls):
            frame = self.ball_box.annotate(frame, balls)
        return frame


def run(input_path: str, output_path: str, weights_path: str = "models/best.pt", conf: float = 0.3):
    detector = Detector(weights_path, conf)
    tracker = Tracker()
    classifier = TeamClassifier()
    annotators = _Annotators()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        persons, balls = detector.split(detections)
        persons = tracker.update(persons)
        team_labels = classifier.classify(frame, persons)
        writer.write(annotators.draw(frame, persons, team_labels, balls))
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[pipeline] {frame_idx}/{total or '?'} frames")

    cap.release()
    writer.release()
    print(f"[pipeline] done: {frame_idx} frames -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Annotate an NBA clip with tracked, team-labeled players.")
    parser.add_argument("--input", required=True, help="input video path")
    parser.add_argument("--output", required=True, help="annotated output video path")
    parser.add_argument("--weights", default="models/best.pt", help="YOLOv8 weights (falls back to yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.3, help="detection confidence threshold")
    args = parser.parse_args()
    run(args.input, args.output, args.weights, args.conf)


if __name__ == "__main__":
    main()
