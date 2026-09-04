"""Fine-tunes YOLOv8 on a Roboflow basketball dataset, exports models/best.pt."""

import argparse
import os
import shutil
from pathlib import Path

DEFAULT_WORKSPACE = "roboflow-universe-projects"
DEFAULT_PROJECT = "basketball-players-fy4c2"


def download_dataset(api_key: str, workspace: str, project_id: str, version: int) -> Path:
    from roboflow import Roboflow

    project = Roboflow(api_key=api_key).workspace(workspace).project(project_id)
    if version == 0:
        version = max(int(v.version.split("/")[-1]) for v in project.versions())
    print(f"[train] downloading {workspace}/{project_id} v{version}")
    dataset = project.version(version).download(
        "yolov8", location=f"datasets/{project_id}-v{version}"
    )
    return Path(dataset.location).resolve()


def check_classes(data_yaml: Path):
    import yaml

    from .detect import BALL_NAMES, PERSON_LIKE_NAMES

    names = yaml.safe_load(data_yaml.read_text())["names"]
    if isinstance(names, dict):
        names = list(names.values())
    print(f"[train] dataset classes: {names}")
    lower = {n.lower() for n in names}
    if not lower & PERSON_LIKE_NAMES:
        print("[train] WARNING: no class matches detect.py's person-like names")
    if not lower & BALL_NAMES:
        print("[train] WARNING: no class matches detect.py's ball names")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for nba-vision.")
    parser.add_argument("--api-key", default=None, help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=0, help="dataset version (0 = latest)")
    parser.add_argument("--model", default="yolov8n.pt", help="base weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960, help="training image size (ball is small)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of dataset to train on")
    parser.add_argument("--device", default=None, help="cuda device or 'cpu' (default: auto)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Roboflow API key required: pass --api-key or set ROBOFLOW_API_KEY "
            "(free at https://app.roboflow.com -> Settings -> API Keys)"
        )

    dataset_dir = download_dataset(api_key, args.workspace, args.project, args.version)
    check_classes(dataset_dir / "data.yaml")

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[train] no CUDA GPU — training on CPU will be very slow (use Colab, see README)")

    results = YOLO(args.model).train(
        data=str(dataset_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=device,
        project="runs",
        name="train",
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy(best, "models/best.pt")
    print(f"[train] exported {best} -> models/best.pt")


if __name__ == "__main__":
    main()
