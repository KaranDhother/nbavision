# nba-vision

Side project messing around with computer vision on NBA footage. Give it a
broadcast clip and it outputs an annotated version: every player boxed and
tracked with a persistent ID, auto-split into the two teams by jersey color.

YOLOv8 finds the people, ByteTrack keeps IDs on them between frames, and a
k-means color trick on the torso region figures out who's on which team.

## Setup

```bash
git clone https://github.com/KaranDhother/nbavision.git && cd nbavision
python -m venv venv
source venv/Scripts/activate   # or venv\Scripts\activate on cmd/powershell
pip install -r requirements.txt
```

For a clip, any short mp4 works. I grab highlights with yt-dlp:

```bash
yt-dlp -f "mp4[height<=720]" --download-sections "*0:10-0:25" <youtube-url> -o clip.mp4
```

## Run

```bash
python -m src.pipeline --input clip.mp4 --output out.mp4
```

If there's no fine-tuned model at `models/best.pt` it falls back to stock
COCO yolov8n (auto-downloads, ~6MB). That means everyone on screen counts as
a "player" — refs, coaches, front-row fans — and there's no ball detection,
but it's enough to see the tracking and team colors work.

## Training the real model

`src/train.py` fine-tunes YOLOv8 on a public basketball dataset from
[Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2)
(player / ball / ref classes) and drops the weights at `models/best.pt`.
Needs a free Roboflow API key, and realistically a GPU — I run it on a free
Colab T4, takes under an hour:

```python
!git clone https://github.com/KaranDhother/nbavision.git
%cd nbavision
!pip install -q -r requirements.txt
import os; os.environ["ROBOFLOW_API_KEY"] = "your-key"
!python -m src.train
```

then download `models/best.pt` from Colab into `models/` locally and rerun
the pipeline.

Some things to know: white jerseys trip up the color classifier sometimes
(low saturation), every camera cut resets the tracker IDs, and a fast-moving
ball is genuinely hard to detect even fine-tuned.
