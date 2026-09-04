# nba-vision

Computer vision pipeline that takes an NBA broadcast clip and outputs an
annotated version: players, the ball, and referees detected with YOLOv8,
tracked across frames with ByteTrack, and players automatically split into
their two teams by jersey color.

Broadcast footage is hard to analyze by hand — players move fast, cameras cut
and pan, and there are 10+ people on screen. This project turns raw footage
into structured, per-player data (who is where, on which team, across time),
which is the foundation for downstream basketball analytics like spacing
analysis, play recognition, and shot detection.

## How it works

```
input video ─→ detect (YOLOv8) ─→ track (ByteTrack) ─→ team classify (k-means) ─→ annotated video
```

- **`src/detect.py`** — YOLOv8 inference per frame. Uses fine-tuned weights at
  `models/best.pt` when present; otherwise falls back to stock COCO
  `yolov8n.pt`, whose `person` class stands in for "player" so the whole
  pipeline runs before a custom model exists.
- **`src/track.py`** — ByteTrack (via `supervision`) assigns each player a
  persistent ID across frames.
- **`src/team_classify.py`** — crops each player's torso, finds the dominant
  jersey color with k-means, then clusters all players in the frame into two
  teams. Team identity is kept stable over time with running color centroids
  and per-player majority voting.
- **`src/pipeline.py`** — ties it together and writes the annotated output
  (Team A red, Team B blue, ball orange, labels like `#12 A`).

## Setup

```bash
git clone <this repo> && cd nba_vision
python -m venv venv
venv\Scripts\activate        # Windows (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

No weights download needed up front — on first run, if `models/best.pt` is
missing, ultralytics auto-downloads `yolov8n.pt` (~6 MB).

**Getting a sample clip:** any short `.mp4` works. For real NBA footage, grab
a highlight with [yt-dlp](https://github.com/yt-dlp/yt-dlp):

```bash
yt-dlp -f "mp4[height<=720]" --download-sections "*0:10-0:25" <youtube-url> -o clip.mp4
```

## Run

```bash
python -m src.pipeline --input clip.mp4 --output out.mp4 --weights models/best.pt
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | (required) | input video path |
| `--output` | (required) | annotated output path |
| `--weights` | `models/best.pt` | YOLOv8 weights; falls back to COCO `yolov8n.pt` |
| `--conf` | `0.3` | detection confidence threshold |

Progress prints every 30 frames; the output is a normal mp4 you can scrub
through to inspect IDs and team assignments.

## Training the custom model

`src/train.py` fine-tunes YOLOv8 on the
[basketball-players](https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2)
dataset from Roboflow Universe (classes include `Player`, `Ball`, `Ref`) and
exports the weights to `models/best.pt`, where the pipeline picks them up
automatically. You need a free Roboflow API key
(app.roboflow.com → Settings → API Keys).

Training needs an NVIDIA GPU to be practical — on Google Colab's free T4 it
takes ~30–60 min:

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com)
   and set **Runtime → Change runtime type → T4 GPU**
2. Run:

   ```
   !git clone <this repo> nba_vision
   %cd nba_vision
   !pip install -q -r requirements.txt
   import os; os.environ["ROBOFLOW_API_KEY"] = "your-key"
   !python -m src.train
   ```

3. Download the result to your machine:

   ```python
   from google.colab import files
   files.download("models/best.pt")
   ```

4. Place it at `models/best.pt` locally and rerun the pipeline — player,
   ball, and referee classes now come from the fine-tuned model.

Useful flags: `--epochs`, `--imgsz` (default 960; higher helps ball
detection), `--model yolov8s.pt` for a bigger base model, and
`--epochs 1 --imgsz 320 --fraction 0.05` as a quick CPU wiring test.

The same command runs locally, but CPU training is very slow and AMD GPUs on
Windows aren't supported for PyTorch training yet.

## Roadmap

- [ ] **Fine-tuned detection** — train YOLOv8 on a Roboflow basketball dataset
      for proper `player` / `ball` / `referee` classes, drop in at
      `models/best.pt`
- [ ] **Ball tracking** — the ball is currently drawn per-frame but not
      tracked; add a motion model tolerant of occlusion and fast flight
- [ ] **Court keypoint homography** — detect court landmarks and project
      player positions into a bird's-eye minimap
- [ ] **Jersey number OCR** — read numbers off torso crops to link tracker IDs
      to actual players
- [ ] **Shot detection** — detect shot attempts and makes from ball + player
      trajectories

## Known limitations

- **Ball detection on fast motion is hard** — motion blur and the ball's small
  size mean even a fine-tuned detector will miss frames mid-flight; COCO
  fallback weights rarely see it at all.
- **Team classifier assumes two clearly distinct jersey colors** — matchups
  with similar palettes (or heavy white/gray uniforms, which are filtered as
  low-saturation) degrade the A/B split.
- **COCO fallback detects every person** — referees, coaches, and courtside
  fans are all "players" until the fine-tuned model lands.
- **No re-identification** — a player who leaves the frame and returns gets a
  new tracker ID.
