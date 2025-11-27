# YOLO11 Delta Robot Conveyor Pick-and-Place

This example upgrades a traditional threshold-and-contour pipeline with a YOLO11
segmentation model to detect parts on a moving conveyor using a monochrome
camera, compute pose (X, Y, rotation), and stream robot-ready millimeter
coordinates for a delta robot pick head.

![Conveyor diagram](https://raw.githubusercontent.com/ultralytics/assets/main/images/blog/YOLOv11/YOLO11-assembly-line.png)

## Why this pipeline

- **Segmentation-driven pose** – the rotation angle comes from the mask's
  minimum-area rectangle instead of brittle contour heuristics.
- **Robot-friendly output** – pixels convert to millimeters using the camera
  center offset and `mm_per_pixel` scale used by the original script.
- **Conveyor prediction** – optional speed and latency fields forecast the pick
  location when the robot moves, reducing slip on fast belts.
- **Jitter reduction** – exponential smoothing stabilizes coordinates for
  lightweight delta arms.
- **Frame-to-frame tracking** – detections are associated across frames to
  de-duplicate the same part and gate robot commands until a track is stable.
- **Encoder-aware motion** – optional `c1Counts` belt encoder data is fused into
  the tracks so predicted pick points follow moving parts instead of stale
  frame positions.
- **Basler ready** – uses `pypylon` when present and falls back to OpenCV video
  capture for quick testing.

## Quick start

Install the dependencies (assumes Python 3.9+):

```bash
pip install ultralytics opencv-python numpy  # plus 'pypylon' for Basler cameras
```

Run the pipeline. It will try to open a Basler camera first, then a default
webcam if `pypylon` is missing.

```bash
python examples/YOLO11-Conveyor-Pick-Place/pick_place_conveyor.py
```

Press **`s`** to persist your tuned values to `vision_config.json` and **`q`**
to exit.

## Configurable parameters

`vision_config.json` is loaded automatically and written when you press `s`.

| Field | Purpose |
| --- | --- |
| `model_path` | YOLO11 segmentation weights (e.g., `yolo11n-seg.pt`). |
| `target_classes` | YOLO class IDs to accept (tuple). |
| `roi` | Pixel ROI `(x1, y1, x2, y2)` to ignore conveyor edges. |
| `mm_per_pixel` | Scale factor for pixel → millimeter conversion. |
| `cam_center_robot_x/y` | Robot coordinates aligned to the camera center. |
| `c1_mm_per_count` | Millimeters per encoder count for `c1Counts` belt data. |
| `conveyor_axis_sign` | Use `1` or `-1` to match belt direction along robot X. |
| `conveyor_speed_mms` | Belt speed in mm/s for pick-point prediction. |
| `robot_latency_s` | Time from detection to pick command. |
| `smoothing_alpha` | Exponential moving average factor for jitter control. |
| `track_match_thresh_mm` | Max distance (mm) to associate detections to a track. |
| `min_track_hits` | Frames required before a track can trigger a robot command. |
| `max_track_age` | Frames a track lives without an update before being dropped. |
| `robot_workspace` | `(xmin, xmax, ymin, ymax)` safety bounds for commands. |
| `command_cooldown_s` | Minimum spacing between robot command prints. |

## Robot coordinate mapping

Pixel coordinates `(px, py)` are converted into robot millimeters `(rx, ry)`
using the camera's optical center as the reference:

```text
rx = cam_center_robot_x + (py - h/2) * mm_per_pixel
ry = cam_center_robot_y + (px - w/2) * mm_per_pixel
```

If you provide `conveyor_speed_mms` and `robot_latency_s`, the script predicts
where the part will be by the time the robot moves, so the pick head meets the
part instead of lagging behind.

When you have encoder feedback from ABB RobotStudio, pass the live `c1Counts`
value into the detection call. The example stubs a counter that increments by
`10` each frame; replacing it with the real encoder lets the tracker slide
existing tracks forward on the belt direction and auto-compute belt speed from
count deltas.

## Notes for production

- Train or fine-tune the YOLO11 segmentation model on your monochrome parts to
  get accurate angles and clean masks.
- Keep the ROI tight around the belt; it reduces false positives and makes the
  smoothing more responsive.
- The command scheduler only emits a pick when a track is both stable and
  inside `robot_workspace`, which helps avoid duplicate picks and out-of-range
  moves.
- For deterministic results on a PLC, export the model to ONNX and run it with
  ONNX Runtime or TensorRT, keeping the same input size and ROI defined here.
- Tune `smoothing_alpha` and the conveyor/latency terms with live telemetry from
  the delta robot controller to match your mechanical response.
