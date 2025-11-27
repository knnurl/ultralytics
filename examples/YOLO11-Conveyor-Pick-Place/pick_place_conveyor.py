"""
YOLO11 conveyor pick-and-place pipeline for monochrome cameras and delta robots.

This example shows how to pair a Basler monochrome camera (via pypylon) with an
Ultralytics YOLO11 segmentation model to detect items on a conveyor, estimate a
rotation angle from the mask, and convert pixel coordinates into robot-friendly
millimeters. It also smooths detections to reduce jitter and predicts where a
part will be when the robot is ready to pick.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:  # Basler cameras are optional; fall back to a regular webcam if missing
    from pypylon import pylon
except Exception:  # pragma: no cover - optional dependency for Basler cameras
    pylon = None


@dataclass
class VisionConfig:
    """Runtime configuration that can be tuned without touching the code."""

    model_path: str = "yolo11n-seg.pt"
    conf: float = 0.4
    iou: float = 0.4
    device: str = "cpu"
    target_classes: Tuple[int, ...] = (0,)
    roi: Tuple[int, int, int, int] = (0, 0, 1280, 1024)
    mm_per_pixel: float = 0.5
    cam_center_robot_x: float = -1412.5
    cam_center_robot_y: float = -29.5
    c1_mm_per_count: float = 1.0
    conveyor_axis_sign: int = 1
    conveyor_speed_mms: float = 0.0
    robot_latency_s: float = 0.0
    smoothing_alpha: float = 0.35
    track_match_thresh_mm: float = 30.0
    min_track_hits: int = 2
    max_track_age: int = 10
    robot_workspace: Tuple[float, float, float, float] = (-2000, 2000, -400, 400)
    command_cooldown_s: float = 0.15

    @classmethod
    def load(cls, path: Path) -> "VisionConfig":
        if not path.exists():
            return cls()
        with path.open() as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


@dataclass
class Detection:
    label: str
    pixel_xy: Tuple[float, float]
    angle_deg: float
    robot_xy: Tuple[float, float]
    predicted_robot_xy: Tuple[float, float]
    track_id: int


@dataclass
class Track:
    track_id: int
    robot_xy: Tuple[float, float]
    pixel_xy: Tuple[float, float]
    angle: float
    hits: int = 1
    age: int = 0
    commanded: bool = False


class BaslerStream:
    """Minimal Basler grabber that yields grayscale frames."""

    def __init__(self, exposure: Optional[int] = None):
        if pylon is None:
            raise RuntimeError("pypylon is required for Basler cameras.")
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found.")
        self.cam = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        self.cam.Open()
        if exposure:
            self.cam.ExposureAuto.SetValue("Off")
            self.cam.ExposureTime.SetValue(exposure)
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while self.cam.IsGrabbing():
            grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                continue
            frame = grab.Array
            grab.Release()
            yield frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()


class OpenCVStream:
    """Fallback webcam grabber when pypylon is not available."""

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open default camera.")
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                continue
            yield frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()


class DetectionPipeline:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.model.to(cfg.device)
        self.track_id_counter = 0
        self.tracks: List[Track] = []
        self.last_c1_counts: Optional[int] = None
        self.last_frame_time: Optional[float] = None
        self.belt_speed_mms: float = cfg.conveyor_speed_mms

    def pixel_to_robot(self, px: float, py: float, w: int, h: int) -> Tuple[float, float]:
        dx = px - w / 2
        dy = py - h / 2
        return (
            self.cfg.cam_center_robot_x + dy * self.cfg.mm_per_pixel,
            self.cfg.cam_center_robot_y + dx * self.cfg.mm_per_pixel,
        )

    def predict_pick_xy(self, robot_xy: Tuple[float, float]) -> Tuple[float, float]:
        speed = self.belt_speed_mms if self.belt_speed_mms != 0 else self.cfg.conveyor_speed_mms
        if speed == 0 or self.cfg.robot_latency_s == 0:
            return robot_xy
        dx = speed * self.cfg.robot_latency_s * self.cfg.conveyor_axis_sign
        return (robot_xy[0] + dx, robot_xy[1])

    def _apply_belt_motion(self, c1_counts: Optional[int]) -> None:
        now = time.time()
        if self.last_frame_time is None:
            self.last_frame_time = now
        elapsed = now - self.last_frame_time
        self.last_frame_time = now

        if c1_counts is None:
            return

        if self.last_c1_counts is None:
            self.last_c1_counts = c1_counts
            return

        delta_counts = c1_counts - self.last_c1_counts
        self.last_c1_counts = c1_counts
        delta_mm = delta_counts * self.cfg.c1_mm_per_count * self.cfg.conveyor_axis_sign
        if elapsed > 0:
            self.belt_speed_mms = delta_mm / elapsed

        if delta_mm == 0 or not self.tracks:
            return

        for track in self.tracks:
            x, y = track.robot_xy
            track.robot_xy = (x + delta_mm, y)

    def _mask_angle_and_center(self, mask: np.ndarray) -> Tuple[Tuple[float, float], float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contour from mask.")
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        (cx, cy), (_, _), angle = rect
        angle = (angle + 180) % 180
        if angle > 90:
            angle -= 180
        return (cx, cy), angle

    def _match_track(self, robot_xy: Tuple[float, float], angle: float, pixel_xy: Tuple[float, float]) -> Track:
        if not self.tracks:
            self.track_id_counter += 1
            return Track(self.track_id_counter, robot_xy, pixel_xy, angle)

        dists = [np.linalg.norm(np.array(t.robot_xy) - np.array(robot_xy)) for t in self.tracks]
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= self.cfg.track_match_thresh_mm:
            track = self.tracks[best_idx]
            smoothed = self.cfg.smoothing_alpha * np.array(robot_xy) + (1 - self.cfg.smoothing_alpha) * np.array(track.robot_xy)
            track.robot_xy = tuple(smoothed.tolist())
            track.pixel_xy = pixel_xy
            track.angle = angle
            track.hits += 1
            track.age = 0
            return track

        self.track_id_counter += 1
        return Track(self.track_id_counter, robot_xy, pixel_xy, angle)

    def _age_tracks(self) -> None:
        surviving = []
        for track in self.tracks:
            track.age += 1
            if track.age <= self.cfg.max_track_age:
                surviving.append(track)
        self.tracks = surviving

    def detect(self, frame: np.ndarray, c1_counts: Optional[int] = None) -> List[Detection]:
        self._apply_belt_motion(c1_counts)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.cfg.roi
        x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 1, w))
        y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 1, h))
        if x2 <= x1 or y2 <= y1:
            return []
        roi = frame[y1:y2, x1:x2]

        results = self.model.predict(
            roi,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=list(self.cfg.target_classes),
            verbose=False,
            imgsz=max(roi.shape[:2]),
            device=self.cfg.device,
        )
        if not results:
            return []

        res = results[0]
        if not res.boxes:
            return []

        masks = res.masks.data if res.masks is not None else None
        detections: List[Detection] = []
        self._age_tracks()

        for idx, box in enumerate(res.boxes):
            cls_id = int(box.cls)
            xyxy = box.xyxy[0].cpu().numpy()
            px, py = (float(xyxy[0] + xyxy[2]) / 2 + x1, float(xyxy[1] + xyxy[3]) / 2 + y1)
            angle = 0.0
            if masks is not None:
                mask = (masks[idx].cpu().numpy() * 255).astype("uint8")
                (mx, my), angle = self._mask_angle_and_center(mask)
                px, py = mx + x1, my + y1

            rx, ry = self.pixel_to_robot(px, py, frame.shape[1], frame.shape[0])

            track = self._match_track((rx, ry), angle, (px, py))
            if track not in self.tracks:
                self.tracks.append(track)

            rx, ry = track.robot_xy
            predicted_xy = self.predict_pick_xy((rx, ry))

            label = (
                f"id={track.track_id} cls={cls_id} rx={rx:.1f} ry={ry:.1f} a={angle:.1f}"
                f" v={self.belt_speed_mms:.1f}mm/s"
            )
            detections.append(
                Detection(label, (px, py), angle, (rx, ry), predicted_xy, track.track_id)
            )

        return detections


class RobotCommandScheduler:
    """Debounce and validate commands before sending them to the robot."""

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.last_command_time = 0.0

    def _in_workspace(self, xy: Tuple[float, float]) -> bool:
        x, y = xy
        xmin, xmax, ymin, ymax = self.cfg.robot_workspace
        return xmin <= x <= xmax and ymin <= y <= ymax

    def select_commands(self, pipeline: DetectionPipeline, detections: List[Detection]) -> List[Detection]:
        now = time.time()
        commands: List[Detection] = []
        for det in detections:
            track = next((t for t in pipeline.tracks if t.track_id == det.track_id), None)
            if track is None:
                continue
            if track.commanded:
                continue
            if track.hits < self.cfg.min_track_hits or track.age > 0:
                continue
            if not self._in_workspace(det.predicted_robot_xy):
                continue
            if now - self.last_command_time < self.cfg.command_cooldown_s:
                continue

            track.commanded = True
            self.last_command_time = now
            commands.append(det)

        return commands

    def draw(self, frame: np.ndarray, detections: List[Detection], pipeline: DetectionPipeline) -> np.ndarray:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = self.cfg.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(
            vis,
            f"belt speed: {pipeline.belt_speed_mms:.1f} mm/s",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 255),
            2,
        )

        for det in detections:
            px, py = map(int, det.pixel_xy)
            color = (0, 255, 0) if det.track_id else (0, 255, 255)
            cv2.drawMarker(vis, (px, py), color, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                vis,
                det.label,
                (max(px - 120, 10), max(py - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        if detections:
            pred_x, pred_y = map(int, detections[0].predicted_robot_xy)
            cv2.putText(
                vis,
                f"next pick -> {pred_x:.1f},{pred_y:.1f} mm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        return vis


def create_stream(prefer_basler: bool = True):
    if prefer_basler and pylon is not None:
        try:
            return BaslerStream()
        except Exception as e:  # pragma: no cover - optional hardware path
            print(f"[Basler] falling back to webcam: {e}")
    return OpenCVStream()


def main(config_path: Path = Path("vision_config.json")):
    cfg = VisionConfig.load(config_path)
    pipeline = DetectionPipeline(cfg)
    scheduler = RobotCommandScheduler(cfg)
    stream = create_stream()
    # Replace this stub with the live encoder reading from RobotStudio: c1Counts
    c1_counts = 0
    print("Press 'q' to quit, 's' to save config.")

    try:
        for frame in stream:
            # Stub encoder update: in production, read c1Counts from RobotStudio
            c1_counts += 10

            detections = pipeline.detect(frame, c1_counts=c1_counts)
            commands = scheduler.select_commands(pipeline, detections)
            output = pipeline.draw(frame, detections, pipeline) if detections else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            for det in commands:
                print("[ROBOT CMD]", det)

            cv2.imshow("Conveyor", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cfg.save(config_path)
                print(f"[✔] Config saved to {config_path}")
    finally:
        stream.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
