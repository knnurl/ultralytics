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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

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
    conveyor_speed_mms: float = 0.0
    robot_latency_s: float = 0.0
    smoothing_alpha: float = 0.35

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
        self.prev_robot_xy: Optional[np.ndarray] = None

    def pixel_to_robot(self, px: float, py: float, w: int, h: int) -> Tuple[float, float]:
        dx = px - w / 2
        dy = py - h / 2
        return (
            self.cfg.cam_center_robot_x + dy * self.cfg.mm_per_pixel,
            self.cfg.cam_center_robot_y + dx * self.cfg.mm_per_pixel,
        )

    def predict_pick_xy(self, robot_xy: Tuple[float, float]) -> Tuple[float, float]:
        if self.cfg.conveyor_speed_mms == 0 or self.cfg.robot_latency_s == 0:
            return robot_xy
        dx = self.cfg.conveyor_speed_mms * self.cfg.robot_latency_s
        return (robot_xy[0] + dx, robot_xy[1])

    def smooth_xy(self, robot_xy: Tuple[float, float]) -> Tuple[float, float]:
        if self.prev_robot_xy is None:
            self.prev_robot_xy = np.array(robot_xy, dtype=float)
            return robot_xy
        arr = np.array(robot_xy, dtype=float)
        self.prev_robot_xy = self.cfg.smoothing_alpha * arr + (1 - self.cfg.smoothing_alpha) * self.prev_robot_xy
        return tuple(self.prev_robot_xy.tolist())

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

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.cfg.roi
        x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 1, w))
        y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 1, h))
        if x2 <= x1 or y2 <= y1:
            return None
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
            return None

        res = results[0]
        if not res.boxes:
            return None

        masks = res.masks.data if res.masks is not None else None

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
            rx, ry = self.smooth_xy((rx, ry))
            predicted_xy = self.predict_pick_xy((rx, ry))
            label = f"cls={cls_id} rx={rx:.1f} ry={ry:.1f} a={angle:.1f}"
            return Detection(label, (px, py), angle, (rx, ry), predicted_xy)
        return None

    def draw(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = self.cfg.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        px, py = map(int, detection.pixel_xy)
        cv2.drawMarker(vis, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            vis,
            detection.label,
            (max(px - 120, 10), max(py - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        pred_x, pred_y = map(int, detection.predicted_robot_xy)
        cv2.putText(
            vis,
            f"pred pick -> {pred_x:.1f},{pred_y:.1f} mm",
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
    stream = create_stream()
    print("Press 'q' to quit, 's' to save config.")

    try:
        for frame in stream:
            detection = pipeline.detect(frame)
            output = pipeline.draw(frame, detection) if detection else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if detection:
                print("[ROBOT] ->", detection)

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
