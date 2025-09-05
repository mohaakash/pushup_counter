import time
import argparse
import os
from collections import defaultdict
from typing import Dict, Tuple, Optional
import cv2
import numpy as np

# make ultralytics optional so module import never raises if package or model is missing
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception:
    YOLO = None

# ---------------- CONFIGURATION ----------------
DEFAULT_MODEL = "yolo11m-pose.pt"
DEFAULT_INPUT = "video/pushup1.mp4"
DEFAULT_OUTPUT = "video/output.mp4"

# Thresholds
DOWN_ANGLE = 90.0       # elbow angle threshold for "down" position
UP_ANGLE = 110.0        # elbow angle threshold for "up" position
MIN_KP_CONF = 0.35      # minimum keypoint confidence
EMA_ALPHA = 0.2         # smoothing factor for angle
MIN_ANGLE_CONF = 0.5    # minimum average confidence for angle calculation
STALE_TTL = 2.5         # seconds before dropping inactive tracks

# Display settings
MAX_DISPLAY_WIDTH = 1280  # Maximum width for display window
MAX_DISPLAY_HEIGHT = 720  # Maximum height for display window

# Font settings
FONT_SCALE = 1.2        # Increased font scale for better readability
FONT_THICKNESS = 3      # Increased font thickness

# ------------------------------------------------

# COCO-17 keypoint indices used by Ultralytics pose models
LEFT = {"shoulder": 5, "elbow": 7, "wrist": 9}
RIGHT = {"shoulder": 6, "elbow": 8, "wrist": 10}


def angle_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point b formed by points a-b-c"""
    ba = a - b
    bc = c - b
    
    # Normalize vectors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0  # Return neutral angle for degenerate cases
    
    nba = ba / norm_ba
    nbc = bc / norm_bc
    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def pick_best_side(kps_xy: np.ndarray, kps_conf: Optional[np.ndarray], min_conf: float) -> Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """Pick the side with highest confidence keypoints for arm tracking"""
    
    def get_side_info(side_name: str, indices: Dict[str, int]):
        # Check if all required keypoints meet confidence threshold
        if kps_conf is not None:
            confidences = [kps_conf[indices[joint]] for joint in ["shoulder", "elbow", "wrist"]]
            if any(conf < min_conf for conf in confidences):
                return side_name, None, 0.0
            avg_conf = np.mean(confidences)
        else:
            avg_conf = 1.0
        
        # Extract keypoints
        shoulder = kps_xy[indices["shoulder"]]
        elbow = kps_xy[indices["elbow"]]
        wrist = kps_xy[indices["wrist"]]
        
        return side_name, (shoulder, elbow, wrist), avg_conf
    
    left_name, left_points, left_conf = get_side_info("left", LEFT)
    right_name, right_points, right_conf = get_side_info("right", RIGHT)

    # If neither side has valid keypoints
    if left_points is None and right_points is None:
        return "none", (None, None, None), 0.0

    # Prefer the side with higher average confidence
    if left_points is None:
        return right_name, right_points, right_conf
    if right_points is None:
        return left_name, left_points, left_conf

    if left_conf >= right_conf:
        return left_name, left_points, left_conf
    return right_name, right_points, right_conf


def draw_skeleton_lines(image: np.ndarray, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray,
                        color=(0, 255, 0), thickness=3):
    """Draw simple arm skeleton (shoulder-elbow-wrist)."""
    try:
        p1 = tuple(map(int, shoulder.tolist()))
        p2 = tuple(map(int, elbow.tolist()))
        p3 = tuple(map(int, wrist.tolist()))
        cv2.line(image, p1, p2, color, thickness)
        cv2.line(image, p2, p3, color, thickness)
    except Exception:
        pass


def draw_angle_indicator(image: np.ndarray, elbow: np.ndarray, angle: float,
                         color=(0, 255, 255), radius=30, thickness=2):
    """Draw a circular angle indicator and angle text near the elbow."""
    try:
        center = tuple(map(int, elbow.tolist()))
        cv2.circle(image, center, radius, color, thickness)
        text = f"{int(angle)}°"
        org = (center[0] - radius, center[1] - radius - 10)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.6, color, FONT_THICKNESS - 1, cv2.LINE_AA)
    except Exception:
        pass


def draw_pose_overlay(image: np.ndarray, kps_xy: np.ndarray, kps_conf: Optional[np.ndarray],
                      min_conf: float = MIN_KP_CONF):
    """Draw the best arm side skeleton and angle indicator on the image."""
    side, pts, avg_conf = pick_best_side(kps_xy, kps_conf, min_conf)
    if side == "none" or pts is None:
        return image

    shoulder, elbow, wrist = pts
    angle = angle_three_points(shoulder, elbow, wrist)
    draw_skeleton_lines(image, shoulder, elbow, wrist)
    draw_angle_indicator(image, elbow, angle)
    return image


class PushUpCounter:
    def __init__(self, down_thresh: float, up_thresh: float, ema_alpha: float, min_angle_conf: float):
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.ema_alpha = ema_alpha
        self.min_angle_conf = min_angle_conf
        
        self.state = {}  # {person_id: {"stage": str, "count": int, "angle": float, "side": str, "last_seen": float}}
        self.counts = defaultdict(int)

    def update(self, person_id: int, angle: float, side: str, confidence: float) -> Tuple[int, str, float]:
        if person_id not in self.state:
            self.state[person_id] = {
                "stage": "up",
                "count": 0,
                "angle": angle,
                "side": side,
                "last_seen": time.time()
            }
        
        person = self.state[person_id]
        person["last_seen"] = time.time()
        
        # Smooth angle with EMA if confidence is high enough
        if confidence >= self.min_angle_conf:
            person["angle"] = (self.ema_alpha * angle) + (1 - self.ema_alpha) * person["angle"]
        
        smooth_angle = person["angle"]
        
        # State machine for push-up counting
        if person["stage"] == "up" and smooth_angle < self.down_thresh:
            person["stage"] = "down"
        elif person["stage"] == "down" and smooth_angle > self.up_thresh:
            person["stage"] = "up"
            person["count"] += 1
            self.counts[person_id] = person["count"]
            
        return person["count"], person["stage"], smooth_angle

    def get_total_reps(self) -> int:
        return sum(self.counts.values())

    def cleanup_stale_tracks(self, ttl: float):
        now = time.time()
        stale_ids = [pid for pid, data in self.state.items() if now - data["last_seen"] > ttl]
        for pid in stale_ids:
            del self.state[pid]
            if pid in self.counts:
                del self.counts[pid]

    def get_person_stats(self, person_id: int) -> Optional[Dict]:
        return self.state.get(person_id)


def run_pushup_counter(video_path: str, model_path: str, output_path: str):
    """Main function to run push-up counter on a video file."""
    if YOLO is None:
        print("Ultralytics YOLO package not found. Please install it to run this script.")
        return

    # Initialize model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize counter
    counter = PushUpCounter(DOWN_ANGLE, UP_ANGLE, EMA_ALPHA, MIN_ANGLE_CONF)

    frame_count = 0
    start_time = time.time()

    # Run tracking
    results_generator = model.track(
        source=video_path,
        stream=True,
        conf=0.5,
        iou=0.5,
        verbose=False,
        tracker="bytetrack.yaml",
        pose=True,
    )

    for result in results_generator:
        frame_count += 1
        frame = result.orig_img.copy()

        # Extract detection data
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        person_ids = None

        # Get tracking IDs if available
        if boxes is not None and hasattr(boxes, 'id') and boxes.id is not None:
            try:
                person_ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                person_ids = None

        current_fps = fps if frame_count < 2 else frame_count / (time.time() - start_time)

        # Process each detected person
        if keypoints is not None and getattr(keypoints, "xy", None) is not None:
            try:
                kps_xy = keypoints.xy.cpu().numpy()
                kps_conf = keypoints.conf.cpu().numpy() if getattr(keypoints, "conf", None) is not None else None
            except Exception:
                kps_xy = None
                kps_conf = None

            if kps_xy is not None:
                for i, person_keypoints in enumerate(kps_xy):
                    person_id = int(person_ids[i]) if person_ids is not None and i < len(person_ids) else i
                    person_conf = kps_conf[i] if kps_conf is not None else None

                    side, pts, avg_confidence = pick_best_side(person_keypoints, person_conf, MIN_KP_CONF)
                    if side == "none" or avg_confidence == 0.0:
                        continue

                    shoulder, elbow, wrist = pts
                    angle = angle_three_points(shoulder, elbow, wrist)

                    count, stage, smooth_angle = counter.update(person_id, angle, side, avg_confidence)

                    # Draw skeleton and angle
                    draw_skeleton_lines(frame, shoulder, elbow, wrist)
                    draw_angle_indicator(frame, elbow, smooth_angle)

                    # Display individual stats
                    anchor_x = max(10, min(width - 460, int(elbow[0])))
                    anchor_y = max(55, int(elbow[1]) - 10)
                    
                    cv2.rectangle(frame, (anchor_x, anchor_y - 50), (anchor_x + 450, anchor_y + 60), (0, 0, 0), -1)
                    cv2.putText(frame, f"ID: {person_id}", (anchor_x + 10, anchor_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (255, 255, 255), FONT_THICKNESS - 1, cv2.LINE_AA)
                    cv2.putText(frame, f"REPS: {count}", (anchor_x + 150, anchor_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (0, 255, 0), FONT_THICKNESS - 1, cv2.LINE_AA)
                    cv2.putText(frame, f"STAGE: {stage.upper()}", (anchor_x + 10, anchor_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (255, 255, 0), FONT_THICKNESS - 1, cv2.LINE_AA)
                    cv2.putText(frame, f"ANGLE: {int(smooth_angle)}", (anchor_x + 250, anchor_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (255, 0, 255), FONT_THICKNESS - 1, cv2.LINE_AA)

        # Draw total counter
        total_reps = counter.get_total_reps()
        cv2.rectangle(frame, (10, 5), (300, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL REPS: {total_reps}", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0),
                   FONT_THICKNESS, cv2.LINE_AA)

        # Clean up stale tracks
        counter.cleanup_stale_tracks(STALE_TTL)

        # Write frame
        writer.write(frame)

        # Display frame (optional)
        display_width = min(width, MAX_DISPLAY_WIDTH)
        display_height = min(height, MAX_DISPLAY_HEIGHT)
        
        # Maintain aspect ratio
        scale = min(display_width / width, display_height / height)
        resized_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        cv2.imshow("Push-up Counter", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Annotated video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push-up counter using YOLOv8 Pose.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to YOLOv8 pose model.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input video file.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save annotated video.")
    args = parser.parse_args()

    run_pushup_counter(args.input, args.model, args.output)
