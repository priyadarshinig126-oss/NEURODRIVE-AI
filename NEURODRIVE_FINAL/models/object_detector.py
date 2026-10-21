from pathlib import Path
import time

import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(model_path)
        # Classes available in the bundled YOLO11n COCO weights. This is not a
        # custom seatbelt, traffic-light-state, or wall/table detector.
        self.allowed = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "suitcase",
            "chair", "couch", "potted plant", "dining table", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator",
        }
        self.memory = {}
        self.next_id = 1

    def detect(self, frame, conf=0.35):
        results = self.model.predict(frame, conf=conf, verbose=False)
        detections = []
        if not results:
            return detections
        names = results[0].names
        for box in results[0].boxes:
            class_name = names[int(box.cls[0])]
            if class_name not in self.allowed:
                continue
            bbox = box.xyxy[0].cpu().numpy().tolist()
            h, w = frame.shape[:2]
            cx = (bbox[0] + bbox[2]) / 2 / max(w, 1)
            zone = "LEFT" if cx < 0.36 else "RIGHT" if cx > 0.64 else "FRONT"
            detections.append({
                "class_name": class_name,
                "confidence": float(box.conf[0]),
                "bbox": [int(value) for value in bbox],
                "direction": zone,
            })
        return detections

    @staticmethod
    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
        area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def track_ids(self, detections):
        used_ids = set()
        updated_memory = {}
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            best_id, best_score = None, 0.0
            for track_id, previous in self.memory.items():
                if track_id in used_ids:
                    continue
                if previous["class_name"] != detection["class_name"]:
                    continue
                score = self.iou(previous["bbox"], detection["bbox"])
                if score > best_score:
                    best_score, best_id = score, track_id
            if best_id is None or best_score < 0.20:
                best_id = self.next_id
                self.next_id += 1
            previous = self.memory.get(best_id)
            detection["track_id"] = best_id
            detection["center"] = center
            if previous:
                old_x, old_y = previous["center"]
                detection["movement_px"] = (center[0] - old_x, center[1] - old_y)
                detection["previous_distance_m"] = previous.get("distance_m")
                detection["previous_timestamp"] = previous.get("timestamp")
            else:
                detection["movement_px"] = (0.0, 0.0)
                detection["previous_distance_m"] = None
                detection["previous_timestamp"] = None
            detection["distance_m"] = None
            detection["ttc_s"] = None
            used_ids.add(best_id)
            updated_memory[best_id] = {
                "bbox": detection["bbox"],
                "class_name": detection["class_name"],
                "center": center,
                "distance_m": previous.get("distance_m") if previous else None,
                "timestamp": previous.get("timestamp") if previous else None,
            }
        self.memory = updated_memory
        return detections

    def update_distances(self, detections):
        now = time.monotonic()
        for detection in detections:
            track_id = detection.get("track_id")
            current_distance = detection.get("distance_m")
            if current_distance is None:
                continue
            previous_distance = detection.get("previous_distance_m")
            previous_timestamp = detection.get("previous_timestamp")
            closing_speed = None
            ttc = None
            if previous_distance is not None and previous_timestamp is not None:
                elapsed = now - previous_timestamp
                if elapsed > 0.05:
                    closing_speed = (previous_distance - current_distance) / elapsed
                    if closing_speed > 0.1:
                        ttc = current_distance / closing_speed
            detection["closing_speed_mps"] = closing_speed
            detection["ttc_s"] = round(ttc, 2) if ttc is not None else None
            if track_id in self.memory:
                self.memory[track_id]["distance_m"] = current_distance
                self.memory[track_id]["timestamp"] = now
        return detections

    def draw(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = f'#{detection["track_id"]} {detection["class_name"]} {detection["confidence"]:.0%}'
            distance = detection.get("distance_m")
            if distance is not None:
                label += f" | {distance:.1f}m"
            ttc = detection.get("ttc_s")
            if ttc is not None:
                label += f" | TTC {ttc:.1f}s"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            text_y = max(20, y1 - 8)
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        return frame
