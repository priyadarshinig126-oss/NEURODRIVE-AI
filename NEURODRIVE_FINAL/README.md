# NEURODRIVE-AI

A Streamlit research prototype for camera-based driver monitoring and road-scene perception. It is not a certified ADAS/autonomous driving system and must not be used to control a real vehicle.

## Run locally

1. Activate your existing virtual environment.
2. Install dependencies if needed: `pip install -r requirements.txt`
3. From this project folder run: `streamlit run app.py`

## Input sources

- **Live Camera**: browser camera through WebRTC; allow camera permission.
- **Upload Video**: process an existing video frame by frame.
- **Take / Upload Image**: take a picture or upload a still image for object detection.
- **Demo Simulation**: run the built-in demo frames.

## Operating contexts

- **Driver-operated car**: runs face/eye/yawn heuristics from the driver camera view. This assumes the camera sees the driver's face; a road-facing camera alone cannot monitor the driver.
- **Driverless car (research simulation)**: suppresses driver-monitoring logic and focuses on visible road-scene objects and estimated risk. It does not steer, brake, or control a vehicle.

## Model capability limits

The included `models/yolo11n.pt` is a standard COCO model. It can detect common classes such as people, cars, bicycles, motorcycles, buses, trucks, some animals, chairs, dining tables, and traffic-light objects. It does **not** by itself provide reliable seatbelt classification, traffic-light color/state, wall/desk detection, lane-level overtaking assessment, or a complete 360-degree view. These need suitable trained models, sensors, or additional data.

Distance is monocular and approximate. TTC is only a rough estimate when repeated detections can be associated across frames. Direction labels (LEFT / FRONT / RIGHT) describe horizontal position in the image, not real-world lane direction. Google Maps directions open externally; the app does not directly read GPS coordinates.

## Tests

Run `pytest -q` from this folder. The included unit tests cover basic risk-engine behavior only; they do not certify perception accuracy or hardware/camera compatibility.


## Consolidated navigation/dashboard update

The sidebar accepts an optional starting location and destination. When a destination is entered, the main page shows an embedded Google Maps preview beside the current input/vehicle-mode context, plus a link to full Google Maps directions. The embedded map is a preview; it does not provide live turn-by-turn navigation or expose browser GPS coordinates to the app.

### Run locally
Activate the project's Python environment and run:
```powershell
streamlit run app.py
```
Keep the model file at `models/yolo11n.pt`. This is a research prototype; estimated distance and risk output are not safety-certified.
