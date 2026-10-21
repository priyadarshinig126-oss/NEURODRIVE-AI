from pathlib import Path
APP_TITLE = "NEURODRIVE-AI | Autonomous Driving Intelligence"
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DEFAULT_MODEL = str(MODEL_DIR / "yolo11n.pt")
RELEVANT_CLASSES = {"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign"}
