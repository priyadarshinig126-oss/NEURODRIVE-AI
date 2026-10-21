from pathlib import Path
class SeatbeltDetector:
    def __init__(self,weights="models/seatbelt.pt"):self.weights=Path(weights);self.available=self.weights.exists()
    def status(self):return "CUSTOM_MODEL_AVAILABLE" if self.available else "HEURISTIC_MODE"
