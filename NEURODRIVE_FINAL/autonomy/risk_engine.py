class RiskEngine:
    """Conservative perception risk estimator for research use only."""
    VEHICLES_AND_OBSTACLES = {"car", "truck", "bus", "motorcycle", "person", "bicycle", "dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear", "zebra", "giraffe"}

    def assess(self, objects, lane):
        candidates = [o for o in objects if o.get("class_name") in self.VEHICLES_AND_OBSTACLES and o.get("distance_m") is not None]
        # Prioritize objects in the central image corridor, while retaining the
        # closest object's direction for an interpretable warning.
        def priority(o):
            x1, _, x2, _ = o.get("bbox", [0, 0, 0, 0])
            center = (x1 + x2) / 2
            return (0 if o.get("direction", "FRONT") == "FRONT" else 1, float(o["distance_m"]))
        closest = min(candidates, key=priority) if candidates else None
        if closest is None:
            return {"risk_level":"UNKNOWN", "risk_score":0.0, "min_distance_m":None, "ttc_s":None, "closing_speed_mps":None, "closest":None, "message":"No measurable road obstacle detected", "direction":"UNKNOWN"}
        distance = max(0.0, float(closest["distance_m"]))
        ttc = closest.get("ttc_s")
        closing = closest.get("closing_speed_mps")
        # Monocular distance is approximate. Do not claim a collision solely
        # from proximity; combine distance, center-corridor position and TTC.
        central = closest.get("direction", "FRONT") == "FRONT"
        proximity = max(0.0, min(75.0, (18.0 - distance) * 4.0)) if central else max(0.0, min(35.0, (12.0 - distance) * 3.0))
        lane_score = min(15.0, abs(float(lane.get("offset", 0.0))) * 30.0)
        score = min(100.0, proximity + lane_score)
        level = "LOW"
        if central and distance < 4: level = "CRITICAL"
        elif central and distance < 7: level = "HIGH"
        elif central and distance < 13: level = "MEDIUM"
        elif not central and distance < 5: level = "MEDIUM"
        if ttc is not None and closing is not None and closing > 0.1:
            if ttc < 1.5 and central: level, score = "CRITICAL", max(score, 90.0)
            elif ttc < 3.0 and central and level != "CRITICAL": level, score = "HIGH", max(score, 70.0)
        direction = closest.get("direction", "UNKNOWN")
        messages = {"CRITICAL":f"Potential urgent hazard toward {direction.lower()} of the camera view", "HIGH":f"Potential high-risk object toward {direction.lower()}; assess surroundings", "MEDIUM":f"Object nearby toward {direction.lower()}; maintain caution", "LOW":"No high risk indicated by current estimates"}
        return {"risk_level":level, "risk_score":round(score,1), "min_distance_m":round(distance,1), "ttc_s":ttc, "closing_speed_mps":closing, "closest":closest, "message":messages[level], "direction":direction}
