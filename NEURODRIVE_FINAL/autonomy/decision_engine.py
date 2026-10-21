
import cv2


class DecisionEngine:
    """Generate research-demo warnings and driver guidance."""

    def result(self, action, reason, confidence, signal):
        return {
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "signal": signal,
        }

    def decide(self, risk, lane, objects, driver):
        driver_state = driver.get("state", "UNKNOWN")
        attention = driver.get("attention", "UNKNOWN")
        risk_level = risk.get("risk_level", "UNKNOWN")
        lane_offset = abs(float(lane.get("offset", 0.0)))

        # Driver-monitoring warnings take priority.
        if driver_state == "DROWSY / EYES CLOSED":
            return self.result(
                "DRIVER_ALERT",
                "Possible drowsiness detected. Stop safely when possible.",
                0.95,
                "RED",
            )

        if attention not in {"FORWARD", "UNKNOWN"}:
            return self.result(
                "ATTENTION_ALERT",
                "Possible driver distraction. Check the road ahead.",
                0.85,
                "AMBER",
            )

        # Avoid making a collision claim when risk is unknown.
        if risk_level == "UNKNOWN":
            return self.result(
                "ASSESSING",
                "Insufficient obstacle-distance information to assess risk.",
                0.50,
                "AMBER",
            )

        # Display warnings rather than claiming to control the vehicle.
        if risk_level == "CRITICAL":
            return self.result(
                "URGENT_WARNING",
                "Potential immediate hazard. Assess the road and respond safely.",
                0.95,
                "RED",
            )

        if risk_level == "HIGH":
            return self.result(
                "SLOW_DOWN",
                "Potential high-risk obstacle. Reduce speed if safe and assess surroundings.",
                0.85,
                "RED",
            )

        if risk_level == "MEDIUM":
            return self.result(
                "CAUTION",
                "Potential obstacle detected. Maintain a safe distance.",
                0.75,
                "AMBER",
            )

        if lane_offset > 0.18:
            return self.result(
                "CHECK_LANE_POSITION",
                "Possible lane deviation. Check lane markings and road conditions.",
                0.70,
                "AMBER",
            )

        return self.result(
            "MONITOR_ROAD",
            "No high-risk condition identified by the current estimates.",
            0.75,
            "GREEN",
        )

    def draw(self, frame, decision):
        action = decision.get("action", "UNKNOWN")
        reason = decision.get("reason", "")

        message = f"ACTION: {action} | {reason}"

        # Keep the overlay inside the video frame.
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        max_width = max(100, width - 24)

        words = message.split()
        lines = []
        current_line = ""

        for word in words:
            candidate = (
                f"{current_line} {word}".strip()
            )

            text_width = cv2.getTextSize(
                candidate, font, font_scale, thickness
            )[0][0]

            if text_width <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        line_height = 22
        y = height - 15 - (len(lines) - 1) * line_height

        for line in lines:
            cv2.putText(
                frame,
                line,
                (12, max(20, y)),
                font,
                font_scale,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (12, max(20, y)),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += line_height

        return frame