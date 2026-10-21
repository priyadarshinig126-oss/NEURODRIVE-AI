import os
import tempfile
import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from config.settings import APP_TITLE, DEFAULT_MODEL
from models.object_detector import ObjectDetector
from models.lane_detector import LaneDetector
from models.distance_estimator import DistanceEstimator
from models.driver_monitor import DriverMonitor
from autonomy.risk_engine import RiskEngine
from autonomy.decision_engine import DecisionEngine
from simulation.scenarios import DemoSimulation
from monitoring.database import EventLogger
from monitoring.performance import PerformanceMonitor
from ui.dashboard import (
    render_css,
    render_header,
    render_metrics,
    
    render_decision,
    render_driver_card,
    render_event_table,
)
from utils.report import build_pdf_report

st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="wide")
render_css()


@st.cache_resource
def load_models():
    return (
        ObjectDetector(DEFAULT_MODEL),
        LaneDetector(),
        DistanceEstimator(),
        DriverMonitor(),
        RiskEngine(),
        DecisionEngine(),
    )


detector, lane, distance, driver, risk_engine, decision = load_models()

if "logger" not in st.session_state:
    st.session_state.logger = EventLogger()
if "perf" not in st.session_state:
    st.session_state.perf = PerformanceMonitor()
if "events" not in st.session_state:
    st.session_state.events = []
if "frame_id" not in st.session_state:
    st.session_state.frame_id = 0


class NeuroDriveProcessor(VideoProcessorBase):
    """Real-time WebRTC processor for the laptop/USB camera."""

    def __init__(self, detector_obj, lane_obj, distance_obj, driver_obj, risk_obj, decision_obj, conf, ref, focal):
        self.detector = detector_obj
        self.lane = lane_obj
        self.distance = distance_obj
        self.driver = driver_obj
        self.risk_engine = risk_obj
        self.decision = decision_obj
        self.conf = conf
        self.ref = ref
        self.focal = focal

        self.lock = threading.Lock()
        self.frame_id = 0
        self.latest = {
            "action": "WAITING",
            "reason": "Start the camera",
            "confidence": 0.0,
            "signal": "AMBER",
            "risk_level": "LOW",
            "risk_score": 0.0,
            "min_distance_m": None,
            "ttc_s": None,
            "driver_state": "WAITING",
            "attention": "UNKNOWN",
            "seatbelt": "UNKNOWN",
            "eye_score": 0.0,
            "yawn_score": 0.0,
            "fps": 0.0,
            "objects": 0,
            "lane_offset": 0.0,
            "lane_confidence": 0.0,
        }
        self.pending_events = []
        self.last_event_time = 0.0
        self.last_action = None
        self.last_output = None
        self.process_every = 2  # Process every 2nd camera frame for smoother live video.

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_id += 1

        # Keep the camera responsive by processing every second frame.
        if self.frame_id % self.process_every != 0:
            if self.last_output is not None:
                return av.VideoFrame.from_ndarray(self.last_output, format="bgr24")
            return frame

        start = time.perf_counter()

        try:
            detections = self.detector.track_ids(self.detector.detect(image, self.conf))

            for obj in detections:
                obj["distance_m"] = self.distance.estimate(
                    obj["bbox"], image.shape, self.ref, self.focal, obj["class_name"]
                )

            lane_info = self.lane.detect(image)
            driver_info = self.driver.analyze(image)
            risk_info = self.risk_engine.assess(detections, lane_info)
            decision_info = self.decision.decide(
                risk_info, lane_info, detections, driver_info
            )

            output = self.detector.draw(image.copy(), detections)
            output = self.lane.draw(output, lane_info)
            output = self.driver.draw(output, driver_info)
            output = self.decision.draw(output, decision_info)

            elapsed = max(time.perf_counter() - start, 1e-6)
            fps = 1.0 / elapsed

            # Add a compact live HUD directly on the video.
            self.draw_hud(output, decision_info, risk_info, driver_info, fps, len(detections))

            event = {
                "frame": self.frame_id,
                "action": decision_info["action"],
                "risk": risk_info["risk_level"],
                "risk_score": risk_info["risk_score"],
                "min_distance_m": risk_info["min_distance_m"],
                "ttc_s": risk_info["ttc_s"],
                "driver_state": driver_info["state"],
                "fps": round(fps, 1),
            }

            now = time.time()
            important = decision_info["action"] not in ("CRUISE", "KEEP_LANE")
            changed = decision_info["action"] != self.last_action
            if important and (changed or now - self.last_event_time >= 2.0):
                with self.lock:
                    self.pending_events.append(event)
                self.last_event_time = now

            self.last_action = decision_info["action"]

            with self.lock:
                self.latest = {
                    "action": decision_info["action"],
                    "reason": decision_info["reason"],
                    "confidence": decision_info["confidence"],
                    "signal": decision_info["signal"],
                    "risk_level": risk_info["risk_level"],
                    "risk_score": risk_info["risk_score"],
                    "min_distance_m": risk_info["min_distance_m"],
                    "ttc_s": risk_info["ttc_s"],
                    "driver_state": driver_info["state"],
                    "attention": driver_info["attention"],
                    "seatbelt": driver_info["seatbelt"],
                    "eye_score": driver_info["eye_score"],
                    "yawn_score": driver_info["yawn_score"],
                    "fps": round(fps, 1),
                    "objects": len(detections),
                    "lane_offset": round(float(lane_info["offset"]), 3),
                    "lane_confidence": lane_info["confidence"],
                }

            self.last_output = output
            return av.VideoFrame.from_ndarray(output, format="bgr24")

        except Exception as exc:
            # Keep the camera alive even if one frame causes an AI error.
            error_frame = image.copy()
            cv2.rectangle(error_frame, (10, 10), (min(error_frame.shape[1] - 10, 900), 55), (0, 0, 180), -1)
            cv2.putText(
                error_frame,
                f"AI ERROR: {str(exc)[:100]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            self.last_output = error_frame
            return av.VideoFrame.from_ndarray(error_frame, format="bgr24")

    @staticmethod
    def draw_hud(frame, decision_info, risk_info, driver_info, fps, object_count):
        h, w = frame.shape[:2]
        width = min(w - 20, 440)
        cv2.rectangle(frame, (10, 58), (width, 155), (10, 18, 30), -1)
        cv2.rectangle(frame, (10, 58), (width, 155), (80, 180, 255), 1)

        distance_text = (
            "--" if risk_info["min_distance_m"] is None
            else f"{risk_info['min_distance_m']:.1f} m"
        )
        ttc_text = "--" if risk_info["ttc_s"] is None else f"{risk_info['ttc_s']:.1f} s"

        lines = [
            f"RISK: {risk_info['risk_level']} ({risk_info['risk_score']:.0f})",
            f"ACTION: {decision_info['action']}",
            f"DIST: {distance_text}   TTC: {ttc_text}",
            f"DRIVER: {driver_info['state']}   OBJECTS: {object_count}   FPS: {fps:.1f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, 82 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def drain_events(self):
        with self.lock:
            events = list(self.pending_events)
            self.pending_events.clear()
        return events


render_header()

with st.sidebar:
    st.header("⚙️ Mission Control")
    mode = st.selectbox(
        "Input Mode",
        ["Demo Simulation", "Upload Video", "🔴 Live Camera"],
    )
    conf = st.slider("Detection Confidence", 0.20, 0.80, 0.35, 0.05)
    ref = st.number_input("Reference object height (m)", 0.8, 3.0, 1.5, 0.1)
    focal = st.number_input("Virtual focal length (px)", 300.0, 1600.0, 850.0, 25.0)
    st.caption("Distance is an estimated monocular value.")
    if mode == "🔴 Live Camera":
        st.info("Click START below and allow browser camera permission.")

# ------------------------------------------------------------
# LIVE CAMERA MODE
# ------------------------------------------------------------
if mode == "🔴 Live Camera":
    st.subheader("🔴 Live AI Camera")
    st.caption("Real-time perception runs in the browser camera stream. This is an educational/research demo, not a vehicle controller.")

    processor_factory = lambda: NeuroDriveProcessor(
        detector,
        lane,
        distance,
        driver,
        risk_engine,
        decision,
        conf,
        ref,
        focal,
    )

    ctx = webrtc_streamer(
        key="neurodrive-live-camera",
        video_processor_factory=processor_factory,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    metrics_placeholder = st.empty()
    decision_placeholder = st.empty()
    driver_placeholder = st.empty()
    event_placeholder = st.empty()

    if ctx.state.playing:
        while ctx.state.playing:
            processor = ctx.video_processor
            if processor is not None:
                data = processor.snapshot()
                new_events = processor.drain_events()

                for event in new_events:
                    st.session_state.events.append(event)
                    st.session_state.logger.log(event)
                    st.session_state.perf.update(event["fps"])

                if data["fps"]:
                    st.session_state.perf.update(data["fps"])

                metrics_placeholder.markdown(
                    f"""
                    <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:12px;'>
                    <div><small>Autonomy</small><h3>{data['action']}</h3></div>
                    <div><small>Risk</small><h3>{data['risk_level']} ({data['risk_score']:.0f})</h3></div>
                    <div><small>Distance</small><h3>{'—' if data['min_distance_m'] is None else f"{data['min_distance_m']:.1f} m"}</h3></div>
                    <div><small>Objects</small><h3>{data['objects']}</h3></div>
                    <div><small>FPS</small><h3>{data['fps']:.1f}</h3></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with decision_placeholder.container():
                    st.subheader("🧠 Autonomous Decision")
                    st.markdown(f"### `{data['action']}`")
                    st.write(data["reason"])
                    st.progress(min(1.0, data["confidence"]), text=f"Decision confidence: {data['confidence']:.0%}")
                    st.caption(
                        f"Estimated TTC: {data['ttc_s']:.2f}s"
                        if data["ttc_s"] is not None
                        else "TTC: not available"
                    )

                with driver_placeholder.container():
                    st.subheader("👤 Driver Monitor")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("State", data["driver_state"])
                    c2.metric("Attention", data["attention"])
                    c3.metric("Seatbelt", data["seatbelt"])
                    c4.metric("Objects", data["objects"])
                    st.caption(f"Eye openness: {data['eye_score']}  |  Yawn score: {data['yawn_score']}")

                with event_placeholder.container():
                    st.subheader("📊 Recent Safety Events")
                    render_event_table(st.session_state.events[-10:])

            time.sleep(0.15)

    else:
        st.info("Click **START** in the camera widget to begin live detection.")

    st.divider()
    st.caption("Live mode uses WebRTC for continuous browser-camera streaming.")

# ------------------------------------------------------------
# EXISTING DEMO / UPLOAD MODES
# ------------------------------------------------------------
else:
    source = None
    if mode == "Upload Video":
        source = st.file_uploader(
            "Upload road/driving video",
            type=["mp4", "avi", "mov", "mkv"],
        )

    cols = st.columns(4)
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("🎥 NeuroDrive Perception")
        frame_box = st.empty()
    with right:
        st.subheader("🧠 Autonomous Decision")
        decision_box = st.empty()
        st.subheader("👤 Driver Monitor")
        driver_box = st.empty()

    if mode == "Demo Simulation":
        frames = DemoSimulation().frames()
    elif mode == "Upload Video" and source:
        os.makedirs("uploads", exist_ok=True)
        suffix = os.path.splitext(source.name)[1] or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix, dir="uploads")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(source.getvalue())

        cap = cv2.VideoCapture(path)

        def video_frames():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()

        frames = video_frames()
    else:
        frames = []

    if not frames:
        st.markdown(
            "### 🚦 Ready for autonomous perception\n\n"
            "Start with **Demo Simulation** or choose **🔴 Live Camera**.\n\n"
            "`Perceive → Track → Estimate → Predict → Assess Risk → Decide → Act → Explain`"
        )
    else:
        for frame in frames:
            if frame is None:
                continue

            t0 = time.perf_counter()
            st.session_state.frame_id += 1

            ds = detector.track_ids(detector.detect(frame, conf))
            for d in ds:
                d["distance_m"] = distance.estimate(
                    d["bbox"], frame.shape, ref, focal, d["class_name"]
                )

            li = lane.detect(frame)
            di = driver.analyze(frame)
            ri = risk_engine.assess(ds, li)
            de = decision.decide(ri, li, ds, di)

            out = decision.draw(
                driver.draw(lane.draw(detector.draw(frame.copy(), ds), li), di),
                de,
            )
            fps = 1 / max(time.perf_counter() - t0, 1e-6)
            st.session_state.perf.update(fps)

            ev = {
                "frame": st.session_state.frame_id,
                "action": de["action"],
                "risk": ri["risk_level"],
                "risk_score": ri["risk_score"],
                "min_distance_m": ri["min_distance_m"],
                "ttc_s": ri["ttc_s"],
                "driver_state": di["state"],
                "fps": round(fps, 1),
            }

            if de["action"] not in ("CRUISE", "KEEP_LANE"):
                st.session_state.events.append(ev)
                st.session_state.logger.log(ev)

            frame_box.image(
                cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            render_metrics(cols, de, ri, li, fps)
            render_decision(decision_box, de, ri)
            render_driver_card(driver_box, di)

    st.divider()
    a, b = st.columns([2, 1])
    with a:
        st.subheader("📊 Event Timeline")
        render_event_table(st.session_state.events[-30:])
    with b:
        st.subheader("📄 Research Report")
        if st.button("Generate PDF Report"):
            report_path = build_pdf_report(
                st.session_state.events,
                st.session_state.perf.summary(),
                "reports/neurodrive_report.pdf",
            )
            with open(report_path, "rb") as report_file:
                st.download_button(
                    "Download Report",
                    report_file,
                    file_name="neurodrive_report.pdf",
                    mime="application/pdf",
                )
