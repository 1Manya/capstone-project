"""
demo_dashboard.py
===================
Presentation-layer dashboard for the safety mechanism, built on top of the
already-tested modules (trust_engine.py, temporal_validator.py,
failsafe_state_machine.py, hash_chain_logger.py) as real imports - same
logic that runs in the ROS 2 pipeline, no ROS dependency needed here.

This is deliberately SELF-CONTAINED and manually controlled (step-by-step,
attack toggle) rather than auto-playing on a timer, so a live demo in front
of a panel is fully presenter-controlled and can't desync or lag.

Usage:
    streamlit run demo_dashboard.py -- --frames-dir path/to/frames

Or point --frames-dir at extracted nuScenes camera frames (e.g. export a
scene's CAM_FRONT images as PNGs first, or point this at any folder of
sequential images for a quick test/demo run).
"""

import argparse
import random
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trust_engine import TrustEngine
from temporal_validator import TemporalValidator
from failsafe_state_machine import FailSafeStateMachine
from hash_chain_logger import HashChainLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="frames")
    ap.add_argument("--attacked-frames-dir", default="frames_attacked")
    ap.add_argument("--log-path", default="demo_audit_log.jsonl")
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args


def init_state(args):
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.frame_paths = sorted(Path(args.frames_dir).glob("*.png")) or \
                                    sorted(Path(args.frames_dir).glob("*.jpg"))
    st.session_state.attacked_frame_paths = sorted(Path(args.attacked_frames_dir).glob("*.png")) or \
                                             sorted(Path(args.attacked_frames_dir).glob("*.jpg"))
    st.session_state.frame_idx = 0
    st.session_state.trust_engine = TrustEngine()
    st.session_state.validator = TemporalValidator()
    st.session_state.state_machine = FailSafeStateMachine()
    st.session_state.logger = HashChainLogger(args.log_path)
    st.session_state.attack_mode = False
    st.session_state.history = []  # list of dicts for the trend chart


def step_frame():
    ss = st.session_state
    if not ss.frame_paths:
        return
    ss.frame_idx = (ss.frame_idx + 1) % len(ss.frame_paths)

    # simulate the CNN's per-frame output - deliberately controlled by the
    # presenter's attack toggle, not random, so the demo is predictable live
    if ss.attack_mode:
        label = 1
        confidence = round(random.uniform(0.82, 0.98), 3)
    else:
        label = 0
        confidence = round(random.uniform(0.80, 0.97), 3)

    confirmed = ss.validator.update(label)
    weights = ss.trust_engine.update("camera", label, confidence)
    state = ss.state_machine.update(confirmed, weights["camera"])

    if confirmed and state != "NORMAL":
        ss.logger.log_event(
            frame_id=f"frame_{ss.frame_idx:05d}",
            vehicle_position={"lat": 0.0, "lon": 0.0},  # placeholder, wire to real ego_pose later
            cnn_confidence=confidence,
            trust_weights=weights,
            decision=state,
        )

    ss.history.append({"frame": ss.frame_idx, **weights, "state": state})
    if len(ss.history) > 30:
        ss.history.pop(0)


def tamper_log():
    """Demo moment: edit an old entry directly, then re-verify to show detection."""
    ss = st.session_state
    lines = ss.logger._read_lines()
    if len(lines) < 1:
        st.warning("No log entries yet to tamper with - trigger an attack first.")
        return
    import json
    idx = 0
    entry = json.loads(lines[idx])
    entry["cnn_confidence"] = 0.01  # attacker tries to hide the real confidence
    lines[idx] = json.dumps(entry)
    with open(ss.logger.log_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    st.set_page_config(page_title="Safety Mechanism Demo", layout="wide")
    init_state(args)
    ss = st.session_state

    st.title("Rolling-Shutter Attack Safety Mechanism - Live Demo")

    with st.sidebar:
        st.subheader("Presenter controls")
        ss.attack_mode = st.toggle("Simulate attack ON", value=ss.attack_mode)
        if st.button("Next frame ▶", use_container_width=True):
            step_frame()
        if st.button("Tamper log (demo)", use_container_width=True):
            tamper_log()
        if st.button("Reset demo", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("Camera feed - clean (ground truth)")
        if ss.frame_paths:
            img = Image.open(ss.frame_paths[ss.frame_idx % len(ss.frame_paths)])
            st.image(img, use_container_width=True)
        else:
            st.info("No frames found in --frames-dir")

        st.subheader("Camera feed - what the CNN sees")
        if ss.attack_mode and ss.attacked_frame_paths:
            atk_img = Image.open(ss.attacked_frame_paths[ss.frame_idx % len(ss.attacked_frame_paths)])
            st.image(atk_img, use_container_width=True)
            st.caption("Rolling-shutter stripes injected via the real simulator "
                       "(generate_attacked_frames.py) - this is what's actually "
                       "corrupting the frame right now.")
        elif ss.attack_mode:
            st.warning("Attack ON, but no attacked frames found - run "
                       "generate_attacked_frames.py first.")
        else:
            st.image(img if ss.frame_paths else None, use_container_width=True)
            st.caption("Attack is OFF - CNN sees the same clean frame.")

        st.subheader("Vehicle state")
        state = ss.state_machine.state
        color = {"NORMAL": "green", "DEGRADED": "orange", "EMERGENCY_STOP": "red"}[state]
        st.markdown(f"### :{color}[{state}]")

    with col2:
        st.subheader("Sensor trust weights")
        weights = ss.trust_engine.weights
        st.bar_chart(weights)

        st.subheader("Audit log (hash-chain)")
        report = ss.logger.verify_chain()
        if report["valid"]:
            st.success(f"Chain valid - {report['total_entries']} entries")
        else:
            st.error(f"CHAIN BROKEN at entry {report['broken_at']} - tampering detected!")

        lines = ss.logger._read_lines()
        if lines:
            import json
            recent = [json.loads(l) for l in lines[-5:]]
            st.dataframe(
                [{"frame": e["frame_id"], "decision": e["decision"],
                  "confidence": round(e["cnn_confidence"], 2),
                  "hash": e["entry_hash"][:10] + "..."} for e in recent],
                use_container_width=True,
            )
        else:
            st.caption("No confirmed attack events logged yet.")


if __name__ == "__main__":
    main()
