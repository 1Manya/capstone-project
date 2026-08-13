#!/usr/bin/env python3
"""
stub_cnn_publisher.py
======================
Fakes the CNN detector's output on /detector/attack_status, so the safety
mechanism (trust engine, temporal validator, fail-safe state machine) can be
built and tested WITHOUT the real CNN being ready or reachable.

Publishes std_msgs/String containing a small JSON payload:
    {"frame_id": "...", "label": 0 or 1, "confidence": 0.0-1.0, "timestamp": float}

NOTE: this uses std_msgs/String + JSON for simplicity so you don't need a
custom .msg package to get started today. If your teammate's real CNN later
publishes a custom message type instead, only parse_cnn_output() in your
downstream nodes needs to change - see the blueprint's integration notes.

Usage:
    python3 stub_cnn_publisher.py --pattern mixed --rate 2.0
    python3 stub_cnn_publisher.py --pattern clean_streak
    python3 stub_cnn_publisher.py --pattern sustained_attack
    python3 stub_cnn_publisher.py --pattern single_noisy_frame
"""

import argparse
import json
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Each pattern is a list of (label, confidence) tuples, played in a loop.
# Designed to exercise specific behaviors your trust engine / temporal
# validator need to handle correctly.
PATTERNS = {
    # Long clean run -> trust should stay high / fully recover if it dipped
    "clean_streak": [(0, round(random.uniform(0.85, 0.99), 3)) for _ in range(20)],

    # A SINGLE noisy frame in the middle of clean ones -> temporal validator
    # should NOT confirm an attack from this alone (majority-of-6 rule)
    "single_noisy_frame": (
        [(0, round(random.uniform(0.85, 0.99), 3)) for _ in range(8)]
        + [(1, round(random.uniform(0.6, 0.95), 3))]  # the one noisy frame
        + [(0, round(random.uniform(0.85, 0.99), 3)) for _ in range(8)]
    ),

    # 4+ of 6 frames attacked -> temporal validator SHOULD confirm an attack,
    # trust engine should shift weight away from camera
    "sustained_attack": (
        [(0, round(random.uniform(0.85, 0.99), 3)) for _ in range(5)]
        + [(1, round(random.uniform(0.85, 0.99), 3)) for _ in range(8)]
        + [(0, round(random.uniform(0.85, 0.99), 3)) for _ in range(5)]  # then recovers
    ),

    # Randomized mix of everything, for general stress-testing
    "mixed": None,  # generated dynamically, see below
}


class StubCNNPublisher(Node):
    def __init__(self, pattern: str, rate_hz: float):
        super().__init__("stub_cnn_publisher")
        self.pub = self.create_publisher(String, "/detector/attack_status", 10)
        self.pattern_name = pattern
        self.sequence = PATTERNS.get(pattern)
        self.frame_idx = 0

        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self.publish_next)
        self.get_logger().info(f"Stub CNN publisher started, pattern='{pattern}', rate={rate_hz}Hz")

    def next_label_confidence(self):
        if self.pattern_name == "mixed":
            label = random.choices([0, 1], weights=[0.75, 0.25])[0]
            confidence = round(random.uniform(0.6, 0.99), 3)
            return label, confidence
        else:
            item = self.sequence[self.frame_idx % len(self.sequence)]
            return item

    def publish_next(self):
        label, confidence = self.next_label_confidence()
        msg_dict = {
            "frame_id": f"frame_{self.frame_idx:05d}",
            "label": label,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(msg_dict)
        self.pub.publish(msg)
        self.get_logger().info(f"Published: {msg_dict}")
        self.frame_idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=list(PATTERNS.keys()), default="mixed")
    ap.add_argument("--rate", type=float, default=2.0)
    args = ap.parse_args()

    rclpy.init()
    node = StubCNNPublisher(args.pattern, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
