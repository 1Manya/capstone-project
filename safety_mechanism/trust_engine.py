"""
trust_engine.py
================
Bayesian sensor trust engine, standalone (no ROS) so it can be built and
unit-tested independently of everything else in the pipeline.

Design grounded in: R. S. Hallyburton & M. Pajic, "Bayesian Methods for
Trust in Collaborative Multi-Agent Autonomy" (arxiv.org/abs/2403.16956).
Core idea adapted from their framework: sensor trust is a value in [0, 1],
updated multiplicatively based on how strongly current evidence agrees or
disagrees with "this sensor is trustworthy right now" - trust drops sharply
on strong disagreement (high-confidence attack signal), barely moves on
weak/ambiguous evidence, and recovers GRADUALLY (not instantly) on sustained
agreement (clean signal) - this asymmetry (fast drop, slow recovery) is a
deliberate safety property, not an implementation detail: a compromised
camera should lose trust quickly, but regaining full trust should require
sustained good behavior, not just one clean frame.

This is NOT a reimplementation of AVstack - it's a smaller, purpose-built
adaptation of their general trust-fusion concept for this project's specific
scope (camera/LiDAR/Radar/IMU, one CNN attack-confidence signal).
"""

from dataclasses import dataclass, field


@dataclass
class TrustEngine:
    # current trust weight per sensor, must sum to 1.0 at all times
    weights: dict = field(default_factory=lambda: {
        "camera": 0.40, "lidar": 0.25, "radar": 0.20, "imu": 0.15
    })

    # how aggressively trust reacts to a single frame's evidence.
    # Higher = faster reaction (both drop and recovery), tune via unit tests.
    decay_rate: float = 1.5      # controls how hard camera trust drops on attack
    recovery_rate: float = 0.15  # deliberately much slower than decay_rate

    def update(self, sensor: str, label: int, confidence: float) -> dict:
        """
        Update trust weights given one frame's CNN output for `sensor`
        (normally "camera", since that's what the CNN watches).

        label: 0 = clean, 1 = attacked
        confidence: 0.0-1.0, how confident the CNN is in that label

        Returns the new weights dict (also stored in self.weights).
        """
        if sensor not in self.weights:
            raise ValueError(f"Unknown sensor: {sensor}")

        current = self.weights[sensor]

        if label == 1:
            # attack signal: drop trust proportionally to confidence.
            # confidence near 1.0 -> sharp drop. confidence near 0.5
            # (borderline/ambiguous) -> barely moves.
            severity = max(0.0, (confidence - 0.5) * 2)  # rescale 0.5-1.0 -> 0-1
            new_value = current * (1.0 - self.decay_rate * severity * 0.5)
        else:
            # clean signal: recover GRADUALLY toward this sensor's original
            # baseline share, scaled by confidence, but capped by recovery_rate
            # so a single clean frame can't undo a multi-frame trust drop.
            baseline = self._baseline(sensor)
            new_value = current + (baseline - current) * self.recovery_rate * confidence

        new_value = max(0.01, min(new_value, 0.90))  # never fully 0 or fully dominant
        self.weights[sensor] = new_value
        self._renormalize()
        return dict(self.weights)

    def _baseline(self, sensor: str) -> float:
        """Original default share for a sensor, used as the recovery target."""
        defaults = {"camera": 0.40, "lidar": 0.25, "radar": 0.20, "imu": 0.15}
        return defaults[sensor]

    def _renormalize(self):
        """Redistribute the trust lost/gained by one sensor across the others,
        proportionally to their current weights, so the total always sums to 1.0."""
        total = sum(self.weights.values())
        if total <= 0:
            return
        for k in self.weights:
            self.weights[k] /= total


# --------------------------------------------------------------------------
# Unit tests - run with: python3 trust_engine.py
# --------------------------------------------------------------------------
def run_tests():
    print("Test 1: high-confidence attack -> camera weight drops sharply")
    te = TrustEngine()
    before = te.weights["camera"]
    te.update("camera", label=1, confidence=0.95)
    after = te.weights["camera"]
    print(f"  camera: {before:.3f} -> {after:.3f}")
    assert after < before * 0.85, "camera trust should drop sharply on high-confidence attack"
    print("  PASS\n")

    print("Test 2: low/borderline confidence -> weight barely moves")
    te = TrustEngine()
    before = te.weights["camera"]
    te.update("camera", label=1, confidence=0.52)
    after = te.weights["camera"]
    print(f"  camera: {before:.3f} -> {after:.3f}")
    assert abs(after - before) < 0.03, "borderline confidence should barely change trust"
    print("  PASS\n")

    print("Test 3: sustained clean frames -> weight recovers, but gradually")
    te = TrustEngine()
    te.weights["camera"] = 0.10  # simulate a prior attack having dropped it
    te._renormalize()
    trajectory = [te.weights["camera"]]
    for _ in range(10):
        te.update("camera", label=0, confidence=0.9)
        trajectory.append(te.weights["camera"])
    print(f"  camera trajectory over 10 clean frames: {[round(v, 3) for v in trajectory]}")
    assert trajectory[1] < trajectory[0] + 0.05, "should NOT jump back instantly on one clean frame"
    assert trajectory[-1] > trajectory[0], "should recover meaningfully over sustained clean frames"
    print("  PASS\n")

    print("Test 4: weights always sum to 1.0")
    te = TrustEngine()
    for label, conf in [(1, 0.9), (0, 0.8), (1, 0.6), (0, 0.95)]:
        te.update("camera", label, conf)
        total = sum(te.weights.values())
        assert abs(total - 1.0) < 1e-6, f"weights must sum to 1.0, got {total}"
    print(f"  final weights: { {k: round(v,3) for k,v in te.weights.items()} }, sum={sum(te.weights.values()):.6f}")
    print("  PASS\n")

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
