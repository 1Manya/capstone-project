"""
temporal_validator.py
=======================
Plain Python, no ROS - confirms an attack only when a MAJORITY of the last
N frames (default 6, need 4+) were flagged attacked, so a single noisy
frame from the CNN can't trigger a false state change downstream.
"""
from collections import deque


class TemporalValidator:
    def __init__(self, window_size: int = 6, majority_threshold: int = 4):
        self.window_size = window_size
        self.majority_threshold = majority_threshold
        self.window = deque(maxlen=window_size)

    def update(self, label: int) -> bool:
        """Feed in this frame's label (0/1). Returns True if attack CONFIRMED
        (majority of window is attacked), False otherwise."""
        self.window.append(label)
        attacked_count = sum(self.window)
        return attacked_count >= self.majority_threshold

    def reset(self):
        self.window.clear()


def run_tests():
    print("Test 1: single noisy frame among clean -> NOT confirmed")
    v = TemporalValidator()
    results = [v.update(l) for l in [0,0,0,1,0,0]]
    print(f"  labels=[0,0,0,1,0,0] -> confirmed={results[-1]}")
    assert results[-1] == False
    print("  PASS\n")

    print("Test 2: 4-of-6 attacked -> CONFIRMED")
    v = TemporalValidator()
    results = [v.update(l) for l in [0,1,1,1,1,0]]
    print(f"  labels=[0,1,1,1,1,0] -> confirmed={results[-1]}")
    assert results[-1] == True
    print("  PASS\n")

    print("Test 3: exactly 3-of-6 attacked -> NOT confirmed (below threshold)")
    v = TemporalValidator()
    results = [v.update(l) for l in [1,1,1,0,0,0]]
    print(f"  labels=[1,1,1,0,0,0] -> confirmed={results[-1]}")
    assert results[-1] == False
    print("  PASS\n")

    print("Test 4: sustained attack then recovers -> confirmed then un-confirmed")
    v = TemporalValidator()
    seq = [1,1,1,1,1,1, 0,0,0,0,0,0]
    results = [v.update(l) for l in seq]
    print(f"  mid-sequence (after 6 attacked): {results[5]}")
    print(f"  end-sequence (after 6 clean):    {results[-1]}")
    assert results[5] == True
    assert results[-1] == False
    print("  PASS\n")

    print("All tests passed.")

if __name__ == "__main__":
    run_tests()
