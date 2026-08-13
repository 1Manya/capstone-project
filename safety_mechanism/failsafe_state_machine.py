"""
failsafe_state_machine.py
===========================
Plain Python, no ROS - takes the temporal validator's confirmed/not-confirmed
signal plus current camera trust weight, and decides the vehicle's nav mode:
NORMAL -> DEGRADED -> EMERGENCY_STOP, with GRADUAL (not instant) restore back
to NORMAL once clean frames sustain for a while - avoids flapping/false
recovery on a single clean frame right after an attack.
"""


class FailSafeStateMachine:
    def __init__(self, restore_after: int = 5, emergency_camera_trust_threshold: float = 0.05):
        self.state = "NORMAL"
        self.restore_after = restore_after
        self.emergency_camera_trust_threshold = emergency_camera_trust_threshold
        self._clean_streak = 0

    def update(self, attack_confirmed: bool, camera_trust: float,
               other_sensors_healthy: bool = True) -> str:
        """
        attack_confirmed: from TemporalValidator.update()
        camera_trust: current camera trust weight from TrustEngine
        other_sensors_healthy: True unless LiDAR/Radar/IMU are ALSO
            degraded/unavailable - defaults to True since this project's
            scope doesn't model sensor hardware failure, only camera attack.
        """
        if attack_confirmed:
            self._clean_streak = 0
            if self.state == "NORMAL":
                self.state = "DEGRADED"
            if (self.state == "DEGRADED"
                    and camera_trust < self.emergency_camera_trust_threshold
                    and not other_sensors_healthy):
                self.state = "EMERGENCY_STOP"
        else:
            if self.state in ("DEGRADED", "EMERGENCY_STOP"):
                self._clean_streak += 1
                if self._clean_streak >= self.restore_after:
                    self.state = "NORMAL"
                    self._clean_streak = 0
        return self.state


def run_tests():
    print("Test 1: stays NORMAL while never attacked")
    sm = FailSafeStateMachine()
    states = [sm.update(False, 0.4) for _ in range(5)]
    print(f"  states={states}")
    assert all(s == "NORMAL" for s in states)
    print("  PASS\n")

    print("Test 2: single confirmed attack -> DEGRADED, stays DEGRADED on one clean frame after")
    sm = FailSafeStateMachine()
    s1 = sm.update(True, 0.2)
    s2 = sm.update(False, 0.25)  # just one clean frame - should NOT instantly restore
    print(f"  after attack: {s1}, after 1 clean frame: {s2}")
    assert s1 == "DEGRADED"
    assert s2 == "DEGRADED"
    print("  PASS\n")

    print("Test 3: gradual restore after sustained clean frames")
    sm = FailSafeStateMachine(restore_after=5)
    sm.update(True, 0.2)
    states = [sm.update(False, 0.3 + i*0.05) for i in range(5)]
    print(f"  5 clean frames after attack -> states={states}")
    assert states[-1] == "NORMAL"
    assert states[0] == "DEGRADED"  # not restored on the FIRST clean frame
    print("  PASS\n")

    print("Test 4: escalates to EMERGENCY_STOP only when camera trust critical AND other sensors unhealthy")
    sm = FailSafeStateMachine(emergency_camera_trust_threshold=0.1)
    sm.update(True, 0.5)  # first confirmed attack -> DEGRADED, trust still ok
    s_healthy = sm.update(True, 0.03, other_sensors_healthy=True)
    print(f"  low camera trust but other sensors healthy: {s_healthy}")
    assert s_healthy == "DEGRADED"  # should NOT escalate if other sensors fine

    sm2 = FailSafeStateMachine(emergency_camera_trust_threshold=0.1)
    sm2.update(True, 0.5)
    s_unhealthy = sm2.update(True, 0.03, other_sensors_healthy=False)
    print(f"  low camera trust AND other sensors unhealthy: {s_unhealthy}")
    assert s_unhealthy == "EMERGENCY_STOP"
    print("  PASS\n")

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
