"""
hash_chain_logger.py
======================
Tamper-evident audit log for confirmed attack events, per the project's
Objective 3 spec: each block includes timestamp, vehicle position, frame ID,
CNN confidence, sensor trust weights at time of attack, and the navigation
decision taken - each block also includes the SHA-256 hash of the previous
block, so any edit to a past entry is detectable.

Grounded in Schneier & Kelsey's hash-chain secure audit log design and
Bellare & Yee's forward integrity property (see literature review refs).

Storage: JSONL file (one JSON object per line) - simple, human-readable,
append-friendly, no database dependency needed for this scope.
"""
import hashlib
import json
import time
from pathlib import Path

GENESIS_HASH = "0" * 64


class HashChainLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()

    def _last_hash(self) -> str:
        """Reads the last entry's own hash from the file. GENESIS_HASH if empty."""
        lines = self._read_lines()
        if not lines:
            return GENESIS_HASH
        return json.loads(lines[-1])["entry_hash"]

    def _read_lines(self):
        with open(self.log_path) as f:
            return [line for line in f.read().splitlines() if line.strip()]

    def log_event(self, frame_id: str, vehicle_position: dict, cnn_confidence: float,
                  trust_weights: dict, decision: str) -> dict:
        """Appends one tamper-evident log entry. Returns the entry written."""
        prev_hash = self._last_hash()
        entry_body = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "vehicle_position": vehicle_position,
            "cnn_confidence": cnn_confidence,
            "trust_weights": trust_weights,
            "decision": decision,
            "prev_hash": prev_hash,
        }
        # hash is computed over the canonical (sorted-key) JSON of the body,
        # so it's deterministic regardless of dict insertion order
        canonical = json.dumps(entry_body, sort_keys=True)
        entry_hash = hashlib.sha256(canonical.encode()).hexdigest()
        entry_body["entry_hash"] = entry_hash

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry_body) + "\n")
        return entry_body

    def verify_chain(self) -> dict:
        """Walks the whole chain and checks every link. Returns a report dict:
        {valid: bool, total_entries: int, broken_at: int or None}"""
        lines = self._read_lines()
        prev_hash = GENESIS_HASH
        for i, line in enumerate(lines):
            entry = json.loads(line)
            stored_hash = entry.pop("entry_hash")

            if entry["prev_hash"] != prev_hash:
                return {"valid": False, "total_entries": len(lines), "broken_at": i}

            canonical = json.dumps(entry, sort_keys=True)
            recomputed_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if recomputed_hash != stored_hash:
                return {"valid": False, "total_entries": len(lines), "broken_at": i}

            prev_hash = stored_hash
        return {"valid": True, "total_entries": len(lines), "broken_at": None}


def run_tests():
    import tempfile, os

    print("Test 1: fresh chain of 3 events verifies as valid")
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "audit.jsonl")
        logger = HashChainLogger(log_path)
        logger.log_event("frame_001", {"lat": 1.0, "lon": 2.0}, 0.91,
                          {"camera": 0.2, "lidar": 0.3, "radar": 0.3, "imu": 0.2}, "DEGRADED")
        logger.log_event("frame_002", {"lat": 1.1, "lon": 2.1}, 0.88,
                          {"camera": 0.15, "lidar": 0.35, "radar": 0.3, "imu": 0.2}, "DEGRADED")
        logger.log_event("frame_003", {"lat": 1.2, "lon": 2.2}, 0.10,
                          {"camera": 0.3, "lidar": 0.25, "radar": 0.25, "imu": 0.2}, "NORMAL")
        report = logger.verify_chain()
        print(f"  {report}")
        assert report["valid"] == True
        assert report["total_entries"] == 3
        print("  PASS\n")

        print("Test 2: tampering entry 1 (0-indexed) is DETECTED")
        lines = logger._read_lines()
        tampered = json.loads(lines[1])
        tampered["cnn_confidence"] = 0.01  # attacker tries to hide the real confidence
        lines[1] = json.dumps(tampered)
        with open(log_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        report = logger.verify_chain()
        print(f"  {report}")
        assert report["valid"] == False
        assert report["broken_at"] == 1
        print("  PASS\n")

    print("Test 3: empty log verifies as valid (trivially)")
    with tempfile.TemporaryDirectory() as d:
        logger = HashChainLogger(os.path.join(d, "empty.jsonl"))
        report = logger.verify_chain()
        print(f"  {report}")
        assert report["valid"] == True
        assert report["total_entries"] == 0
        print("  PASS\n")

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
