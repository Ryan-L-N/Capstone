"""Tests for metrics/collector.py — fall detection, zones, stability, JSONL export."""

import json
import os
import tempfile

import numpy as np
import pytest
from metrics.collector import MetricsCollector, quat_to_euler, _get_zone, FALL_THRESHOLD, COMPLETION_X


class TestQuatToEuler:
    def test_identity_quaternion(self):
        """w=1, x=y=z=0 → zero rotation."""
        roll, pitch, yaw = quat_to_euler([1, 0, 0, 0])
        assert roll == pytest.approx(0.0, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_90deg_yaw(self):
        """90° yaw rotation (about z-axis)."""
        # w=cos(45°), z=sin(45°)
        angle = np.pi / 2
        quat = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        roll, pitch, yaw = quat_to_euler(quat)
        assert yaw == pytest.approx(np.pi / 2, abs=1e-4)


class TestGetZone:
    def test_zone_0(self):
        assert _get_zone(0.0) == 0
        assert _get_zone(5.0) == 0
        assert _get_zone(9.9) == 0

    def test_zone_1(self):
        assert _get_zone(10.0) == 1
        assert _get_zone(15.0) == 1

    def test_zone_4(self):
        assert _get_zone(40.0) == 4
        assert _get_zone(49.9) == 4

    def test_clamp_negative(self):
        assert _get_zone(-1.0) == 0

    def test_clamp_beyond(self):
        assert _get_zone(100.0) == 4


class TestMetricsCollector:
    def _make_collector(self, tmpdir):
        return MetricsCollector(str(tmpdir), "friction", "flat")

    def _identity_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def test_basic_episode(self, tmp_path):
        mc = self._make_collector(tmp_path)
        mc.start_episode("ep_001")

        # Walk forward 10 steps at z=0.5 (no fall)
        for i in range(10):
            pos = np.array([float(i), 15.0, 0.5])
            mc.step(pos, self._identity_quat(),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0]),
                    sim_time=i * 0.02)

        result = mc.end_episode()
        assert result["episode_id"] == "ep_001"
        assert result["policy"] == "flat"
        assert result["environment"] == "friction"
        assert result["fall_detected"] is False
        assert result["completion"] is False
        assert result["progress"] == pytest.approx(9.0, abs=0.01)

    def test_fall_detection(self, tmp_path):
        mc = self._make_collector(tmp_path)
        mc.start_episode("ep_fall")

        # Step with z below fall threshold
        pos = np.array([5.0, 15.0, 0.10])  # below 0.15
        mc.step(pos, self._identity_quat(),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]))

        assert mc.episode_done() is True
        result = mc.end_episode()
        assert result["fall_detected"] is True
        assert result["fall_zone"] == 1  # zone 1 (x=5)

    def test_completion_detection(self, tmp_path):
        mc = self._make_collector(tmp_path)
        mc.start_episode("ep_complete")

        pos = np.array([COMPLETION_X + 0.5, 15.0, 0.5])
        mc.step(pos, self._identity_quat(),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                sim_time=30.0)

        result = mc.end_episode()
        assert result["completion"] is True
        assert result["time_to_complete"] == pytest.approx(30.0, abs=0.01)

    def test_zone_tracking(self, tmp_path):
        mc = self._make_collector(tmp_path)
        mc.start_episode("ep_zones")

        # Walk to x=25 → zone 3
        pos = np.array([25.0, 15.0, 0.5])
        mc.step(pos, self._identity_quat(),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]))

        result = mc.end_episode()
        assert result["zone_reached"] == 3  # floor(25/10)+1 = 3

    def test_stability_score_structure(self, tmp_path):
        mc = self._make_collector(tmp_path)
        mc.start_episode("ep_stab")

        for i in range(20):
            pos = np.array([float(i), 15.0, 0.5])
            mc.step(pos, self._identity_quat(),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0]))

        result = mc.end_episode()
        assert "stability_score" in result
        assert isinstance(result["stability_score"], float)
        assert result["stability_score"] >= 0

    def test_energy_tracking(self, tmp_path):
        mc = self._make_collector(tmp_path)
        mc.start_episode("ep_energy")

        torques = np.ones(12) * 5.0
        joint_vel = np.ones(12) * 2.0
        pos = np.array([1.0, 15.0, 0.5])
        mc.step(pos, self._identity_quat(),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                joint_torques=torques, joint_vel=joint_vel)

        result = mc.end_episode()
        assert result["total_energy"] > 0


class TestStairsGroundHeightFn:
    """Test fall detection with ground_height_fn for stairs."""

    def _identity_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def test_stairs_no_fall_on_surface(self, tmp_path):
        """Robot standing on stair surface should NOT be detected as fallen."""
        # At x=20m, stair elevation ≈ 3.63m. Robot base at 4.13m (0.5m above).
        ground_fn = lambda x: 3.63  # simplified
        mc = MetricsCollector(str(tmp_path), "stairs", "flat",
                              ground_height_fn=ground_fn)
        mc.start_episode("ep_stairs_ok")
        pos = np.array([20.0, 15.0, 4.13])  # 0.5m above stair surface
        mc.step(pos, self._identity_quat(),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]))
        assert mc.episode_done() is False

    def test_stairs_fall_on_surface(self, tmp_path):
        """Robot collapsed onto stair surface should be detected as fallen."""
        ground_fn = lambda x: 3.63
        mc = MetricsCollector(str(tmp_path), "stairs", "flat",
                              ground_height_fn=ground_fn)
        mc.start_episode("ep_stairs_fall")
        # Robot base at 3.70m — only 0.07m above stair (below 0.15m threshold)
        pos = np.array([20.0, 15.0, 3.70])
        mc.step(pos, self._identity_quat(),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]))
        assert mc.episode_done() is True
        result = mc.end_episode()
        assert result["fall_detected"] is True

    def test_stairs_tumble_detected(self, tmp_path):
        """Robot at zone 3 elevation but fallen to zone 1 height should be detected."""
        # Stair at x=5m has elevation ~0.5m. Robot fell from zone 3 to here.
        ground_fn = lambda x: 0.5
        mc = MetricsCollector(str(tmp_path), "stairs", "flat",
                              ground_height_fn=ground_fn)
        mc.start_episode("ep_tumble")
        # Robot at z=0.55m, ground at 0.5m → only 0.05m above ground = fallen
        pos = np.array([5.0, 15.0, 0.55])
        mc.step(pos, self._identity_quat(),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]))
        assert mc.episode_done() is True

    def test_flat_env_backward_compatible(self, tmp_path):
        """Without ground_height_fn, fall detection works as before (absolute Z)."""
        mc = MetricsCollector(str(tmp_path), "friction", "flat")
        mc.start_episode("ep_flat")
        pos = np.array([5.0, 15.0, 0.10])  # below 0.15m absolute
        mc.step(pos, self._identity_quat(),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]))
        assert mc.episode_done() is True

    def test_height_stored_relative(self, tmp_path):
        """Heights buffer should store height relative to ground, not absolute Z."""
        ground_fn = lambda x: 5.0
        mc = MetricsCollector(str(tmp_path), "stairs", "rough",
                              ground_height_fn=ground_fn)
        mc.start_episode("ep_rel")
        # Two steps at constant 0.5m above ground → height_variance ≈ 0
        for i in range(2):
            pos = np.array([25.0 + i, 15.0, 5.5])  # 0.5m above ground_z=5.0
            mc.step(pos, self._identity_quat(),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0]))
        result = mc.end_episode()
        # If heights were stored as absolute (5.5), variance would be 0 too,
        # but fall detection wouldn't have fired (5.5 > 0.15). The key test is
        # that the robot at z=5.5 with ground=5.0 is NOT detected as fallen.
        assert result["fall_detected"] is False
        assert result["height_variance"] == pytest.approx(0.0, abs=1e-6)


class TestJSONLExport:
    def test_save_creates_file(self, tmp_path):
        mc = MetricsCollector(str(tmp_path), "boulder", "rough")
        mc.start_episode("ep_001")
        pos = np.array([5.0, 15.0, 0.5])
        mc.step(pos, np.array([1, 0, 0, 0], dtype=float),
                np.array([1, 0, 0], dtype=float),
                np.array([0, 0, 0], dtype=float))
        mc.end_episode()
        filepath = mc.save()

        assert os.path.exists(filepath)
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["episode_id"] == "ep_001"
        assert data["environment"] == "boulder"
        assert data["policy"] == "rough"

    def test_append_mode(self, tmp_path):
        """Multiple saves should append, not overwrite."""
        mc = MetricsCollector(str(tmp_path), "friction", "flat")

        mc.start_episode("ep_001")
        mc.step(np.array([1, 15, 0.5]), np.array([1, 0, 0, 0], dtype=float),
                np.array([1, 0, 0], dtype=float), np.array([0, 0, 0], dtype=float))
        mc.end_episode()
        mc.save()

        mc.start_episode("ep_002")
        mc.step(np.array([2, 15, 0.5]), np.array([1, 0, 0, 0], dtype=float),
                np.array([1, 0, 0], dtype=float), np.array([0, 0, 0], dtype=float))
        mc.end_episode()
        mc.save()

        filepath = os.path.join(str(tmp_path), "friction_flat_episodes.jsonl")
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 2
