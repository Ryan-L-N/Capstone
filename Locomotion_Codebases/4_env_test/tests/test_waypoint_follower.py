"""Tests for waypoint_follower.py — heading controller and waypoint advance."""

import numpy as np
import pytest
from navigation.waypoint_follower import WaypointFollower, WAYPOINTS, KP_YAW, TARGET_VX


class TestWaypointInit:
    def test_six_waypoints(self):
        assert len(WAYPOINTS) == 6

    def test_waypoints_along_centerline(self):
        """All waypoints at y=15.0 (arena center)."""
        for wp in WAYPOINTS:
            assert wp[1] == 15.0

    def test_waypoints_x_ascending(self):
        for i in range(len(WAYPOINTS) - 1):
            assert WAYPOINTS[i][0] < WAYPOINTS[i + 1][0]

    def test_start_at_zero(self):
        assert WAYPOINTS[0][0] == 0.0

    def test_end_at_fifty(self):
        assert WAYPOINTS[-1][0] == 50.0


class TestComputeCommands:
    def setup_method(self):
        self.follower = WaypointFollower()

    def test_output_shape(self):
        pos = np.array([0.0, 15.0, 0.6])
        cmd = self.follower.compute_commands(pos, 0.0)
        assert cmd.shape == (3,)

    def test_forward_velocity_positive(self):
        """Robot heading toward waypoint should have positive vx."""
        pos = np.array([0.0, 15.0, 0.6])
        cmd = self.follower.compute_commands(pos, 0.0)
        assert cmd[0] == pytest.approx(TARGET_VX)

    def test_vy_zero(self):
        """Lateral velocity should always be zero."""
        pos = np.array([5.0, 15.0, 0.6])
        cmd = self.follower.compute_commands(pos, 0.0)
        assert cmd[1] == pytest.approx(0.0)

    def test_straight_ahead_no_turn(self):
        """Robot on centerline heading east → minimal yaw correction."""
        # First call from spawn advances past WP0 to target WP1
        spawn = np.array([0.0, 15.0, 0.6])
        self.follower.compute_commands(spawn, 0.0)
        # Now at x=5, heading east toward WP1 at x=10
        pos = np.array([5.0, 15.0, 0.6])
        cmd = self.follower.compute_commands(pos, 0.0)  # yaw=0 = east
        assert abs(cmd[2]) < 0.1  # near-zero angular velocity

    def test_heading_correction_left(self):
        """Robot drifted south → should turn left (positive omega)."""
        pos = np.array([5.0, 14.0, 0.6])  # south of centerline
        cmd = self.follower.compute_commands(pos, 0.0)
        assert cmd[2] > 0  # positive yaw rate = turn left toward y=15

    def test_heading_correction_right(self):
        """Robot drifted north → should turn right (negative omega)."""
        pos = np.array([5.0, 16.0, 0.6])  # north of centerline
        cmd = self.follower.compute_commands(pos, 0.0)
        assert cmd[2] < 0

    def test_command_clamping_vx(self):
        pos = np.array([0.0, 15.0, 0.6])
        cmd = self.follower.compute_commands(pos, 0.0)
        assert -2.0 <= cmd[0] <= 3.0

    def test_command_clamping_omega(self):
        """Even with large heading error, omega should be clamped."""
        pos = np.array([5.0, 15.0, 0.6])
        # Facing backwards (yaw = pi)
        cmd = self.follower.compute_commands(pos, np.pi)
        assert -2.0 <= cmd[2] <= 2.0


class TestWaypointAdvance:
    def setup_method(self):
        self.follower = WaypointFollower()

    def test_starts_at_wp0(self):
        assert self.follower.current_waypoint_index == 0

    def test_advance_past_wp1(self):
        """Reaching x=9.5 should advance from WP0 to WP1."""
        pos = np.array([9.6, 15.0, 0.6])  # past WP1 threshold (10.0 - 0.5)
        self.follower.compute_commands(pos, 0.0)
        assert self.follower.current_waypoint_index >= 1

    def test_sequential_advance(self):
        """Walk through all waypoints."""
        for x in [9.6, 19.6, 29.6, 39.6, 49.6]:
            pos = np.array([x, 15.0, 0.6])
            self.follower.compute_commands(pos, 0.0)
        assert self.follower.is_done

    def test_is_done_initially_false(self):
        assert not self.follower.is_done

    def test_reset(self):
        pos = np.array([15.0, 15.0, 0.6])
        self.follower.compute_commands(pos, 0.0)
        self.follower.reset()
        assert self.follower.current_waypoint_index == 0
        assert not self.follower.is_done
