"""Tests for zone_params.py — validates zone specifications for all 4 environments."""

import pytest
from configs.zone_params import ZONE_PARAMS, ZONE_LENGTH, ARENA_WIDTH, ARENA_LENGTH, NUM_ZONES, BOULDER_SHAPES


ENVS = ["friction", "grass", "boulder", "stairs"]


class TestZoneConstants:
    def test_zone_length(self):
        assert ZONE_LENGTH == 10.0

    def test_arena_dimensions(self):
        assert ARENA_WIDTH == 30.0
        assert ARENA_LENGTH == 50.0

    def test_num_zones(self):
        assert NUM_ZONES == 5


class TestAllEnvironments:
    @pytest.mark.parametrize("env", ENVS)
    def test_env_exists(self, env):
        assert env in ZONE_PARAMS

    @pytest.mark.parametrize("env", ENVS)
    def test_five_zones(self, env):
        assert len(ZONE_PARAMS[env]) == 5

    @pytest.mark.parametrize("env", ENVS)
    def test_zones_numbered_1_to_5(self, env):
        zones = [z["zone"] for z in ZONE_PARAMS[env]]
        assert zones == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("env", ENVS)
    def test_contiguous_x_ranges(self, env):
        """Zones must tile 0-50m without gaps or overlap."""
        params = ZONE_PARAMS[env]
        assert params[0]["x_start"] == 0.0
        for i in range(len(params) - 1):
            assert params[i]["x_end"] == params[i + 1]["x_start"]
        assert params[-1]["x_end"] == 50.0

    @pytest.mark.parametrize("env", ENVS)
    def test_zone_widths(self, env):
        for z in ZONE_PARAMS[env]:
            assert z["x_end"] - z["x_start"] == pytest.approx(ZONE_LENGTH)

    @pytest.mark.parametrize("env", ENVS)
    def test_labels_nonempty(self, env):
        for z in ZONE_PARAMS[env]:
            assert z["label"] and len(z["label"]) > 0


class TestFrictionZones:
    def test_decreasing_friction(self):
        """Friction should decrease from zone 1 to zone 5 (high→low grip)."""
        params = ZONE_PARAMS["friction"]
        static_vals = [z["mu_static"] for z in params]
        for i in range(len(static_vals) - 1):
            assert static_vals[i] > static_vals[i + 1], \
                f"Zone {i+1} static friction {static_vals[i]} should be > zone {i+2} {static_vals[i+1]}"

    def test_static_gt_dynamic(self):
        """Static friction must always be >= dynamic friction."""
        for z in ZONE_PARAMS["friction"]:
            assert z["mu_static"] >= z["mu_dynamic"]

    def test_positive_friction(self):
        for z in ZONE_PARAMS["friction"]:
            assert z["mu_static"] > 0
            assert z["mu_dynamic"] > 0


class TestGrassZones:
    def test_increasing_density(self):
        params = ZONE_PARAMS["grass"]
        densities = [z["stalk_density"] for z in params]
        for i in range(len(densities) - 1):
            assert densities[i] <= densities[i + 1]

    def test_increasing_drag(self):
        params = ZONE_PARAMS["grass"]
        drags = [z["drag_coeff"] for z in params]
        for i in range(len(drags) - 1):
            assert drags[i] <= drags[i + 1]

    def test_zone1_no_stalks(self):
        """Zone 1 should have no physical stalks (density=0)."""
        assert ZONE_PARAMS["grass"][0]["stalk_density"] == 0
        assert ZONE_PARAMS["grass"][0]["height_range"] is None


class TestBoulderZones:
    def test_increasing_edge_size(self):
        params = ZONE_PARAMS["boulder"]
        for i in range(len(params) - 1):
            assert params[i]["edge_range"][1] <= params[i + 1]["edge_range"][0]

    def test_decreasing_count(self):
        """Fewer boulders as they get larger."""
        params = ZONE_PARAMS["boulder"]
        counts = [z["count"] for z in params]
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1]

    def test_boulder_shapes(self):
        assert len(BOULDER_SHAPES) == 4
        total_weight = sum(s["weight"] for s in BOULDER_SHAPES.values())
        assert total_weight == pytest.approx(1.0)


class TestStairsZones:
    def test_increasing_step_height(self):
        params = ZONE_PARAMS["stairs"]
        heights = [z["step_height"] for z in params]
        for i in range(len(heights) - 1):
            assert heights[i] < heights[i + 1]

    def test_consistent_step_depth(self):
        """All zones use 0.30m step depth."""
        for z in ZONE_PARAMS["stairs"]:
            assert z["step_depth"] == pytest.approx(0.30)

    def test_consistent_step_count(self):
        for z in ZONE_PARAMS["stairs"]:
            assert z["num_steps"] == 33
