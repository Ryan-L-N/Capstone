"""Tests for metrics/reporter.py — JSONL loading, summary stats, CSV output."""

import json
import os

import numpy as np
import pytest
from metrics.reporter import load_episodes, compute_summary_statistics, _save_csv, run_statistical_tests


def _make_episode(env="friction", policy="flat", progress=25.0, completion=False,
                  fall_detected=False, fall_zone=None, stability=0.5,
                  velocity=1.0, zone_reached=3):
    return {
        "episode_id": "test_ep",
        "environment": env,
        "policy": policy,
        "completion": completion,
        "progress": progress,
        "zone_reached": zone_reached,
        "time_to_complete": None,
        "stability_score": stability,
        "mean_roll": 0.01,
        "mean_pitch": 0.02,
        "height_variance": 0.001,
        "mean_ang_vel": 0.5,
        "fall_detected": fall_detected,
        "fall_location": 15.0 if fall_detected else None,
        "fall_zone": fall_zone,
        "mean_velocity": velocity,
        "total_energy": 100.0,
        "episode_length": 60.0,
    }


class TestLoadEpisodes:
    def test_load_jsonl(self, tmp_path):
        """Write a JSONL file and verify load_episodes reads it."""
        ep1 = _make_episode(env="friction", policy="flat", progress=10.0)
        ep2 = _make_episode(env="friction", policy="flat", progress=20.0)

        filepath = tmp_path / "friction_flat_episodes.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps(ep1) + "\n")
            f.write(json.dumps(ep2) + "\n")

        episodes = load_episodes(str(tmp_path))
        assert len(episodes) == 2
        assert episodes[0]["progress"] == 10.0
        assert episodes[1]["progress"] == 20.0

    def test_load_empty_dir(self, tmp_path):
        episodes = load_episodes(str(tmp_path))
        assert episodes == []

    def test_ignores_non_jsonl(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not json")
        (tmp_path / "data.jsonl").write_text(json.dumps(_make_episode()) + "\n")
        episodes = load_episodes(str(tmp_path))
        assert len(episodes) == 1


class TestComputeSummaryStatistics:
    def test_single_group(self):
        eps = [_make_episode(progress=p) for p in [10.0, 20.0, 30.0]]
        summaries = compute_summary_statistics(eps)
        assert len(summaries) == 1
        s = summaries[0]
        assert s["environment"] == "friction"
        assert s["policy"] == "flat"
        assert s["num_episodes"] == 3
        assert s["mean_progress"] == pytest.approx(20.0, abs=0.1)

    def test_multiple_groups(self):
        eps = [
            _make_episode(env="friction", policy="flat"),
            _make_episode(env="friction", policy="rough"),
            _make_episode(env="boulder", policy="flat"),
        ]
        summaries = compute_summary_statistics(eps)
        assert len(summaries) == 3

    def test_completion_rate(self):
        eps = [
            _make_episode(completion=True),
            _make_episode(completion=True),
            _make_episode(completion=False),
        ]
        summaries = compute_summary_statistics(eps)
        assert summaries[0]["completion_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_fall_rate(self):
        eps = [
            _make_episode(fall_detected=True, fall_zone=2),
            _make_episode(fall_detected=False),
        ]
        summaries = compute_summary_statistics(eps)
        assert summaries[0]["fall_rate"] == pytest.approx(0.5, abs=0.01)

    def test_zone_distribution(self):
        eps = [
            _make_episode(zone_reached=1),
            _make_episode(zone_reached=3),
            _make_episode(zone_reached=5),
        ]
        summaries = compute_summary_statistics(eps)
        s = summaries[0]
        assert s["zone_1_count"] == 1
        assert s["zone_3_count"] == 1
        assert s["zone_5_count"] == 1
        assert s["zone_2_count"] == 0


class TestSaveCSV:
    def test_csv_creation(self, tmp_path):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        filepath = str(tmp_path / "test.csv")
        _save_csv(filepath, rows)
        assert os.path.exists(filepath)

        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows
        assert "a,b" in lines[0]

    def test_empty_rows(self, tmp_path):
        filepath = str(tmp_path / "empty.csv")
        _save_csv(filepath, [])
        assert not os.path.exists(filepath)


class TestStatisticalTests:
    def test_with_two_policies(self):
        eps = []
        np.random.seed(42)
        for _ in range(20):
            eps.append(_make_episode(policy="flat", progress=np.random.normal(30, 5)))
            eps.append(_make_episode(policy="rough", progress=np.random.normal(25, 5)))

        results = run_statistical_tests(eps)
        assert len(results) == 1
        r = results[0]
        assert "progress_t_stat" in r
        assert "progress_p_value" in r
        assert "progress_cohens_d" in r
        assert "completion_z_stat" in r

    def test_single_policy_skipped(self):
        eps = [_make_episode(policy="flat") for _ in range(5)]
        results = run_statistical_tests(eps)
        assert len(results) == 0
