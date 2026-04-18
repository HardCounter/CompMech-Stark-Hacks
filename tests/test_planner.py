"""Tests for the Next-Best-View planner (pure numpy — no torch required)."""
import numpy as np
import pytest
from gripper_cv.heapgrasp.reconstruct import default_camera_matrix
from gripper_cv.planner import NextBestViewPlanner


@pytest.fixture
def planner():
    K = default_camera_matrix(320, 240)
    return NextBestViewPlanner(
        camera_matrix=K,
        volume_size=16,
        object_diameter=0.15,
        camera_distance=0.40,
        lam=0.5,
        n_candidates=12,
        image_width=320,
        image_height=240,
    )


@pytest.fixture
def full_voxels():
    return np.ones((16, 16, 16), dtype=bool)


@pytest.fixture
def empty_voxels():
    return np.zeros((16, 16, 16), dtype=bool)


class TestInformationGain:
    def test_zero_ig_on_empty_grid(self, planner, empty_voxels):
        planner._voxels = empty_voxels
        assert planner._information_gain(0.0) == 0.0

    def test_positive_ig_on_full_grid(self, planner, full_voxels):
        planner._voxels = full_voxels
        assert planner._information_gain(0.0) > 0.0

    def test_ig_decreases_after_update(self, planner, full_voxels):
        """IG for a candidate should decrease after it is observed."""
        ig_before = planner._information_gain(0.0)
        planner.update(0.0, full_voxels)
        ig_after = planner._information_gain(0.0)
        assert ig_after < ig_before

    def test_ig_is_nonnegative(self, planner, full_voxels):
        planner._voxels = full_voxels
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            assert planner._information_gain(float(angle)) >= 0.0


class TestScore:
    def test_score_equals_ig_with_no_visits(self, planner, full_voxels):
        """Score == IG when no views have been taken (no trajectory penalty)."""
        planner._voxels = full_voxels
        ig = planner._information_gain(90.0)
        score = planner.score(90.0)
        assert abs(score - ig) < 1e-9

    def test_trajectory_penalty_favours_nearby_angles(self, planner, full_voxels):
        """
        The trajectory term in Score = IG / (1 + λ·dist) penalises long moves.
        With similar IG, nearby angles should score higher than far-away ones.
        """
        planner.update(0.0, full_voxels)
        score_near = planner.score(10.0)     # 10° from visited → low penalty
        score_far = planner.score(180.0)     # 180° from visited → high penalty
        assert score_near >= score_far

    def test_scores_all_returns_n_candidates_entries(self, planner, full_voxels):
        planner._voxels = full_voxels
        scores = planner.scores_all()
        assert len(scores) == planner.n_candidates

    def test_all_scores_nonnegative(self, planner, full_voxels):
        planner._voxels = full_voxels
        for score in planner.scores_all().values():
            assert score >= 0.0


class TestNextBestView:
    def test_returns_valid_angle(self, planner, full_voxels):
        planner.update(0.0, full_voxels)
        nbv = planner.next_best_view()
        assert 0.0 <= nbv < 360.0

    def test_avoids_visited_angle(self, planner, full_voxels):
        planner.update(0.0, full_voxels)
        nbv = planner.next_best_view()
        # Should not suggest 0° again (within 2° tolerance)
        assert abs(nbv) > 2.0 and abs(nbv - 360.0) > 2.0

    def test_sequential_calls_differ(self, planner, full_voxels):
        """Successive NBV calls should return different angles."""
        seen = set()
        for _ in range(4):
            nbv = planner.next_best_view()
            planner.update(nbv, full_voxels)
            seen.add(round(nbv, 1))
        assert len(seen) >= 2

    def test_falls_back_when_all_visited(self, planner, full_voxels):
        """When all candidates are visited, next_best_view still returns a value."""
        step = 360.0 / planner.n_candidates
        for i in range(planner.n_candidates):
            planner.update(i * step, full_voxels)
        nbv = planner.next_best_view()
        assert 0.0 <= nbv < 360.0


class TestUpdate:
    def test_update_increments_times_seen(self, planner, full_voxels):
        """After one update, at least one voxel should have times_seen >= 1."""
        assert planner._times_seen.max() == 0
        planner.update(0.0, full_voxels)
        assert planner._times_seen.max() >= 1

    def test_update_appends_to_visited(self, planner, full_voxels):
        assert len(planner._visited) == 0
        planner.update(45.0, full_voxels)
        planner.update(90.0, full_voxels)
        assert planner._visited == [45.0, 90.0]

    def test_update_stores_voxels(self, planner, full_voxels):
        partial = full_voxels.copy()
        partial[0, 0, 0] = False
        planner.update(0.0, partial)
        assert not planner._voxels[0, 0, 0]
        assert planner._voxels[1, 1, 1]


class TestAngularDiff:
    def test_wrap_near_360(self):
        from gripper_cv.planner.next_best_view import _angular_diff
        assert _angular_diff(359.0, 1.0) == pytest.approx(2.0)

    def test_same_angle_is_zero(self):
        from gripper_cv.planner.next_best_view import _angular_diff
        assert _angular_diff(90.0, 90.0) == pytest.approx(0.0)

    def test_opposite_angles(self):
        from gripper_cv.planner.next_best_view import _angular_diff
        assert _angular_diff(0.0, 180.0) == pytest.approx(180.0)


class TestSuggestViewSchedule:
    def test_returns_correct_length(self, planner):
        angles = planner.suggest_view_schedule(n_total=8, n_initial=4)
        assert len(angles) == 8

    def test_initial_angles_evenly_spaced(self, planner):
        angles = planner.suggest_view_schedule(n_total=8, n_initial=4)
        expected_initial = [0.0, 90.0, 180.0, 270.0]
        for expected, actual in zip(expected_initial, angles[:4]):
            assert actual == pytest.approx(expected)

    def test_does_not_mutate_planner_state(self, planner, full_voxels):
        """suggest_view_schedule must not change the live planner state."""
        planner.update(0.0, full_voxels)
        visited_before = list(planner._visited)
        planner.suggest_view_schedule(n_total=6, n_initial=3)
        assert planner._visited == visited_before
