"""Edge-case tests for the 3D forward and inverse kinematics.

Place this file at ``tests/test_edge_cases_3d.py`` and run ``python -m pytest``
from the repository root.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.forward_kinematics_3d import forward_kinematics_3d
from src.inverse_kinematics_3d import inverse_kinematics_3d

LINKS = [0.5, 1.0, 0.8]


def _end_effector(angles, links=LINKS):
    positions, _ = forward_kinematics_3d(angles, links)
    return np.asarray(positions)[-1]


# --- forward kinematics -----------------------------------------------------


def test_fk_output_shapes():
    positions, transform = forward_kinematics_3d([0.1, 0.2, 0.3], LINKS)
    assert np.asarray(positions).shape == (4, 3)
    assert np.asarray(transform).shape == (4, 4)


def test_fk_home_pose_is_fully_extended():
    tip = _end_effector([0.0, 0.0, 0.0])
    np.testing.assert_allclose(tip, [LINKS[1] + LINKS[2], 0.0, LINKS[0]], atol=1e-9)


def test_fk_base_yaw_preserves_reach_and_height():
    straight = _end_effector([0.0, 0.3, 0.4])
    rotated = _end_effector([np.pi / 2, 0.3, 0.4])
    assert np.hypot(*rotated[:2]) == pytest.approx(np.hypot(*straight[:2]))
    assert rotated[2] == pytest.approx(straight[2])


@pytest.mark.parametrize("angles", [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
def test_fk_rejects_wrong_number_of_angles(angles):
    with pytest.raises(ValueError):
        forward_kinematics_3d(angles, LINKS)


@pytest.mark.parametrize("links", [[0.5, 1.0], [0.5, 0.0, 0.8], [0.5, -1.0, 0.8]])
def test_fk_rejects_invalid_link_lengths(links):
    with pytest.raises(ValueError):
        forward_kinematics_3d([0.0, 0.0, 0.0], links)


# --- inverse kinematics -----------------------------------------------------


@pytest.mark.parametrize(
    "angles",
    [
        [0.4, 0.3, 0.5],
        [-1.2, 0.6, 0.9],
        [2.5, -0.2, 1.1],
        [0.0, 0.5, -0.7],
    ],
)
def test_ik_round_trip_reaches_target(angles):
    target = _end_effector(angles)
    solution = inverse_kinematics_3d(target, LINKS)
    np.testing.assert_allclose(_end_effector(solution), target, atol=1e-5)


def test_ik_solution_respects_joint_limits():
    limits = [[-np.pi, np.pi], [-0.5, 0.5], [0.0, 2.0]]
    target = _end_effector([0.3, 0.2, 0.8])
    solution = inverse_kinematics_3d(target, LINKS, joint_limits=limits)
    for angle, (low, high) in zip(solution, limits):
        assert low - 1e-9 <= angle <= high + 1e-9


def test_ik_unreachable_target_raises():
    with pytest.raises(ValueError):
        inverse_kinematics_3d([10.0, 0.0, 0.0], LINKS)


def test_ik_target_blocked_by_joint_limits_raises():
    # Reachable with free joints, but the elbow is locked straight so the
    # arm cannot fold down to this point.
    locked = [[-np.pi, np.pi], [0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        inverse_kinematics_3d([0.5, 0.0, 0.5], LINKS, joint_limits=locked)


@pytest.mark.parametrize(
    "target",
    [[np.nan, 0.0, 1.0], [0.0, np.inf, 1.0], [1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
)
def test_ik_rejects_invalid_targets(target):
    with pytest.raises(ValueError):
        inverse_kinematics_3d(target, LINKS)


@pytest.mark.parametrize(
    "limits",
    [
        [[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]],  # lower > upper
        [[0.0, 0.0], [-1.0, 1.0], [-1.0, 1.0]],  # lower == upper
        [[-1.0, 1.0], [-1.0, 1.0]],  # wrong shape
    ],
)
def test_ik_rejects_invalid_joint_limits(limits):
    with pytest.raises(ValueError):
        inverse_kinematics_3d([1.0, 0.0, 1.0], LINKS, joint_limits=limits)
