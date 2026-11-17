import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.forward_kinematics_3d import forward_kinematics_3d
from src.interactive_3d import create_interactive_3d


def get_joint_label(axis, joint_name: str) -> str:
    """Return the annotation text for a named joint."""
    return next(
        text.get_text()
        for text in axis.texts
        if text.get_text().startswith(f"  {joint_name}\n")
    )


def test_interactive_plot_labels_all_initial_joint_positions():
    figure, axis, _ = create_interactive_3d([0.5, 1.0, 0.8])

    assert len(axis.texts) == 4
    assert get_joint_label(axis, "Base") == "  Base\n  (0.000, 0.000, 0.000)"
    assert (
        get_joint_label(axis, "Shoulder")
        == "  Shoulder\n  (0.000, 0.000, 0.500)"
    )
    assert get_joint_label(axis, "Elbow") == "  Elbow\n  (1.000, 0.000, 0.500)"
    assert (
        get_joint_label(axis, "End effector")
        == "  End effector\n  (1.800, 0.000, 0.500)"
    )

    figure.canvas.draw()


def test_interactive_controls_update_end_effector_position_and_label():
    figure, axis, sliders = create_interactive_3d([0.5, 1.0, 0.8])

    assert set(sliders) == {"yaw", "shoulder", "elbow"}

    sliders["yaw"].set_val(np.pi / 2)

    end_effector = axis.lines[0].get_data_3d()

    np.testing.assert_allclose(
        [end_effector[0][-1], end_effector[1][-1], end_effector[2][-1]],
        [0.0, 1.8, 0.5],
        atol=1e-12,
    )

    assert len(axis.texts) == 4
    assert (
        get_joint_label(axis, "Base")
        == "  Base\n  (0.000, 0.000, 0.000)"
    )
    assert (
        get_joint_label(axis, "End effector")
        == "  End effector\n  (0.000, 1.800, 0.500)"
    )

    figure.canvas.draw()


def test_interactive_labels_match_kinematics_after_slider_changes():
    link_lengths = [0.5, 1.0, 0.8]
    figure, axis, sliders = create_interactive_3d(link_lengths)
    joint_angles = [0.4, -0.3, 0.5]

    sliders["yaw"].set_val(joint_angles[0])
    sliders["shoulder"].set_val(joint_angles[1])
    sliders["elbow"].set_val(joint_angles[2])

    positions, _ = forward_kinematics_3d(joint_angles, link_lengths)
    joint_names = ("Base", "Shoulder", "Elbow", "End effector")

    assert len(axis.texts) == 4

    for joint_name, position in zip(joint_names, positions):
        expected_label = (
            f"  {joint_name}\n"
            f"  ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
        )

        assert get_joint_label(axis, joint_name) == expected_label

    figure.canvas.draw()
