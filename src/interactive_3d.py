"""Interactive Matplotlib controls for the spatial 3R robotic arm."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from .forward_kinematics_3d import forward_kinematics_3d

DEFAULT_LINK_LENGTHS = (0.5, 1.0, 0.8)


def create_interactive_3d(
    link_lengths: Sequence[float] = DEFAULT_LINK_LENGTHS,
):
    """Create a 3D arm figure with sliders and return its controls.

    The sliders control base yaw, shoulder pitch, and elbow pitch in radians.
    The returned figure is not shown automatically, which keeps this function
    usable in tests and notebooks.
    """
    lengths = np.asarray(link_lengths, dtype=float)
    forward_kinematics_3d((0.0, 0.0, 0.0), lengths)

    figure = plt.figure(figsize=(9, 8))
    arm_axis = figure.add_axes((0.08, 0.40, 0.84, 0.53), projection="3d")
    position_label = figure.text(0.08, 0.35, "", fontsize=11)

    slider_specs = (
        ("yaw", "Base yaw", -np.pi, np.pi, 0.28),
        ("shoulder", "Shoulder pitch", -np.pi / 2, np.pi / 2, 0.20),
        ("elbow", "Elbow pitch", -np.pi, np.pi, 0.12),
    )
    sliders: dict[str, Slider] = {}

    for key, label, lower, upper, vertical_position in slider_specs:
        slider_axis = figure.add_axes((0.20, vertical_position, 0.68, 0.03))
        sliders[key] = Slider(slider_axis, label, lower, upper, valinit=0.0)

    reset_axis = figure.add_axes((0.08, 0.035, 0.12, 0.05))
    reset_button = Button(reset_axis, "Reset")

    def update(_: float | None = None) -> None:
        joint_angles = np.array(
            [
                sliders["yaw"].val,
                sliders["shoulder"].val,
                sliders["elbow"].val,
            ]
        )
        positions, _ = forward_kinematics_3d(joint_angles, lengths)

        _draw_arm(arm_axis, positions, lengths)

        position_label.set_text(
            "End-effector position: "
            f"x={positions[-1, 0]:.3f}, "
            f"y={positions[-1, 1]:.3f}, "
            f"z={positions[-1, 2]:.3f}"
        )
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)

    def reset(_: object) -> None:
        for slider in sliders.values():
            slider.reset()

    reset_button.on_clicked(reset)
    update()

    figure._interactive_controls = {  # type: ignore[attr-defined]
        "sliders": sliders,
        "reset_button": reset_button,
    }

    return figure, arm_axis, sliders


def _draw_arm(
    arm_axis,
    positions: np.ndarray,
    link_lengths: np.ndarray,
) -> None:
    """Redraw the arm and preserve equal-scale XYZ bounds."""
    arm_axis.clear()

    arm_axis.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "o-",
        color="#1565c0",
        linewidth=3,
        markersize=8,
        label="Arm links",
    )

    arm_axis.scatter(
        *positions[0],
        color="#212121",
        s=65,
        label="Base",
    )
    arm_axis.scatter(
        *positions[-1],
        color="#d32f2f",
        s=75,
        label="End effector",
    )

    joint_names = ("Base", "Shoulder", "Elbow", "End effector")

    for joint_name, position in zip(joint_names, positions):
        color = "#b71c1c" if joint_name == "End effector" else "#102a43"

        arm_axis.text(
            *position,
            f"  {joint_name}\n"
            f"  ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})",
            color=color,
            fontsize=8,
        )

    reach = float(np.sum(link_lengths))
    center = (np.min(positions, axis=0) + np.max(positions, axis=0)) / 2
    half_span = (
        max(float(np.ptp(positions, axis=0).max()) / 2, reach * 0.35)
        + 0.2
    )

    arm_axis.set_xlim(center[0] - half_span, center[0] + half_span)
    arm_axis.set_ylim(center[1] - half_span, center[1] + half_span)
    arm_axis.set_zlim(center[2] - half_span, center[2] + half_span)

    arm_axis.set_box_aspect((1, 1, 1))
    arm_axis.set_xlabel("X position")
    arm_axis.set_ylabel("Y position")
    arm_axis.set_zlabel("Z position")
    arm_axis.set_title("Interactive Spatial 3R Robotic Arm")
    arm_axis.grid(True, alpha=0.3)
    arm_axis.legend(loc="upper right")


def main() -> None:
    """Launch the interactive slider interface."""
    parser = argparse.ArgumentParser(
        description="Interactively control a spatial 3R arm."
    )
    parser.add_argument(
        "--lengths",
        type=float,
        nargs=3,
        default=DEFAULT_LINK_LENGTHS,
        metavar=("BASE_HEIGHT", "UPPER_ARM", "FOREARM"),
        help="positive base-height, upper-arm, and forearm lengths",
    )

    arguments = parser.parse_args()

    create_interactive_3d(arguments.lengths)
    plt.show()


if __name__ == "__main__":
    main()
