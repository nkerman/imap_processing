"""Build a solid angle map for a given spacing in degrees."""

import typing

import numpy as np


@typing.no_type_check
def build_solid_angle_map(
    spacing: float, input_degrees: bool = False, output_degrees: bool = False
) -> np.ndarray:
    """
    Build a solid angle map for a given spacing in degrees.

    Parameters
    ----------
    spacing : float
        The bin spacing in the specified units.
    input_degrees : bool, optional
        If True, the input spacing is in degrees
        (default is False for radians).
    output_degrees : bool, optional
        If True, the output solid angle map is in square degrees
        (default is False for steradians).

    Returns
    -------
    solid_angle_grid : np.ndarray
        The solid angle map grid in steradians or square degrees.
        First index is latitude, second index is longitude.
    """
    if input_degrees:
        spacing = np.deg2rad(spacing)

    if spacing <= 0:
        raise ValueError("Spacing must be positive valued, non-zero.")

    if not np.isclose((np.pi / spacing) % 1, 0):
        raise ValueError("Spacing must divide evenly into pi radians.")

    solid_angle_lats = np.abs(
        spacing
        * (
            np.sin(
                np.arange(
                    start=(-np.pi / 2 + spacing),
                    stop=(np.pi / 2 + spacing),
                    step=spacing,
                )
            )
            - np.sin(np.arange(start=(-np.pi / 2), stop=(np.pi / 2), step=spacing))
        )
    )

    solid_angle_grid = np.repeat(
        solid_angle_lats[:, np.newaxis], (2 * np.pi) / spacing, axis=1
    )

    if output_degrees:
        solid_angle_grid = solid_angle_grid * ((180 / np.pi) ** 2)

    return solid_angle_grid
