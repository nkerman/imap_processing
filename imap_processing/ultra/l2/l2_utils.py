"""Build a solid angle map for a given spacing in degrees."""

import numpy as np
from numpy.typing import NDArray


def build_solid_angle_map(
    spacing: float, input_degrees: bool = False, output_degrees: bool = False
) -> NDArray:
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


def build_az_el_grid(
    spacing: float,
    input_degrees: bool = False,
    output_degrees: bool = False,
    centered_azimuth: bool = False,
    centered_elevation: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Build a 2D grid of azimuth and elevation angles.

    Azimuth and Elevation values represent the center of each grid cell,
    so the grid is offset by half the spacing.

    Parameters
    ----------
    spacing : float
        Spacing of the grid in degrees if `input_degrees` is True, else radians.
    input_degrees : bool, optional
        Whether the spacing is specified in degrees and must be converted to radians,
        by default False (indicating radians).
    output_degrees : bool, optional
        Whether the output azimuth and elevation angles should be in degrees,
        by default False (indicating radians).
    centered_azimuth : bool, optional
        Whether the azimuth grid should be centered around 0 degrees/0 radians,
        i.e. from -pi to pi radians, by default False, indicating 0 to 2pi radians.
        If True, the azimuth grid will be from -pi to pi radians.
    centered_elevation : bool, optional
        Whether the elevation grid should be centered around 0 degrees/0 radians,
        i.e. from -pi/2 to pi/2 radians, by default True.
        If False, the elevation grid will be from 0 to pi radians.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        - The evenly spaced, 1D range of azimuth angles
        e.g.(0, 0.5, 1, ..., 359.5) deg.
        - The evenly spaced, 1D range of elevation angles
        e.g.(-90, -89.5, ..., 89.5) deg.
        - The 2D grid of azimuth angles (azimuths for each elevation).
        This grid will be constant along the elevation (0th) axis.
        - The 2D grid of elevation angles (elevations for each azimuth).
        This grid will be constant along the azimuth (1st) axis.

    Raises
    ------
    ValueError
        If the spacing is not positive or does not divide evenly into pi radians.
    """
    if input_degrees:
        spacing = np.deg2rad(spacing)

    if spacing <= 0:
        raise ValueError("Spacing must be positive valued, non-zero.")

    if not np.isclose((np.pi / spacing) % 1, 0):
        raise ValueError("Spacing must divide evenly into pi radians.")

    el_range = np.arange(spacing / 2, np.pi, spacing)
    az_range = np.arange(spacing / 2, 2 * np.pi, spacing)
    if centered_azimuth:
        az_range = az_range - np.pi
    if centered_elevation:
        el_range = el_range - np.pi / 2

    # Reverse the elevation range so that the grid is in the order
    # defined by the Ultra prototype code (`build_dps_grid.m`).
    el_range = el_range[::-1]

    az_grid, el_grid = np.meshgrid(az_range, el_range)

    if output_degrees:
        az_range = np.rad2deg(az_range)
        el_range = np.rad2deg(el_range)
        az_grid = np.rad2deg(az_grid)
        el_grid = np.rad2deg(el_grid)

    return az_range, el_range, az_grid, el_grid
