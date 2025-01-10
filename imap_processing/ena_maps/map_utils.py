"""Utilities for generating ENA maps."""

from __future__ import annotations

import logging
import typing

import numpy as np
from numpy.typing import NDArray

from imap_processing.spice import geometry
from imap_processing.ultra.utils import spatial_utils

logger = logging.getLogger(__name__)


# Ignore linting rule to allow for 6 unrelated args
# Also need to manually specify allowed str literals for order parameter
def match_rect_indices_frame_to_frame(  # noqa: PLR0913
    input_frame: geometry.SpiceFrame,
    projection_frame: geometry.SpiceFrame,
    event_time: float,
    input_frame_spacing_deg: float,
    projection_frame_spacing_deg: float,
    order: typing.Literal["C"] | typing.Literal["F"] = "F",
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Match flattened rectangular grid indices between two reference frames.

    Parameters
    ----------
    input_frame : geometry.SpiceFrame
        The frame in which the input grid is defined.
    projection_frame : geometry.SpiceFrame
        The frame to which the input grid will be projected.
    event_time : float
        The time at which to project the grid.
    input_frame_spacing_deg : float, optional
        The spacing of the input frame grid in degrees,
        by default DEFAULT_SPACING_DEG.
    projection_frame_spacing_deg : float, optional
        The spacing of the projection frame grid in degrees,
        by default DEFAULT_SPACING_DEG.
    order : str, optional
        The order of unraveling/re-raveling the grid points,
        by default "F" for column-major Fortran ordering.
        See numpy.ravel documentation for more information.

    Returns
    -------
    tuple[NDArray]
        Tuple of the following arrays:
        - Ordered 1D array of pixel indices covering the entire input grid.
        Guaranteed to have one index for each point on the input grid,
        (meaning it will cover the entire input grid).
        Size is number of pixels on input grid
        `= (az_grid_input.size * el_grid_input.size)`.
        - Ordered 1D array of indices in the projection frame
        corresponding to each index in the input grid.
        Not guaranteed to cover the entire projection grid.
        Size is the same as the input indices
        = `(az_grid_input.size * el_grid_input.size)`.
        - Azimuth values of the input grid points in the input frame, raveled.
        Same size as the input indices.
        - Elevation values of the input grid points in the input frame, raveled.
        Same size as the input indices.
        - Azimuth values of the input grid points in the projection frame, raveled.
        Same size as the input indices.
        - Elevation values of the input grid points in the projection frame, raveled.
        Same size as the input indices.
        - Azimuth indices of the input grid points in the projection frame az grid.
        Same size as the input indices.
        - Elevation indices of the input grid points in the projection frame el grid.
        Same size as the input indices.
    """
    if input_frame_spacing_deg > projection_frame_spacing_deg:
        logger.warning(
            "Input frame has a larger spacing than the projection frame."
            f"\nReceived: input = {input_frame_spacing_deg} degrees "
            f"and {projection_frame_spacing_deg} degrees."
        )
    # Build the azimuth, elevation grid in the inertial frame
    az_grid_in, el_grid_in = spatial_utils.build_az_el_grid(
        spacing=input_frame_spacing_deg,
        input_degrees=True,
        output_degrees=False,
        centered_azimuth=False,  # (0, 2pi rad = 0, 360 deg)
        centered_elevation=True,  # (-pi/2, pi/2 rad = -90, 90 deg)
    )[2:4]

    # Unwrap the grid to a 1D array for same interface as tessellation
    az_grid_input_raveled = az_grid_in.ravel(order=order)
    el_grid_input_raveled = el_grid_in.ravel(order=order)

    # Get the flattened indices of the grid points in the input frame (Fortran order)
    # TODO: Discuss with Nick/Tim/Laura: Should this just be an arange?
    flat_indices_in = np.ravel(
        np.arange(az_grid_input_raveled.size).reshape(az_grid_in.shape),
        order=order,
    )

    radii_in = geometry.spherical_to_cartesian(
        np.stack(
            (
                np.ones_like(az_grid_input_raveled),
                az_grid_input_raveled,
                el_grid_input_raveled,
            ),
            axis=-1,
        )
    )

    # Project the grid points from the input frame to the projection frame
    # radii_proj are cartesian (x,y,z) radii vectors in the projection frame
    # corresponding to the grid points in the input frame
    radii_proj = geometry.frame_transform(
        et=event_time,
        position=radii_in,
        from_frame=input_frame,
        to_frame=projection_frame,
    )

    # Convert the (x,y,z) vectors to spherical coordinates in the projection frame
    # Then extract the azimuth, elevation angles in the projection frame. Ignore radius.
    input_grid_in_proj_spherical_coord = geometry.cartesian_to_spherical(
        radii_proj, degrees=False
    )

    input_az_in_proj_az = input_grid_in_proj_spherical_coord[:, 1]
    input_el_in_proj_el = input_grid_in_proj_spherical_coord[:, 2]

    # Create bin edges for azimuth (0 to 2pi) and elevation (-pi/2 to pi/2)
    proj_frame_az_bin_edges, proj_frame_el_bin_edges = spatial_utils.build_az_el_grid(
        spacing=projection_frame_spacing_deg,
        input_degrees=True,
        output_degrees=False,
        centered_azimuth=False,  # (0, 2pi rad = 0, 360 deg)
        centered_elevation=True,  # (-pi/2, pi/2 rad = -90, 90 deg)
    )[4:6]

    # Use digitize to find indices (-1 since digitize returns 1-based indices)
    input_az_in_proj_az_indices = (
        np.digitize(input_az_in_proj_az, proj_frame_az_bin_edges) - 1
    )
    input_el_in_proj_el_indices = (
        np.digitize(input_el_in_proj_el, proj_frame_el_bin_edges[::-1]) - 1
    )

    # NOTE: This method of matching indices 1:1 is rectangular grid-focused
    # It will not necessarily work with a tessellation (e.g. Healpix)
    flat_indices_proj = np.ravel_multi_index(
        multi_index=(input_az_in_proj_az_indices, input_el_in_proj_el_indices),
        dims=(
            int(360 // projection_frame_spacing_deg),
            int(180 // projection_frame_spacing_deg),
        ),
        order=order,
    )
    return (
        flat_indices_in,
        flat_indices_proj,
        az_grid_input_raveled,
        el_grid_input_raveled,
        input_az_in_proj_az,
        input_el_in_proj_el,
        input_az_in_proj_az_indices,
        input_el_in_proj_el_indices,
    )
