"""Calculate ULTRA Level 2 (L2) Product."""

from __future__ import annotations

import logging
import typing
from functools import lru_cache

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.spice import geometry
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins
from imap_processing.ultra.utils import spatial_utils

# Set default values:
DEFAULT_SPACING_DEG = 0.5

logger = logging.getLogger(__name__)
logger.info("Importing ultra_l2 module")

# Use lru_cache to wrap build_az_el_grid, allowing for efficient reuse of the same
# az/el grid without passing through all functions
build_az_el_grid_cached = lru_cache(maxsize=10)(spatial_utils.build_az_el_grid)


def is_ultra45(
    str_or_dataset: str | xr.Dataset,
) -> bool:
    """
    Determine if the input is a 45 sensor (return True) or 90 sensor (False) product.

    Parameters
    ----------
    str_or_dataset : str | xr.Dataset
        Either the string descriptor or the xarray.Dataset object
        which contains the descriptor as an attribute "Logical_file_id".

    Returns
    -------
    bool
        True if the descriptor contains '45sensor', else False.

    Raises
    ------
    ValueError
        If the input is not a string or xarray.Dataset.

    Notes
    -----
    Issues a logger warning if neither '45sensor' nor '90sensor'
    is found in the descriptor string.
    """
    # Get the global attr which should contain the substring '45sensor' or '90sensor'
    if isinstance(str_or_dataset, str):
        descriptor_str = str_or_dataset
    elif isinstance(str_or_dataset, xr.Dataset):
        descriptor_str = str_or_dataset.attrs["Logical_file_id"]
    else:
        raise ValueError("Input must be a string or xarray.Dataset")

    if "45sensor" in descriptor_str:
        return True
    elif "90sensor" not in descriptor_str:
        logger.warning(
            f"Found neither 45, nor 90 in descriptor string: {descriptor_str}"
        )
    return False


def map_indices_frame_to_frame(
    input_frame: geometry.SpiceFrame,
    projection_frame: geometry.SpiceFrame,
    event_time: float,
    input_frame_spacing_deg: float = DEFAULT_SPACING_DEG,
    projection_frame_spacing_deg: float = DEFAULT_SPACING_DEG,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Match flattened rectangular grid indices between two frames.

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

    Returns
    -------
    tuple[NDArray]
        Tuple of the following arrays:
        - Flat indices of the grid points in the input frame.
        - Flat indices of the grid points in the projection frame.
        - Azimuth values of the input grid points in the input frame, raveled.
        - Elevation values of the input grid points in the input frame, raveled.
        - Azimuth values of the input grid points in the projection frame, raveled.
        - Elevation values of the input grid points in the projection frame, raveled.
        - Azimuth indices of the input grid points in the projection frame az grid.
        - Elevation indices of the input grid points in the projection frame el grid.
    """
    if input_frame_spacing_deg > projection_frame_spacing_deg:
        logger.warning(
            "Input frame has a larger spacing than the projection frame."
            f"\nReceived: input = {input_frame_spacing_deg} degrees "
            f"and {projection_frame_spacing_deg} degrees."
        )
    # Build the azimuth, elevation grid in the inertial frame
    (_, __, az_grid_in, el_grid_in) = build_az_el_grid_cached(
        spacing=input_frame_spacing_deg,
        input_degrees=True,
        output_degrees=False,
        centered_azimuth=False,  # (0, 2pi rad = 0, 360 deg)
        centered_elevation=True,  # (-pi/2, pi/2 rad = -90, 90 deg)
    )

    # Unwrap the grid to a 1D array for same interface as tessellation
    az_grid_input_raveled = az_grid_in.ravel(order="F")
    el_grid_input_raveled = el_grid_in.ravel(order="F")

    # Get the flattened indices of the grid points in the input frame (Fortran order)
    # TODO: Discuss with Nick/Tim/Laura: Should this just be an arange?
    flat_indices_in = np.ravel(
        np.arange(az_grid_input_raveled.size).reshape(az_grid_in.shape), order="F"
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
        et=[
            event_time,
        ],
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
    az_bins = np.linspace(0, 2 * np.pi, int(360 / projection_frame_spacing_deg) + 1)
    el_bins = np.linspace(
        -np.pi / 2, np.pi / 2, int(180 / projection_frame_spacing_deg) + 1
    )

    # Use digitize to find indices (-1 since digitize returns 1-based indices)
    input_az_in_proj_az_indices = np.digitize(input_az_in_proj_az, az_bins) - 1
    input_el_in_proj_el_indices = np.digitize(input_el_in_proj_el, el_bins[::-1]) - 1

    # NOTE: This method of matching indices 1:1 is rectangular grid-focused
    # It will not necessarily work with a tessellation (e.g. Healpix)
    flat_indices_proj = np.ravel_multi_index(
        multi_index=(input_az_in_proj_az_indices, input_el_in_proj_el_indices),
        dims=(
            int(360 // projection_frame_spacing_deg),
            int(180 // projection_frame_spacing_deg),
        ),
        order="F",  # MATLAB uses Fortran order, which we replicate here
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


@typing.no_type_check
def build_dps_combined_exposure_time(
    l1c_products: list[xr.Dataset],
    spacing_deg: float = DEFAULT_SPACING_DEG,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Build exposure time and time-averaged sensitivity maps on ecliptic inertial grid.

    Parameters
    ----------
    l1c_products : list[xr.Dataset]
        List of xarray Datasets containing the l1c products to combine.
    spacing_deg : float, optional
        The spacing of the grid in degrees, by default DEFAULT_SPACING_DEG.

    Returns
    -------
    tuple[NDArray]
        Tuple of the following arrays:
        - Exposure time (combined across l1c products) for 45 degree sensor
        - Exposure time (combined across l1c products) for 90 degree sensor
        - Exposure time (combined across l1c products) for both sensors added together
    """
    num_dps_pointings = len(l1c_products)
    (num_el_bins_l1c, num_az_bins_l1c, num_energy_bins_l1c) = l1c_products[
        0
    ].counts.shape
    logger.info(
        f"Number of DPS Pointings: {num_dps_pointings}. "
        f"Shape is {num_el_bins_l1c} el x {num_az_bins_l1c} "
        f"az x {num_energy_bins_l1c} energy"
    )

    # Setup the combined-across-pointings exp time arrays, in ecliptic inertial frame
    combined_exptime_45 = np.zeros(num_el_bins_l1c * num_az_bins_l1c)
    combined_exptime_90 = np.zeros(num_el_bins_l1c * num_az_bins_l1c)

    # Create ecliptic inertial grid onto which we will project the individual DPS frames
    # Build the azimuth and elevation grid in the heliocentric ecliptic frame (HAE)
    (az_range, el_range, az_grid, el_grid) = build_az_el_grid_cached(
        spacing=spacing_deg,
        input_degrees=True,
        output_degrees=False,
        centered_azimuth=False,  # (0, 2pi rad = 0, 360 deg)
        centered_elevation=True,  # (-pi/2, pi/2 rad = -90, 90 deg)
    )

    for prod_num, l1c_prod in enumerate(l1c_products):
        time = float(l1c_prod.epoch.values)
        logger.info(
            f"Projecting exposure time from product at time index {prod_num} "
            f"with epoch {time}."
        )

        (
            flat_indices_in,
            flat_indices_proj,
            az_grid_input_raveled,
            el_grid_input_raveled,
            input_az_in_proj_az,
            input_el_in_proj_el,
            input_az_in_proj_az_indices,
            input_el_in_proj_el_indices,
        ) = map_indices_frame_to_frame(
            input_frame=geometry.SpiceFrame.ECLIPJ2000,
            projection_frame=geometry.SpiceFrame.IMAP_DPS,
            event_time=time,
            input_frame_spacing_deg=spacing_deg,
            projection_frame_spacing_deg=spacing_deg,
        )

        # Add to the combined exposure time arrays
        pointing_projected_exptime = l1c_prod.exposure_time.values.ravel(order="F")[
            flat_indices_proj
        ]
        if is_ultra45(l1c_prod):
            combined_exptime_45 += pointing_projected_exptime
        else:
            combined_exptime_90 += pointing_projected_exptime

    combined_exptime_total = combined_exptime_45 + combined_exptime_90

    return (combined_exptime_45, combined_exptime_90, combined_exptime_total)


@typing.no_type_check
def build_flux_maps(
    l1c_products: list[xr.Dataset], spacing_deg: float = DEFAULT_SPACING_DEG
) -> tuple[NDArray]:
    """
    Build maps of particle counts, flux, etc. from multiple l1c observations.

    Parameters
    ----------
    l1c_products : list[xr.Dataset]
        List of l1c product datasets.
    spacing_deg : float, optional
        The spacing in degrees of the output maps, by default DEFAULT_SPACING_DEG.

    Returns
    -------
    tuple[NDArray]
        Tuple of the following arrays:
        - Combined counts for each energy bin
        - Combined exposure time for both sensors
        - Combined exposure time for 45 degree sensor
        - Combined exposure time for 90 degree sensor
        - Frame epochs
        - Energy bin widths
        - Energy bin edges
        - Energy bin midpoints
        - Solid angle grid
        - Developer dictionary containing intermediate results
        NOTE: Many of these are temporary, used in dev, and will be removed.

    Raises
    ------
    ValueError
        If the az, el, or energy bin centers don't match across l1c products.
    ValueError
        _description_
    """
    # Check that all l1c products have the same shape of the az, el, energy bin coords
    for l1c_product in l1c_products:
        if (
            not l1c_product.azimuth_bin_center.values.shape
            == l1c_products[0].azimuth_bin_center.values.shape
        ):
            raise ValueError("Azimuth bin center shape mismatch.")
        if (
            not l1c_product.elevation_bin_center.values.shape
            == l1c_products[0].elevation_bin_center.values.shape
        ):
            raise ValueError("Elevation bin center shape mismatch.")
        if (
            not l1c_product.energy_bin_center.values.shape
            == l1c_products[0].energy_bin_center.values.shape
        ):
            raise ValueError("Energy bin center shape mismatch.")

    (num_el, num_az, num_energy) = l1c_products[0].counts.shape

    # This will contain the counts for each energy bin
    combined_counts = np.zeros((num_el * num_az, num_energy))

    # Build the azimuth and elevation grid in the heliocentric ecliptic frame (HAE)
    (az_range, el_range, az_grid, el_grid) = build_az_el_grid_cached(
        spacing=spacing_deg,
        input_degrees=True,
        output_degrees=False,
        centered_azimuth=False,  # (0, 2pi rad = 0, 360 deg)
        centered_elevation=True,  # (-pi/2, pi/2 rad = -90, 90 deg)
    )
    # Create cartesian (x,y,z) vectors corresponding to the
    # grid points in the helio inertial frame
    radii_helio = geometry.spherical_to_cartesian(
        np.stack((np.ones_like(az_grid), az_grid, el_grid), axis=-1)
    )
    # Unravel to [n, 3] shape
    radii_helio = radii_helio.reshape(-1, 3, order="C")

    solid_angle_grid = spatial_utils.build_solid_angle_map(
        spacing=spacing_deg, input_degrees=True, output_degrees=False
    )

    frame_epochs = np.unique([l1c_product.epoch for l1c_product in l1c_products])
    if len(frame_epochs) != len(l1c_products):
        logger.warning(
            "Frame epochs are not unique for each l1c product: "
            f"{len(frame_epochs)} epochs vs {len(l1c_products)} products."
        )

    (combined_exptime_45, combined_exptime_90, combined_exptime_total) = (
        build_dps_combined_exposure_time(l1c_products, spacing_deg)
    )

    energy_centers_from_file = l1c_products[0].energy_bin_center.values
    energy_bin_edges, energy_midpoints = build_energy_bins()

    # The function we call adds an extra bin edge at the start of the array.
    # We remove it here.
    energy_bin_edges = energy_bin_edges[1:]
    energy_midpoints = energy_midpoints[1:]

    if not np.allclose(energy_centers_from_file, energy_midpoints):
        raise ValueError(
            f"Energy bin centers from file: {energy_centers_from_file}\n"
            f"Energy midpoints from function: {energy_midpoints}"
        )

    energy_delta = np.diff(energy_bin_edges)
    if not np.all(energy_delta >= 0):
        raise ValueError("Energy bin edges must be monotonically increasing.")

    for prod_num, l1c_prod in enumerate(l1c_products):
        time = float(l1c_prod.epoch.values)
        logger.info(
            f"Projecting exposure time from product at time index {prod_num} "
            f"with epoch {time}."
        )

        (flat_indices_hae, flat_indices_dps) = map_indices_frame_to_frame(
            input_frame=geometry.SpiceFrame.ECLIPJ2000,
            projection_frame=geometry.SpiceFrame.IMAP_DPS,
            event_time=time,
        )[:2]

        # TODO: Replace with sum of counts across all l1c products on an inertial grid
        pointing_projected_counts = l1c_prod.counts.values.reshape(
            -1, l1c_prod.counts.shape[-1], order="F"
        )[flat_indices_dps, :]

        combined_counts += pointing_projected_counts

    # TODO: this will be removed in the final version
    # Output the last pointing's data for use in development and testing
    developer_dict = {
        # "az_grid": az_grid,
        # "el_grid": el_grid,
        # "az_grid_raveled": az_grid_raveled,
        # "el_grid_raveled": el_grid_raveled,
        # "hae_az_in_dps_az": hae_az_in_dps_az,
        # "hae_el_in_dps_el": hae_el_in_dps_el,
        # "hae_az_in_dps_az_indices": hae_az_in_dps_az_indices,
        # "hae_el_in_dps_el_indices": hae_el_in_dps_el_indices,
        # "radii_helio": radii_helio,
        # "all_dps_index_map_dict": all_dps_index_map_dict,
    }

    return (
        combined_counts,
        combined_exptime_total,
        combined_exptime_45,
        combined_exptime_90,
        frame_epochs,
        energy_delta,
        energy_bin_edges,
        energy_midpoints,
        solid_angle_grid,
        developer_dict,
    )


def ultra_l2(l1c_products: list) -> xr.Dataset:
    """
    Generate Ultra L2 Product from L1C Products.

    NOTE: This function is a placeholder and will be implemented in the future.

    Parameters
    ----------
    l1c_products : list
        List of l1c products or paths to l1c products.

    Returns
    -------
    xr.Dataset
        L2 output dataset.
    """
    # TODO: come back to this! its just a sketch so far, and in wrong place
    # Map indices across all l1c products,
    all_dps_index_map_dict = {}
    for prod_num, l1c_prod in enumerate(l1c_products):
        time = float(l1c_prod.epoch.values)
        logger.info(
            f"Generating index maps for product at time index {prod_num} "
            f"with epoch {time}."
        )
        product_index_map_dict = {}

        # First 'pull' from DPS frame to HAE
        (
            indices_hae_input,
            indices_dps_proj,
        ) = map_indices_frame_to_frame(
            input_frame=geometry.SpiceFrame.ECLIPJ2000,
            projection_frame=geometry.SpiceFrame.IMAP_DPS,
            event_time=time,
        )[:2]
        product_index_map_dict["indices_hae_input"] = indices_hae_input
        product_index_map_dict["indices_dps_proj"] = indices_dps_proj

        # Then 'push' from DPS frame to HAE
        (
            indices_dps_input,
            indices_hae_proj,
        ) = map_indices_frame_to_frame(
            input_frame=geometry.SpiceFrame.IMAP_DPS,
            projection_frame=geometry.SpiceFrame.ECLIPJ2000,
            event_time=time,
        )[:2]
        product_index_map_dict["indices_hae_proj"] = indices_hae_proj
        product_index_map_dict["indices_dps_input"] = indices_dps_input
        all_dps_index_map_dict[time] = product_index_map_dict

    logger.info("Running ultra_l2 function")
    num_dps_pointings = len(l1c_products)
    logger.info(f"Number of DPS Pointings: {num_dps_pointings}")

    # TODO: Replace with full Dataset creation
    (
        combined_counts,
        combined_exptime_total,
        combined_exptime_45,
        combined_exptime_90,
        frame_epochs,
        energy_delta,
        energy_bin_edges,
        energy_midpoints,
        solid_angle_grid,
        developer_dict,
    ) = build_flux_maps(l1c_products, spacing_deg=DEFAULT_SPACING_DEG)

    # Rewrap the vars into grids:
    combined_counts = spatial_utils.rewrap_even_spaced_el_az_grid(
        combined_counts, extra_axis=True
    )
    combined_exptime_total = spatial_utils.rewrap_even_spaced_el_az_grid(
        combined_exptime_total, extra_axis=False
    )

    ds_l2 = xr.Dataset(
        {
            "counts": (["el", "az", "energy"], combined_counts),
            "exposure_time": (["el", "az"], combined_exptime_total),
        },
        attrs={"epochs": frame_epochs},
    )

    return ds_l2
