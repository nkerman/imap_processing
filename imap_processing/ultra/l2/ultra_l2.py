"""Calculate ULTRA Level 2 (L2) Product."""

from __future__ import annotations

import logging
import typing

import numpy as np
import spiceypy as spice
import xarray as xr
from numpy.typing import NDArray

from imap_processing.spice import geometry
from imap_processing.spice.kernels import ensure_spice
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins
from imap_processing.ultra.l2 import l2_utils

# Set default values:
DEFAULT_SPACING_DEG = 0.5

logger = logging.getLogger(__name__)
logger.info("Importing ultra_l2 module")


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


@typing.no_type_check
@ensure_spice
def project_inertial_frame_to_dps(
    event_time: float,
    existing_grids: tuple[NDArray, NDArray] | None = None,
    # Only used if no existing grids are provided
    spacing_deg: float = DEFAULT_SPACING_DEG,
) -> tuple[NDArray]:
    """
    Project an az/el grid in an inertial frame (ECLIPJ2000) to the DPS frame.

    # TODO: Likely replace J2000 with HAE frame (everywhere it occurs)

    Parameters
    ----------
    event_time : float
        Time at which to project the grid.
    existing_grids : tuple[NDArray, NDArray], optional
        Existing azimuth_array, elevation_array grids to use, by default None
        in which case the grids will be built from scratch.
        Each of these grids should be 2D arrays of either azimuth or elevation values.
    spacing_deg : float, optional
        Spacing of the grid in degrees, by default DEFAULT_SPACING_DEG.

    Returns
    -------
    tuple[NDArray]
        Tuple of following arrays and dimensions:
        - Azimuth values of all points on inertial grid (2D)
        - Elevation grid of all points on inertial grid (2D)
        - Flat indices of these grid points in the heliocentric frame (1D)
        - Flat indices of these grid points projected into the DPS frame (1D)
        - Azimuth values of original points in the inertial frame frame,
        raveled and projected into DPS azimuth coords (1D)
        - Elevation values of original points in the inertial frame frame,
        raveled and projected into DPS elevation coords (1D)
        - Azimuth indices of original points in the DPS azimuth grid (1D)
        - Elevation indices of original points in the DPS elevation grid (1D)
    """
    if not existing_grids:
        # Build the azimuth, elevation grid in the inertial frame
        (_, __, az_grid, el_grid) = l2_utils.build_az_el_grid(
            spacing=spacing_deg,
            input_degrees=True,
            output_degrees=False,
            centered_azimuth=False,  # (0, 2pi rad = 0, 360 deg)
            centered_elevation=True,  # (-pi/2, pi/2 rad = -90, 90 deg)
        )
    else:
        az_grid, el_grid = existing_grids

    # Unwrap the grid to a 1D array for same interface as tessellation
    az_grid_raveled = az_grid.ravel(order="F")
    el_grid_raveled = el_grid.ravel(order="F")

    radii_helio = geometry.spherical_to_cartesian(
        np.stack(
            (np.ones_like(az_grid_raveled), az_grid_raveled, el_grid_raveled),
            axis=-1,
        )
    )
    flat_indices_helio = np.arange(az_grid_raveled.size)

    # Project the grid points from the heliocentric frame to the DPS frame
    # radii_dps are cartesian (x,y,z) radii vectors in the DPS frame corresponding to
    # the grid points in the helio inertial frame
    radii_dps = geometry.frame_transform(
        et=[
            event_time,
        ],
        position=radii_helio,
        from_frame=geometry.SpiceFrame.ECLIPJ2000,
        to_frame=geometry.SpiceFrame.IMAP_DPS,
    )

    # Convert the (x,y,z) vectors to spherical coordinates in the DPS frame
    # Then extract the azimuth and elevation angles in the DPS frame. Ignore radius.
    hae_grid_in_dps_spherical_coord = geometry.cartesian_to_spherical(
        radii_dps, degrees=False
    )

    hae_az_in_dps_az = hae_grid_in_dps_spherical_coord[:, 1]
    hae_el_in_dps_el = hae_grid_in_dps_spherical_coord[:, 2]

    # Wrap the azimuth to the range (0, 2pi) rad = (0, 360) degrees
    hae_az_in_dps_az = np.mod(hae_az_in_dps_az, 2 * np.pi)

    # Wrap the elevation to the range (-pi/2, pi/2) rad = (-90, 90) degrees
    hae_az_in_dps_az_indices = np.floor(
        (hae_az_in_dps_az / (2 * np.pi)) * (360 / spacing_deg)
    ).astype(int)
    hae_el_in_dps_el_indices = np.floor(
        # NOTE: equiv to: (np.pi/2 - hae_el_in_dps_el) * ((180 / np.pi) / spacing_deg)
        (0.5 - hae_el_in_dps_el / np.pi) * (180 / spacing_deg)
    ).astype(int)

    # TODO: This method of matching indices 1:1 is rectangular grid-focused
    # It will not work with a tessellation (e.g. Healpix)
    flat_indices_dps = np.ravel_multi_index(
        multi_index=(hae_az_in_dps_az_indices, hae_el_in_dps_el_indices),
        dims=az_grid.T.shape,
        order="F",  # MATLAB uses Fortran order, which we replicate here
    )

    return (
        az_grid_raveled,
        el_grid_raveled,
        flat_indices_helio,
        flat_indices_dps,
        hae_az_in_dps_az,
        hae_el_in_dps_el,
        hae_az_in_dps_az_indices,
        hae_el_in_dps_el_indices,
    )


@typing.no_type_check
@ensure_spice
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
    (az_range, el_range, az_grid, el_grid) = l2_utils.build_az_el_grid(
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
            f"with epoch {time} = {spice.et2utc(time, 'ISOC', 3)}"
        )
        # TODO: Reset the frame definition for the appropriate Pointing

        (
            az_grid_raveled,
            el_grid_raveled,
            flat_indices_helio,
            flat_indices_dps,
            hae_az_in_dps_az,
            hae_el_in_dps_el,
            hae_az_in_dps_az_indices,
            hae_el_in_dps_el_indices,
        ) = project_inertial_frame_to_dps(
            event_time=time,
            existing_grids=(az_grid, el_grid),
            spacing_deg=spacing_deg,
        )

        # Add to the combined exposure time arrays
        pointing_projected_exptime = l1c_prod.exposure_time.values.ravel(order="F")[
            flat_indices_dps
        ]
        if is_ultra45(l1c_prod):
            combined_exptime_45 += pointing_projected_exptime
        else:
            combined_exptime_90 += pointing_projected_exptime

    combined_exptime_total = combined_exptime_45 + combined_exptime_90

    return (combined_exptime_45, combined_exptime_90, combined_exptime_total)


@typing.no_type_check
@ensure_spice
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
    (az_range, el_range, az_grid, el_grid) = l2_utils.build_az_el_grid(
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

    solid_angle_grid = l2_utils.build_solid_angle_map(
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
            f"with epoch {time} = {spice.et2utc(time, 'ISOC', 3)}"
        )

        (
            az_grid_raveled,
            el_grid_raveled,
            flat_indices_helio,
            flat_indices_dps,
            hae_az_in_dps_az,
            hae_el_in_dps_el,
            hae_az_in_dps_az_indices,
            hae_el_in_dps_el_indices,
        ) = project_inertial_frame_to_dps(
            event_time=time,
            existing_grids=(az_grid, el_grid),
        )

        # TODO: Replace with sum of counts across all l1c products on an inertial grid
        pointing_projected_counts = l1c_prod.counts.values.reshape(
            -1, l1c_prod.counts.shape[-1], order="F"
        )[flat_indices_dps, :]

        combined_counts += pointing_projected_counts

    # TODO: this will be removed in the final version
    # Output the last pointing's data for use in development and testing
    developer_dict = {
        "az_grid": az_grid,
        "el_grid": el_grid,
        "az_grid_raveled": az_grid_raveled,
        "el_grid_raveled": el_grid_raveled,
        "hae_az_in_dps_az": hae_az_in_dps_az,
        "hae_el_in_dps_el": hae_el_in_dps_el,
        "hae_az_in_dps_az_indices": hae_az_in_dps_az_indices,
        "hae_el_in_dps_el_indices": hae_el_in_dps_el_indices,
        "radii_helio": radii_helio,
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
    combined_counts = l2_utils.rewrap_even_spaced_el_az_grid(
        combined_counts, extra_axis=True
    )
    combined_exptime_total = l2_utils.rewrap_even_spaced_el_az_grid(
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
