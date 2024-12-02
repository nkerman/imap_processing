"""Tests coverage for ultra_l2.py"""

import logging
from os import environ

import numpy as np
import spiceypy as spice
import xarray as xr

from imap_processing.spice.kernels import ensure_spice
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins
from imap_processing.ultra.l2 import ultra_l2

environ["SPICE_METAKERNEL"] = (
    "/Users/nake7532/Projects/IMAP/imap_processing/imap_processing/tests/"
    "spice/test_data/imap_ena_sim_metakernel_nkerman.tm"
)
DEFAULT_SPACING_DEG = 0.5
# %%

# TODO: remove this where we set logger info to show
logging.basicConfig(level=logging.INFO)


@ensure_spice
def mock_l1c_pset_product(
    num_lat_bins: int = int(180 / DEFAULT_SPACING_DEG),
    num_lon_bins: int = int(360 / DEFAULT_SPACING_DEG),
    stripe_center_lon_bin: int = 0,
    timestr: str = "2025-01-01T00:00:00",
    head: str = "45",
):
    """Mock the L1C PSET product.

    Will be an xarray.Dataset with at least the variables and shapes:
    counts: (num_lat_bins, num_lon_bins, num_energy_bins)
    exposure_time: (num_lat_bins, num_lon_bins)
    sensitivity: (num_lat_bins, num_lon_bins, num_energy_bins)

    and the coordinate variables:
    latitude: (num_lat_bins)
    longitude: (num_lon_bins)
    energy: (determined by build_energy_bins function)
    head: Either '45' or '90'. Default is '45'.

    as well as the epoch (assumed to be a single time for each product).
    """

    _, energy_bin_midpoints = build_energy_bins()
    energy_bin_midpoints = energy_bin_midpoints[1:]
    num_energy_bins = len(energy_bin_midpoints)

    grid_shape = (num_lat_bins, num_lon_bins, num_energy_bins)

    def get_binomial_counts(distance_scaling, lon_bin, central_lon_bin):
        # Note, this is not quite correct, as it won't wrap around at 720
        distance_lon_bin = np.abs(lon_bin - central_lon_bin)
        return np.random.binomial(
            n=50,
            p=np.maximum(1 - (distance_lon_bin / distance_scaling), 0.01),
        )

    counts = np.fromfunction(
        lambda lat_bin, lon_bin, energy_bin: get_binomial_counts(
            distance_scaling=20, lon_bin=lon_bin, central_lon_bin=stripe_center_lon_bin
        ),
        shape=grid_shape,
    )

    exposure_time = np.zeros((num_lat_bins, num_lon_bins))
    if head == "90":
        exposure_time[
            :,
            stripe_center_lon_bin : stripe_center_lon_bin
            + int(20 / DEFAULT_SPACING_DEG),
        ] = 1
    else:
        exposure_time[
            : int(90 / DEFAULT_SPACING_DEG),
            stripe_center_lon_bin : stripe_center_lon_bin
            + int(70 / DEFAULT_SPACING_DEG),
        ] = 1

    sensitivity = np.ones(grid_shape)

    pset_product = xr.Dataset(
        {
            "counts": (
                ["elevation_bin_center", "azimuth_bin_center", "energy_bin_center"],
                counts,
            ),
            "exposure_time": (
                ["elevation_bin_center", "azimuth_bin_center"],
                exposure_time,
            ),
            "sensitivity": (
                ["elevation_bin_center", "azimuth_bin_center", "energy_bin_center"],
                sensitivity,
            ),
            "epoch": spice.str2et(timestr),
        },
        coords={
            "elevation_bin_center": np.linspace(-90, 90, num_lat_bins),
            "azimuth_bin_center": np.linspace(-180, 180, num_lon_bins),
            "energy_bin_center": energy_bin_midpoints,
        },
        attrs={
            "Logical_file_id": (
                f"imap_ultra_l1c_{head}sensor-pset_{timestr[:4]}"
                f"{timestr[5:7]}{timestr[8:10]}-repointNNNNN_vNNN"
            )
        },
    )

    return pset_product


def test_is_ultra45(caplog):
    """Test is_ultra45 function."""
    assert ultra_l2.is_ultra45(mock_l1c_pset_product(head="45"))
    assert not ultra_l2.is_ultra45(mock_l1c_pset_product(head="90"))
    assert not ultra_l2.is_ultra45(mock_l1c_pset_product(head="90"))

    # If neither 45 nor 90: Raises warning, returns False
    with caplog.at_level(logging.WARNING):
        assert not ultra_l2.is_ultra45(mock_l1c_pset_product(head="123456789"))
        assert "Found neither 45, nor 90 in descriptor string" in caplog.text
