"""Tests coverage for ultra_l2.py"""

import numpy as np
import xarray as xr

from imap_processing.ultra.l2 import ultra_l2


def mock_l1c_pset_product(
    num_lat_bins: int = int(180 / 2),
    num_lon_bins: int = int(360 / 2),
    num_energy_bins: int = 90,
    stripe_center_lon_bin: int = 0,
    num_counts_total: int = int(1e5),
):
    """Mock the L1C PSET product.

    Will be an xarray.Dataset with at least the variables and shapes:
    counts: (num_lat_bins, num_lon_bins, num_energy_bins)
    exposure_time: (num_lat_bins, num_lon_bins)
    sensitivity: (num_lat_bins, num_lon_bins, num_energy_bins)

    and the coordinate variables:
    latitude: (num_lat_bins)
    longitude: (num_lon_bins)
    energy: (num_energy_bins)
    """

    grid_shape = (num_lat_bins, num_lon_bins, num_energy_bins)

    def get_binomial_counts(distance_scaling, lon_bin, central_lon_bin):
        distance_lon_bin = np.abs(lon_bin - central_lon_bin)
        return np.random.binomial(
            n=20, p=np.maximum(1 - (distance_lon_bin / distance_scaling), 0.01)
        )

    counts = np.fromfunction(
        lambda lat_bin, lon_bin, energy_bin: get_binomial_counts(
            distance_scaling=180, lon_bin=lon_bin, central_lon_bin=stripe_center_lon_bin
        ),
        shape=grid_shape,
    )

    exposure_time = np.ones((num_lat_bins, num_lon_bins))
    sensitivity = np.ones(grid_shape)

    pset_product = xr.Dataset(
        {
            "counts": (["latitude", "longitude", "energy"], counts),
            "exposure_time": (["latitude", "longitude"], exposure_time),
            "sensitivity": (["latitude", "longitude", "energy"], sensitivity),
        },
        coords={
            "latitude": np.linspace(-90, 90, num_lat_bins),
            "longitude": np.linspace(-180, 180, num_lon_bins),
            "energy": np.linspace(0.1, 10, num_energy_bins),
        },
    )

    return pset_product


def test_ultra_l2():
    """Tests ultra_l2 function."""
    l1c_products = [
        mock_l1c_pset_product(stripe_center_lon_bin=mid_longitude)
        for mid_longitude in np.arange(-180, 180, 45)
    ]
    l2_output = ultra_l2.ultra_l2(l1c_products)

    assert l2_output is None
