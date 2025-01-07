"""Tests coverage for ultra_l2.py"""

import logging
import typing

import numpy as np
import pytest
import spiceypy as spice
import xarray as xr

from imap_processing.spice import geometry
from imap_processing.spice.kernels import ensure_spice
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins
from imap_processing.ultra.l2 import ultra_l2

DEFAULT_SPACING_DEG = 0.5

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


class TestUltraL2:
    @typing.no_type_check
    @pytest.fixture()
    def _setup_l1c_pset_products(self, l1c_spacing_deg=1, l2_spacing_deg=1):
        # Default to a courser spacing for faster tests
        self.l1c_spatial_bin_spacing_deg = l1c_spacing_deg
        self.l1c_products = [
            mock_l1c_pset_product(
                stripe_center_lon_bin=mid_longitude,
                num_lat_bins=int(180 / self.l1c_spatial_bin_spacing_deg),
                num_lon_bins=int(360 / self.l1c_spatial_bin_spacing_deg),
                timestr=f"2025-09-{i + 1:02d}T12:00:00",
                head=("45" if (i % 2 == 0) else "90"),
            )
            for i, mid_longitude in enumerate(
                np.arange(0, int(360 / self.l1c_spatial_bin_spacing_deg), 90)
            )
        ]

        # Create a simple and unrealistic mapping of indices for testing where
        # all DPS indices map to the 0th HAE index in the 'push' method,
        # and all HAE indices map to the 0th DPS index in the 'pull' method.
        max_input_index_dps = (180 * 360) / (l1c_spacing_deg**2)
        indices_dps_input = np.arange(0, max_input_index_dps, dtype=int)
        indices_hae_proj = np.zeros_like(indices_dps_input)
        max_input_index_hae = (180 * 360) / (l2_spacing_deg**2)
        indices_hae_input = np.arange(0, max_input_index_hae, dtype=int)
        indices_dps_proj = np.zeros_like(indices_hae_input)

        # Create a dictionary containing the mapping of indices
        pointing_indices_map_dict = {
            "indices_dps_input": indices_dps_input,
            "indices_dps_proj": indices_dps_proj,
            "indices_hae_input": indices_hae_input,
            "indices_hae_proj": indices_hae_proj,
        }
        self.all_dps_index_map_dict_simple = {
            float(prod.epoch.values): pointing_indices_map_dict
            for prod in self.l1c_products
        }

    def test_is_ultra45(self, caplog):
        """Test is_ultra45 function."""
        assert ultra_l2.is_ultra45(mock_l1c_pset_product(head="45"))
        assert not ultra_l2.is_ultra45(mock_l1c_pset_product(head="90"))

        # If neither 45 nor 90: Raises warning, returns False
        with caplog.at_level(logging.WARNING):
            assert not ultra_l2.is_ultra45(mock_l1c_pset_product(head="123456789"))
            assert "Found neither 45, nor 90 in descriptor string" in caplog.text

    @pytest.mark.external_kernel()
    @pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
    @pytest.mark.usefixtures("_setup_l1c_pset_products")
    def test_build_dps_combined_exposure_time(self):
        # TODO: Determine if the code works w different spacing for target/source grid
        spacing_deg = self.l1c_spatial_bin_spacing_deg

        (combined_exptime_45, combined_exptime_90, combined_exptime_total) = (
            ultra_l2.build_dps_combined_exposure_time(
                self.l1c_products,
                all_pointings_matched_indices=self.all_dps_index_map_dict_simple,
            )
        )

        assert (
            combined_exptime_45.shape
            == combined_exptime_90.shape
            == combined_exptime_total.shape
            == ((180 / spacing_deg) * (360 / spacing_deg),)
        )

        # TODO: Make this test more stringent by checking the actual values expected
        # # Because we set exptime to be bands of 1 on background of 0:
        # # The min exptime should be >= 0*, the max should be <= number of l1c_products
        # # The max of the 45 and 90 combined should be <= the number of l1c_products
        # # with that head.
        # # NOTE: *The min exptime is 0 currently, but could be unexpectedly
        # # made >0 if the l1c_products geometry are changed in the fixture above.
        # assert combined_exptime_total.min() == 0
        # assert combined_exptime_total.max() <= len(self.l1c_products)
        # assert combined_exptime_45.min() == 0
        # assert combined_exptime_45.max() <= len(
        #     [p for p in self.l1c_products if ultra_l2.is_ultra45(p)]
        # )
        # assert combined_exptime_90.min() == 0
        # assert combined_exptime_90.max() <= len(
        #     [p for p in self.l1c_products if not ultra_l2.is_ultra45(p)]
        # )

        # assert np.array_equal(
        #     combined_exptime_total,
        #     combined_exptime_45 + combined_exptime_90,
        # )

    @pytest.mark.external_kernel()
    @pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
    def test_map_indices_frame_to_frame_same_frame(self):
        et = ensure_spice(spice.utc2et)("2025-09-30T12:00:00.000")
        spacing_deg = 1
        (
            flat_indices_in,
            flat_indices_proj,
            az_grid_input_raveled,
            el_grid_input_raveled,
            input_az_in_proj_az,
            input_el_in_proj_el,
        ) = ultra_l2.map_indices_frame_to_frame(
            input_frame=geometry.SpiceFrame.IMAP_DPS,
            projection_frame=geometry.SpiceFrame.IMAP_DPS,
            event_time=et,
            input_frame_spacing_deg=spacing_deg,
            projection_frame_spacing_deg=spacing_deg,
        )[:6]

        # The two arrays should be the same if the input and projection
        # frames are the same and the spacing is the same.
        np.testing.assert_array_equal(flat_indices_in, flat_indices_proj)

        # The input and projection azimuth and elevation grids should be the same
        np.testing.assert_allclose(az_grid_input_raveled, input_az_in_proj_az)
        np.testing.assert_allclose(el_grid_input_raveled, input_el_in_proj_el)

        # The projection azimuth and elevation, when reshaped to a 2D grid,
        # should be the same as the input grid created with build_az_el_grid.
        np.testing.assert_allclose(
            input_az_in_proj_az.reshape(
                (180 // spacing_deg, 360 // spacing_deg), order="F"
            ),
            ultra_l2.build_az_el_grid_cached(spacing=np.deg2rad(spacing_deg))[2],
        )
        np.testing.assert_allclose(
            input_el_in_proj_el.reshape(
                (180 // spacing_deg, 360 // spacing_deg), order="F"
            ),
            ultra_l2.build_az_el_grid_cached(spacing=np.deg2rad(spacing_deg))[3],
        )

    @pytest.mark.external_kernel()
    @pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
    @pytest.mark.parametrize("input_spacing_deg", [1 / 4, 1 / 3, 1 / 2, 1])
    def test_map_indices_frame_to_frame_different_spacings(self, input_spacing_deg):
        et = ensure_spice(spice.utc2et)("2025-09-30T12:00:00.000")
        proj_spacing_deg = 1

        (
            flat_indices_in,
            flat_indices_proj,
        ) = ultra_l2.map_indices_frame_to_frame(
            input_frame=geometry.SpiceFrame.IMAP_DPS,
            projection_frame=geometry.SpiceFrame.IMAP_DPS,
            event_time=et,
            input_frame_spacing_deg=input_spacing_deg,
            projection_frame_spacing_deg=proj_spacing_deg,
        )[:2]

        # The input indices should still be the same length as the projection indices
        assert len(flat_indices_in) == len(flat_indices_proj)

        # However, the min and max of the input indices should go from:
        # 0 to (180/spacing_deg of the input)*(360/spacing_deg of the input)
        # and the min and max of the projection indices should go from:
        # 0 to (180/spacing_deg of the projection)*(360/spacing_deg of the projection)
        assert flat_indices_in.min() == 0
        assert flat_indices_in.max() == (
            (180 / (input_spacing_deg)) * (360 / (input_spacing_deg)) - 1
        )
        assert flat_indices_proj.min() == 0
        assert (
            flat_indices_proj.max()
            == ((180 / proj_spacing_deg) * (360 / proj_spacing_deg)) - 1
        )

        # There should be (proj_spacing_deg/input_spacing_deg)**2
        # instances of each projection index, and proj_spacing_deg = 1.
        assert np.all(np.bincount(flat_indices_proj) == 1 / (input_spacing_deg**2))

    @pytest.mark.external_kernel()
    @pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
    @pytest.mark.usefixtures("_setup_l1c_pset_products")
    def test_ultra_l2(self):
        ds_l2 = ultra_l2.ultra_l2(
            l1c_products=self.l1c_products,
            l2_spacing_deg=1,
        )

        # Check exposure time
        np.testing.assert_equal(ds_l2["exposure_time"].shape, (180, 360))
        np.testing.assert_equal(
            ds_l2["exposure_time"].values.sum(),
            np.sum(
                [ds_l1c["exposure_time"].values.sum() for ds_l1c in self.l1c_products]
            ),
        )
