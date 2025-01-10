"""Tests coverage for ENA Map Mapping Util functions."""

from unittest import mock

import numpy as np
import pytest

from imap_processing.ena_maps import map_utils
from imap_processing.spice import geometry
from imap_processing.ultra.utils import spatial_utils


class TestENAMapMappingUtils:
    @mock.patch("imap_processing.ena_maps.map_utils.geometry.frame_transform")
    def test_match_rect_indices_frame_to_frame_same_frame(self, mock_frame_transform):
        # Mock frame transform to return the input position vectors (no transform)
        mock_frame_transform.side_effect = (
            lambda et, position, from_frame, to_frame: position
        )
        et = -1
        spacing_deg = 1
        (
            flat_indices_in,
            flat_indices_proj,
            az_grid_input_raveled,
            el_grid_input_raveled,
            input_az_in_proj_az,
            input_el_in_proj_el,
        ) = map_utils.match_rect_indices_frame_to_frame(
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
            spatial_utils.build_az_el_grid(spacing=np.deg2rad(spacing_deg))[2],
        )
        np.testing.assert_allclose(
            input_el_in_proj_el.reshape(
                (180 // spacing_deg, 360 // spacing_deg), order="F"
            ),
            spatial_utils.build_az_el_grid(spacing=np.deg2rad(spacing_deg))[3],
        )

    @mock.patch("imap_processing.ena_maps.map_utils.geometry.frame_transform")
    @pytest.mark.parametrize("input_spacing_deg", [1 / 4, 1 / 3, 1 / 2, 1])
    def test_match_rect_indices_frame_to_frame_different_spacings(
        self, mock_frame_transform, input_spacing_deg
    ):
        # Mock frame transform to return the input position vectors (no transform)
        mock_frame_transform.side_effect = (
            lambda et, position, from_frame, to_frame: position
        )

        et = -1
        proj_spacing_deg = 1

        (
            flat_indices_in,
            flat_indices_proj,
        ) = map_utils.match_rect_indices_frame_to_frame(
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
