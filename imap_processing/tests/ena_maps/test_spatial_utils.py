"""Test creation of solid angle map and other spatial utils."""

import numpy as np
import numpy.testing as npt
import pytest

from imap_processing.ena_maps.utils import spatial_utils

# Parameterize with spacings (degrees here):
valid_spacings = [0.25, 0.5, 1, 5, 10, 20]
invalid_spacings = [0, -1, 11]
invalid_spacings_match_str = [
    "Spacing must be positive valued, non-zero.",
    "Spacing must be positive valued, non-zero.",
    "Spacing must divide evenly into pi radians.",
]


def test_build_spatial_bins():
    """Tests build_spatial_bins function."""
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        spatial_utils.build_spatial_bins()
    )

    assert az_bin_edges[0] == 0
    assert az_bin_edges[-1] == 360
    assert len(az_bin_edges) == 721

    assert el_bin_edges[0] == -90
    assert el_bin_edges[-1] == 90
    assert len(el_bin_edges) == 361

    assert len(az_bin_midpoints) == 720
    np.testing.assert_allclose(az_bin_midpoints[0], 0.25, atol=1e-4)
    np.testing.assert_allclose(az_bin_midpoints[-1], 359.75, atol=1e-4)

    assert len(el_bin_midpoints) == 360
    np.testing.assert_allclose(el_bin_midpoints[0], -89.75, atol=1e-4)
    np.testing.assert_allclose(el_bin_midpoints[-1], 89.75, atol=1e-4)


@pytest.mark.parametrize("spacing", valid_spacings)
def test_build_solid_angle_map_integration(spacing):
    """Test build_solid_angle_map function integrates to 4 pi steradians."""
    solid_angle_map_steradians = spatial_utils.build_solid_angle_map(
        spacing, input_degrees=True, output_degrees=False
    )
    assert np.isclose(np.sum(solid_angle_map_steradians), 4 * np.pi, atol=0, rtol=1e-9)

    solid_angle_map_sqdeg = spatial_utils.build_solid_angle_map(
        np.deg2rad(spacing), input_degrees=False, output_degrees=True
    )
    assert np.isclose(
        np.sum(solid_angle_map_sqdeg), 4 * np.pi * (180 / np.pi) ** 2, atol=0, rtol=1e-9
    )


@pytest.mark.parametrize("spacing", valid_spacings)
def test_build_solid_angle_map_equal_at_equal_el(spacing):
    """Test build_solid_angle_map function produces equal solid angle at equal el."""
    solid_angle_map = spatial_utils.build_solid_angle_map(
        spacing, input_degrees=True, output_degrees=False
    )
    el_grid = spatial_utils.AzElSkyGrid(
        spacing_deg=spacing,
        centered_azimuth=False,
        centered_elevation=True,
        reversed_elevation=False,
        angular_units="deg",
    ).el_grid
    for unique_el in np.unique(el_grid):
        solid_angles = solid_angle_map[el_grid == unique_el]
        np.testing.assert_allclose(solid_angles, solid_angles[0])


@pytest.mark.parametrize(
    "spacing, match_str", zip(invalid_spacings, invalid_spacings_match_str)
)
def test_build_solid_angle_map_invalid_spacing(spacing, match_str):
    """Test build_solid_angle_map function raises error for invalid spacing."""
    with pytest.raises(ValueError, match=match_str):
        _ = spatial_utils.build_solid_angle_map(
            spacing, input_degrees=True, output_degrees=False
        )


def test_rewrap_even_spaced_el_az_grid_1d():
    """Test rewrap_even_spaced_el_az_grid function, without extra axis."""
    orig_shape = (180 * 12, 360 * 12)
    orig_grid = np.fromfunction(lambda i, j: i**2 + j, orig_shape, dtype=int)
    raveled_values = orig_grid.ravel(order="F")
    rewrapped_grid_infer_shape = spatial_utils.rewrap_even_spaced_el_az_grid(
        raveled_values
    )
    rewrapped_grid_known_shape = spatial_utils.rewrap_even_spaced_el_az_grid(
        raveled_values, shape=orig_shape
    )

    assert np.array_equal(rewrapped_grid_infer_shape, orig_grid)
    assert np.array_equal(rewrapped_grid_known_shape, orig_grid)


def test_rewrap_even_spaced_el_az_grid_2d():
    """Test rewrap_even_spaced_el_az_grid function, with extra axis."""
    orig_shape = (180 * 12, 360 * 12, 5)
    orig_grid = np.fromfunction(lambda i, j, k: i**2 + j + k, orig_shape, dtype=int)
    raveled_values = orig_grid.reshape(-1, 5, order="F")
    rewrapped_grid_infer_shape = spatial_utils.rewrap_even_spaced_el_az_grid(
        raveled_values, extra_axis=True
    )
    rewrapped_grid_known_shape = spatial_utils.rewrap_even_spaced_el_az_grid(
        raveled_values, shape=orig_shape, extra_axis=True
    )
    assert raveled_values.shape == (180 * 12 * 360 * 12, 5)
    assert np.array_equal(rewrapped_grid_infer_shape, orig_grid)
    assert np.array_equal(rewrapped_grid_known_shape, orig_grid)


class TestAzElSkyGrid:
    @pytest.mark.parametrize("spacing", valid_spacings)
    def test_instantiate(self, spacing):
        grid = spatial_utils.AzElSkyGrid(
            spacing_deg=spacing,
            centered_azimuth=False,
            centered_elevation=True,
            reversed_elevation=False,
            angular_units="deg",
        )

        # Size checks
        assert grid.az_range.size == int(360 / spacing)
        assert grid.el_range.size == int(180 / spacing)
        assert grid.az_range.size == grid.az_grid.shape[0]
        assert grid.el_range.size == grid.el_grid.shape[1]

        # Check grid values
        expected_az_range = np.arange((spacing / 2), 360 + (spacing / 2), spacing)
        expected_el_range = np.arange(-90 + (spacing / 2), 90 + (spacing / 2), spacing)

        npt.assert_allclose(grid.az_range, expected_az_range, atol=1e-12)
        npt.assert_allclose(grid.el_range, expected_el_range, atol=1e-12)

        # Check bin edges
        expected_az_bin_edges = np.arange(0, 360 + spacing, spacing)
        expected_el_bin_edges = np.arange(-90, 90 + spacing, spacing)
        npt.assert_allclose(grid.az_bin_edges, expected_az_bin_edges, atol=1e-11)
        npt.assert_allclose(grid.el_bin_edges, expected_el_bin_edges, atol=1e-11)

    @pytest.mark.parametrize("spacing", valid_spacings)
    @pytest.mark.parametrize("starting_unit", ["deg", "rad"])
    def test_angular_unit_conversions(self, spacing, starting_unit):
        # Begins in whatever angular unit is specified, then convert to deg, then to rad
        grid = spatial_utils.AzElSkyGrid(
            spacing_deg=spacing,
            centered_azimuth=False,
            centered_elevation=True,
            reversed_elevation=False,
            angular_units=starting_unit,
        )

        # Convert to degrees and check grid center values and edges
        grid.to_degrees()
        expected_az_range_rad_deg = np.arange(
            (spacing / 2), 360 + (spacing / 2), spacing
        )
        expected_el_range_rad_deg = np.arange(
            -90 + (spacing / 2), 90 + (spacing / 2), spacing
        )
        expected_az_bin_edges_deg = np.arange(0, 360 + spacing, spacing)
        expected_el_bin_edges_deg = np.arange(-90, 90 + spacing, spacing)

        for attr, expected in zip(
            [grid.az_range, grid.el_range, grid.az_bin_edges, grid.el_bin_edges],
            [
                expected_az_range_rad_deg,
                expected_el_range_rad_deg,
                expected_az_bin_edges_deg,
                expected_el_bin_edges_deg,
            ],
        ):
            npt.assert_allclose(attr, expected, atol=1e-10)

        # Convert back to radians and check grid values
        grid.to_radians()
        spacing = np.deg2rad(spacing)
        expected_az_range_rad = np.arange(
            (spacing / 2), (2 * np.pi) + (spacing / 2), spacing
        )
        expected_el_range_rad = np.arange(
            -(np.pi / 2) + (spacing / 2), (np.pi / 2) + (spacing / 2), spacing
        )
        expected_az_bin_edges_rad = np.arange(0, (2 * np.pi) + spacing, spacing)
        expected_el_bin_edges_rad = np.arange(
            -(np.pi / 2), (np.pi / 2) + spacing, spacing
        )

        for attr, expected in zip(
            [grid.az_range, grid.el_range, grid.az_bin_edges, grid.el_bin_edges],
            [
                expected_az_range_rad,
                expected_el_range_rad,
                expected_az_bin_edges_rad,
                expected_el_bin_edges_rad,
            ],
        ):
            npt.assert_allclose(attr, expected, atol=1e-10)
