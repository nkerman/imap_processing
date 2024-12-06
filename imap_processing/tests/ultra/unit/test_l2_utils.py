"""Test creation of solid angle map."""

import numpy as np
import pytest

from imap_processing.ultra.l2 import l2_utils

# Parameterize with spacings (degrees here):
valid_spacings = [0.1, 0.25, 0.5, 1, 5, 10, 20]
invalid_spacings = [0, -1, 11]
invalid_spacings_match_str = [
    "Spacing must be positive valued, non-zero.",
    "Spacing must be positive valued, non-zero.",
    "Spacing must divide evenly into pi radians.",
]


@pytest.mark.parametrize("spacing", valid_spacings)
def test_build_solid_angle_map(spacing):
    """Test build_solid_angle_map function."""
    solid_angle_map_steradians = l2_utils.build_solid_angle_map(
        spacing, input_degrees=True, output_degrees=False
    )
    assert np.isclose(np.sum(solid_angle_map_steradians), 4 * np.pi, atol=0, rtol=1e-9)

    solid_angle_map_sqdeg = l2_utils.build_solid_angle_map(
        np.deg2rad(spacing), input_degrees=False, output_degrees=True
    )
    assert np.isclose(
        np.sum(solid_angle_map_sqdeg), 4 * np.pi * (180 / np.pi) ** 2, atol=0, rtol=1e-9
    )


@pytest.mark.parametrize(
    "spacing, match_str", zip(invalid_spacings, invalid_spacings_match_str)
)
def test_build_solid_angle_map_invalid_spacing(spacing, match_str):
    """Test build_solid_angle_map function raises error for invalid spacing."""
    with pytest.raises(ValueError, match=match_str):
        _ = l2_utils.build_solid_angle_map(
            spacing, input_degrees=True, output_degrees=False
        )


@pytest.mark.parametrize("spacing", valid_spacings)
def test_build_az_el_grid(spacing):
    """Test build_az_el_grid function."""
    az_range, el_range, az_grid, el_grid = l2_utils.build_az_el_grid(
        spacing=spacing,
        input_degrees=True,
        output_degrees=True,
        centered_azimuth=False,
        centered_elevation=True,
    )

    # Size checks
    assert az_range.size == int(360 / spacing)
    assert el_range.size == int(180 / spacing)
    assert az_range.size == az_grid.shape[1]
    assert el_range.size == el_grid.shape[0]

    # Check grid values
    expected_az_range = np.arange((spacing / 2), 360 + (spacing / 2), spacing)
    expected_el_range = np.arange(-90 + (spacing / 2), 90 + (spacing / 2), spacing)[
        ::-1
    ]  # Note el order is reversed
    assert np.allclose(az_range, expected_az_range), (
        f"Expected azimuth range: {expected_az_range}, " f"but got: {az_range}"
    )
    assert np.allclose(el_range, expected_el_range), (
        f"Expected elevation range: {expected_el_range}, " f"but got: {el_range}"
    )


def test_rewrap_even_spaced_el_az_grid_1d():
    """Test rewrap_even_spaced_el_az_grid function, without extra axis."""
    orig_shape = (180 * 12, 360 * 12)
    orig_grid = np.fromfunction(lambda i, j: i**2 + j, orig_shape, dtype=int)
    raveled_values = orig_grid.ravel(order="F")
    rewrapped_grid_infer_shape = l2_utils.rewrap_even_spaced_el_az_grid(raveled_values)
    rewrapped_grid_known_shape = l2_utils.rewrap_even_spaced_el_az_grid(
        raveled_values, shape=orig_shape
    )

    assert np.array_equal(rewrapped_grid_infer_shape, orig_grid)
    assert np.array_equal(rewrapped_grid_known_shape, orig_grid)


def test_rewrap_even_spaced_el_az_grid_2d():
    """Test rewrap_even_spaced_el_az_grid function, with extra axis."""
    orig_shape = (180 * 12, 360 * 12, 5)
    orig_grid = np.fromfunction(lambda i, j, k: i**2 + j + k, orig_shape, dtype=int)
    raveled_values = orig_grid.reshape(-1, 5, order="F")
    rewrapped_grid_infer_shape = l2_utils.rewrap_even_spaced_el_az_grid(
        raveled_values, extra_axis=True
    )
    rewrapped_grid_known_shape = l2_utils.rewrap_even_spaced_el_az_grid(
        raveled_values, shape=orig_shape, extra_axis=True
    )
    assert raveled_values.shape == (180 * 12 * 360 * 12, 5)
    assert np.array_equal(rewrapped_grid_infer_shape, orig_grid)
    assert np.array_equal(rewrapped_grid_known_shape, orig_grid)