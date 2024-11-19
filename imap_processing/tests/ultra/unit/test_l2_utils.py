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
