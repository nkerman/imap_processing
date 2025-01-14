import numpy as np
import pytest

from imap_processing.ena_maps import map_utils


class TestENAMapMappingUtils:
    def test_bin_single_array_at_indices(
        self,
    ):
        value_array = np.array([1, 2, 3, 4, 5, 6])
        input_indices = np.array([0, 1, 2, 2, 1, 0])
        projection_indices = np.array([1, 2, 3, 1, 2, 3])
        projection_grid_shape = (5,)
        expected_projection_values = np.array([0, 4, 4, 4, 0])
        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )
        np.testing.assert_equal(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_extra_axis(
        self,
    ):
        # Binning will occur along axis 0 (combining 1, 2, 3 and 4, 5, 6 separately)
        value_array = np.array(
            [
                [1, 4],
                [2, 5],
                [3, 6],
            ]
        )
        input_indices = np.array([0, 1, 2, 2])
        projection_indices = np.array([1, 0, 1, 6])
        projection_grid_shape = (7, 1)
        expected_projection_values = np.array(
            [
                [2, 5],
                [4, 10],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [3, 6],
            ]
        )
        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        np.testing.assert_equal(projection_values, expected_projection_values)

    # Parameterize by the size of the projection grid,
    # which is not necessarily same size as input grid
    @pytest.mark.parametrize("projection_grid_shape", [(1, 1), (10, 10), (360, 720)])
    def test_bin_values_at_indices_1d_collapse_to_idx_zero(self, projection_grid_shape):
        wrapped_input_values = [
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        # unwrap to 0,0,0,1,2,3,4,5,6,7,8,9
        unwrapped_input_values = np.ravel(wrapped_input_values, order="C")
        scale_factor = 1.5
        input_values_to_bin = {
            "sum_variable_1": unwrapped_input_values,
            "sum_variable_2": unwrapped_input_values * scale_factor,
        }

        input_indices = np.arange(len(unwrapped_input_values))
        projection_indices = np.zeros_like(input_indices)
        expected_projection_values_1 = np.zeros(projection_grid_shape).ravel()
        expected_projection_values_1[0] = np.sum(unwrapped_input_values)
        expected_projection_values_2 = expected_projection_values_1 * scale_factor

        output_dict = map_utils.bin_values_at_indices(
            input_values_to_bin,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        np.testing.assert_equal(
            output_dict["sum_variable_1"], expected_projection_values_1
        )
        np.testing.assert_equal(
            output_dict["sum_variable_2"], expected_projection_values_2
        )
