"""Utilities for generating ENA maps."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def bin_values_at_indices(
    input_values_to_bin: dict[str, NDArray],
    input_indices: NDArray,
    projection_indices: NDArray,
    projection_grid_shape: tuple[int, int],
) -> dict[str, NDArray]:
    """
    Project values from input grid to projection grid based on matched indices.

    Parameters
    ----------
    input_values_to_bin : dict[str, NDArray]
        Dict matching variable names to arrays of values to bin.
    input_indices : NDArray
        Ordered indices for input grid, corresponding to indices in projection grid.
        1 dimensional. May be non-unique, depending on the projection method.
    projection_indices : NDArray
        Ordered indices for projection grid, corresponding to indices in input grid.
        1 dimensional. May be non-unique, depending on the projection method.
    projection_grid_shape : tuple[int, int]
        The shape of the grid onto which values are projected (rows, columns).
        This size of the resulting grid (rows * columns) will be the size of the
        projected values contained in the output dictionary.

    Returns
    -------
    dict[str, NDArray]
        Dict matching the input variable names to the binned values
        on the projection grid.

    Raises
    ------
    ValueError
        If the input indices are not 1D arrays with the same number of elements.
    ValueError
        If the number of i
    """
    # Both sets of indices must be 1D with the same number of elements
    if input_indices.ndim != 1 or projection_indices.ndim != 1:
        raise ValueError(
            "Indices must be 1D arrays. "
            "If using a rectangular grid, the indices must be unwrapped."
        )
    if input_indices.size != projection_indices.size:
        raise ValueError(
            "The number of input and projection indices must be the same. "
            f"Received {input_indices.size} input indices and \
                         {projection_indices.size} projection indices."
        )

    num_projection_indices = np.multiply(*projection_grid_shape)

    binned_values_dict = {}
    for value_name, value_array in input_values_to_bin.items():
        if value_array.size != input_indices.size:
            # TODO: This assumption doesn't hold for 'Pull' method. Re-evaluate.
            # If the same size on axis 0, but values has extra axis (e.g. energy bin),
            # we can carry the extra axis through, adding each bin separately
            if value_array.shape[0] == input_indices.size:
                # TODO: Allow for extra axis (e.g. energy bins)
                # which is summed separately
                extra_axis = True
                logger.info(f"Extra axis detected for {value_name}.")
            else:
                raise ValueError(
                    f"Value array for {value_name} "
                    "must have the same size as the indices."
                )
        if extra_axis:
            # TODO: how can we handle the bincount in case of extra axis?
            pass
        binned_values = np.bincount(
            projection_indices, weights=value_array, minlength=num_projection_indices
        )
        binned_values_dict[value_name] = binned_values

    return binned_values_dict
