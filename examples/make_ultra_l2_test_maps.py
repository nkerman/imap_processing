"""Test data for Menlo Ultra L2."""

# %% testing ultra L2 code
from os import environ, path
from pathlib import Path

import healpy as hp_vis
import numpy as np

from imap_processing.cdf.utils import write_cdf
from imap_processing.ena_maps import ena_maps
from imap_processing.tests.ultra.mock_data import mock_l1c_pset_product_healpix
from imap_processing.ultra.l2 import ultra_l2

# %% Make fake L1c products - healpix style of Ultra PSETs
# Spatial Parameters for the fake L1c products
l1c_nside = 128

# Get path of the metakernel file
current_dir = path.dirname(path.abspath(__file__))

# TODO: !!! MENLO you'll need to change the user directory within this file !!!
metakernel_path = path.join(
    current_dir,
    "local_test_data",
    "imap_ena_sim_metakernel_nkerman.tm",
)
environ["SPICE_METAKERNEL"] = path.abspath(metakernel_path)

manual_timestrs = [
    "2025-05-15T12:00:00",
    "2025-06-15T12:00:00",
    "2025-07-15T12:00:00",
    "2025-07-20T12:00:00",
]

fake_l1c_products_ultra = [
    mock_l1c_pset_product_healpix(
        nside=l1c_nside,
        # mid_latitude is the center of the horizontal stripe of counts
        stripe_center_lat=mid_latitude,
        width_scale=2,
        counts_scaling_params=(50, 0.5),
        peak_exposure=100,
        timestr=manual_timestrs[i],
        head=("90"),
        energy_dependent_exposure=True,
    )
    for i, mid_latitude in enumerate(
        np.arange(
            -90 + 22.5,
            90,
            45,
        )
    )
]

# %% Visualize the fake L1c products
hp_vis.mollview(fake_l1c_products_ultra[1]["counts"][0, 0])
# %%
# Add some additional metadata to the fake L1c products
for pset in fake_l1c_products_ultra:
    pset.attrs["type"] = "science"

# %%  BEGIN - TESTING FOR ULTRA L2 CODE
data_dict = {f"test_pset{i}": pset for i, pset in enumerate(fake_l1c_products_ultra)}

for name, pset in data_dict.items():
    # Write a CDF file for each pset with a unique name
    outpath = Path(
        f"/Users/nake7532/Projects/IMAP/imap_processing/imap_processing/ena_maps/data/imap/ultra/l1c/{name}.cdf"
    )
    write_cdf(pset, istp=True, compression=4)

# %%
hp_output_map_structure = ena_maps.AbstractSkyMap.from_properties_dict(
    {
        "sky_tiling_type": "HEALPIX",
        "spice_reference_frame": "ECLIPJ2000",
        "projection_method_and_values": {
            "PUSH": [],
            "PULL": [],
        },
        "nside": 32,
        "nested": False,
    }
)

rect_output_map_structure = ena_maps.AbstractSkyMap.from_properties_dict(
    {
        "sky_tiling_type": "RECTANGULAR",
        "spice_reference_frame": "ECLIPJ2000",
        "projection_method_and_values": {
            "PUSH": [
                "counts",
                "exposure_factor",
                "sensitivity",
                "background_rates",
            ],
        },
        "spacing_deg": 6,
    }
)
# %%
[
    hp_map_ds,
] = ultra_l2.ultra_l2(
    data_dict=data_dict,
)
print(hp_map_ds)

# %%
simple_pset_date_range = np.nanstd(
    [p["epoch"].values[0] for p in fake_l1c_products_ultra]
)

print(
    f"Simple PSET date range: {simple_pset_date_range} = \
{simple_pset_date_range / (86400 * 1e9)} days"
)

mean_obs_date_range = hp_map_ds["obs_date_range"].mean().values
print(
    f"Mean observation date range: {mean_obs_date_range}"
    f"= {mean_obs_date_range / (86400 * 1e9)} days"
)

hp_vis.mollview((hp_map_ds["obs_date_range"])[0, 0])
# %%
for name in [
    "flux",
    "flux_uncertainty",
    "ena_intensity",
    "ena_intensity_uncertainty",
    "counts",
    "exposure_factor",
    "sensitivity",
    "background_rates",
    "num_pointing_set_pixel_members",
    "obs_date",
    "pointing_set_exposure_times_solid_angle",
]:
    try:
        da = hp_map_ds[name]
        dims_to_mean = [dim for dim in da.dims if "epoch" in dim or "energy" in dim]
        plot_val = da.mean(dim=dims_to_mean)
        hp_vis.mollview(plot_val, max=np.quantile(plot_val, 0.98), title=name)
    except KeyError:
        print(f"KeyError: {name} not in hp_map_ds")
