"""SWE L2 processing module."""

import numpy as np
import numpy.typing as npt
import xarray as xr

from imap_processing.swe.utils.swe_utils import read_lookup_table

# TODO: add these to instrument status summary
ENERGY_CONVERSION_FACTOR = 4.75
# 7 CEMs geometric factors in cm^2 sr eV/eV units.
GEOMETRIC_FACTORS = np.array(
    [
        435e-6,
        599e-6,
        808e-6,
        781e-6,
        876e-6,
        548e-6,
        432e-6,
    ]
)
ELECTRON_MASS = 9.10938356e-31  # kg

# See doc string of calculate_phase_space_density() for more details.
VELOCITY_CONVERSION_FACTOR = 1.237e31


def get_particle_energy() -> npt.NDArray:
    """
    Get particle energy.

    Calculate particle energy and add to the lookup table.
    To convert Volts to Energy, multiply ESA voltage in Volts by
    energy conversion factor to get electron energy in eV.

    Returns
    -------
    lookup_table : pandas.DataFrame
        Lookup table with energy column added.
    """
    # The lookup table gives voltage applied to analyzers.
    lookup_table = read_lookup_table()

    # Convert voltage to electron energy in eV by apply conversion factor.
    lookup_table["energy"] = lookup_table["esa_v"].values * ENERGY_CONVERSION_FACTOR
    return lookup_table


def calculate_phase_space_density(l1b_dataset: xr.Dataset) -> npt.NDArray:
    """
    Convert counts to phase space density.

    Calculate phase space density, fv, in units of s^3/ (cm^6 * ster).
        fv = 2 * (C/tau) / (G * v^4)
        where:
            C / tau = corrected count rate. L1B science data.
            G = geometric factor, in (cm^2 * ster). 7 CEMS value.
            v = electron speed, computed from energy, in cm/s.
                We need to use this formula to convert energy to speed:
                    E = 0.5 * m * v^2
                    where E is electron energy, in eV
                    (result from get_particle_energy() function),
                    m is mass of electron (9.10938356e-31 kg),
                    and v is what we want to calculate. Reorganizing above
                    formula result in v = sqrt(2 * E / m). This will be used
                    to calculate electron speed, v.

                Now to convert electron speed units to cm/s:
                    v = sqrt(2 * E / m)
                      = sqrt(2 * E(eV) * 1.60219e-19(J/eV) / 9.10938e-31 kg)
                        where J = kg * m^2 / s^2.
                      = sqrt(2 * 1.60219 * 10e−19 m^2/s^2 * E(eV) / 9.10938e-31)
                      = sqrt(2 * 1.60219 * 10e−19 * 10e4 cm^2/s^2 * E(eV) / 9.10938e-31)
                      = sqrt(3.20438 * 10e-15 * E(eV) / 9.10938e-31) cm/s
                      = sqrt( (3.20438 * 10e-15 / 9.10938e-31) * E(eV) ) cm/s
        fv = 2 * (C/tau) / (G * v^4)
           = 2 * (C/tau) / (G * (sqrt( (3.20438 * 10e-15 / 9.10938e-31) * E(eV) ))^4)
           = 2 * (C/tau) / (G * (sqrt(3.5176e16)^4 * E(eV)^2)
           = 2 * (C/tau) / (G * 1.237e31 * E(eV)^2)
        Ruth Skoug also got the same result, 1.237e31.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The L1B dataset to process.

    Returns
    -------
    density : numpy.ndarray
        Phase space density.
    """
    # Get esa_table_num for each full sweep.
    esa_table_nums = l1b_dataset["esa_table_num"].values[:, 0]
    # Get energy values from lookup table.
    particle_energy = get_particle_energy()
    # Get 720 (24 energy steps x 30 angle) particle energy for each full
    # sweep data.
    particle_energy_data = np.array(
        [
            particle_energy[particle_energy["table_index"] == val]["energy"].tolist()
            for val in esa_table_nums
        ]
    )
    particle_energy_data = particle_energy_data.reshape(-1, 24, 30)

    # Calculate phase space density using formula:
    #   2 * (C/tau) / (G * 1.237e31 * E(eV)^2)
    # See doc string for more details.
    density = (2 * l1b_dataset["science_data"]) / (
        GEOMETRIC_FACTORS[np.newaxis, np.newaxis, np.newaxis, :]
        * VELOCITY_CONVERSION_FACTOR
        * particle_energy_data[:, :, :, np.newaxis] ** 2
    )

    return density
