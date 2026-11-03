"""A module for reading in cloudy outputs and creating a synthesizer grid.

This module reads in cloudy outputs and creates a synthesizer grid. The
synthesizer grid is a HDF5 file that contains the spectra and line luminosities
from cloudy outputs.

Example usage:

"""

import os
import re

import h5py
import numpy as np
import submission_scripts
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.photoionisation import cloudy17, cloudy23
from synthesizer.units import has_units
from unyt import Angstrom, Hz, cm, dimensionless, erg, eV, km, s, yr
from utils import (
    get_cloudy_params,
    get_grid_props_cloudy,
)

from syncretize.grid_io import GridFile
from syncretize.parser import Parser

# Create dictionary of commonly used units for grid axes
axes_units = {
    "reference_ionisation_parameter": dimensionless,
    "fixed_ionisation_parameter": dimensionless,
    "depletion_scale": dimensionless,
    "z": dimensionless,
    "ionisation_parameter": dimensionless,
    "ages": yr,
    "metallicities": dimensionless,
    "hydrogen_density": cm ** (-3),
    "stop_column_density": dimensionless,
    "alpha_enhancement": dimensionless,
    "abundance_scalings.nitrogen_to_oxygen": dimensionless,
    "abundance_scalings.carbon_to_oxygen": dimensionless,
    "turbulence": km / s,
}

log_on_read_dict = {
    "reference_ionisation_parameter": True,
    "fixed_ionisation_parameter": True,
    "depletion_scale": False,
    "z": False,
    "ionisation_parameter": True,
    "hydrogen_density": True,
    "stop_column_density": False,
    "alpha_enhancement": False,
    "abundance_scalings.nitrogen_to_oxygen": False,
    "abundance_scalings.carbon_to_oxygen": False,
}

pluralisation_of_axes = {
    "z": "redshifts",
    "metallicities": "metallicities",
    "ionisation_parameter": "ionisation_parameters",
    "turbulence": "turbulences",
    "hydrogen_density": "hydrogen_densities",
    "depletion_scale": "depletion_scales",
    "stop_column_density": "column_densities",
    "alpha_enhancement": "alpha_enhancements",
    "reference_ionisation_parameter": "reference_ionisation_parameters",
    "abundance_scalings.nitrogen_to_oxygen": (
        "abundance_scalings.nitrogen_to_oxygen"
    ),
    "abundance_scalings.carbon_to_oxygen": (
        "abundance_scalings.carbon_to_oxygen"
    ),
}

# Non-ionising wavelength band (Angstrom) used to normalise each cloudy run to
# its input spectrum. Cloudy reproduces the input SED here, unlike at the
# ionisation edges, where it depends on how the input is sampled.
NORMALISATION_BAND = (1500.0, 5000.0)

# Warn if the Q of a normalised incident spectrum differs from the incident
# grid's stored Q by more than this (dex)
Q_CONSISTENCY_TOLERANCE_DEX = 0.02


def band_luminosity(lam, lnu, band=NORMALISATION_BAND):
    """Integrate L_nu over a wavelength band.

    The band edges are interpolated, so the result does not depend on where
    the grid's points fall relative to them.

    Args:
        lam (np.ndarray):
            Wavelengths in Angstrom, increasing, along the last axis of lnu.
        lnu (np.ndarray):
            Spectral luminosity density, shape (..., len(lam)).
        band (tuple of float):
            The (lower, upper) wavelength limits in Angstrom.

    Returns:
        np.ndarray or float:
            The integral of lnu over the band (lnu units x Angstrom), with
            the leading shape of lnu.
    """
    lo, hi = band
    lnu = np.asarray(lnu, dtype=float)
    inside = (lam > lo) & (lam < hi)
    flat = lnu.reshape(-1, lnu.shape[-1])
    y_lo = np.array([np.interp(lo, lam, row) for row in flat])
    y_hi = np.array([np.interp(hi, lam, row) for row in flat])
    x = np.concatenate([[lo], lam[inside], [hi]])
    y = np.concatenate([y_lo[:, None], flat[:, inside], y_hi[:, None]], axis=1)
    out = np.trapezoid(y, x, axis=1).reshape(lnu.shape[:-1])
    return out if out.ndim else float(out)


def continuum_normalisation(
    cloudy_lam,
    cloudy_incident,
    input_band_luminosity,
    band=NORMALISATION_BAND,
):
    """Normalisation that matches a cloudy run to its input spectrum.

    Cloudy rescales the input SED to the requested ionisation parameter. This
    returns the factor that maps its outputs back to the units of the input
    grid, by matching the incident continuum in a non-ionising band. Unlike
    matching the ionising photon rate Q, this doesn't depend on how the input
    samples the ionisation edges (see syncretize#151).

    Args:
        cloudy_lam (np.ndarray):
            Cloudy's wavelengths in Angstrom (increasing).
        cloudy_incident (np.ndarray):
            Cloudy's incident L_nu at cloudy_lam.
        input_band_luminosity (float):
            band_luminosity of the input spectrum for this grid point.
        band (tuple of float):
            The (lower, upper) wavelength limits in Angstrom.

    Returns:
        float or None:
            The normalisation, or None if either spectrum has no flux in the
            band (e.g. ionising-only incident grids), in which case the
            caller should fall back to another normalisation.
    """
    cloudy_band = band_luminosity(cloudy_lam, cloudy_incident, band)
    if not (
        np.isfinite(input_band_luminosity)
        and np.isfinite(cloudy_band)
        and input_band_luminosity > 0
        and cloudy_band > 0
    ):
        return None
    return float(input_band_luminosity / cloudy_band)


def read_incident_band_luminosity(incident_grid_file, band=NORMALISATION_BAND):
    """Read the band luminosity of every spectrum in an incident grid file.

    The spectra are read straight from the file, not from a loaded Grid,
    because synthesizer rescales luminosity weighted grids on load and the
    cloudy outputs must be matched to the values stored in the file.

    Args:
        incident_grid_file (str):
            Path to the incident grid HDF5 file.
        band (tuple of float):
            The (lower, upper) wavelength limits in Angstrom.

    Returns:
        np.ndarray or None:
            The band luminosity, with the incident grid's axes shape, or None
            if the incident grid doesn't cover the band.
    """
    with h5py.File(incident_grid_file, "r") as hf:
        lam = hf["spectra/wavelength"][...]
        if lam[0] > band[0] or lam[-1] < band[1]:
            return None
        i0 = max(int(np.searchsorted(lam, band[0])) - 1, 0)
        i1 = min(int(np.searchsorted(lam, band[1])) + 1, len(lam))
        lnu = hf["spectra/incident"][..., i0:i1]
    return band_luminosity(lam[i0:i1], lnu, band)


def create_empty_grid(
    grid_dir,
    incident_grid_name,
    new_grid_name,
    new_axes,
    new_axes_values,
    photoionisation_parameters,
):
    """
    Create an empty Synthesizer grid file based on an existing incident grid.

    Args:
        grid_dir (str):
            Directory where the grid file will be saved.
        incident_grid_name (str):
            Name of the incident grid file we ran through cloudy.
        new_grid_name (str):
            Name of the new grid file.
        new_axes (list):
            List of the new axes.
        new_axes_values (dict):
            Dictionary of the new axes values (including incident).
        photoionisation_parameters (dict):
            Dictionary of photoionisation modelling parameters.

    Returns:
        out_grid (GridFile):
            The newly created grid file.
        incident_grid (GridFile):
            The incident grid file.
        shape (tuple)
            The shape of the new grid.
    """
    # Open the parent incident grid
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=grid_dir,
        ignore_lines=True,
    )

    # Make the new grid
    out_grid = GridFile(f"{grid_dir}/{new_grid_name}")

    # Add attribute with the original incident grid axes
    out_grid.write_attribute(
        group="/",
        attr_key="incident_axes",
        data=incident_grid.axes,
    )

    # Copy over the model metadata
    out_grid.write_model_metadata(incident_grid._model_metadata)

    # Get properties of the grid
    (
        n_axes,
        shape,
        n_models,
        mesh,
        model_list,
        index_list,
    ) = get_grid_props_cloudy(new_axes, new_axes_values, verbose=False)

    # We want to copy over log10_specific_ionising_luminosity from the
    # incident grid to allow us to normalise the cloudy outputs.
    # However, the axes of the incident grid may be different from the
    # cloudy grid due to additional parameters, in which case we need to
    # extend the axes of log10_specific_ionising_luminosity.

    # Read log10_specific_ionising_luminosity directly from the incident
    # grid file rather than from the loaded Grid. synthesizer rescales
    # luminosity weighted grids to its internal luminosity unit on load, but
    # the cloudy outputs must be normalised to the values stored in the file.
    with h5py.File(incident_grid.grid_filename, "r") as hf:
        log10_specific_ionising_lum = {
            ion: hf["log10_specific_ionising_luminosity"][ion][...]
            for ion in hf["log10_specific_ionising_luminosity"].keys()
        }

    # Compute the amount we may have to expand the axes (if we don't need to
    # expand this is harmless)
    expansion = int(
        np.prod(shape) / np.prod(log10_specific_ionising_lum["HI"].shape)
    )

    # Loop over ions
    for ion in log10_specific_ionising_lum.keys():
        # If there are no additional axes simply copy over the incident
        # .log10_specific_ionising_lum .
        if len(new_axes) == len(incident_grid.axes):
            out_grid.write_dataset(
                key=f"log10_specific_ionising_luminosity/{ion}",
                data=log10_specific_ionising_lum[ion] * dimensionless,
                description="The specific ionising photon luminosity for"
                f" {ion}",
                log_on_read=False,
            )

        # Otherwise, we need to expand the axis to account for these additional
        # parameters.
        else:
            # Create new array with repeated elements
            log10_specific_ionising_luminosity = np.repeat(
                log10_specific_ionising_lum[ion] * dimensionless,
                expansion,
                axis=-1,
            )
            out_grid.write_dataset(
                f"log10_specific_ionising_luminosity/{ion}",
                np.reshape(log10_specific_ionising_luminosity, shape)
                * dimensionless,
                description="The specific ionising photon luminosity"
                f" for {ion}",
                log_on_read=False,
            )

    # Copy over the weight attribute from the incident grid
    weight_var = getattr(incident_grid, "_weight_var", None)
    if weight_var is None:
        weight_var = "None"
    out_grid.write_attribute(
        group="/",
        attr_key="WeightVariable",
        data=weight_var,
    )

    # Now write out the grid axes, we first do the incident grid axes so we
    # can extract their metadata and then any extras
    for axis, extract_axis in zip(
        incident_grid.axes, incident_grid._extract_axes
    ):
        # Write out this axis (we can get everything we need from the incident)
        if extract_axis != axis:
            log_on_read = True
        else:
            log_on_read = False

        out_grid.write_dataset(
            key=f"axes/{axis}",
            data=getattr(incident_grid, axis),
            description=f"Grid axes {axis} (taken from incident grid)",
            log_on_read=log_on_read,
        )

    # Now write out the new grid axes if there are any
    # List of new grid axes including pluralised form
    new_axes_for_grid = []
    for axis in new_axes:
        # Skip the one we did above
        if axis in incident_grid.axes:
            new_axes_for_grid.append(axis)
            continue

        if axis in pluralisation_of_axes.keys():
            pluralised_axis = pluralisation_of_axes[axis]
            new_axes_for_grid.append(pluralised_axis)
        else:
            raise ValueError(f"No pluralisation set for {axis} axis")

        # If we have units in the grid_params dictionary already then use
        # these, otherwise use the axes_units dictionary, if we don't have
        # units for the axis then raise an error.

        # Determine log_on_read safety
        log_on_read = log_on_read_dict.get(axis, False)

        if has_units(new_axes_values[axis]):
            description = f"grid axes {axis} (introduced during cloudy run)"
            out_grid.write_dataset(
                key=f"axes/{pluralised_axis}",
                data=new_axes_values[axis],
                description=description,
                log_on_read=log_on_read,
            )

        # Ok, maybe its just a common axis that we have units for
        elif axis in axes_units:
            # Multiply values with the corresponding unit
            values_with_units = new_axes_values[axis] * axes_units[axis]

            out_grid.write_dataset(
                key=f"axes/{pluralised_axis}",
                data=values_with_units,
                description=f"grid axes {axis} (introduced during cloudy run)",
                log_on_read=log_on_read,
            )

        else:
            raise ValueError(
                f"Unit for axis '{axis}' needs to be added to "
                f"create_synthesizer_grid.py"
            )

    # Add attribute with the full grid axes
    out_grid.write_attribute(
        group="/", attr_key="axes", data=new_axes_for_grid
    )

    # Write out cloudy parameters
    out_grid.write_cloudy_metadata(photoionisation_parameters)

    return out_grid, incident_grid


def get_grid_properties_hf(hf, verbose=True):
    """
    Get the properties of a grid from an HDF5 file.

    Arguments:
        hf : h5py.File
            The HDF5 file containing the grid.
        verbose : boolean
            Print out the properties of the grid.

    Returns:
        axes : list
            List of axes in the correct order.
        n_axes : int
            Number of axes in the grid.
        shape : tuple
            Shape of the grid.
        n_models : int
            Number of models in the grid.
        mesh : tuple
            Mesh of the grid.
        model_list : list
            List of model parameters.
        index_list : list
            List of model indices.
    """
    # Unpack the axes
    axes = hf.attrs["axes"]
    axes_values = {axis: hf[f"axes/{axis}"][:] for axis in axes}

    # Get the properties of the grid including the dimensions etc.
    return axes, *get_grid_props_cloudy(axes, axes_values, verbose=verbose)


def check_if_failed(
    model_to_check,
    extensions_to_check=["emergent_elin"],
):
    """
    Function to check if a particular model has failed.

    Args:
        model_to_check (str)
            The model, including path, to check.
        extensions_to_check (list)
            List of file extensions to check to check for each model.

    """

    failed = False

    # Check if files exist
    for ext in extensions_to_check:
        if not os.path.isfile(model_to_check + "." + ext):
            failed = True

    # If they exist also check they have size >0
    if not failed:
        for ext in extensions_to_check:
            if os.path.getsize(model_to_check + "." + ext) < 100:
                failed = True

    return failed


def check_cloudy_runs(
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    extensions_to_check=["emergent_elin"],
    machine=None,
    cloudy_executable_path=None,
):
    """
    Check that all the cloudy runs have run properly.

    Args:
        new_grid_name
            The name of the new grid (also the subdirectory wher the cloudy
            runs are stored)
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        replace (boolean)
            If a run has failed simply replace the model with the previous one.
        extensions_to_check (list)
            List of files to check for each model.

    Returns:
        failed_list : list
            List of models that have failed.
    """

    # List of tuples describing failed models
    failed_list = []

    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # The model to check, including the full path
            model_to_check = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            # Check if it failed
            failed = check_if_failed(
                model_to_check,
                extensions_to_check=extensions_to_check,
            )

            # Record models that have failed
            if failed:
                # The path to the cloudy out file
                outfile = (
                    f"{cloudy_dir}/{new_grid_name}/"
                    f"{incident_index}/{photoionisation_index}.out"
                )

                # Grab the final line of
                with open(outfile, "r") as file:
                    lines = file.readlines()
                    last_line = lines[-1].strip()

                # Append the incident and photoionisation index to an array
                failed_list.append([incident_index, photoionisation_index])

                # If at least one model fails print a header
                if len(failed_list) == 1:
                    print("Failed models:")
                    print(
                        "incident_index, photoionisation_index, last line from"
                        "cloudy"
                    )

                # Print the indices and the last line of the cloudy out file
                print("    ", incident_index, photoionisation_index, last_line)

    number_of_failed_models = len(failed_list)

    if number_of_failed_models > 0:
        print("number of failed cloudy runs:", number_of_failed_models)

        # Save list of failed runs
        failed_filename = f"{new_grid_name}.failures"
        np.savetxt(failed_filename, np.array(failed_list).T, fmt="%d")

        # If a specific machine is specified run the function in
        # submission_scripts to generate a submission script.
        if machine:
            getattr(submission_scripts, machine)(
                new_grid_name,
                number_of_jobs=number_of_failed_models,
                cloudy_output_dir=cloudy_dir,
                cloudy_executable_path=cloudy_executable_path,
                memory="4G",
                from_list=failed_filename,
            )

    # Return number_of_failed_models
    return number_of_failed_models


def add_spectra(
    new_grid,
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    new_shape,
    spec_names=("incident", "transmitted", "nebular", "linecont"),
    norm_by_q=True,
    norm_method="continuum",
    incident_band_luminosity=None,
):
    """
    Open cloudy spectra and add them to the grid.

    Each cloudy run is rescaled by a single factor (applied to all spectra and,
    via spectra["normalisation"], to the line luminosities) that maps it back
    to the units of the incident grid.

    Args:
        new_grid (GridFile)
            The grid file to add the spectra to.
        new_grid_name (str)
            The name of the new grid.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        spec_names (list)
            The names of the spectra to save (default is incident, transmitted,
            nebular and linecont).
        shape (ndarray)
            The shape of the new grid.
        norm_by_q (bool)
            Deprecated, kept for backwards compatibility. False disables
            normalisation entirely (same as norm_method=None).
        norm_method (str or None)
            How to normalise each run:
            "continuum" (default): match cloudy's incident spectrum to the
            input over NORMALISATION_BAND (needs incident_band_luminosity).
            Falls back to "q" for grid points with no flux in the band.
            "q": match the ionising photon rate Q(HI) of cloudy's incident
            spectrum to the incident grid's stored value. This is the
            previous behaviour; it is sensitive to how the input samples
            the ionisation edges (syncretize#151).
            None: no normalisation.
        incident_band_luminosity (np.ndarray)
            band_luminosity of each incident grid spectrum (incident grid
            shape), see read_incident_band_luminosity.
    """
    if not norm_by_q:
        norm_method = None
    if norm_method not in ("continuum", "q", None):
        raise ValueError(f"Unknown norm_method: {norm_method}")
    if norm_method == "continuum" and incident_band_luminosity is None:
        print(
            "No incident band luminosities available; normalising by Q "
            "instead of the continuum."
        )
        norm_method = "q"
    n_q_fallback = 0
    q_offsets = []

    # Determine the cloudy version...
    cloudy_version = new_grid.read_attribute(
        "cloudy_version",
        group="CloudyParams",
    )

    # ... and use to select the correct module.
    if cloudy_version.split(".")[0] == "c23":
        cloudy = cloudy23
    elif cloudy_version.split(".")[0] == "c17":
        cloudy = cloudy17

    # Read first spectra from the first grid point to get length and
    # wavelength grid.
    lam = cloudy.read_wavelength(f"{cloudy_dir}/{new_grid_name}/0/0")

    # Write the spectra names... for some reason (not sure why since .keys()
    # gives the same result)
    new_grid.write_attribute(group="/", attr_key="spec_names", data=spec_names)

    # Number of wavelength points
    nlam = len(lam)

    # Set up the spectra dictionary
    spectra = {}

    # Make spectral grids and set them to zero
    for spec_name in spec_names:
        spectra[spec_name] = np.zeros((*new_shape, nlam))

    # Array for holding the normalisation which is calculated below and
    # used by lines
    spectra["normalisation"] = np.ones(new_shape)

    # Array for recording cloudy failures
    failures = np.zeros(new_shape)

    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # Get the indices of the grid_point
            indices = tuple(incident_index_tuple) + tuple(
                photoionisation_index_tuple
            )

            # The infile
            infile = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            # Check to see if the model failed or not, using the default
            # extensions
            failed = check_if_failed(infile)

            # If the model has failed save the spectra as an array of zeros
            if failed:
                # Set all the spectra to zero but with the correct shape
                for spec_name in spec_names:
                    spectra[spec_name][indices] = np.zeros(nlam)

                # Record failures in array
                failures[indices] = 1

            else:
                # Read the continuum file containing the spectra
                spec_dict = cloudy.read_continuum(infile, return_dict=True)

                # Work out the factor that maps this run back to the units of
                # the incident grid
                normalisation = None
                if norm_method == "continuum":
                    normalisation = continuum_normalisation(
                        lam,
                        spec_dict["incident"],
                        incident_band_luminosity[tuple(incident_index_tuple)],
                    )
                    if normalisation is None:
                        n_q_fallback += 1

                # Q normalisation: requested, or as the fallback for points
                # with no continuum in the band. Also used as a diagnostic.
                if norm_method is not None:
                    # Create sed object
                    sed = Sed(
                        lam=lam * Angstrom,
                        lnu=spec_dict["incident"] * erg / s / Hz,
                    )

                    # Calculate Q
                    ionising_photon_production_rate = (
                        sed.calculate_ionising_photon_production_rate(
                            ionisation_energy=13.6 * eV, limit=100
                        )
                    )
                    log10_q_stored = new_grid.read_dataset(
                        "log10_specific_ionising_luminosity/HI",
                        indices=indices,
                    )
                    log10_q_norm = log10_q_stored - np.log10(
                        ionising_photon_production_rate
                    )

                    if normalisation is None:
                        normalisation = 10**log10_q_norm
                    elif np.isfinite(log10_q_norm):
                        # How far the Q of the continuum-normalised incident
                        # spectrum is from the stored Q (dex)
                        q_offsets.append(
                            float(log10_q_norm - np.log10(normalisation))
                        )

                    # Save normalisation for later use (rescaling lines)
                    spectra["normalisation"][indices] = normalisation

                # Save the normalised spectrum to the correct grid point
                for spec_name in spec_names:
                    spectra[spec_name][indices] = (
                        spec_dict[spec_name]
                        * spectra["normalisation"][indices]
                    )

    # Report on the normalisation
    if norm_method == "continuum":
        if n_q_fallback:
            print(
                f"{n_q_fallback} grid points had no incident continuum in "
                f"{NORMALISATION_BAND} A and were normalised by Q instead."
            )
        if q_offsets:
            worst = float(np.max(np.abs(q_offsets)))
            print(
                "Continuum-normalised incident spectra vs stored Q: max "
                f"|offset| = {worst:.3f} dex, median "
                f"{np.median(np.abs(q_offsets)):.3f} dex."
            )
            if worst > Q_CONSISTENCY_TOLERANCE_DEX:
                print(
                    "WARNING: the ionising photon rate of some normalised "
                    "incident spectra differs from the incident grid's stored "
                    f"Q by more than {Q_CONSISTENCY_TOLERANCE_DEX} dex. The "
                    "incident grid may sample its ionising spectrum too "
                    "coarsely (e.g. around the 912 A edge); the stellar "
                    "continuum is still normalised correctly."
                )

    # Apply units to the spectra
    for spec in spectra:
        spectra[spec] *= erg / s / Hz

    # Write the spectra out
    new_grid.write_spectra(
        spectra,
        wavelength=lam * Angstrom,
    )

    # Need to write the dataset containing the failures
    new_grid.write_dataset(
        "failures",
        failures * dimensionless,
        "array of model failures",
        False,
        verbose=False,
    )

    return lam, spectra


def add_lines(
    new_grid,
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    new_shape,
    lam=None,
    spectra=None,
    line_type="linelist",
):
    """
    Open cloudy lines and add them to the HDF5 grid

    Arguments:
        new_grid (GridFile)
            The grid file to add the spectra to.
        new_grid_name (str)
            The name of the new grid.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        new_shape (ndarray)
            The shape of the new grid.
        lam (ndarray)
            The wavelength grid of the spectra.
        spectra (bool)
            The spectra.
        line_type (str)
            The type of line file to use (linelist, lines)
    """

    if spectra is not None:
        calculate_continuum = True
    else:
        calculate_continuum = False

    # Determine the cloudy version...
    cloudy_version = new_grid.read_attribute(
        "cloudy_version",
        group="CloudyParams",
    )

    # ... and use to select the correct module.
    if cloudy_version.split(".")[0] == "c23":
        cloudy = cloudy23
    elif cloudy_version.split(".")[0] == "c17":
        cloudy = cloudy17

    # Open the first linelist and use it to populate the keys
    with open(f"{cloudy_dir}/{new_grid_name}/0/linelist.dat") as file:
        linelist = file.readlines()
        # strip escape character
        lines_to_include = [
            re.sub(r"[\x00-\x1F\x7F]", "", s) for s in linelist
        ]

    # Calculate number of lines
    nlines = len(lines_to_include)

    print(f"number of lines included: {nlines}")

    # Set up the lines dictionary
    lines = {}

    # Define output quantities

    # We always save luminosity...
    lines["luminosity"] = np.empty((*new_shape, nlines))

    # ... but only save continuum values if spectra are provided.
    continuum_quantities = [
        "transmitted",
        "incident",
        "nebular_continuum",
        "total_continuum",
    ]

    if calculate_continuum:
        spectra_ = {}

        # Calculate incideted_contiuum
        spectra_["incident"] = spectra["incident"]

        # Calculate transmitted_contiuum
        spectra_["transmitted"] = spectra["transmitted"]

        # Calculate nebular_contiuum
        spectra_["nebular_continuum"] = (
            spectra["nebular"] - spectra["linecont"]
        )

        # Calculate total_continuum. Note this is not the same as adding
        #  nebular and transmitted since that would include emission lines.
        spectra_["total_continuum"] = (
            spectra_["nebular_continuum"] + spectra_["transmitted"]
        )

        # Define output arrays
        for continuum_quantity in continuum_quantities:
            lines[continuum_quantity] = np.empty((*new_shape, nlines))

    # Array for recording cloudy failures
    failures = np.zeros(new_shape)

    # Loop over incident models and photoionisation models
    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # Get the indices of the grid_point
            indices = tuple(incident_index_tuple) + tuple(
                photoionisation_index_tuple
            )

            # The infile
            infile = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            # Check to see if the model failed or not, using the default
            # extensions
            failed = check_if_failed(infile)

            # If the model has failed save the luminosities as an array of
            # zeros
            if failed:
                # Set the line luminosities to zero
                lines["luminosity"][indices] = np.zeros(nlines)

                # Set the continuum luminosities to zero
                if calculate_continuum:
                    for continuum_quantity in continuum_quantities:
                        lines[continuum_quantity][indices] = np.zeros(nlines)

                # Record failures in array
                failures[indices] = 1

            else:
                # Read line quantities
                ids, line_wavelengths, luminosities = cloudy.read_linelist(
                    infile, extension="emergent_elin"
                )

                # re-order by wavelength
                sorted_indices = np.argsort(line_wavelengths)
                ids = ids[sorted_indices]
                luminosities = luminosities[sorted_indices]
                line_wavelengths = line_wavelengths[sorted_indices]

                # If we're on the first grid point save the wavelength grid
                if (incident_index == 0) and (photoionisation_index == 0):
                    lines["wavelength"] = line_wavelengths
                    # for id, lam in zip(ids, sorted_indices):
                    #     print(id, lam)

                    # Record list of IDs
                    lines["id"] = ids

                # If spectra have been calculated extract the normalisation ..
                if calculate_continuum:
                    normalisation = spectra["normalisation"][indices]

                # Otherwise set the normalisation to unity
                else:
                    normalisation = 1.0

                # Calculate line luminosity and save it. Uses normalisation
                # from spectra. Units are erg/s
                lines["luminosity"][indices] = luminosities * normalisation

                # Calculate continuum luminosities. Units are erg/s/Hz.
                if calculate_continuum:
                    for continuum_quantity in continuum_quantities:
                        lines[continuum_quantity][indices] = np.interp(
                            line_wavelengths,
                            lam,
                            spectra_[continuum_quantity][indices],
                        )

    # Apply units to the wavelength
    lines["wavelength"] *= Angstrom

    # Apply units to the lines
    lines["luminosity"] *= erg / s

    # If continuum values are calculated add units
    for continuum_quantity in continuum_quantities:
        if continuum_quantity in lines:
            lines[continuum_quantity] *= erg / s / Hz

    # Write the lines out
    new_grid.write_lines(
        lines,
    )

    # Need to write the dataset containing the failures
    # This is repeated if spectra were made.
    if spectra is not None:
        print("Dataset of failed models already written out!")
    else:
        new_grid.write_dataset(
            "failures",
            failures * dimensionless,
            "array of model failures",
            False,
            verbose=False,
        )


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(
        description="Create Synthesizer HDF5 grid files from cloudy outputs.",
        cloudy_args=True,
    )

    # Add additional parameters which are specific to this script

    # Machine (for submission script generation for failed models)
    parser.add_argument("--machine", type=str, required=False, default=None)

    # Path to cloudy directory (not the executable; this is assumed to
    # {cloudy}/{cloudy_version}/source/cloudy.exe)
    parser.add_argument(
        "--cloudy-executable-path", type=str, required=False, default=None
    )

    # Should we include the spectra in the grid?
    parser.add_argument(
        "--include-spectra",
        action="store_true",
        help="Should the spectra be included in the grid?",
        default=False,
        required=False,
    )

    # Should we normalise by the specific ionising luminosity?
    parser.add_argument(
        "--norm-by-Q",
        "-Q",
        default=True,
        required=False,
        help="Should the grid be normalised by the specific "
        "ionising luminosity? (default is True)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Unpack arguments
    grid_dir = args.grid_dir
    cloudy_dir = args.cloudy_output_dir
    incident_grid_name = args.incident_grid
    cloudy_param_file = args.cloudy_paramfile
    extra_cloudy_param_file = args.cloudy_paramfile_extra
    include_spectra = args.include_spectra
    machine = args.machine
    cloudy_executable_path = args.cloudy_executable_path
    verbose = args.verbose

    print(" " * 80)
    print("-" * 80)
    print("incident_grid_name:", incident_grid_name)
    print("cloudy_param_file:", cloudy_param_file)
    print("extra_cloudy_param_file:", extra_cloudy_param_file)

    # Check for extensions
    # Create _file and _name versions (with and w/o extensions)

    if cloudy_param_file.split(".")[-1] != "yaml":
        cloudy_param_name = cloudy_param_file
        cloudy_param_file += ".yaml"
    else:
        cloudy_param_name = "".join(cloudy_param_file.split(".")[:-1])

    if extra_cloudy_param_file is not None:
        if extra_cloudy_param_file.split(".")[-1] != "yaml":
            extra_cloudy_param_name = extra_cloudy_param_file
            extra_cloudy_param_file += ".yaml"
        else:
            extra_cloudy_param_name = "".join(
                extra_cloudy_param_file.split(".")[:-1]
            )

    if incident_grid_name.split(".")[-1] != "hdf5":
        incident_grid_file = incident_grid_name + ".hdf5"
    else:
        incident_grid_file = incident_grid_name
        incident_grid_name = "".join(incident_grid_name.split(".")[:-1])

    # Get name of new grid (concatenation of incident_grid and cloudy
    # parameter file)
    new_grid_name = f"{incident_grid_name}_cloudy-{cloudy_param_name}"

    # If an additional parameter set append this to the new grid name
    if extra_cloudy_param_file:
        # Ignore the directory part if it exists
        extra_cloudy_param_name = extra_cloudy_param_name.split("/")[-1]
        # Append the new_grid_name with the additional name
        new_grid_name += "-" + extra_cloudy_param_name

    new_grid_file = new_grid_name + ".hdf5"

    # Open the incident grid using synthesizer
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=grid_dir,
        ignore_lines=True,
    )

    # Extract axes and axes values from the Grid
    incident_axes = incident_grid.axes
    incident_axes_values = incident_grid.axes_values

    # Get properties of the incident grid
    (
        incident_n_axes,
        incident_shape,
        incident_n_models,
        incident_mesh,
        incident_model_list,
        incident_index_list,
    ) = get_grid_props_cloudy(
        incident_axes, incident_axes_values, verbose=False
    )

    # Load the cloudy parameters you are going to run
    fixed_photoionisation_params, variable_photoionisation_params = (
        get_cloudy_params(cloudy_param_file)
    )

    # If an additional parameter set is provided supersede the default
    # parameters with these.
    if extra_cloudy_param_file:
        (photoionisation_fixed_params_, photoionisation_variable_params_) = (
            get_cloudy_params(extra_cloudy_param_file)
        )
        fixed_photoionisation_params |= photoionisation_fixed_params_
        variable_photoionisation_params |= photoionisation_variable_params_

    print("fixed photoionisation parameters:")
    for k, v in fixed_photoionisation_params.items():
        print(f"    {k}: {v}")
    print("variable photoionisation parameters:")
    for k, v in variable_photoionisation_params.items():
        print(f"    {k}: {v}")

    # If we have photoionisation parameters that vary we need to calculate the
    # model list
    if len(variable_photoionisation_params) > 0:
        # Sort the axes to ensure consistent ordering
        photoionisation_axes = sorted(variable_photoionisation_params.keys())

        # get properties of the photoionsation grid
        (
            photoionisation_n_axes,
            photoionisation_shape,
            photoionisation_n_models,
            photoionisation_mesh,
            photoionisation_model_list,
            photoionisation_index_list,
        ) = get_grid_props_cloudy(
            photoionisation_axes,
            variable_photoionisation_params,
            verbose=False,
        )

    # Else, we need to specify that the only photonionisation model is 0
    else:
        photoionisation_n_models = 1
        photoionisation_index_list = [[]]
        photoionisation_shape = ()

    # Define the axes and values of the output grid
    new_axes = list(incident_axes)
    new_axes_values = incident_axes_values.copy()

    new_shape = tuple(incident_shape) + tuple(photoionisation_shape)
    if verbose:
        print("new shape:", new_shape)

    # If there are photoionisation axes add these as well
    if len(variable_photoionisation_params) > 0:
        new_axes += photoionisation_axes
        new_axes_values |= variable_photoionisation_params

    for key, value in new_axes_values.items():
        print(key, value)

    # Create empty synthesizer grid, this returns a GridFile object we can
    # add spectra and lines to below
    new_grid, incident_grid = create_empty_grid(
        grid_dir,
        incident_grid_name,
        new_grid_file,
        new_axes,
        new_axes_values,
        fixed_photoionisation_params,
    )

    # Check cloudy runs and potentially replace them by the nearest grid point
    # if they fail.
    number_of_failed_models = check_cloudy_runs(
        new_grid_name,
        cloudy_dir,
        incident_index_list,
        photoionisation_index_list,
        machine=machine,
        cloudy_executable_path=cloudy_executable_path,
    )

    # Even if a grid point has failed the code will now still create a grid,
    # recording failures with zeros in the spectra and line quantities. The
    # code will also save an array of failure flags.

    # Now add spectra.
    if include_spectra:
        # Band luminosity of each input spectrum, read from the file, used to
        # normalise each cloudy run back to the incident grid's units
        incident_band_luminosity = read_incident_band_luminosity(
            incident_grid.grid_filename
        )

        lam, spectra = add_spectra(
            new_grid,
            new_grid_name,
            cloudy_dir,
            incident_index_list,
            photoionisation_index_list,
            new_shape,
            spec_names=("incident", "transmitted", "nebular", "linecont"),
            norm_method="continuum",
            incident_band_luminosity=incident_band_luminosity,
        )
        print("Added spectra")
        print(spectra.keys())
    else:
        lam = None
        spectra = None

    # Now add lines
    add_lines(
        new_grid,
        new_grid_name,
        cloudy_dir,
        incident_index_list,
        photoionisation_index_list,
        new_shape,
        lam,
        spectra,
        line_type="linelist",
    )
    print("Added lines")
