"""Tests for normalising cloudy runs by the non-ionising continuum."""

from pathlib import Path

import numpy as np
import pytest
from unyt import Angstrom, Hz, dimensionless, erg, eV, s, unyt_array

from syncretize.grid_io import GridFile

CLOUDY_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "syncretize" / "cloudy"
)
LYMAN = 911.65  # 13.6 eV in Angstrom


@pytest.fixture
def csg(monkeypatch):
    """Import create_synthesizer_grid with its sibling modules importable.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Used to put the cloudy directory on sys.path for this test only.

    Returns:
        module:
            syncretize.cloudy.create_synthesizer_grid.
    """
    pytest.importorskip("synthesizer.grid")
    monkeypatch.syspath_prepend(str(CLOUDY_DIR))
    from syncretize.cloudy import create_synthesizer_grid

    return create_synthesizer_grid


def _step_sed(lam, jump=1e3):
    """A gently sloped SED that drops by `jump` blueward of 912 A."""
    return np.where(lam < LYMAN, 1.0 / jump, 1.0) * (lam / 1000.0) ** -1.5


def _loglog(new_lam, lam, lnu):
    """Interpolate an SED in log-log space, as cloudy does for a table."""
    return 10 ** np.interp(np.log10(new_lam), np.log10(lam), np.log10(lnu))


def _q(lam, lnu):
    """Ionising photon rate with synthesizer."""
    from synthesizer.emissions import Sed

    sed = Sed(lam=lam * Angstrom, lnu=lnu * erg / s / Hz)
    return float(
        sed.calculate_ionising_photon_production_rate(
            ionisation_energy=13.6 * eV
        )
    )


def test_band_luminosity_flat(csg):
    """A flat spectrum integrates to its value times the band width."""
    lam = np.linspace(1000.0, 6000.0, 37)
    lo, hi = csg.NORMALISATION_BAND
    assert csg.band_luminosity(lam, np.full_like(lam, 2.0)) == pytest.approx(
        2.0 * (hi - lo)
    )


def test_band_luminosity_independent_of_sampling(csg):
    """Coarse and fine grids give the same band integral for a smooth SED."""
    coarse = np.arange(1005.0, 6000.0, 20.0)
    fine = np.geomspace(1000.0, 6000.0, 5000)
    a = csg.band_luminosity(coarse, _step_sed(coarse))
    b = csg.band_luminosity(fine, _step_sed(fine))
    assert a == pytest.approx(b, rel=2e-3)


def test_continuum_normalisation_recovers_scale(csg):
    """The continuum factor is exact where the Q factor is not.

    Mimic a cloudy run: the input SED on a 10 A grid (points at 905 and
    915 A) is rescaled by an arbitrary factor and re-gridded log-log onto a
    finer mesh, which smears the Lyman edge.
    """
    lam_in = np.arange(500.0, 6000.0, 10.0) + 5.0
    lnu_in = _step_sed(lam_in)
    mesh = np.sort(
        np.concatenate(
            [np.geomspace(lam_in[0], lam_in[-1], 4000), [911.55, 913.27]]
        )
    )
    scale = 3.7e-12
    cloudy_incident = scale * _loglog(mesh, lam_in, lnu_in)

    norm = csg.continuum_normalisation(
        mesh, cloudy_incident, csg.band_luminosity(lam_in, lnu_in)
    )
    assert norm * scale == pytest.approx(1.0, rel=2e-3)

    # The Q based factor, as used before, is off for this input (~14%)
    q_norm = _q(lam_in, lnu_in) / _q(mesh, cloudy_incident)
    assert abs(q_norm * scale - 1) > 0.05


def test_continuum_normalisation_none_without_band_flux(csg):
    """No flux in the band (e.g. ionising-only grids) returns None."""
    mesh = np.geomspace(100.0, 6000.0, 1000)
    incident = np.where(mesh < LYMAN, 1.0, 0.0)
    assert csg.continuum_normalisation(mesh, incident, 1.0) is None
    assert csg.continuum_normalisation(mesh, np.ones_like(mesh), 0.0) is None


def _write_incident_grid(path, wavelength):
    """Write a minimal two point incident grid with L_nu = 1 + index."""
    grid = GridFile(str(path))
    axes = {"alpha": unyt_array([-1.6, -1.2], dimensionless)}
    lnu = np.array([np.full(len(wavelength), 1.0 + i) for i in range(2)])
    grid.write_grid_common(
        axes=axes,
        wavelength=unyt_array(wavelength, Angstrom),
        spectra={"incident": unyt_array(lnu, erg / s / Hz)},
        log_on_read={"alpha": False},
        descriptions={"alpha": "Power-law slope"},
        model={"model": "unit-test"},
    )


def test_read_incident_band_luminosity(csg, tmp_path):
    """Band luminosities are read from the file, one per grid point."""
    path = tmp_path / "incident.hdf5"
    _write_incident_grid(path, np.geomspace(100.0, 1e5, 500))
    lo, hi = csg.NORMALISATION_BAND
    out = csg.read_incident_band_luminosity(str(path))
    np.testing.assert_allclose(out, [1.0 * (hi - lo), 2.0 * (hi - lo)])


def test_read_incident_band_luminosity_band_not_covered(csg, tmp_path):
    """An incident grid that doesn't cover the band returns None."""
    path = tmp_path / "incident.hdf5"
    _write_incident_grid(path, np.geomspace(10.0, 1000.0, 200))
    assert csg.read_incident_band_luminosity(str(path)) is None
