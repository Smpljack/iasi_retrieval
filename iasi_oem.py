import numpy as np

from typhon.arts.workspace import Workspace, arts_agenda
from typhon.physics import wavelength2frequency, wavenumber2frequency


def load_abs_lookup(ws, atm_path, freq_ranges):
    """
    Loads existing absorption lookup table or creates one in case it does
    not exist yet for given batch of atmospheres and frequency range.
    """


def setup_sensor(ws, freq_ranges, f_backend_width):
    """
    Setup the sensor properties, mainly including the frequency grid.
    """
    ws.sensor_pos = np.array([[850e3]])  # 850km
    ws.sensor_time = np.array([0.0])
    ws.sensor_los = np.array([[180.0]])  # nadir viewing
    ws.f_grid = np.concatenate([np.arange(wavenumber2frequency(freq[0] * 0.99),
                                          wavenumber2frequency(freq[1] * 1.01),
                                          f_backend_width) for freq in freq_ranges])
    ws.f_backend_width = np.array([f_backend_width])
    # Sensor settings
    ws.FlagOn(ws.sensor_norm)
    ws.AntennaOff()
    ws.sensor_responseInit()
    ws.sensor_responseBackend()
    return ws


def load_generic_settings(ws):
    """Load generic agendas and set generic settings."""
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")

    # (standard) emission calculation
    ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission)

    # cosmic background radiation
    ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground)

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)

    ws.Copy(ws.surface_rtprop_agenda, ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface)

    # clearsky agenda
    ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__LookUpTable)
    # sensor-only path
    ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)
    # no refraction
    ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)

    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    ws.stokes_dim = 1
    ws.jacobian_quantities = []
    ws.iy_unit = "PlanckBT"
    ws.cloudboxOff()
    return ws

def iasi_obs_batch_simulation(ws, atm_batch_path):


def atm_batch_and_abs_setup(ws, atm_batch_path):
    """
    Load the batch of atmospheric fields and setup absorption species
    for the RT calculation.
    """
    ws.ReadXML(ws.batch_atm_fields_compact,
               atm_batch_path)
    ws.batch_atm_fields_compactAddConstant(name="abs_species-O2",
                                           value=0.2095,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-N2",
                                           value=0.7808,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-CO2",
                                           value=0.0004,
                                           condensibles=["abs_species-H2O"])
    # define absorbing species and load lines for given frequency range from HITRAN
    ws.abs_speciesSet(species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                               "O2, O2-CIAfunCKDMT100",
                               "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                               "O3",
                               "CO2, CO2-CKDMT252"])
    ws.abs_lineshapeDefine(shape='Voigt_Kuntz6',
                           forefactor='VVH',
                           cutoff=750e9)
    # Read HITRAN catalog / ARTS catalog
    ws.abs_linesReadFromSplitArtscat(
        basename='/scratch/uni/u237/data/catalogue/hitran/hitran_split_artscat5/',
        fmin=np.min(ws.f_backend.value),
        fmax=np.max(ws.f_backend.value))
    return ws