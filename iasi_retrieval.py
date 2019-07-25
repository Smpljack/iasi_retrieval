import numpy as np
import matplotlib.pyplot as plt

from typhon.arts.workspace import Workspace, arts_agenda
from typhon.arts import xml
from typhon.physics import wavelength2frequency, wavenumber2frequency
from poem import oem
from retrieval_plotting import plot_retrieval_profiles
from arts_with_python.arts_data_handler import gridded_field_to_xr_ds

ws = Workspace(verbosity = 1)
indir = outdir = "a_priori/"

###########################################################################
# General setup
###########################################################################
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


def read_iasi_abs_lines(ws, spectral_limit1, spectral_limit2, spectral_str):
    """
    Read absorption lines from HITRAN catalogue into workspace for given spectral range, which can be
    in units of frequency, wavelength or wavenumber.
    :param ws: ARTS workspace object
    :param spectral_limit1: lower spectral limit
    :param spectral_limit2: upper spectral limit
    :param spectral_str: Can be "frequency", "wavelength" or "wavenumber"
    :return:
    """
    if spectral_str == "wavelength":
        spectral_limit2 = wavelength2frequency(spectral_limit1 * 1e-6)
        spectral_limit1 = wavelength2frequency(spectral_limit2 * 1e-6)
    elif spectral_str == "wavenumber":
        spectral_limit1 = wavenumber2frequency(spectral_limit1 * 100)
        spectral_limit2 = wavenumber2frequency(spectral_limit2 * 100)

    # define absorbing species and load lines for given frequency range from HITRAN
    ws.abs_speciesSet(species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                               "O2, O2-CIAfunCKDMT100",
                               "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                               "O3",
                               "CO2, CO2-CKDMT252"])
    # Read HITRAN catalog (needed for O3):
    ws.abs_linesReadFromHitran("/scratch/uni/u237/data/catalogue/hitran/hitran_online/HITRAN2012.par",
                               spectral_limit1,
                               spectral_limit2)
    ws.abs_lines_per_speciesCreateFromLines()
    return ws

ws = read_iasi_abs_lines(ws, 595, 1405, "wavenumber")
# ws = read_abs_lines(ws, 595, 755, "wavenumber")

###########################################################################
# Atmospheric Setup
###########################################################################

ws.atmosphere_dim = 1  # for 1DVAR
#p = np.array(
#    [1000., 975., 950., 925., 900., 850., 800., 750., 700., 650., 600., 550., 500., 400., 300., 200., 100.])*100.0
garand = gridded_field_to_xr_ds(
    xml.load("/scratch/uni/u237/users/mprange/arts/controlfiles/testdata/garand_profiles.xml.gz")[0], "z")
p = garand["z"].values
ws.p_grid = p
ws.AtmRawRead(basename="testdata/tropical") #tropical atmosphere assumed
ws.lat_grid = []
ws.lon_grid = []
ws.AtmFieldsCalc()
ws.AbsInputFromAtmFields()

ws.z_surface = np.asarray(ws.z_field)[0]
ws.t_surface = np.asarray(ws.t_field)[0]
alt = np.asarray(ws.z_field).ravel() # altitude in [m]
xml.save(ws.z_field.value, outdir+"z_field.xml")
xml.save(ws.p_grid.value, outdir+"p_grid.xml")

###########################################################################
# Sensor Settings
###########################################################################
ws.sensor_pos  = np.array([[850e3]]) # 850km
ws.sensor_time = np.array([0.0])
ws.sensor_los  = np.array([[180.0]]) # nadir viewing


h2o_band = np.arange(wavenumber2frequency(1185*100), wavenumber2frequency(1405*100), 7.5e9)
co2_band = np.arange(wavenumber2frequency(595*100), wavenumber2frequency(755*100), 7.5e9)
# The below lines are important to select frequency range and resolution.
# Grid spacing and FWHM of the Gaussian response should match!
ws.f_grid.value = np.concatenate((co2_band, h2o_band))
# load external f_backend
ws.ReadXML(ws.f_backend, "sensor_specs/IASI/f_backend.xml" )
ws.VectorCreate("f_backend_width")
ws.ReadXML(ws.f_backend_width, "sensor_specs/IASI/f_backend_width.xml" )
ws.backend_channel_responseGaussian(ws.f_backend_width)

# Sensor settings
ws.FlagOn(ws.sensor_norm)
ws.AntennaOff()
ws.sensor_responseInit()
ws.sensor_responseBackend()
ws.sensor_checkedCalc()

ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
ws.abs_xsec_agenda_checkedCalc()
ws.abs_lookupSetup()
ws.abs_lookupCalc()

###########################################################################
# Surface Settings
###########################################################################
ws.surface_scalar_reflectivity = np.array([0.5]) # nominal albedo for surface
ws.propmat_clearsky_agenda_checkedCalc()
ws.atmgeom_checkedCalc()
ws.cloudbox_checkedCalc()
ws.sensor_checkedCalc()
ws.jacobianOff()

###########################################################################
# Load "observations" and save a priori state
###########################################################################
ws.y = xml.load("/scratch/uni/u237/users/mprange/phd/iasi_retrieval/iasi_obs_simulation/iasi_obs.xml")[0]
a_priori_vmr = ws.vmr_field.value
a_priori_T = ws.t_field.value
xml.save(a_priori_vmr, "a_priori/vmr_apriori.xml")
xml.save(a_priori_T, "a_priori/T_apriori.xml")
###########################################################################
# Prepare OEM retrieval
###########################################################################
ws.retrievalDefInit()

retrieval_species = ["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                     "T"]
oem.save_covariances(retrieval_species)

# Add H2O as retrieval quantity.
ws.retrievalAddAbsSpecies(species=retrieval_species[0],
                          unit="rel",
                          g1=ws.p_grid,
                          g2=ws.lat_grid,
                          g3=ws.lon_grid)
ws.covmat_sxAddBlock(block=xml.load("a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml"))
ws.retrievalAddTemperature(
                           g1=ws.p_grid,
                           g2=ws.lat_grid,
                           g3=ws.lon_grid)
ws.covmat_sxAddBlock(block=xml.load("a_priori/covariance_T.xml"))
# Setup observation error covariance matrix.

Se_covmat = oem.iasi_nedt()
ws.covmat_seAddBlock(block=1.0 ** 2 * np.diag(np.ones(ws.y.value.size)))
ws.retrievalDefClose()


# define inversion iteration as function object within python
@arts_agenda
def inversion_agenda(ws):
    ws.Ignore(ws.inversion_iteration_counter)
    ws.x2artsAtmAndSurf()
    ws.atmfields_checkedCalc()  # to be safe, rerun checks dealing with atmosph.
    ws.atmgeom_checkedCalc()
    ws.yCalc()  # calculate yf and jacobian matching x
    ws.Copy(ws.yf, ws.y)
    ws.jacobianAdjustAndTransform()


ws.Copy(ws.inversion_iterate_agenda, inversion_agenda)

ws.xaStandard() # a_priori vector is current state of retrieval fields in ws.
ws.x  = np.array([]) # create empty vector for retrieved state vector?
ws.yf = np.array([]) # create empty vector for simulated TB?
ws.jacobian = np.array([[]])

###########################################################################
# Conduct OEM retrieval
###########################################################################
ws.oem_errors = []
ws.OEM(method="gn",
       max_iter=1000,
       display_progress=1,
       max_start_cost=1e5)
print(ws.oem_errors.value)
ws.x2artsAtmAndSurf() # convert from ARTS coords back to user-defined grid

###########################################################################
# Save retrieval results
###########################################################################
retrieved_vmr = np.copy(ws.vmr_field.value)
retrieved_T = np.copy(ws.t_field.value)
xml.save(retrieved_vmr, "retrieved/vmr_retrieved.xml")
xml.save(retrieved_T, "retrieved/T_retrieved.xml")

###########################################################################
# Plot retrieval results
###########################################################################
# HUMIDITY
plot_retrieval_profiles(a_priori, retrieved, garand["abs_species-H2O"].values, alt, "H2O VMR")
plt.savefig("plots/moist_layer_retrieval_profile.pdf")

plt.figure()
plt.imshow(xml.load("a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml"))
plt.colorbar()
plt.savefig("plots/covmat_H2O.pdf")

# TEMPERATURE
retrieved_T = np.copy(ws.t_field)[:,0,0]
plot_retrieval_profiles(a_priori_T, retrieved_T, garand["T"].values, alt, "Temperature")
plt.savefig("plots/temp_retrieval_profile.pdf")

plt.figure()
plt.imshow(xml.load("a_priori/covariance_T.xml"))
plt.colorbar()
plt.savefig("plots/covmat_T.pdf")