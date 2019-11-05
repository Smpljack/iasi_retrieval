import numpy as np
import os

from typhon.arts.workspace import arts_agenda
from typhon.physics import wavenumber2frequency, frequency2wavenumber


def load_abs_lookup(ws, atm_batch_path=None, f_ranges=None, abs_lookup_table_path=None):
    """
    Loads existing absorption lookup table or creates one in case it does
    not exist yet for given batch of atmospheres and frequency range.
    :param ws:
    :param atm_batch_path:
    :param f_ranges:
    :param abs_lookup_table_path:
    :return:
    """
    if abs_lookup_table_path is not None:
        ws.ReadXML(ws.abs_lookup, abs_lookup_table_path)
        ws = abs_setup(ws)
        ws.abs_lookupAdapt()
    else:
        ws.ReadXML(ws.batch_atm_fields_compact,
                   atm_batch_path)
        ws = abs_setup(ws)
        ws.abs_lineshapeDefine(shape='Voigt_Kuntz6',
                               forefactor='VVH',
                               cutoff=750e9)
        ws.abs_linesReadFromSplitArtscat(
            basename='/scratch/uni/u237/data/catalogue/hitran/hitran_split_artscat5/',
            fmin=f_ranges[0][0],
            fmax=f_ranges[-1][-1])
        ws.abs_lines_per_speciesCreateFromLines()
        ws.abs_lookupSetupBatch()
        ws.abs_xsec_agenda_checkedCalc()
        ws.abs_lookupCalc()
        f_str = "_".join([f"{int(frequency2wavenumber(freq[0]/100))}_"
                          f"{int(frequency2wavenumber(freq[1])/100)}" for freq in f_ranges])
        ws.WriteXML("binary", ws.abs_lookup,
                    f"a_priori/abs_lookup_{os.path.splitext(os.path.basename(atm_batch_path))[0]}"
                    f"_{f_str}_cm-1.xml")
    return ws


def setup_sensor(ws, f_ranges, f_backend_width):
    """
    Setup the sensor properties, mainly including the frequency grid.
    :param ws:
    :param f_ranges:
    :param f_backend_width:
    :return:
    """
    ws.sensor_pos = np.array([[850e3]])  # 850km
    ws.sensor_time = np.array([0.0])
    ws.sensor_los = np.array([[180.0]])  # nadir viewing
    ws.f_grid = np.concatenate([np.arange(freq[0] * 0.99,
                                          freq[1] * 1.01,
                                          f_backend_width) for freq in f_ranges])
    ws.f_backend = np.concatenate([np.arange(freq[0],
                                             freq[1],
                                             f_backend_width) for freq in f_ranges])
    ws.f_backend_width = np.array([f_backend_width])
    ws.backend_channel_responseGaussian(ws.f_backend_width)

    # Sensor settings
    ws.FlagOn(ws.sensor_norm)
    ws.AntennaOff()
    ws.sensor_responseInit()
    ws.sensor_responseBackend()
    return ws


def load_generic_settings(ws):
    """
    Load generic agendas and set generic settings.
    :param ws:
    :return:
    """
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
    ws.atmosphere_dim = 1 #1D VAR
    ws.jacobian_quantities = []
    ws.iy_unit = "PlanckBT"
    ws.cloudboxOff()
    ws.surface_scalar_reflectivity = np.array([0.05])  # nominal albedo for surface
    return ws


def iasi_observation(ws, atm_batch_path, n_atmospheres, f_ranges, iasi_obs_path=None):
    """
    iasi observation simulation
    :param ws:
    :param atm_batch_path:
    :param n_atmospheres:
    :param f_ranges:
    :param iasi_obs_path:
    :return:
    """
    if iasi_obs_path:
        ws.ReadXML(ws.ybatch, iasi_obs_path)
    else:
        ws.ReadXML(ws.batch_atm_fields_compact, atm_batch_path)
        ws = abs_setup(ws)
        ws.propmat_clearsky_agenda_checkedCalc()
        @arts_agenda
        def ybatch_calc_agenda(ws):
            ws.Extract(ws.atm_fields_compact,
                       ws.batch_atm_fields_compact,
                       ws.ybatch_index)
            ws.AtmFieldsFromCompact()
            # ws.jacobianOff()
            ws.jacobianInit()
            ws.jacobianAddAbsSpecies(species="H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                                     g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
            ws.jacobianAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
            ws.jacobianClose()
            ws.Extract(ws.z_surface, ws.z_field, 0)
            ws.Extract(ws.t_surface, ws.t_field, 0)
            ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
            ws.atmgeom_checkedCalc()
            ws.cloudbox_checkedCalc()
            ws.sensor_checkedCalc()
            ws.yCalc()

        ws.Copy(ws.ybatch_calc_agenda, ybatch_calc_agenda)
        ws.IndexSet(ws.ybatch_n, n_atmospheres)  # Amount of atmospheres
        ws.ybatchCalc()
        # Add measurement noise to synthetic observation
        for i in range(n_atmospheres):
            ws.ybatch.value[i][ws.f_backend.value < wavenumber2frequency(175000)] += \
                np.random.normal(loc=0.0, scale=0.1)
            ws.ybatch.value[i][ws.f_backend.value >= wavenumber2frequency(175000)] += \
                np.random.normal(loc=0.0, scale=0.2)
        f_str = "_".join([f"{int(frequency2wavenumber(freq[0]/100))}_"
                          f"{int(frequency2wavenumber(freq[1])/100)}" for freq in f_ranges])
        if not os.path.isdir(f"iasi_obs_{os.path.splitext(os.path.basename(atm_batch_path))[0]}"):
            os.mkdir(f"iasi_obs_{os.path.splitext(os.path.basename(atm_batch_path))[0]}/")
        ws.WriteXML("ascii", ws.ybatch.value,
                    f"iasi_obs_{os.path.splitext(os.path.basename(atm_batch_path))[0]}/"
                    f"{os.path.splitext(os.path.basename(atm_batch_path))[0]}_{f_str}_cm-1.xml")
        ws.WriteXML("ascii", ws.ybatch_jacobians.value,
                    f"iasi_obs_{os.path.splitext(os.path.basename(atm_batch_path))[0]}/"
                    f"{os.path.splitext(os.path.basename(atm_batch_path))[0]}_{f_str}_cm-1_jacobian.xml")
    return ws


def abs_setup(ws):
    """
    Load the batch of atmospheric fields and setup absorption species
    for the RT calculation.
    :param ws:
    :return:
    """
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
    return ws


def setup_oem_retrieval(ws, a_priori_atm_batch_path, a_priori_atm_index, cov_h2o_vmr, cov_t, retrieval_quantities):
    """

    :param ws:
    :param t_a_priori:
    :param vmr_a_priori:
    :param cov_h2o_vmr:
    :param cov_t:
    :param retrieval_quantities:
    :return:
    """
    ws.ReadXML(ws.batch_atm_fields_compact, a_priori_atm_batch_path)
    ws = abs_setup(ws)
    ws.Extract(ws.atm_fields_compact,
               ws.batch_atm_fields_compact,
               a_priori_atm_index)
    ws.AtmFieldsFromCompact()
    ws.lat_grid = []
    ws.lon_grid = []
    ws.p_grid = ws.atm_fields_compact.value.grids[1]
    # ws.AtmFieldsCalc()
    ws.AbsInputFromAtmFields()
    ws.z_surface = np.asarray(ws.z_field)[0]
    ws.t_surface = np.asarray(ws.t_field)[0]

    ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    ws.retrievalDefInit()
    if "Temperature" in retrieval_quantities:
        ws.retrievalAddTemperature(
            g1=ws.p_grid,
            g2=ws.lat_grid,
            g3=ws.lon_grid)
        ws.covmat_sxAddBlock(block=cov_t)
    if "H2O" in retrieval_quantities:
        ws.retrievalAddAbsSpecies(species="H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                                  unit="vmr",
                                  g1=ws.p_grid,
                                  g2=ws.lat_grid,
                                  g3=ws.lon_grid)
        ws.jacobianSetFuncTransformation(transformation_func="log10")
        ws.covmat_sxAddBlock(block=cov_h2o_vmr)
    cov_y = np.diag(np.ones(ws.f_backend.value.size))
    low_noise_ind = ws.f_backend.value < wavenumber2frequency(175000)
    high_noise_ind = ws.f_backend.value >= wavenumber2frequency(175000)
    cov_y[low_noise_ind, low_noise_ind] *= 0.1 ** 2
    cov_y[high_noise_ind, high_noise_ind] *= 0.2 ** 2
    ws.covmat_seAddBlock(block=cov_y)
    ws.retrievalDefClose()
    return ws

def oem_retrieval(ws, ybatch_indices, inversion_method="lm", max_iter=20, gamma_start=1000,
                  gamma_dec_factor=2.0, gamma_inc_factor=2.0, gamma_upper_limit=1e20,
                  gamma_lower_limit=1.0, gamma_upper_convergence_limit=99.0):
    """

    :param ws:
    :param ybatch_indices:
    :param inversion_method:
    :param max_iter:
    :param gamma_start:
    :param gamma_dec_factor:
    :param gamma_inc_factor:
    :param gamma_upper_limit:
    :param gamma_lower_limit:
    :param gamma_upper_convergence_limit:
    :return:
    """
    # define inversion iteration as function object within python
    @arts_agenda
    def inversion_agenda(ws):
        ws.Ignore(ws.inversion_iteration_counter)
        ws.x2artsAtmAndSurf()
        # to be safe, rerun checks dealing with atmosph.
        # Allow negative vmr? Allow temperatures < 150 K and > 300 K?
        ws.atmfields_checkedCalc(
             #negative_vmr_ok=1,
             bad_partition_functions_ok=1,
        )
        ws.atmgeom_checkedCalc()
        ws.yCalc()  # calculate yf and jacobian matching x
        ws.Copy(ws.yf, ws.y)
        ws.jacobianAdjustAndTransform()
    ws.Copy(ws.inversion_iterate_agenda, inversion_agenda)

    retrieved_h2o_vmr = []
    retrieved_t = []
    vmr_a_priori = np.copy(ws.vmr_field.value)
    t_a_priori = np.copy(ws.t_field.value)
    for obs in np.array(ws.ybatch.value)[ybatch_indices]:
        ws.y = np.copy(obs)
        ws.vmr_field.value = np.copy(vmr_a_priori)
        ws.t_field.value = np.copy(t_a_priori)
        ws.xaStandard()  # a_priori vector is current state of retrieval fields in ws, but transformed
        ws.x = np.array([])  # create empty vector for retrieved state vector?
        ws.yf = np.array([])  # create empty vector for simulated TB?
        ws.jacobian = np.array([[]])
        ws.oem_errors = []
        ws.OEM(method=inversion_method,
               max_iter=max_iter,
               display_progress=1,
               max_start_cost=1e5,
               # start value for gamma, decrease/increase factors,
               # upper limit for gamma, lower gamma limit which causes gamma=0
               # Upper gamma limit, above which no convergence is accepted
               lm_ga_settings=np.array([gamma_start, gamma_dec_factor, gamma_inc_factor, gamma_upper_limit,
                                        gamma_lower_limit, gamma_upper_convergence_limit]))
        print(ws.oem_errors.value)
        ws.x2artsAtmAndSurf()  # convert from ARTS coords back to user-defined grid
        retrieved_h2o_vmr.append(ws.vmr_field.value[0, :, 0, 0])
        retrieved_t.append(ws.t_field.value[:, 0, 0])
    return retrieved_h2o_vmr, retrieved_t
