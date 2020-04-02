import os
import numpy as np
from scipy.linalg import inv
from scipy.signal import convolve

from typhon.arts.workspace import arts_agenda
from typhon.physics import wavenumber2frequency, frequency2wavenumber, \
    constants, planck, radiance2planckTb, moist_lapse_rate, relative_humidity2vmr, e_eq_mixed_mk


def setup_retrieval_paths(project_path, project_name):
    """
    Setup the directory infrastructure for the retrieval project I am working on.
    """
    if not os.path.isdir(project_path + project_name):
        os.mkdir(project_path + project_name)
    for dir_name in ["plots", "sensor", "a_priori", "observations", "retrieval_output"]:
        if not os.path.isdir(project_path + project_name + f"/{dir_name}"):
            os.mkdir(project_path + project_name + f"/{dir_name}")
    os.chdir(project_path + project_name)


def load_abs_lookup(ws, hitran_split_artscat5_path, atm_batch_path=None,
                    abs_lookup_base_path=None, abs_lookup_path=None, f_ranges=None,
                    line_shape='Voigt_Kuntz6'):
    """
    Loads existing absorption lookup table or creates one in case it does
    not exist yet for given batch of atmospheres and frequency range.
    :param ws:
    :param atm_batch_path:
    :param f_ranges:
    :param abs_lookup_table_path:
    :return:
    """
    f_str = "_".join([f"{int(frequency2wavenumber(freq[0] / 100))}_"
                      f"{int(frequency2wavenumber(freq[1]) / 100)}" for freq in f_ranges])
    if not abs_lookup_path:
        abs_lookup_path = f"{abs_lookup_base_path}" \
                          f"abs_lookup_{os.path.basename(atm_batch_path).split(os.extsep, 1)[0]}_{f_str}_cm-1.xml"
    if os.path.isfile(abs_lookup_path):
        ws.ReadXML(ws.abs_lookup, abs_lookup_path)
        ws = abs_setup(ws)
        ws.abs_lookupAdapt()
    else:
        ws.ReadXML(ws.batch_atm_fields_compact,
                   atm_batch_path)
        ws = abs_setup(ws)
        ws.ReadSplitARTSCAT(
            basename=hitran_split_artscat5_path,
            fmin=(np.min(ws.f_grid.value) - ws.f_backend_width * 10)[0],
            fmax=(np.max(ws.f_grid.value) + ws.f_backend_width * 10)[0],
            globalquantumnumbers="",
            localquantumnumbers="",
            ignore_missing=0,
        )
        ws.abs_lines_per_speciesCreateFromLines()
        ws.abs_lookupSetupBatch()
        ws.abs_xsec_agenda_checkedCalc()
        ws.lbl_checkedCalc()
        ws.abs_lookupCalc()
        ws.WriteXML("binary", ws.abs_lookup, abs_lookup_path)
    return ws


def setup_sensor(ws, f_backend_width, f_ranges=None, add_frequencies=None):
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
    if f_ranges is not None:
        ws.f_backend = np.concatenate([np.arange(freq[0],
                                                 freq[1],
                                                 f_backend_width) for freq in f_ranges])
    else:
        ws.f_backend = np.array([])

    if add_frequencies is not None:
        ws.f_backend = np.unique(np.sort(np.append(ws.f_backend.value, add_frequencies)))
    ws.f_grid = ws.f_backend
    ws.f_grid = np.append(ws.f_grid.value,
                          np.arange(np.min(ws.f_grid.value) - 10 * f_backend_width,
                                    np.min(ws.f_grid.value), f_backend_width))
    ws.f_grid = np.append(ws.f_grid.value,
                          np.arange(np.max(ws.f_grid.value) + f_backend_width,
                                    np.max(ws.f_grid.value) + 11 * f_backend_width, f_backend_width))
    ws.f_grid = np.sort(ws.f_grid.value)
    ws.f_backend_width = np.array([f_backend_width])
    ws.backend_channel_responseGaussian(ws.f_backend_width)

    # Sensor settings
    ws.FlagOn(ws.sensor_norm)
    ws.AntennaOff()
    ws.sensor_responseInit()
    ws.sensor_responseBackend()
    ws.WriteXML("binary", ws.f_backend, "sensor/f_backend.xml")
    return ws


def load_generic_settings(ws, ):
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

    @arts_agenda
    def iy_surface_agenda_PY(ws):
        ws.SurfaceBlackbody()
        ws.iySurfaceRtpropCalc()

    ws.Copy(ws.iy_surface_agenda, iy_surface_agenda_PY)
    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    # ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)

    ws.Copy(ws.surface_rtprop_agenda, ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface)

    # clearsky agenda
    ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__LookUpTable)
    # sensor-only path
    ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)
    # no refraction
    ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)

    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    ws.stokes_dim = 1
    ws.atmosphere_dim = 1  # 1D VAR
    ws.jacobian_quantities = []
    ws.iy_unit = "PlanckBT"
    ws.cloudboxOff()
    ws.surface_scalar_reflectivity = np.array([0.])  # nominal albedo for surface
    return ws


def iasi_observation(ws, f_ranges, atm_batch_path, ybatch_start=None, ybatch_n=None,
                     iasi_obs_path=None, iasi_obs_data=None, add_measurement_noise=True):
    """
    iasi observation simulation
    :param ws:
    :param atm_batch_path:
    :param n_atmospheres:
    :param f_ranges:
    :param iasi_obs_path:
    :return:
    """
    f_str = "_".join([f"{int(frequency2wavenumber(freq[0] / 100))}_"
                      f"{int(frequency2wavenumber(freq[1]) / 100)}" for freq in f_ranges])
    batch_atm_fields_name = os.path.basename(atm_batch_path).split(os.extsep, 1)[0]

    if iasi_obs_path:
        ws.ReadXML(ws.ybatch, iasi_obs_path)
    elif iasi_obs_data is not None:
        ws.Copy(ws.ybatch, iasi_obs_data)
    else:
        ws.ReadXML(ws.batch_atm_fields_compact, atm_batch_path)
        ws = abs_setup(ws)
        ws.propmat_clearsky_agenda_checkedCalc()
        atm_ind = 0
        # ws.execute_controlfile("/scratch/uni/u237/users/mprange/phd/iasi_retrieval/ybatch_agenda.arts")

        ws.VectorCreate("t_surface_vector")
        ws.NumericCreate("t_surface_numeric")

        @arts_agenda
        def ybatch_calc_agenda(ws):
            ws.Extract(ws.atm_fields_compact,
                       ws.batch_atm_fields_compact,
                       ws.ybatch_index)
            ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
            ws.jacobianInit()
            ws.jacobianAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
            ws.jacobianAddAbsSpecies(species="H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                                     unit="rel",
                                     g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
            ws.jacobianClose()
            ws.Extract(ws.z_surface, ws.z_field, 0)
            ws.Extract(ws.t_surface, ws.t_field, 0)
            ws.Copy(ws.surface_props_names, ["Skin temperature"])
            ws.VectorExtractFromMatrix(ws.t_surface_vector, ws.t_surface, 0, "row")
            ws.Extract(ws.t_surface_numeric, ws.t_surface_vector, 0)
            ws.Tensor3SetConstant(ws.surface_props_data, 1, 1, 1, ws.t_surface_numeric)
            ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
            ws.atmgeom_checkedCalc()
            ws.cloudbox_checkedCalc()
            ws.sensor_checkedCalc()
            ws.yCalc()

        ws.Copy(ws.ybatch_calc_agenda, ybatch_calc_agenda)
        if ybatch_start is not None:
            ws.IndexSet(ws.ybatch_start, ybatch_start)
        ws.IndexSet(ws.ybatch_n, ybatch_n)  # Amount of atmospheres
        ws.ybatchCalc()
        # Add measurement noise to synthetic observation
        if add_measurement_noise:
            for i in range(ybatch_n):
                ws.ybatch.value[i][ws.f_backend.value < wavenumber2frequency(175000)] += \
                    np.array([np.random.normal(loc=0.0, scale=0.1)
                              for i in range(np.sum(ws.f_backend.value < wavenumber2frequency(175000)))])
                ws.ybatch.value[i][ws.f_backend.value >= wavenumber2frequency(175000)] += \
                    np.array([np.random.normal(loc=0.0, scale=0.2)
                              for i in range(np.sum(ws.f_backend.value >= wavenumber2frequency(175000)))])
        ws.WriteXML("ascii", ws.ybatch_jacobians,
                    f"observations/{batch_atm_fields_name}_{f_str}_cm-1_jacobian"
                    f"{ws.ybatch_start.value}-{ws.ybatch_start.value + ws.ybatch_n.value}.xml")
        ws.WriteXML("ascii", ws.batch_atm_fields_compact,
                    f"observations/{batch_atm_fields_name}_atm_fields.xml")
    ws.WriteXML("ascii", ws.ybatch,
                f"observations/{batch_atm_fields_name}_{f_str}_cm-1"
                f"{ws.ybatch_start.value}-{ws.ybatch_start.value + ws.ybatch_n.value}.xml")
    return ws


def abs_setup(ws):
    """
    Load the batch of atmospheric fields and setup absorption species
    for the RT calculation.
    :param ws:
    :return:
    """
    # define absorbing species and load lines for given frequency range from HITRAN
    ws.abs_speciesSet(species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                               "O2, O2-CIAfunCKDMT100",
                               "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                               "O3",
                               "CO2, CO2-CKDMT252"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-O2",
                                           value=0.2095,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-N2",
                                           value=0.7808,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-CO2",
                                           value=0.0004,
                                           condensibles=["abs_species-H2O"])
    return ws


def setup_apriori_state(ws, a_priori_atm_batch_path, batch_ind,
                        moist_adiabat=False, t_surface_std=None, stratospheric_temperature=None, RH=None):
    ws.ReadXML(ws.batch_atm_fields_compact, a_priori_atm_batch_path)
    ws = abs_setup(ws)
    ws.Extract(ws.atm_fields_compact,
               ws.batch_atm_fields_compact,
               int(batch_ind))
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
    ws.lat_grid = []
    ws.lon_grid = []
    # ws.AtmFieldsCalc()
    if moist_adiabat:
        ws = t_profile_to_moist_adiabat(ws, t_surface_std, stratospheric_temperature)
    if RH is not None:
        ws = h2o_vmr_from_rh(ws, RH)
    ws.Extract(ws.t_surface, ws.t_field, 0)
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.AbsInputFromAtmFields()
    ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()
    snames = ["Skin temperature"]
    sdata = np.array([ws.t_surface.value])
    ws.Copy(ws.surface_props_names, snames)
    ws.Copy(ws.surface_props_data, sdata)
    return ws


def setup_retrieval_quantities(ws, retrieval_quantities, cov_cross=None, cov_h2o_vmr=None,
                               cov_t=None, cov_t_surface=None):
    if "t_surface_python" in retrieval_quantities:
        ws = get_transmittance(ws)
        ws.MatrixCreate("cov_t_surface")
        ws.MatrixCreate("cov_y_t_surface")
        ws.cov_t_surface = cov_t_surface
        ws.cov_y_t_surface = 0.1 ** 2 * np.diag(np.ones(ws.f_backend.value.shape))
        ws.WriteXML("ascii", ws.cov_y_t_surface.value, "sensor/covariance_y.xml")
        ws.WriteXML("ascii", cov_t_surface, "a_priori/covmat_sx.xml")

    else:
        ws.retrievalDefInit()
        if "t_surface" in retrieval_quantities:
            ws.retrievalAddSurfaceQuantity(
                g1=ws.lat_grid, g2=ws.lon_grid, quantity="Skin temperature")
            ws.covmat_sxAddBlock(block=cov_t_surface)

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

        if cov_cross is not None:
            for S_dict in cov_cross:
                ws.covmat_sxAddBlock(block=S_dict["S"], i=S_dict["i"], j=S_dict["j"])
        cov_y = np.diag(np.ones(ws.f_backend.value.size))
        low_noise_ind = ws.f_backend.value < wavenumber2frequency(175000)
        high_noise_ind = ws.f_backend.value >= wavenumber2frequency(175000)
        cov_y[low_noise_ind, low_noise_ind] *= 0.1 ** 2
        cov_y[high_noise_ind, high_noise_ind] *= 0.2 ** 2
        ws.covmat_seAddBlock(block=cov_y)
        ws.retrievalDefClose()
        ws.WriteXML("ascii", cov_y, "sensor/covmat_sy.xml")
        ws.WriteXML("ascii", ws.covmat_sx, "a_priori/covmat_sx.xml")
    return ws


def save_atm_state(ws, dirname):
    ws.WriteXML("ascii", ws.vmr_field, f"{dirname}/vmr.xml")
    ws.WriteXML("ascii", ws.t_field, f"{dirname}/temperature.xml")
    ws.WriteXML("ascii", ws.t_surface, f"{dirname}/t_surface.xml")
    ws.WriteXML("ascii", ws.p_grid, f"{dirname}/p.xml")


def save_z_p_grids(ws, dirname):
    ws.WriteXML("ascii", ws.p_grid, f"{dirname}/p.xml")
    ws.WriteXML("ascii", ws.z_field.value[:, 0, 0], f"{dirname}/z.xml")


def t_profile_to_moist_adiabat(ws, t_surface_std, stratospheric_temperature):
    T_moist = np.copy(ws.t_surface.value[0, :] + 1.)#np.random.normal(0., t_surface_std))
    for p, z_diff in zip(ws.p_grid.value[1:], np.diff(ws.z_field.value[:, 0, 0])):
        if p < 10000.:
            break
        T_moist = np.append(T_moist,
                            T_moist[-1] -
                            moist_lapse_rate(p, T_moist[-1]) * z_diff)
    full_new_T = np.append(T_moist, stratospheric_temperature + (T_moist[-1] - stratospheric_temperature[0]))
    convolve_ind = ws.p_grid.value < 15000.
    convolve_ind &= ws.p_grid.value > 5000.
    K = np.ones(10) / 10.
    avg_T = np.append(
        np.append(
            full_new_T[ws.p_grid.value > 15000.],
            convolve_inner(full_new_T[convolve_ind], full_new_T, K)),
        full_new_T[ws.p_grid.value < 5000.])
    ws.t_field.value[:, 0, 0] = np.copy(avg_T)
    return ws


def convolve_inner(x, x_full, K):
    x_full_ind = np.where(np.isin(x_full, x))[0]
    x_convolve = np.append(
        np.append(
            x_full[x_full_ind[0] - int(len(K) / 2):x_full_ind[0]],
            x
        ),
        x_full[x_full_ind[-1] + 1:x_full_ind[-1] + int(len(K) / 2)]
    )
    return convolve(x_convolve, K, mode='valid')


def h2o_vmr_from_rh(ws, RH):
    new_vmr = relative_humidity2vmr(RH, ws.p_grid.value, ws.t_field.value[:, 0, 0],
                                    e_eq=e_eq_mixed_mk)
    new_vmr[ws.p_grid.value < 10000.] = new_vmr[np.argmin(np.abs(ws.p_grid.value - 10000.))]
    ws.vmr_field.value[0, :, 0, 0] = np.copy(new_vmr)
    return ws


def retrieve_ybatch_for_a_priori_batch(ws, retrieval_batch_indices, a_priori_batch_indices, a_priori_atm_batch_path,
                                       Sa_T, Sa_h2o, t_surface_std=None, stratospheric_temperature=None,
                                       moist_adiabat=False, RH=False, inversion_method="lm", max_iter=15,
                                       gamma_start=10,
                                       gamma_dec_factor=2.0, gamma_inc_factor=2.0, gamma_upper_limit=1e20,
                                       gamma_lower_limit=1.0,
                                       gamma_upper_convergence_limit=99.0):
    retrieved_h2o_vmr = []
    retrieved_t = []
    retrieved_ts = []
    retrieved_y = []
    retrieved_jacobian = []
    apriori_h2o_vmr = []
    apriori_t = []
    apriori_ts = []
    oem_diagnostics = []
    for retr_batch_ind, apriori_batch_ind in zip(retrieval_batch_indices, a_priori_batch_indices):
        ws.y = ws.ybatch.value[retr_batch_ind - ws.ybatch_start.value]
        ws = setup_apriori_state(ws, a_priori_atm_batch_path, apriori_batch_ind, moist_adiabat,
                                 t_surface_std, stratospheric_temperature, RH)
        ws = setup_retrieval_quantities(ws,
                                        retrieval_quantities=["Temperature", "H2O", "t_surface"],
                                        cov_t_surface=np.array([[100.]]),
                                        cov_t=Sa_T,
                                        cov_h2o_vmr=Sa_h2o,
                                        # cov_cross=Sa_cross,
                                        )
        apriori_h2o_vmr.append(np.copy(ws.vmr_field.value[0, :, 0, 0]))
        apriori_t.append(np.copy(ws.t_field.value[:, 0, 0]))
        apriori_ts.append(np.copy(ws.surface_props_data.value[0]))
        print(f"Retrieving batch profile {retr_batch_ind + 1}.")
        print(f"Profile {retr_batch_ind - retrieval_batch_indices[0] + 1} "
              f"out of {len(retrieval_batch_indices)} in this job.")
        try:
            ws = oem_retrieval(ws, inversion_method, max_iter, gamma_start,
                               gamma_dec_factor, gamma_inc_factor, gamma_upper_limit,
                               gamma_lower_limit, gamma_upper_convergence_limit)
        except Exception:
            print(ws.oem_errors.value)
            retrieved_h2o_vmr.append(np.nan * np.ones(ws.vmr_field.value[0, :, 0, 0].shape))
            retrieved_t.append(np.nan * np.ones(ws.t_field.value[:, 0, 0].shape))
            retrieved_ts.append(np.nan * np.ones(ws.surface_props_data.value[0].shape))
            retrieved_y.append(np.nan * np.ones(ws.y.value.shape))
            retrieved_jacobian.append(np.nan * np.ones((len(ws.y.value), len(ws.p_grid.value) * 2 + 1)))
            oem_diagnostics.append(np.nan * np.ones(5))
            continue
        retrieved_h2o_vmr.append(np.copy(ws.vmr_field.value[0, :, 0, 0]))
        retrieved_t.append(np.copy(ws.t_field.value[:, 0, 0]))
        retrieved_ts.append(ws.surface_props_data.value[0])
        retrieved_y.append(np.copy(ws.y.value))
        retrieved_jacobian.append(np.copy(ws.jacobian.value))
        oem_diagnostics.append(ws.oem_diagnostics.value)
    ws.WriteXML("ascii", retrieved_h2o_vmr, f"retrieval_output/h2o_vmr_batch_profiles_"
                                            f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", retrieved_t, f"retrieval_output/temperature_batch_profiles_"
                                      f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", retrieved_ts, f"retrieval_output/surface_temperature_batch_profiles_"
                                       f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", retrieved_y, f"retrieval_output/ybatch_profiles_"
                                      f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", retrieved_jacobian, f"retrieval_output/jacobian_batch_profiles_"
                                             f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", oem_diagnostics, f"retrieval_output/oem_diagnostics_batch_profiles_"
                                          f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", apriori_h2o_vmr, f"a_priori/h2o_vmr_batch_profiles_"
                                          f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", apriori_t, f"a_priori/temperature_batch_profiles_"
                                    f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", apriori_ts, f"a_priori/surface_temperature_batch_profiles_"
                                     f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")


def oem_retrieval(ws, inversion_method="lm", max_iter=20, gamma_start=1000,
                  gamma_dec_factor=2.0, gamma_inc_factor=2.0, gamma_upper_limit=1e20,
                  gamma_lower_limit=1.0, gamma_upper_convergence_limit=99.0):
    # define inversion iteration as function object within python
    @arts_agenda
    def inversion_agenda(ws):
        ws.Ignore(ws.inversion_iteration_counter)
        ws.x2artsAtmAndSurf()
        # ws.Extract(ws.t_surface, ws.surface_props_data, 0)
        # print(ws.t_surface.value)
        # to be safe, rerun checks dealing with atmosph.
        # Allow negative vmr? Allow temperatures < 150 K and > 300 K?
        ws.atmfields_checkedCalc(
            # negative_vmr_ok=1,
            bad_partition_functions_ok=1,
        )
        ws.atmgeom_checkedCalc()
        ws.yCalc()  # calculate yf and jacobian matching x
        ws.Copy(ws.yf, ws.y)
        ws.jacobianAdjustAndTransform()

    ws.Copy(ws.inversion_iterate_agenda, inversion_agenda)

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
    ws.x2artsAtmAndSurf()  # convert from ARTS coords back to user-defined grid
    return ws


def radiance2planck_bt_wavenumber(r, wavenumber):
    c = constants.speed_of_light
    k = constants.boltzmann
    h = constants.planck
    return h / k * c * wavenumber / np.log(np.divide(2 * h * c ** 2 * wavenumber ** 3, r) + 1)


def get_transmittance(ws):
    ws.MatrixCreate("t_surface_a")
    ws.t_surface_a = np.copy(ws.t_surface.value)
    ws.jacobianOff()
    ws.yCalc()
    ws.MatrixCreate("planck_a")
    ws.MatrixPlanck(ws.planck_a, ws.stokes_dim, ws.f_backend.value, ws.t_surface_a.value[0, 0])
    ws.VectorCreate("transmittance")
    if ws.iy_unit.value == "PlanckBT":
        ws.transmittance = planck(ws.f_backend.value, ws.y.value) / ws.planck_a.value[:, 0] / \
                           (1 - ws.surface_scalar_reflectivity.value)
    else:
        ws.transmittance = ws.y.value / ws.planck_a.value[:, 0] / (1 - ws.surface_scalar_reflectivity.value)
    return ws


def oem_t_surface(ws, ybatch_indices, max_iter):
    Sa = ws.cov_t_surface.value
    Sy = ws.cov_y_t_surface.value
    ws.MatrixCreate("jacobian_Ts")
    retrieved_Ts = []
    for obs in np.array(ws.ybatch.value)[ybatch_indices]:
        ws.y = np.copy(obs)
        ws.t_surface.value = np.copy(ws.t_surface_a.value)
        not_converged = True
        print(f"t_surface_apriori={ws.t_surface.value}")
        iter_n = 0
        while not_converged:
            iter_n += 1
            ws.jacobian_Ts = np.array([planck_derivative_T(ws.f_backend.value, ws.t_surface.value[0, 0]) *
                                       ws.transmittance.value * (1 - ws.surface_scalar_reflectivity.value)]).T
            ws.t_surface = gauss_newton_t_surface(ws, Sa, Sy)
            print(f"iter{iter_n}, t_surface={ws.t_surface.value}")
            print(f"iter{iter_n}, cost={eval_cost_function(ws, Sa, Sy)}")
            if iter_n == max_iter:
                break
        retrieved_Ts.append(np.array([np.copy(ws.t_surface.value[0, 0])]))
    ws.WriteXML("ascii", ws.jacobian_Ts, "retrieval_output/jacobian_t_surface.xml")
    ws.WriteXML("ascii", retrieved_Ts, "retrieval_output/retrieved_t_surface.xml")


def planck_derivative_T(f, T):
    c = constants.speed_of_light
    k = constants.boltzmann
    h = constants.planck
    return 2 * h ** 2 * f ** 4 / (c ** 2 * k * T ** 2) * np.exp(h * f / (k * T)) / (np.exp(h * f / (k * T)) - 1) ** 2


def gauss_newton_t_surface(ws, Sa, Sy):
    yi = radiance2planckTb(ws.f_backend.value, planck(ws.f_backend.value, ws.t_surface.value[0, 0]) *
                           ws.transmittance.value * (1 - ws.surface_scalar_reflectivity.value))
    return \
        ws.t_surface_a.value + inv(inv(Sa) + ws.jacobian_Ts.value.T @ inv(Sy) @ ws.jacobian_Ts.value) @ \
        ws.jacobian_Ts.value.T @ inv(Sy) @ \
        (ws.y.value - yi +
         ws.jacobian_Ts.value @ (ws.t_surface.value[:, 0] - ws.t_surface_a.value[:, 0]))


def eval_cost_function(ws, Sa, Sy):
    yi = radiance2planckTb(ws.f_backend.value, planck(ws.f_backend.value, ws.t_surface.value[0, 0]) *
                           ws.transmittance.value * (1 - ws.surface_scalar_reflectivity.value))
    return (ws.y.value - yi).T @ inv(Sy) @ (ws.y.value - yi) + \
           (ws.t_surface.value[:, 0] - ws.t_surface_a.value[:, 0]) ** 2 @ inv(Sa)


# def plot_retrieval_results():


def corr_length_cov(z, trpp=12.5e3):
    """Return correlation lengths for given altitudes.
    Parameters:
        z (np.array): Height levels [m]
        trpp (float): Tropopause height [m]

    Returns:
        np.array: Correlation length for each heigth level.

    """
    f = np.poly1d(np.polyfit(x=(0, trpp), y=(2.5e3, 10e3), deg=1))
    cl = f(z)
    cl[z > trpp] = 10e3

    return cl


def covmat_cross(covmat1, covmat2, z_grid, corr_height=1500.):
    """
    Return cross-covariance block for given H2O and Temperature
    covariance matrices. The cross-covariances drop exponentially
    with height (1/e at corr_height) and correlation length approach
    is used for determining non-diagonal entries.
    """
    S = np.zeros((covmat1.shape[0], covmat2.shape[0]))
    if np.any(np.array(S.shape) == 1):
        S = np.array([np.sqrt(covmat1[0, 0] * covmat2[0, 0]) * np.exp(-1 / corr_height * z_grid[i])
                      for i in range(len(z_grid))]).reshape(1, len(z_grid))
        return S

    S[np.diag_indices_from(S)] = [np.sqrt(covmat1[0, 0] * covmat2[0, 0]) * np.exp(-1 / corr_height * z_grid[i])
                                  for i in range(len(z_grid))]
    cl = corr_length_cov(z_grid)
    for i in range(S.shape[1]):
        for j in range(S.shape[0]):
            cl_mean = (cl[i] + cl[j]) / 2
            s = (S[j, j] + S[i, i]) / 2
            S[i, j] = s * np.exp(-np.abs(z_grid[i] - z_grid[j]) / cl_mean)

    return S
