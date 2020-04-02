import numpy as np
import sys
from typhon.arts import xml
from typhon.arts.workspace import Workspace
from typhon.physics import wavelength2frequency, wavenumber2frequency

import iasi_oem as ioem
from poem import oem

# Provide start index of ybatch atmosphere and number of subsequent atmsospheres to retrieve in
# first and second command line arguments.
arg_list = sys.argv
if len(arg_list) == 3:
    ybatch_start = int(arg_list[1])
    ybatch_n = int(arg_list[2])
elif len(arg_list) == 2:
    ybatch_n = int(arg_list[1])
elif len(arg_list) == 1:
    ybatch_start = 2  # If not provided in command line, set here.
    ybatch_n = 1  # If not provided in command line, set here.

ws = Workspace(verbosity=1)
project_path = "/Users/mprange/PycharmProjects/iasi_retrieval/"
project_name = "eresmaa_retrieval"
ioem.setup_retrieval_paths(project_path, project_name)

f_backend_range = np.array([[wavenumber2frequency(1190. * 100),
                             wavenumber2frequency(1400. * 100)],
                            [wavenumber2frequency(2150. * 100),
                             wavenumber2frequency(2400. * 100)],
                            ])
f_backend_width = wavenumber2frequency(25.)
full_spec = np.arange(wavenumber2frequency(645.0 * 100),
                      wavenumber2frequency(2760.0 * 100),
                      f_backend_width)
fascod_path = '/Users/mprange/PycharmProjects/iasi_retrieval/eresmaa_retrieval/'\
              '/a_priori/garand0_eresmaa_profile.xml'
eresmaa_path = "/Users/mprange/data/tropical_ocean_clearsky_eresmaal137_all_q.xml"
eresmaa = xml.load(eresmaa_path)
z = eresmaa[1][1][:, 0, 0]
p = eresmaa[1].grids[1]

# oem.save_covariances(z=z, p=p, abs_species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252", "T"])
Sa_T = xml.load(project_path + project_name +
                "/a_priori/covariance_T.xml")
Sa_h2o = xml.load(project_path + project_name +
                  "/a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml")
Sa_cross_h2o_T = ioem.covmat_cross(Sa_h2o, Sa_T, z, corr_height=2000.)
Sa_cross_T_Ts = ioem.covmat_cross(Sa_T, np.array([[100.]]), z, corr_height=100.)
Sa_cross = [
    {'S': Sa_cross_h2o_T,
     'i': 1,
     'j': 2,
     },
    {'S': Sa_cross_T_Ts,
     'i': 0,
     'j': 1,
     },
]

ws = ioem.load_generic_settings(ws)

ws = ioem.setup_sensor(ws, f_backend_width, f_ranges=f_backend_range,
                       add_frequencies=full_spec[np.array([1026, 1190, 1193, 1270, 1883])],
                       )

ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=eresmaa_path,
                          f_ranges=f_backend_range,
                          abs_lookup_path="/Users/mprange/data/abs_lookup_tables/"
                                          "abs_lookup_tropical_ocean_eresmaal137_all_q_645_2760_cm-1.xml",
                          hitran_split_artscat5_path="/Users/mprange/data/hitran_split_artscat5"
                          )

ws = ioem.iasi_observation(ws,
                           atm_batch_path=eresmaa_path,
                           ybatch_start=ybatch_start,
                           ybatch_n=ybatch_n,
                           f_ranges=f_backend_range,
                           add_measurement_noise=True,
                           )

ws = ioem.setup_apriori_state(ws,
                              a_priori_atm_batch_path=eresmaa_path,
                              batch_ind=0,
                              )
ioem.save_z_p_grids(ws, 'a_priori')

ws = ioem.setup_retrieval_quantities(ws,
                                     retrieval_quantities=["Temperature", "H2O", "t_surface"],
                                     cov_t_surface=np.array([[100.]]),
                                     cov_t=Sa_T,
                                     cov_h2o_vmr=Sa_h2o,
                                     # cov_cross=Sa_cross,
                                     )
stratospheric_temperature = xml.load('/Users/mprange/PycharmProjects/iasi_retrieval/'
                                     'stratospheric_temperature_for_moist_adiabat.xml')
ws = ioem.retrieve_ybatch_for_a_priori_batch(ws,
                                             a_priori_atm_batch_path=eresmaa_path,
                                             retrieval_batch_indices=np.arange(ybatch_start, ybatch_start + ybatch_n),
                                             a_priori_batch_indices=np.arange(ybatch_start, ybatch_start + ybatch_n),
                                             Sa_T=Sa_T,
                                             Sa_h2o=Sa_h2o,
                                             moist_adiabat=True,
                                             t_surface_std=1.5,
                                             stratospheric_temperature=stratospheric_temperature,
                                             RH=0.8*np.ones(p.shape),
                                             inversion_method="lm",
                                             max_iter=15,
                                             gamma_start=10,
                                             gamma_dec_factor=2.0,
                                             gamma_inc_factor=2.0,
                                             gamma_upper_limit=1e20,
                                             gamma_lower_limit=1.0,
                                             gamma_upper_convergence_limit=99.0
                                             )
