import numpy as np

from typhon.arts import xml
from typhon.arts.workspace import Workspace, arts_agenda
from typhon.physics import wavelength2frequency, wavenumber2frequency

import iasi_oem as ioem
from poem import oem


ws = Workspace(verbosity=1)
project_path = "/scratch/uni/u237/users/mprange/phd/iasi_retrieval/"
project_name = "surface_temperature_retrieval"
ioem.setup_retrieval_paths(project_path, project_name)

f_backend_range = np.array([])
f_backend_width = wavenumber2frequency(25.)
full_spec = np.arange(wavenumber2frequency(645.0 * 100),
                      wavenumber2frequency(2760.0 * 100),
                      f_backend_width)

garand_path = "/scratch/uni/u237/users/mprange/arts/controlfiles/testdata/garand_profiles.xml.gz"
garand = xml.load(garand_path)
z = garand[1][1][:, 0, 0]
p = garand[1].grids[1]

#oem.save_covariances(z=z, p=p, abs_species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252", "T"])
Sa_T = xml.load(project_path + project_name +
                "/a_priori/covariance_T.xml")
Sa_h2o = xml.load(project_path + project_name +
                  "/a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml")

ws = ioem.load_generic_settings(ws, py_surface_agenda=False)
ws = ioem.setup_sensor(ws, f_backend_width, f_ranges=None,
                       add_frequencies=full_spec[np.array([1026, 1190, 1193, 1270, 1883])])
ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=garand_path,
                          f_ranges=f_backend_range,
                          abs_lookup_path="/scratch/uni/u237/users/mprange/phd/iasi_retrieval/abs_lookup_tables/"
                                          "abs_lookup_garand_profiles_645_2760_cm-1.xml")

ws = ioem.iasi_observation(ws,
                           atm_batch_path=garand_path,
                           n_atmospheres=7,
                           f_ranges=f_backend_range,
                           )

ws = ioem.setup_oem_retrieval(ws,
                              a_priori_atm_batch_path=garand_path,
                              a_priori_atm_index=4,
                              cov_h2o_vmr=Sa_h2o,
                              cov_t=Sa_T,
                              retrieval_quantities=["t_surface"],)

ws = ioem.oem_retrieval(ws,
                        ybatch_indices=np.array([6]),
                        inversion_method="lm",
                        max_iter=7,
                        gamma_start=1000.0,
                        gamma_inc_factor=2.0,
                        gamma_dec_factor=2.0)


f_backend_range = np.array([[wavenumber2frequency(1190. * 100),
                            wavenumber2frequency(1400. * 100)],
                            [wavenumber2frequency(2150. * 100),
                            wavenumber2frequency(2400. * 100)]
                            ])
ws = ioem.setup_sensor(ws, f_backend_width, f_ranges=f_backend_range,
                       add_frequencies=full_spec[np.array([1026, 1190, 1193, 1270, 1883])])
ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=garand_path,
                          f_ranges=f_backend_range,
                          abs_lookup_path="/scratch/uni/u237/users/mprange/phd/iasi_retrieval/abs_lookup_tables/"
                                          "abs_lookup_garand_profiles_645_2760_cm-1.xml")
ws = ioem.iasi_observation(ws,
                           atm_batch_path=garand_path,
                           n_atmospheres=7,
                           f_ranges=f_backend_range,
                           )
ws = ioem.setup_oem_retrieval(ws,
                              a_priori_atm_batch_path=garand_path,
                              a_priori_atm_index=4,
                              cov_h2o_vmr=Sa_h2o,
                              cov_t=Sa_T,
                              retrieval_quantities=["Temperature", "H2O"])


t_retrieved = xml.load(f"{project_path}{project_name}/retrieval_output/retrieved_temperature.xml")
ws.t_field.value[:, 0, 0] = t_retrieved[0]

ws = ioem.oem_retrieval(ws,
                        ybatch_indices=np.array([6]),
                        inversion_method="lm",
                        max_iter=20,
                        gamma_start=1000.0,
                        gamma_inc_factor=2.0,
                        gamma_dec_factor=2.0)