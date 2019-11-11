import numpy as np

from typhon.arts import xml
from typhon.arts.workspace import Workspace, arts_agenda
from typhon.physics import wavelength2frequency, wavenumber2frequency

import iasi_oem as ioem
from poem import oem

ws = Workspace(verbosity=1)
project_path = "/scratch/uni/u237/users/mprange/phd/iasi_retrieval/"
project_name = "reproduce_first_eml_retrieval"
f_backend_range = np.array([[wavenumber2frequency(1190 * 100),
                             wavenumber2frequency(1400 * 100)],
                            [wavenumber2frequency(2250 * 100),
                             wavenumber2frequency(2400 * 100)]])
f_backend_width = 7.5e9

tropical_ocean_eresmaa_path = "/scratch/uni/u237/user_data/mprange/tropical_ocean_eresmaal137_all_q.xml"
tropical_ocean_eresmaa = xml.load(tropical_ocean_eresmaa_path)
garand_path = "/scratch/uni/u237/users/mprange/arts/controlfiles/testdata/garand_profiles.xml.gz"
garand = xml.load(garand_path)
z = garand[0][1][:, 0, 0]
p = garand[0].grids[1]

oem.save_covariances(z=z, p=p, abs_species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252", "T"])

Sa_T = xml.load(project_path + project_name +
                "/a_priori/covariance_T.xml")
Sa_h2o = xml.load(project_path + project_name +
                  "/a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml")

ioem.setup_retrieval_paths(project_path, project_name)
ws = ioem.load_generic_settings(ws)
ws = ioem.setup_sensor(ws, f_backend_range, f_backend_width)
ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=garand_path,
                          f_ranges=f_backend_range)
ws = ioem.iasi_observation(ws,
                           atm_batch_path=garand_path,
                           n_atmospheres=42,
                           f_ranges=f_backend_range,)
                           #iasi_obs_path="/scratch/uni/u237/users/mprange/phd/iasi_retrieval/"
                           #              "iasi_obs_tropical_ocean_eresmaal137_all_q/"
                           #              "tropical_ocean_eresmaal137_all_q_1190_1192_1300_1302_cm-1.xml")
ws = ioem.setup_oem_retrieval(ws,
                              a_priori_atm_batch_path=garand_path,
                              a_priori_atm_index=2,
                              cov_h2o_vmr=Sa_h2o,
                              cov_t=Sa_T,
                              retrieval_quantities=["Temperature", "H2O"])

ws = ioem.oem_retrieval(ws,
                        ybatch_indices=np.arange(1, 2),
                        inversion_method="lm",
                        max_iter=10,
                        gamma_start=3000)
