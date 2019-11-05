import numpy as np

from typhon.arts import xml
from typhon.arts.workspace import Workspace, arts_agenda
from typhon.physics import wavelength2frequency, wavenumber2frequency

import iasi_oem as ioem

ws = Workspace(verbosity=1)
f_backend_range = np.array([[wavenumber2frequency(1190 * 100),
                             wavenumber2frequency(1192 * 100)],
                            [wavenumber2frequency(1300 * 100),
                             wavenumber2frequency(1302 * 100)]])

f_backend_width = 7.5e9
tropical_ocean_eresmaa = "/scratch/uni/u237/user_data/mprange/tropical_ocean_eresmaal137_all_q.xml"
Sa_T = xml.load("a_priori/covariance_T.xml")
Sa_h2o = xml.load("a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml")


ws = ioem.load_generic_settings(ws)
ws = ioem.setup_sensor(ws, f_backend_range, f_backend_width)
ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=tropical_ocean_eresmaa,
                          f_ranges=f_backend_range,
                          abs_lookup_table_path="a_priori/abs_lookup_tropical_ocean_eresmaal137_"
                                                "all_q_1190_1192_1300_1302_cm-1.xml")
ws = ioem.iasi_observation(ws,
                           atm_batch_path=tropical_ocean_eresmaa,
                           n_atmospheres=100,
                           f_ranges=f_backend_range,
                           iasi_obs_path="/scratch/uni/u237/users/mprange/phd/iasi_retrieval/"
                                         "iasi_obs_tropical_ocean_eresmaal137_all_q/"
                                         "tropical_ocean_eresmaal137_all_q_1190_1192_1300_1302_cm-1.xml")
ws = ioem.setup_oem_retrieval(ws,
                              a_priori_atm_batch_path=tropical_ocean_eresmaa,
                              a_priori_atm_index=1,
                              cov_h2o_vmr=Sa_h2o,
                              cov_t=Sa_T,
                              retrieval_quantities=["Temperature", "H2O"])

retrieved_h2o, retrieved_t = \
    ioem.oem_retrieval(ws,
                       ybatch_indices=np.arange(2, 3),
                       inversion_method="lm",
                       max_iter=2,
                       gamma_start=3000)