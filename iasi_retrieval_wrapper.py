import numpy as np

from typhon.arts.workspace import Workspace, arts_agenda
from typhon.physics import wavelength2frequency, wavenumber2frequency

import iasi_oem as ioem

ws = Workspace(verbosity = 1)
f_backend_range = np.array([[wavenumber2frequency(1190*100),
                             wavenumber2frequency(1192*100)],
                            [wavenumber2frequency(1300*100),
                             wavenumber2frequency(1302*100)]])

f_backend_width = 7.5e9

ws = ioem.load_generic_settings(ws)
ws = ioem.setup_sensor(ws, f_backend_range, f_backend_width)
abs_lookup_atm_batch = "/scratch/uni/u237/user_data/mprange/tropical_ocean_eresmaal137_all_q.xml"
ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=abs_lookup_atm_batch,
                          f_ranges=f_backend_range,
                          abs_lookup_table_path="a_priori/abs_lookup_tropical_ocean_eresmaal137_"
                                                "all_q_1190_1192_1300_1302_cm-1.xml")
ws = ioem.iasi_obs_batch_simulation(ws,
                                    atm_batch_path="/scratch/uni/u237/user_data/mprange/"
                                                   "tropical_ocean_eresmaal137_all_q.xml",
                                    n_atmospheres=100,
                                    f_ranges=f_backend_range)

