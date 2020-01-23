import numpy as np

from typhon.arts import xml
from typhon.arts.workspace import Workspace, arts_agenda
from typhon.physics import wavelength2frequency, wavenumber2frequency, frequency2wavenumber
import xarray as xr

import iasi_oem as ioem
from poem import oem

ws = Workspace(verbosity=1)
project_path = "/scratch/uni/u237/users/mprange/phd/iasi_retrieval/"
project_name = "bco_eml_retrieval"
ioem.setup_retrieval_paths(project_path, project_name)

f_backend_range = np.array([[wavenumber2frequency(1190. * 100),
                            wavenumber2frequency(1400. * 100)],
                            [wavenumber2frequency(2150. * 100),
                            wavenumber2frequency(2400. * 100)]
                            ])

f_backend_width = wavenumber2frequency(25.)
full_spec = np.arange(wavenumber2frequency(645.0 * 100),
                      wavenumber2frequency(2760.0 * 100),
                      f_backend_width)

lidar_gridded_field_path = "fascod_lidar_temp_h2o_adjusted.xml"
lidar_data = xml.load(lidar_gridded_field_path)
eresmaa_path = "/scratch/uni/u237/user_data/mprange/tropical_ocean_eresmaal137_all_q.xml"
garand_path = "/scratch/uni/u237/users/mprange/arts/controlfiles/testdata/garand_profiles.xml.gz"
garand = xml.load(garand_path)
z = garand[0][1][:, 0, 0]
p = garand[0].grids[1]
# oem.save_covariances(z=z, p=p, abs_species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252", "T"])
Sa_T = xml.load(project_path + project_name +
                "/a_priori/covariance_T.xml")
Sa_h2o = xml.load(project_path + project_name +
                  "/a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml")

ws = ioem.load_generic_settings(ws)

ws = ioem.setup_sensor(ws, f_backend_width, f_ranges=f_backend_range,
                       add_frequencies=full_spec[np.array([1026, 1190, 1193, 1270, 1883])])

iasi_obs = xr.open_dataset("/scratch/uni/u237/users/mprange/phd/iasi_retrieval/bco_collocated_iasi_obs.nc")
spectral_ind = np.isin(iasi_obs["wavenumber"].values, frequency2wavenumber(ws.f_backend.value))
iasi_obs_list = list(
    ioem.radiance2planck_bt_wavenumber(
        (iasi_obs["spectral_radiance"][:, :, spectral_ind].stack(num=("along_track", "across_track"))).values.T /
         iasi_obs["scale_factor"][spectral_ind].values, iasi_obs["wavenumber"][spectral_ind].values))

ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=eresmaa_path,
                          f_ranges=f_backend_range,
                          abs_lookup_path="/scratch/uni/u237/users/mprange/phd/iasi_retrieval/abs_lookup_tables/"
                                           "abs_lookup_tropical_ocean_eresmaal137_all_q_645_2760_cm-1.xml"
                          )

ws = ioem.iasi_observation(ws,
                           iasi_obs_data=iasi_obs_list,
                           atm_batch_path="iasi_bco_collocations",
                           f_ranges=f_backend_range,
                           add_measurement_noise=False)

ws = ioem.setup_oem_retrieval(ws,
                              a_priori_atm_batch_path=garand_path,
                              a_priori_atm_index=0,
                              cov_t_surface=np.array([[100.]]),
                              cov_t=Sa_T,
                              cov_h2o_vmr=Sa_h2o,
                              retrieval_quantities=["Temperature", "H2O", "t_surface"])

ws = ioem.oem_retrieval(ws,
                        ybatch_indices=np.array([10]),
                        inversion_method="li",
                        max_iter=1,
                        gamma_start=10.0,
                        gamma_inc_factor=2.0,
                        gamma_dec_factor=2.0)