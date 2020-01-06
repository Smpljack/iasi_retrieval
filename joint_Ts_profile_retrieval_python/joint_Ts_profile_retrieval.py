import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

from typhon.arts import xml
from typhon.arts.workspace import Workspace, arts_agenda
from typhon.physics import wavelength2frequency, wavenumber2frequency
from typhon.plots import profile_p


import iasi_oem as ioem
from poem import oem

# Project setup
ws = Workspace(verbosity=1)
project_path = "/scratch/uni/u237/users/mprange/phd/iasi_retrieval/"
project_name = "joint_Ts_profile_retrieval_python"
ioem.setup_retrieval_paths(project_path, project_name)

# frequency ranges for sensor setup
f_backend_range = np.array([[wavenumber2frequency(1190. * 100),
                            wavenumber2frequency(1400. * 100)],
                            [wavenumber2frequency(2150. * 100),
                            wavenumber2frequency(2400. * 100)]
                            ])
f_backend_width = wavenumber2frequency(25.)
full_spec = np.arange(wavenumber2frequency(645.0 * 100),
                      wavenumber2frequency(2760.0 * 100),
                      f_backend_width)
surface_channels = full_spec[np.array([1026, 1190, 1193, 1270, 1883])]

# atmospheric data
garand_path = "/scratch/uni/u237/users/mprange/arts/controlfiles/testdata/garand_profiles.xml.gz"
garand_eml_path = "/scratch/uni/u237/user_data/mprange/garand_second_atm_eml_disturbed.xml"
garand_eml = xml.load(garand_eml_path)
garand = xml.load(garand_path)
z = garand[1][1][:, 0, 0]
p = garand[1].grids[1]

# Covariances
#oem.save_covariances(z=z, p=p, abs_species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252", "T"])
Sa_T = xml.load(project_path + project_name +
                "/a_priori/covariance_T.xml")
Sa_h2o = xml.load(project_path + project_name +
                  "/a_priori/covariance_H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252.xml")

# Setup generic agendas
ws = ioem.load_generic_settings(ws)
# Sensor setup
ws = ioem.setup_sensor(ws, f_backend_width, f_ranges=f_backend_range,
                       add_frequencies=surface_channels)
# Absorption lookup table
ws = ioem.load_abs_lookup(ws,
                          atm_batch_path=garand_path,
                          f_ranges=f_backend_range,
                          abs_lookup_path="/scratch/uni/u237/users/mprange/phd/iasi_retrieval/abs_lookup_tables/"
                                          "abs_lookup_garand_profiles_645_2760_cm-1.xml")

# Forward simulation for synthetic observation
ws = ioem.iasi_observation(ws,
                           atm_batch_path=garand_eml_path,
                           n_atmospheres=2,
                           f_ranges=f_backend_range,
                           )
# Setup of the OEM retrieval, including a priori state
ws = ioem.setup_oem_retrieval(ws,
                              a_priori_atm_batch_path=garand_path,
                              a_priori_atm_index=1,
                              cov_t_surface=np.array([[50.]]),
                              cov_y_t_surface=0.1 ** 2 * np.diag(np.ones(ws.f_backend.value.shape)),
                              retrieval_quantities=["t_surface"])

# Forward simulation for a priori state, including Jacobians
ws.jacobianInit()
#ws.jacobianAddAbsSpecies(species="H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
#                         unit="rel",
#                         g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
ws.jacobianAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
ws.jacobianClose()
ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
ws.yCalc()
# Transform relative Jacobian (ln-space) to log10 space
#ws.jacobian.value[:, :43] = ws.jacobian.value[:, :43] * np.log(10)
# Add t_surface jacobian
Ts_jacobian = np.array([(ws.transmittance.value * (1 - ws.surface_scalar_reflectivity.value))]).T
ws.jacobian = np.concatenate([ws.jacobian.value, Ts_jacobian], axis=1)

xa = np.concatenate([ws.t_field.value[:, 0, 0], ws.t_surface.value[0, :]],
                    axis=0)
Sx = np.block(
    [
        [Sa_h2o, np.zeros(Sa_h2o.shape), np.zeros((43, 1))],
        [np.zeros(Sa_h2o.shape), Sa_T, np.zeros((43, 1))],
        [np.zeros((1, 43)), np.zeros((1, 43)), ws.cov_t_surface.value]
    ])

y_obs = xml.load("observations/garand_second_atm_eml_disturbed_1190_1400_2150_2400_cm-1.xml")[1]


def linear_retrieval(xa, y, ya, K, Sy, Sa):
    return xa + inv(K.T @ inv(Sy) @ K + inv(Sa)) @ K.T @ inv(Sy) @ (y - ya)

x_retrieved = linear_retrieval(xa, y_obs, ws.y.value, ws.jacobian.value, ws.cov_y_t_surface.value, Sx)

profile_p(p, 10**xa[:43])
profile_p(p, 10**x_retrieved[:43])
profile_p(p, garand_eml[1][2][:,0,0])
plt.show()

profile_p(p, xa[43:86])
profile_p(p, x_retrieved[43:86])
profile_p(p, garand_eml[1][0][:,0,0])
plt.scatter(x_retrieved[-1], p[0], color="red")
plt.show()

plt.show()