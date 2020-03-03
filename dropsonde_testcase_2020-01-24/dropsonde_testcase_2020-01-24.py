import numpy as np

from typhon.arts import xml
from typhon.arts.workspace import Workspace
from typhon.physics import wavelength2frequency, wavenumber2frequency

import iasi_oem as ioem
from poem import oem


ws = Workspace(verbosity=1)
project_path = "/Users/mprange/PycharmProjects/iasi_retrieval/"
project_name = "dropsonde_testcase_2020-01-24"
ioem.setup_retrieval_paths(project_path, project_name)