
from cauchy import SemiCircular; sc = SemiCircular()

import numpy as np
a = 5*np.ones(36, np.complex)
for i in range(18):
    a[i] =4.
A = np.diag(a)
sc.plot_density_info_plus_noise( A,1)
