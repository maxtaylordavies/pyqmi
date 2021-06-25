import os
import numpy as np
import jax.numpy as jnp

from src.utils import log
from src.mapping.mapping import _estimate_rf

data_dir = "/rds/general/user/mt4217/home/fyp/data"

stimulus = jnp.array(np.loadtxt(os.path.join(data_dir, "matlab_stimulus.txt")))
log(f"stimulus.shape = {stimulus.shape}")

response = jnp.array(np.loadtxt(os.path.join(data_dir, "matlab_counts.txt")))
log(f"response.shape = {response.shape}")

rfv, qmi_history, rfv_history = _estimate_rf(stimulus, response, H=5)
log(f"RF estimation complete")

np.savetxt("./rfv.txt", rfv)
np.savetxt("./qmi_history.txt", qmi_history)
np.savetxt("./rfv_history.txt", rfv_history)
