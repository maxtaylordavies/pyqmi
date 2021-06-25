import os
import numpy as np
import jax.numpy as jnp

from src.utils.utils import normalise, repeat_stimulus, pad_stimulus, log
from src.model.model_neuron import generate_synthetic_spike_frequencies
from src.mapping.mapping import map_receptive_field, map_receptive_field_sta


# constants
data_dir = "/rds/general/user/mt4217/home/fyp/data"
num_frames, height, width, frame_history = 900, 304, 608, 1

log("loading stimulus...")

# load natural movie stimulus
stimulus = normalise(
    jnp.array(np.loadtxt(os.path.join(data_dir, "natural_movie_one.txt")))
)

stimulus = stimulus.reshape((num_frames, height, width))
# stimulus = repeat_stimulus(stimulus, num_repeats).reshape(
#     (num_frames * num_repeats, height, width)
# )

log(f"stimulus.shape = {stimulus.shape}")

# load template rfv
rfv = normalise(
    jnp.array(
        np.loadtxt(os.path.join(data_dir, "template.txt"), delimiter=",")
    ).reshape((height * width * frame_history))
)

log(f"rfv.shape = {rfv.shape}")

# generate synthetic response using stimulus and template rfv
response = generate_synthetic_spike_frequencies(
    stimulus, frame_history=frame_history, rfv=rfv, norm=False
).reshape((num_frames))

log(f"response.shape = {response.shape}")

# attempt to recover template rfv through QMI optimisation
# rfv_estimate, _, _ = map_receptive_field(stimulus, response, H=frame_history)
rfv_estimate, qmi_history = map_receptive_field(
    stimulus, response, options={"max_stimulus_repeats": 1, "smooth": True}
)

# save result
np.savetxt("./rfv_estimate.txt", rfv_estimate)
np.savetxt("./qmi_history.txt", qmi_history)
