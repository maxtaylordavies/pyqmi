import numpy as np
import jax.numpy as jnp

from src.utils import flatten_stimulus, pad_stimulus, normalise

_square = lambda x: x ** 2

_exp = lambda x: np.exp(x)

_exp_of_square = lambda x: _exp(_square(x))

_half_wave_rectification_squared = lambda x: _square(max(x, 0))


def generate_synthetic_spike_frequencies(
    stimulus, rfv, frame_history=1, bin_width=1, nonlinear_func=None, norm=True
):
    # if no nonlinear function supplied, use default square
    if nonlinear_func is None:
        nonlinear_func = _square

    # flatten stimulus to set of vectors each representing one frame
    flattened_stimulus = pad_stimulus(flatten_stimulus(stimulus), frame_history)

    # apply linear filter (template receptive field) to flattened stimulus
    linear_filter_output = jnp.matmul(flattened_stimulus, jnp.transpose(rfv))

    # pass output through some nonlinear function to obtain bin-wise spiking frequencies (in Hz)
    frequencies = nonlinear_func(linear_filter_output)

    # number of spikes per bin is then frequency (Hz) * bin width (s)
    spike_counts = frequencies * bin_width

    # normalise
    return normalise(spike_counts) if norm else spike_counts
